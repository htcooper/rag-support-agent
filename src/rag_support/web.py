"""Local web UI: one page and one endpoint around answer_question.

Run it with ``python -I -m rag_support ui``, which binds to 127.0.0.1 only.
The endpoints return everything the developer view shows; the page decides
what the end-user view hides. /api/ask returns one JSON answer; /api/ask/stream
sends the same answer as server-sent events, with filtered text snapshots first.
Traces are sent to Langfuse after the answer, off the user's wait.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from importlib.resources import files
from typing import Annotated, Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

# _hit_summary stays private: answer.py is fingerprinted by the eval harness gate,
# so renaming it would force a harness re-approval for no behavior change.
from rag_support.answer import Answer, answer_question
from rag_support.answer import _hit_summary as hit_summary
from rag_support.errors import API_ERRORS, describe_api_error, http_status_for
from rag_support.retrieve import SupportsRetrieval
from rag_support.stream import FilteredText, StreamingClient
from rag_support.tracing import Tracer, cost_details

logger = logging.getLogger(__name__)

# Caps the cost of a single request: a question is a few hundred tokens at most.
MAX_QUESTION_CHARS = 1000


class AskRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS
        ),
    ]
    rewrite: bool = True
    strict: bool = True


def answer_cost(answer: Answer) -> float | None:
    """Dollar cost of the rewrite and answer calls, or None if a price is unknown.

    The query embedding is left out: it costs under $0.00001.
    """
    calls = [(answer.model, answer.usage)]
    if answer.rewrite is not None:
        calls.append((answer.rewrite.model, answer.rewrite.usage))
    total = 0.0
    for model, usage in calls:
        cost = cost_details(
            model, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        )
        if not cost:
            return None
        total += cost["total"]
    return round(total, 4)


def answer_payload(answer: Answer, latency_s: float) -> dict[str, Any]:
    cited = set(answer.cited_ranks)
    notes = []
    if answer.fallback_used:
        notes.append("served by fallback model")
    if answer.invalid_citations:
        notes.append(f"invalid citations dropped: {answer.invalid_citations}")
    if answer.links_removed:
        notes.append(f"links removed: {answer.links_removed}")
    return {
        "question": answer.question,
        "text": answer.text,
        "grounded": answer.grounded,
        "refused": answer.refused,
        "citations": [
            {
                "rank": c.rank,
                "heading": c.chunk.heading,
                "url": c.chunk.url,
                "cited_text": c.cited_text,
            }
            for c in answer.citations
        ],
        "hits": [
            {**hit_summary(rank, hit), "cited": rank in cited}
            for rank, hit in enumerate(answer.hits, 1)
        ],
        "rewrite": answer.rewrite.text if answer.rewrite is not None else None,
        "model": answer.model,
        "stop_reason": answer.stop_reason,
        "usage": answer.usage,
        "cost_usd": answer_cost(answer),
        "latency_s": round(latency_s, 1),
        "notes": notes,
        "trace_url": answer.trace_url,
    }


def sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


def create_app(
    retriever: SupportsRetrieval,
    client: Any,
    tracer_factory: Callable[[], Tracer],
) -> FastAPI:
    """Build the app. Dependencies are passed in so tests can use fakes."""
    # No interactive API docs: the page is the only interface.
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    page = files("rag_support").joinpath("static/index.html").read_text("utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    # A plain def, so FastAPI runs the blocking pipeline in its threadpool.
    @app.post("/api/ask")
    def ask(request: AskRequest, background: BackgroundTasks) -> dict[str, Any]:
        tracer = tracer_factory()
        start = time.perf_counter()
        try:
            answer = answer_question(
                request.question,
                retriever,
                client,
                tracer=tracer,
                rewrite=request.rewrite,
                strict=request.strict,
            )
        except API_ERRORS as e:
            tracer.flush()  # send the trace even when the call failed
            message = describe_api_error(e)
            if message is None:
                raise
            raise HTTPException(status_code=http_status_for(e), detail=message) from e
        except Exception:
            tracer.flush()
            raise
        # Sent after the response, so the upload isn't part of the wait.
        background.add_task(tracer.flush)
        return answer_payload(answer, time.perf_counter() - start)

    @app.post("/api/ask/stream")
    def ask_stream(request: AskRequest) -> StreamingResponse:
        """Server-sent events, each a JSON object with a "type":

        - status: {"stage": "searching" | "thinking"}
        - text: {"text": ...}, the answer so far, link-filtered; replaces the last
        - answer: {"answer": ...}, the validated payload /api/ask returns
        - error: {"status": ..., "detail": ...}, in place of an answer
        """
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        threading.Thread(
            target=stream_answer, args=(request, events), daemon=True
        ).start()

        def body() -> Iterator[str]:
            yield sse({"type": "status", "stage": "searching"})
            while (event := events.get()) is not None:
                yield sse(event)

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    def stream_answer(
        request: AskRequest, events: queue.Queue[dict[str, Any] | None]
    ) -> None:
        tracer = tracer_factory()
        text = FilteredText()

        def on_text(delta: str) -> None:
            snapshot = text.add(delta)
            if snapshot is not None:
                events.put({"type": "text", "text": snapshot})

        streaming = StreamingClient(
            client,
            on_text,
            on_generate=lambda: events.put({"type": "status", "stage": "thinking"}),
        )
        start = time.perf_counter()
        try:
            answer = answer_question(
                request.question,
                retriever,
                streaming,
                tracer=tracer,
                rewrite=request.rewrite,
                strict=request.strict,
            )
            payload = answer_payload(answer, time.perf_counter() - start)
            events.put({"type": "answer", "answer": payload})
        except Exception as e:  # noqa: BLE001 - the stream must always end
            message = describe_api_error(e) if isinstance(e, API_ERRORS) else None
            if message is None:
                logger.exception("streamed answer failed")
                events.put({"type": "error", "status": 500, "detail": None})
            else:
                status = http_status_for(e)
                events.put({"type": "error", "status": status, "detail": message})
        finally:
            # The page has its answer by now; the upload is off the wait.
            tracer.flush()
            events.put(None)

    return app
