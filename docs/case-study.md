# Case study: measuring a retrieval fix in a traced RAG agent

## The question

Does restating a user's question in the documentation's vocabulary before searching make a FastAPI support agent answer better, and by how much? The answer had to come from measurement: a written prediction, a held-out set, repeated runs, and confidence intervals.

## The eval

39 cases, each graded automatically:

| Case type | n | Passes when |
|---|---|---|
| Should answer, dev | 22 | An expected page is retrieved and cited, and the judge finds every claim supported and the question answered |
| Should answer, held-out | 10 | Same; never used to design a change, so it tests whether gains carry over to new questions |
| Should decline | 5 | The agent declines (Django admin, CSS, Excel, and similar) |
| Prompt injection | 2 | A planted malicious chunk is ranked first; the agent ignores it and nothing from it reaches the answer |

The planted chunk skips ingest screening on purpose, so the injection cases test the last layer of defense: the model and the output filter.

The judge is Claude Sonnet 5.5 with structured output. It sees the question, the same document blocks the agent saw, and the answer. Correct details from the model's own knowledge count as unsupported: a support agent that cites the docs should only say what the docs say. Before it graded anything, the judge was checked on three known-bad answers (empty, "I don't know", and an answer to a different question) and rejected all three.

The runner refuses to run if any eval code or grading file changed since it was last approved, so a result can't be produced by a quietly edited grader.

## Baseline

Opus 5.5 answering from the top 5 chunks, two runs per case: **74% pass**. 
- 11 cases failed at least once
- 8 of them were retrieval misses on both runs: no expected page in the top 5.

The misses shared a pattern. The questions used the user's words, and the docs used feature names:

| Question | Page that answers it |
|---|---|
| "My React app on localhost:3000 gets blocked when it calls my FastAPI backend." | CORS (`CORSMiddleware`) |
| "How do I send a large file without loading the whole thing into memory?" | `StreamingResponse` |
| "My API is returning the user's hashed password in the response." | `response_model` |

## The change and the prediction

To close that vocabulary gap, v1 adds one step before the search. Claude Haiku 5.5 restates the question as a one-line query in the docs' terms; it's a short, simple task, so the smallest model is enough, at about $0.00004 and 1.5 seconds per question. The search embeds the question plus that query rather than the query alone, so the user's own words still count, and a failed rewrite falls back to the question. Opus still receives only the original question, so a bad rewrite can't change what gets answered, and the answer step stays identical to the baseline's, which keeps the comparison about retrieval alone. The rewriter's instructions name no FastAPI features, and a test enforces this, so the rewriter is never fed the eval's answers.

Prediction, written before the run: dev Hit@5 from 17/22 to at least 20/22, and dev pass rate from 68% to about 82%.

## Retrieval first, for $0.0015

Retrieval doesn't need answers or a judge, so it was measured first across the 32 should-answer questions plus a wording set: 5 topics, each asked three ways.

| Queries | Hit@5 before | Hit@5 after |
|---|---|---|
| Should answer, dev (22) | 17/22 | 22/22 |
| Should answer, held-out (10) | 7/10 | 9/10 |
| Wording: docs' own words (5) | 5/5 | 5/5 |
| Wording: user's words (5) | 2/5 | 5/5 |
| Wording: vague (5) | 2/5 | 4/5 |

12 queries were fixed and none broke. Without rewriting, retrieval fell from 5/5 to 2/5 as wording moved away from the docs' terms; with it, retrieval stayed at 5, 5, and 4. The [retrieval page](../evals/retrieval.md) lists each rewrite.

## A grader bug, found by reading failures

The first full v1 run marked `live-updates` unsupported because the answer linked the Server-Sent Events page, a URL the judge said appeared in no document. The URL was in the documents: each one starts with a `Source: <url>` line. The agent's document blocks had that line, and the judge's copy didn't, because the two were built by separate code.

The fix builds the judge's documents from the same blocks sent to the agent, and a regression test checks that every field the agent sees (title, source line, and text) reaches the judge. Results from the old judge were archived, not mixed in, and both variants were re-run from scratch.

## Results

Two runs per case for each variant. Each case's runs are averaged first, and intervals are computed over cases, because two runs of one question aren't independent.

| Cases | Baseline | v1 | Change (95% CI) |
|---|---|---|---|
| All 39 | 74% | 87% | +13 pts (−1 to +27) |
| Should answer, dev (22) | 68% | 89% | +20 pts (−2 to +42) |
| Should answer, held-out (10) | 70% | 75% | +5 pts (−22 to +32) |
| Should decline (5) | 100% | 100% | 0 |
| Prompt injection (2) | 100% | 100% | 0 |
| Retrieval Hit@5 (32) | 75% | 95% | +20 pts (+6 to +34) |

- **Retrieval improved clearly.** The interval excludes zero, and the gain held on the held-out questions.
- **The dev pass rate passed the prediction,** 89% against a predicted 82%. The pass-rate intervals still include zero, so with 39 cases the answer-level gain is likely but not proven.
- **Cost and speed:** $0.0331 to $0.0345 per answer, with median latency 5.9 s to 6.8 s. The rewrite itself costs about $0.00004.
- **Noise:** 37 of 39 baseline cases and 33 of 39 v1 cases got the same result on both runs. The rest flipped between runs of identical code, so a single run of one case is weak evidence.

## What's left

With retrieval fixed, the main failure moved to the answer. In 6 of v1's 8 imperfect cases the right page was retrieved, and the answer added a small detail the excerpts don't state: "check `content_type` yourself," "the `finally` block makes sure `close()` runs," "the threadpool hop has a cost." Whether or not each detail is correct, it isn't in the retrieved docs, so under this eval's rule it counts as unsupported.

The other two are retrieval misses: `login-form` on both runs and `nested-list` on one.

Next steps, in order:

1. Tighten the answer prompt against adding outside details, with the expected effect written first, and measure it the same way.
2. Look at `login-form` retrieval: the question is about a login form, and all 5 results come from the OAuth2 and HTTP Basic security pages instead of the form-data page, even though the rewrite names `Form` parameters.
3. Add cases. At 39, the interval on the overall change is about ±14 points; about 100 cases would narrow it to roughly ±9.
