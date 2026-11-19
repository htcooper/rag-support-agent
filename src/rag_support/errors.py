"""Messages for Anthropic and OpenAI API errors, shared by the CLI and the web UI.

Claude answers the question (Anthropic); the question is embedded for search
with OpenAI, so either service can fail a request.
"""

from __future__ import annotations

import anthropic
import openai

API_ERRORS = (anthropic.APIError, openai.APIError)


def describe_api_error(exc: Exception) -> str | None:
    """Return a message for an API error, or None for any other error.

    Subclasses are checked before their parents: RateLimitError,
    AuthenticationError, and PermissionDeniedError are all APIStatusErrors.
    """
    if isinstance(exc, anthropic.AuthenticationError):
        return "Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env."
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"The API key can't use this model or feature: {exc.message}"
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited or spend cap reached. Try again later."
    if isinstance(exc, anthropic.APIStatusError):
        return f"Anthropic API error {exc.status_code}: {exc.message}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check the network connection."

    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env."
    if isinstance(exc, openai.PermissionDeniedError):
        return f"The OpenAI key can't use the embedding model: {exc.message}"
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached. Try again later."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI API error {exc.status_code}: {exc.message}"
    if isinstance(exc, openai.APIConnectionError):
        return "Couldn't reach the OpenAI API. Check the network connection."
    return None


def http_status_for(exc: Exception) -> int:
    """HTTP status the web UI returns for an API error that has a message."""
    if isinstance(exc, (anthropic.RateLimitError, openai.RateLimitError)):
        return 429
    if isinstance(exc, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return 503
    return 502  # the upstream service rejected or failed the request
