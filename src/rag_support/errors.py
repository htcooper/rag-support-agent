"""Messages for Anthropic API errors, shared by the CLI and the web UI."""

from __future__ import annotations

import anthropic


def describe_api_error(exc: Exception) -> str | None:
    """Return a message for an Anthropic API error, or None for any other error.

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
    return None
