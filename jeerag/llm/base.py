"""Abstract base classes for LLM providers.

Defines the provider-agnostic interface that every adapter implements:

    class ProviderAdapter:
        name: str                      # "openai", "anthropic", ...
        configured() -> bool
        health() -> dict               # {ok, reason, model, ms}
        stream_chat(messages, temperature, max_tokens) -> Iterator[str]
        complete(messages, temperature, max_tokens) -> str

Application code never imports a specific provider — it goes through
``LLMManager`` which routes by provider name / model string.
"""
from __future__ import annotations

from typing import Iterator


class LLMProviderError(RuntimeError):
    """Raised by a provider when a request fails."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class LLMProvider:
    """Base class for an LLM provider adapter.

    Every provider uses its NATIVE API (OpenAI Responses/Chat Completions,
    Anthropic Messages, etc.) — never a lowest-common-denominator shim.
    """

    name: str = "base"

    # --- Configuration -------------------------------------------------

    def configured(self) -> bool:
        """Return True if this provider has the credentials/endpoint it needs."""
        raise NotImplementedError

    def health(self) -> dict:
        """Cheap liveness check. Returns {ok, reason, model, ms}."""
        raise NotImplementedError

    # --- Generation ----------------------------------------------------

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield content deltas from a streaming endpoint."""
        raise NotImplementedError

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Non-streaming completion (default drains stream_chat)."""
        return "".join(self.stream_chat(messages, temperature, max_tokens)).strip()
