"""LLM provider registry.

Built-in providers are registered automatically when this package is imported.
Add a new provider by creating ``jeerag/llm/providers/<name>.py`` with a class
``<Name>Provider(LLMProvider)`` whose ``name`` attribute matches — it will be
auto-discovered and registered here.
"""
from __future__ import annotations

from ..base import LLMProvider, LLMProviderError

_PROVIDER_CLASSES: dict[str, type[LLMProvider]] = {}


def register(cls: type[LLMProvider]) -> type[LLMProvider]:
    """Register a provider class (decorator or manual call)."""
    _PROVIDER_CLASSES[cls.name] = cls
    return cls


def get(name: str) -> type[LLMProvider]:
    if name not in _PROVIDER_CLASSES:
        available = ", ".join(sorted(_PROVIDER_CLASSES)) or "<none>"
        raise KeyError(
            f"Unknown LLM provider '{name}'. Available: {available}"
        )
    return _PROVIDER_CLASSES[name]


def list_providers() -> list[str]:
    return sorted(_PROVIDER_CLASSES)


# ---- Built-in providers (auto-registered on import) ----------------------
from .anthropic import AnthropicProvider  # noqa: F401
from .openai_compat import OpenAICompatibleProvider  # noqa: F401

register(OpenAICompatibleProvider)   # "openai" — OpenAI, opencode2api, Groq, Together, llama.cpp
register(AnthropicProvider)          # "anthropic" — Claude models
