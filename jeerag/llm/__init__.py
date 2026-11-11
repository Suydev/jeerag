"""jeerag LLM provider layer — dependency-free, provider-based.

Architecture (four layers, per strongdm/unified-llm-spec):

  Layer 4: High-Level API      stream_chat(), complete(), configured(), health()
  Layer 3: Core Client         LLMManager — provider routing, key rotation
  Layer 2: Provider Utilities  HTTP client helpers, SSE parsing, retry logic
  Layer 1: Provider Specification — abstract interfaces below

Each provider adapter uses the provider's NATIVE API (not a lowest-common
denominator compatibility layer). This preserves access to provider-specific
features like reasoning tokens, extended thinking, prompt caching, etc.

Configuration (config.json / env ``JEERAG_*``)::

    # Single provider
    llm_provider   : "openai" | "anthropic" | "opencode2api" | ...
    llm_base_url   : endpoint base URL
    llm_model      : model id
    llm_api_key    : single key or comma-separated list
    llm_keys       : list of keys rotated round-robin on auth errors
    temperature    : default 0.2
    timeout        : 120
    connect_timeout: 10
    read_timeout   : 60
    first_event_timeout: 15

    # Multi-LLM ensemble ("RAG of LLMs")
    llm_providers  : list of {provider, base_url, model, key, temperature}

Usage::

    from jeerag.llm import stream_chat, complete, configured, health
    from jeerag.llm.manager import llm_manager

    mgr = llm_manager()                        # builds from config
    answer = mgr.complete([{"role": "user", "content": "hi"}])
    for delta in mgr.stream_chat([...], 0.2):
        print(delta, end="")
"""
from __future__ import annotations

import time
from typing import Iterator

from .manager import LLMManager, LLMManagerError, llm_manager, reset_manager

__all__ = [
    "LLMManager",
    "LLMManagerError",
    "llm_manager",
    "reset_manager",
    "stream_chat",
    "complete",
    "image_part",
    "configured",
    "health",
]


# Module-level convenience functions backed by the singleton manager.
# Existing code (caption.py, formulas.py, webapp.py, advanced_retrieve.py,
# llm_ensemble.py) calls these directly; they delegate to the manager.


def stream_chat(messages, cfg=None, temperature=0.2, max_tokens=None):
    mgr = llm_manager(cfg)
    return mgr.stream_chat(messages, temperature=temperature, max_tokens=max_tokens)


def complete(messages, cfg=None, temperature=0.1, max_tokens=512):
    mgr = llm_manager(cfg)
    return mgr.complete(messages, temperature=temperature, max_tokens=max_tokens)


def image_part(png_bytes: bytes, mime="image/png") -> dict:
    import base64

    b64 = base64.b64encode(png_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def configured(cfg=None) -> bool:
    mgr = llm_manager(cfg)
    return mgr.configured()


def health(cfg=None) -> dict:
    mgr = llm_manager(cfg)
    return mgr.health()
