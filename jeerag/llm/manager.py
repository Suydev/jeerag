"""LLM manager — builds providers from config, handles key rotation & fallback.

Config (config.json / env ``JEERAG_*``)::

    # Single provider (one key or many)
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

    from jeerag.llm.manager import llm_manager
    mgr = llm_manager()                 # builds from config.json
    answer = mgr.complete([...])
    for delta in mgr.stream_chat([...], 0.2):
        print(delta, end="")
    health = mgr.health()
"""
from __future__ import annotations

import time
from typing import Iterator

from .base import LLMProvider, LLMProviderError
from .config import _keys
from .providers import get, register


class LLMManagerError(RuntimeError):
    pass


class LLMManager:
    """Manages one or more LLM providers with key rotation.

    With a single provider: rotates through ``llm_keys`` on auth errors.
    With multiple providers (ensemble): tries each provider in order on
    failure; ``chat`` returns the first successful answer.
    """

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers
        self._cursor = 0  # round-robin cursor for key rotation

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "LLMManager":
        from .. import config as _config

        cfg = cfg or _config.load()

        providers: list[LLMProvider] = []

        # Multi-provider ensemble ("RAG of LLMs")
        raw_providers = cfg.get("llm_providers")
        if raw_providers:
            for p in raw_providers:
                prov_name = p.get("provider") or "openai"
                cls = get(prov_name)
                providers.append(cls(p))
        else:
            # Single provider — one entry per key for clean rotation
            prov_name = cfg.get("llm_provider") or "openai"
            base_url = cfg.get("llm_base_url") or ""
            model = cfg.get("llm_model") or ""
            key_list = _keys(cfg) or [""]

            cls = get(prov_name)
            for key in key_list:
                providers.append(cls({
                    "provider": prov_name,
                    "base_url": base_url,
                    "model": model,
                    "key": key,
                    "temperature": float(cfg.get("llm_temperature", 0.2)),
                    "timeout": int(cfg.get("llm_timeout", 120)),
                    "connect_timeout": int(
                        cfg.get("llm_connect_timeout", 10)
                    ),
                    "read_timeout": int(
                        cfg.get("llm_read_timeout", 60)
                    ),
                    "first_event_timeout": int(
                        cfg.get("llm_first_event_timeout", 15)
                    ),
                }))

        if not providers:
            cls = get("openai")
            providers.append(cls({}))

        return LLMManager(providers)

    def configured(self) -> bool:
        return any(p.configured() for p in self.providers)

    def health(self) -> dict:
        """Aggregate health across all providers."""
        results = []
        any_ok = False
        for p in self.providers:
            try:
                h = p.health()
            except Exception as e:  # noqa: BLE001
                h = {"ok": False,
                     "reason": f"{type(e).__name__}: {e}",
                     "model": getattr(p, "model", "?")}
            results.append({"provider": p.name, **h})
            if h.get("ok"):
                any_ok = True
        return {"ok": any_ok, "providers": results}

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        prefer_provider: str | None = None,
    ) -> Iterator[str]:
        """Stream chat, rotating providers on failure.

        Multiple providers: tries each in turn until one yields content.
        Single provider + multiple keys: rotates keys via manager cursor.
        """
        last_err: LLMManagerError | None = None
        order = list(range(len(self.providers)))
        if prefer_provider:
            order.sort(key=lambda i: 0 if self.providers[i].name == prefer_provider else 1)

        for idx in order:
            p = self.providers[idx]
            try:
                temp = temperature if temperature is not None else float(
                    getattr(p, "temperature", 0.2) or 0.2
                )
                yielded = False
                for delta in p.stream_chat(messages, temp, max_tokens):
                    yielded = True
                    yield delta
                if yielded:
                    return  # success — stop trying other providers
            except LLMProviderError as e:
                last_err = LLMManagerError(f"provider {p.name}: {e}")
                continue
            except Exception as e:  # noqa: BLE001
                last_err = LLMManagerError(
                    f"provider {p.name}: {type(e).__name__}: {e}"
                )
                continue

        raise last_err or LLMManagerError("no provider produced output")

    def complete(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 512,
    ) -> str:
        return "".join(self.chat(messages, temperature, max_tokens)).strip()


# ---------------------------------------------------------------------------
# Module-level singleton (built lazily from config)
_manager: LLMManager | None = None


def llm_manager(cfg: dict | None = None) -> LLMManager:
    global _manager
    if _manager is None or cfg is not None:
        _manager = LLMManager.from_config(cfg)
    return _manager


def reset_manager() -> None:
    """Clear the cached singleton (mostly for tests)."""
    global _manager
    _manager = None
