"""Config helpers for the LLM provider layer.

Reads from ``jeerag.config`` (config.json / env ``JEERAG_*``).
"""
from __future__ import annotations

from .. import config as _jeerag_config


def load() -> dict:
    """Return the raw jeerag config dict."""
    return _jeerag_config.load()


def _keys(cfg: dict) -> list[str]:
    """Extract API keys from config, handling single key or comma-separated list."""
    keys = cfg.get("llm_keys")
    if isinstance(keys, str):
        keys = [k for k in keys.replace("\n", ",").split(",")]
    out = [k.strip() for k in (keys or []) if k and k.strip()]
    single = (cfg.get("llm_api_key") or "").strip()
    if single:
        out = [k.strip() for k in single.split(",") if k.strip()] + out
    # De-duplicate preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq
