"""Dependency-free OpenAI-compatible client with multi-key rotation.

Why hand-rolled: the official `openai` SDK pulls httpx/pydantic and would not
install on this Termux device; an SSE-over-urllib client is ~100 lines and has
no moving parts.  Everything a JEE tutor chat needs (streaming deltas, vision
image parts, retries, key rotation) is right here.

Configuration (first match wins):
  config.json / env ``JEERAG_*`` ::

      llm_base_url : https://openrouter.ai/api/v1     (or "" = retrieval-only)
      llm_model    : model id                          (provider default used)
      llm_api_key  : single key, or comma-separated list
      llm_keys     : list of keys rotated round-robin on 401/402/429

A key list is rotated in order per request; the first key that gets a non-auth
error is kept as "current" so a dead key costs at most one retry.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from typing import Iterable, Iterator

from . import config

OPENROUTER = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = {
    "openrouter.ai": "google/gemini-2.5-flash",
    "omnirush": "gpt-6-astra",
}
TIMEOUT = 120


class LLMError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def _keys(cfg) -> list[str]:
    keys = cfg.get("llm_keys")
    if isinstance(keys, str):
        keys = [k for k in keys.replace("\n", ",").split(",")]
    out = [k.strip() for k in (keys or []) if k and k.strip()]
    single = (cfg.get("llm_api_key") or "").strip()
    if single:
        out = [k.strip() for k in single.split(",") if k.strip()] + out
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def _model(cfg) -> str:
    m = (cfg.get("llm_model") or "").strip()
    if m:
        return m
    base = (cfg.get("llm_base_url") or "").lower()
    for needle, model in DEFAULT_MODELS.items():
        if needle in base:
            return model
    return "gpt-4o-mini"


def _headers(cfg, key) -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    base = (cfg.get("llm_base_url") or "").lower()
    if "openrouter.ai" in base:  # OpenRouter attribution headers
        h["HTTP-Referer"] = "http://127.0.0.1:8613"
        h["X-Title"] = "jeerag"
    return h


def _url(cfg) -> str:
    base = (cfg.get("llm_base_url") or "").rstrip("/")
    if not base:
        raise LLMError("no LLM endpoint configured")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _post(url, headers, payload, timeout=TIMEOUT):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _should_rotate(status) -> bool:
    return status in (401, 402, 403, 429)


def _key_order(cfg):
    keys = _keys(cfg)
    if not keys:
        return [""]
    start = int(cfg.get("_key_cursor", 0)) % len(keys)
    return keys[start:] + keys[:start]


def _advance_cursor(cfg, used_key):
    """Persist which key worked so the next request starts there."""
    keys = _keys(cfg)
    if used_key in keys:
        cfg["_key_cursor"] = keys.index(used_key)
        try:
            config.save({"_key_cursor": cfg["_key_cursor"]})
        except Exception:
            pass


def stream_chat(messages, cfg=None, temperature=0.2, max_tokens=None) -> Iterator[str]:
    """Yield content deltas from an OpenAI-compatible streaming endpoint."""
    cfg = cfg or config.load()
    url = _url(cfg)
    payload = {"model": _model(cfg), "messages": messages,
               "stream": True, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    last_err = None
    for key in _key_order(cfg):
        try:
            resp = _post(url, _headers(cfg, key), payload)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            last_err = LLMError(f"HTTP {e.code}: {body}", e.code)
            if _should_rotate(e.code):
                continue
            raise last_err from None
        except Exception as e:  # network: try the next key once
            last_err = LLMError(f"{type(e).__name__}: {e}")
            continue
        _advance_cursor(cfg, key)
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in obj and obj["error"]:
                    raise LLMError(str(obj["error"]))
                for ch in obj.get("choices") or []:
                    delta = (ch.get("delta") or {}).get("content")
                    if delta:
                        yield delta
        return
    raise last_err or LLMError("no API key configured")


def complete(messages, cfg=None, temperature=0.1, max_tokens=512) -> str:
    """Non-streaming completion (used by the captioner / formula extractor)."""
    return "".join(stream_chat(messages, cfg, temperature, max_tokens)).strip()


def image_part(png_bytes: bytes, mime="image/png") -> dict:
    b64 = base64.b64encode(png_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def configured(cfg=None) -> bool:
    cfg = cfg or config.load()
    return bool((cfg.get("llm_base_url") or "").strip())


def health(cfg=None) -> dict:
    """Cheap liveness check used by /api/health and the UI pill."""
    cfg = cfg or config.load()
    if not configured(cfg):
        return {"ok": False, "reason": "not configured", "mode": "retrieval-only"}
    t0 = time.time()
    try:
        first = next(iter(stream_chat(
            [{"role": "user", "content": "reply with the single word OK"}],
            cfg, 0, 8)))
        return {"ok": True, "ms": int((time.time() - t0) * 1000),
                "model": _model(cfg), "reply": first[:40]}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:300],
                "model": _model(cfg)}
