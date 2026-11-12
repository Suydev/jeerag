"""OpenAI-compatible LLM provider.

Covers: OpenAI API, opencode2api (OpenCode Zen), Groq, Together, local
llama.cpp servers, and any endpoint speaking
``POST /v1/chat/completions`` with SSE streaming.

No hardcoded credentials, models, or URLs — everything comes from config.
Uses the native /v1/chat/completions endpoint for every provider.
"""
from __future__ import annotations

import json
import time
from http.client import HTTPConnection, HTTPSConnection
from typing import Iterator
from urllib.parse import urlparse

from ..base import LLMProvider, LLMProviderError


# ---- timeouts (configurable per-request via manager config) ----------------
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
FIRST_EVENT_TIMEOUT = 15  # upstreams that accept but send no SSE events


def _open_connection(url: str, timeout: int):
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return HTTPSConnection(parsed.hostname, parsed.port or 443, timeout=timeout)
    return HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def _post_streaming(url: str, headers: dict, payload: dict,
                    connect_timeout: int, read_timeout: int,
                    first_event_timeout: int):
    """POST JSON; return (http.client response, connection). Sets per-read timeout."""
    parsed = urlparse(url)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    body = json.dumps(payload).encode()
    headers = dict(headers)
    headers["Content-Length"] = str(len(body))

    conn = _open_connection(url, connect_timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        try:
            resp.fp.sock.settimeout(read_timeout)
        except Exception:
            pass
        return resp, conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise


def _post_complete(url: str, headers: dict, payload: dict,
                   connect_timeout: int, read_timeout: int) -> tuple[int, str]:
    """Non-streaming POST; return (status, body_text)."""
    parsed = urlparse(url)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    body = json.dumps(payload).encode()
    headers = dict(headers)
    headers["Content-Length"] = str(len(body))
    conn = _open_connection(url, connect_timeout)
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        try:
            resp.fp.sock.settimeout(read_timeout)
        except Exception:
            pass
        data = resp.read().decode("utf-8", "replace")
        return resp.status, data
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _should_rotate(status: int) -> bool:
    return status in (401, 402, 403, 429)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible provider: OpenAI, opencode2api, Groq, Together,
    local llama.cpp servers, anything speaking /v1/chat/completions + SSE."""

    name = "openai"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.base_url = (
            self.cfg.get("base_url") or self.cfg.get("llm_base_url") or ""
        ).rstrip("/")
        self.model = self.cfg.get("model") or self.cfg.get("llm_model") or ""
        self.key = self.cfg.get("key") or self.cfg.get("llm_api_key") or ""
        self.temperature = float(
            self.cfg.get("temperature", self.cfg.get("llm_temperature", 0.2))
        )
        self.top_p = self._num_cfg("top_p", "llm_top_p")
        self.default_max_tokens = self._num_cfg("max_tokens", "llm_max_tokens", int)
        self.timeout = int(
            self.cfg.get("timeout", self.cfg.get("llm_timeout", 120))
        )
        self.connect_timeout = int(
            self.cfg.get(
                "connect_timeout",
                self.cfg.get("llm_connect_timeout", CONNECT_TIMEOUT),
            )
        )
        self.read_timeout = int(
            self.cfg.get("read_timeout", self.cfg.get("llm_read_timeout", READ_TIMEOUT))
        )
        self.first_event_timeout = int(
            self.cfg.get(
                "first_event_timeout",
                self.cfg.get("llm_first_event_timeout", FIRST_EVENT_TIMEOUT),
            )
        )

    def _num_cfg(self, key: str, alt_key: str, cast=float):
        """Read a numeric config value (float/int or None) from either key."""
        raw = self.cfg.get(key, self.cfg.get(alt_key))
        if raw is None or raw == "":
            return None
        try:
            return cast(raw)
        except (TypeError, ValueError):
            return None

    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def health(self) -> dict:
        if not self.configured():
            return {
                "ok": False,
                "reason": "no base_url or model",
                "model": self.model,
            }
        t0 = time.time()
        try:
            for delta in self.stream_chat(
                [{"role": "user", "content": "OK"}], 0, 8
            ):
                if delta.strip():
                    return {
                        "ok": True,
                        "ms": int((time.time() - t0) * 1000),
                        "model": self.model,
                        "reply": delta[:40],
                    }
            return {"ok": False, "reason": "no content", "model": self.model}
        except LLMProviderError as e:
            return {"ok": False,
                    "reason": f"{type(e).__name__}: {e}"[:300],
                    "model": self.model}
        except Exception as e:  # noqa: BLE001
            return {"ok": False,
                    "reason": f"{type(e).__name__}: {e}"[:300],
                    "model": self.model}

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        elif self.default_max_tokens:
            payload["max_tokens"] = int(self.default_max_tokens)
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        # OpenRouter attribution headers (harmless elsewhere)
        if "openrouter.ai" in (self.base_url or "").lower():
            headers["HTTP-Referer"] = "http://127.0.0.1:8613"
            headers["X-Title"] = "jeerag"

        last_err: LLMProviderError | None = None
        try:
            resp, conn = _post_streaming(
                url, headers, payload,
                self.connect_timeout, self.read_timeout,
                self.first_event_timeout,
            )
        except Exception as e:
            raise LLMProviderError(f"{type(e).__name__}: {e}") from e

        first_event_time = None
        yielded_any = False
        try:
            with resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if first_event_time is None:
                        first_event_time = time.time()
                    if data == "[DONE]":
                        return
                    if data.startswith("{") and data.endswith("}"):
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict) and obj.get("error"):
                            err = obj["error"]
                            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                            raise LLMProviderError(f"SSE error: {msg}")
                        for ch in obj.get("choices") or []:
                            delta = (ch.get("delta") or {}).get("content")
                            if delta:
                                yielded_any = True
                                yield delta
                    elif data:
                        yielded_any = True
                        yield data
                    if first_event_time is not None:
                        if time.time() - first_event_time > self.first_event_timeout:
                            raise LLMProviderError(
                                f"no SSE data event within {self.first_event_timeout}s "
                                f"(upstream accepted but streamed nothing)"
                            )
                return
        except LLMProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMProviderError(
                f"stream read error: {type(e).__name__}: {e}"
            ) from e

        if yielded_any:
            return

        # Non-streaming fallback with the same key.
        try:
            status, data = _post_complete(
                url, headers, payload, self.connect_timeout, self.read_timeout
            )
        except Exception as e:
            raise LLMProviderError(f"{type(e).__name__}: {e}") from e

        if status == 200:
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                yield data.strip()
                return
            for ch in obj.get("choices") or []:
                delta = (
                    (ch.get("delta") or {}).get("content")
                    or (ch.get("message") or {}).get("content")
                    or ""
                )
                if delta:
                    yield delta
            return
        elif status in (401, 402, 403, 429):
            raise LLMProviderError(f"HTTP {status}: {data[:200]}")
        else:
            raise LLMProviderError(f"HTTP {status}: {data[:200]}")
