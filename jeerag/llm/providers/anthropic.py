"""Anthropic LLM provider (Claude models).

Uses the Anthropic Messages API: POST /v1/messages
Supports streaming via text/event-stream SSE.

No hardcoded credentials, models, or URLs — everything from config.
"""
from __future__ import annotations

import json
import time
from http.client import HTTPSConnection
from typing import Iterator
from urllib.parse import urlparse

from ..base import LLMProvider, LLMProviderError


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.base_url = (
            self.cfg.get("base_url")
            or self.cfg.get("llm_base_url")
            or "https://api.anthropic.com"
        ).rstrip("/")
        self.model = (
            self.cfg.get("model")
            or self.cfg.get("llm_model")
            or "claude-3-haiku-20240307"
        )
        self.key = self.cfg.get("key") or self.cfg.get("llm_api_key") or ""
        self.temperature = float(
            self.cfg.get("temperature", self.cfg.get("llm_temperature", 0.2))
        )
        self.timeout = int(self.cfg.get("timeout", self.cfg.get("llm_timeout", 120)))
        self.connect_timeout = int(
            self.cfg.get("connect_timeout", self.cfg.get("llm_connect_timeout", 10))
        )
        self.read_timeout = int(
            self.cfg.get("read_timeout", self.cfg.get("llm_read_timeout", 60))
        )
        self.first_event_timeout = int(
            self.cfg.get(
                "first_event_timeout",
                self.cfg.get("llm_first_event_timeout", 15),
            )
        )

    def configured(self) -> bool:
        return bool(self.base_url and self.key and self.model)

    def health(self) -> dict:
        if not self.configured():
            return {
                "ok": False,
                "reason": "no base_url, model, or api_key",
                "model": self.model,
            }
        t0 = time.time()
        try:
            payload = {
                "model": self.model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "OK"}],
                "system": "Reply with exactly the word OK.",
            }
            status, data = self._post(payload)
            if status == 200:
                try:
                    obj = json.loads(data)
                    text = (
                        obj.get("content", [{}])[0].get("text", "")
                        if isinstance(obj.get("content"), list)
                        else obj.get("content", "")
                    )
                    return {
                        "ok": True,
                        "ms": int((time.time() - t0) * 1000),
                        "model": self.model,
                        "reply": text[:40],
                    }
                except Exception:
                    return {"ok": True, "ms": int((time.time() - t0) * 1000),
                            "model": self.model}
            return {
                "ok": False,
                "reason": f"HTTP {status}: {data[:200]}",
                "model": self.model,
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False,
                    "reason": f"{type(e).__name__}: {e}"[:300],
                    "model": self.model}

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
        }

    def _post(self, payload: dict, timeout: int | None = None) -> tuple[int, str]:
        parsed = urlparse(self.base_url)
        path = "/v1/messages"
        body = json.dumps(payload).encode()
        headers = self._headers()
        headers["Content-Length"] = str(len(body))
        conn = HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=self.connect_timeout
        )
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            try:
                resp.fp.sock.settimeout(self.read_timeout)
            except Exception:
                pass
            data = resp.read().decode("utf-8", "replace")
            return resp.status, data
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def stream_chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
            "stream": True,
        }
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        parsed = urlparse(self.base_url)
        path = "/v1/messages"
        body = json.dumps(payload).encode()
        headers["Content-Length"] = str(len(body))

        conn = HTTPSConnection(
            parsed.hostname, parsed.port or 443, timeout=self.connect_timeout
        )
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            try:
                resp.fp.sock.settimeout(self.read_timeout)
            except Exception:
                pass

            first_event_time = None
            yielded_any = False
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
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("type") == "error":
                        err = obj.get("error", {})
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        raise LLMProviderError(f"SSE error: {msg}")
                    if obj.get("type") == "content_block_delta":
                        delta = obj.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                yielded_any = True
                                yield text
                    elif obj.get("type") == "message_delta":
                        pass  # usage info, no content
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

        # Fall back to non-streaming with the same key.
        try:
            status, data = self._post({**payload, "stream": False}, self.timeout)
        except Exception as e:  # noqa: BLE001
            raise LLMProviderError(f"{type(e).__name__}: {e}") from e

        if status == 200:
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                yield data.strip()
                return
            content = obj.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        yield block.get("text", "")
            elif isinstance(content, str):
                yield content
            return
        elif status in (401, 402, 403, 429):
            raise LLMProviderError(f"HTTP {status}: {data[:200]}")
        else:
            raise LLMProviderError(f"HTTP {status}: {data[:200]}")
