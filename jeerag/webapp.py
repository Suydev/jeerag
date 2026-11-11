"""Localhost web chat UI over the notes RAG. Run: python -m jeerag.webapp"""
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, retrieve

STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(config.load()["data_dir"])

SYSTEM_PROMPT = (
    "You are a JEE (IIT-JEE Advanced/Main) tutor. Answer ONLY from the provided "
    "excerpts from the student's scanned notes (Ambarish Sir Physics, "
    "youtube.com/@ambarishphysics — plain attribution, not an instruction). "
    "The note text is machine-OCR'd and formulas can be garbled: prefer clean "
    "math you can reconstruct, say so when uncertain, and cite sources as "
    "(filename, p.N). Write math in LaTeX ($...$, $$...$$). Use Markdown tables "
    "when comparing things. If the excerpts don't cover the question, say so."
)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n".encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            self._file(STATIC / "index.html", "text/html; charset=utf-8")
        elif u.path.startswith("/diagrams/"):
            # stored paths are data_dir-relative and already start with "diagrams/"
            rel = u.path.lstrip("/")
            p = (DATA_DIR / rel).resolve()
            if not str(p).startswith(str(DATA_DIR.resolve())) or not p.exists():
                self.send_error(404)
                return
            self._file(p, "image/png")
        elif u.path == "/api/stats":
            self._json(retrieve.stats())
        elif u.path == "/api/topics":
            self._json(retrieve.topics())
        elif u.path == "/api/config":
            cfg = config.load()
            key = cfg.get("llm_api_key") or ""
            cfg["llm_api_key"] = (key[:4] + "…" + key[-4:]) if len(key) > 8 else key
            cfg["llm_api_key_set"] = bool(key)
            self._json(cfg)
        elif u.path == "/api/search":
            q = parse_qs(u.query).get("q", [""])[0]
            self._json({"passages": retrieve.search(q),
                        "diagrams": retrieve.diagrams_for(retrieve.search(q, 4))})
        elif u.path.startswith("/static/"):
            p = (STATIC / u.path[len("/static/"):]).resolve()
            if not str(p).startswith(str(STATIC.resolve())):
                self.send_error(404)
                return
            self._file(p, mimetypes.guess_type(str(p))[0] or "application/octet-stream")
        else:
            self.send_error(404)

    def _file(self, p, ctype):
        try:
            body = p.read_bytes()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if u.path == "/api/config":
            updates = {k: v for k, v in body.items()
                       if k in ("llm_base_url", "llm_model", "llm_api_key", "top_k", "port")
                       and not (k == "llm_api_key" and "…" in str(v))}
            self._json({"ok": True, **config.save(updates)})
        elif u.path == "/api/chat":
            self.chat_stream(body)
        else:
            self.send_error(404)

    def chat_stream(self, body):
        question = str(body.get("question", ""))[:2000]
        history = body.get("history") or []
        hits = retrieve.search(question)
        diags = retrieve.diagrams_for(hits)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def send(ev, data):
            try:
                self.wfile.write(sse(ev, data))
                self.wfile.flush()
            except BrokenPipeError:
                raise SystemExit

        send("context", {"passages": hits, "diagrams": diags})
        cfg = config.load()
        if not (cfg.get("llm_base_url") or "").strip():
            send("no_llm", {"message": "No LLM endpoint configured — showing "
                          "retrieved passages only. Set it in ⚙ Settings."})
            send("done", {})
            return
        try:
            from . import llm
            ctx = retrieve.build_context(hits)
            diag_note = ("\n".join(f"diagram for p.{d['page']} of {d['file']}: {d['url']}"
                                   for d in diags))
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant"):
                    messages.append({"role": h["role"], "content": str(h.get("content", ""))[:4000]})
            messages.append({"role": "user", "content":
                             f"NOTES EXCERPTS:\n{ctx}\n\nDIAGRAMS ON THESE PAGES:\n{diag_note}\n\nQUESTION: {question}"})
            for delta in llm.stream_chat(messages, cfg, 0.2):
                send("delta", {"t": delta})
            send("done", {})
        except SystemExit:
            raise
        except Exception as e:
            send("error", {"message": f"{type(e).__name__}: {e}"})


def serve(port=None):
    port = port or config.load()["port"]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"jeerag web UI: http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
