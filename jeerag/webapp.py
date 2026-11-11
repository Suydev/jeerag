"""Localhost web chat UI over the notes RAG. Run: python -m jeerag.webapp"""
import collections
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config, retrieve, advanced_retrieve

STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(config.load()["data_dir"])

# --- rendered PDF pages: bounded in-memory LRU (never touches disk) ----------
PAGE_DPI = 120
PAGE_CACHE_MAX = 50
_page_cache = collections.OrderedDict()
_page_lock = threading.Lock()


def pdf_path_for(name):
    """Resolve an indexed PDF by file name (absolute path, else notes_dir)."""
    con = retrieve.connect()
    row = con.execute("SELECT path FROM files WHERE file=?", (name,)).fetchone()
    con.close()
    if row and row["path"]:
        p = Path(row["path"])
        if p.exists():
            return p
    p = Path(config.load()["notes_dir"]) / name
    return p if p.exists() else None


def render_page(name, page, dpi=PAGE_DPI):
    """PNG bytes of a 1-based page, cached in RAM (max PAGE_CACHE_MAX pages)."""
    key = (name, page, dpi)
    with _page_lock:
        hit = _page_cache.get(key)
        if hit is not None:
            _page_cache.move_to_end(key)
            return hit
    pdf = pdf_path_for(name)
    if not pdf:
        return None
    import fitz
    with fitz.open(pdf) as doc:
        if not (1 <= page <= doc.page_count):
            return None
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        png = pix.tobytes("png")
    with _page_lock:
        _page_cache[key] = png
        while len(_page_cache) > PAGE_CACHE_MAX:
            _page_cache.popitem(last=False)
    return png

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
            kind = parse_qs(u.query).get("kind", [None])[0]
            self._json(retrieve.topics(kind=kind))
        elif u.path == "/api/config":
            cfg = config.load()
            key = cfg.get("llm_api_key") or ""
            cfg["llm_api_key"] = (key[:4] + "…" + key[-4:]) if len(key) > 8 else key
            cfg["llm_api_key_set"] = bool(key)
            self._json(cfg)
        elif u.path == "/api/search":
            q = parse_qs(u.query).get("q", [""])[0]
            kind = parse_qs(u.query).get("kind", [None])[0]
            hits = retrieve.search(q, kind=kind)
            self._json({"passages": hits, "diagrams": retrieve.diagrams_for(hits)})
        elif u.path == "/api/diagrams":
            q = parse_qs(u.query).get("q", [""])[0]
            try:
                from . import caption
                self._json({"diagrams": caption.search(q)})
            except Exception as e:
                self._json({"diagrams": [], "error": f"{type(e).__name__}: {e}"})
        elif u.path == "/api/cards":
            from . import review
            self._json({"due": review.due(), "stats": review.stats()})
        elif u.path.startswith("/pdfpage/"):
            # /pdfpage/<url-encoded file>/<1-based page>.png
            rest = u.path[len("/pdfpage/"):]
            from urllib.parse import unquote
            fname, _, tail = rest.rpartition("/")
            fname = unquote(fname)
            m = re.match(r"^(\d+)\.png$", tail)
            if not m:
                self.send_error(404)
                return
            png = render_page(fname, int(m.group(1)))
            if png is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
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
            allowed = ("llm_base_url", "llm_model", "llm_api_key", "llm_keys",
                       "llm_vision_model", "top_k", "port", "spaced_repeat")
            updates = {k: v for k, v in body.items()
                       if k in allowed
                       and not (k in ("llm_api_key", "llm_keys") and "…" in str(v))}
            self._json({"ok": True, **config.save(updates)})
        elif u.path == "/api/cards":
            from . import review
            try:
                if str(body.get("action", "add")) == "grade":
                    self._json({"ok": True, "card": review.grade(
                        int(body["id"]), int(body.get("score", 4)))})
                else:
                    cid = review.add(str(body.get("question", ""))[:500],
                                     str(body.get("answer", ""))[:4000],
                                     body.get("file"), body.get("page"),
                                     body.get("diagram"), body.get("topic"))
                    self._json({"ok": True, "id": cid})
            except Exception as e:
                self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)
        elif u.path == "/api/chat":
            self.chat_stream(body)
        else:
            self.send_error(404)

    def chat_stream(self, body):
        question = str(body.get("question", ""))[:2000]
        history = body.get("history") or []
        # --- Advanced retrieval with agentic follow-loop ---
        cfg = config.load()
        top_k = int(cfg.get("top_k", 8))
        try:
            ares = advanced_retrieve.advanced_search(
                question, k=top_k, kind=None, mode="auto", use_loop=True)
            hits = ares["passages"]
            diags = ares["diagrams"]
            method = ares.get("method", "advanced")
            loops = ares.get("loops", 1)
        except Exception:
            # Fallback to basic BM25 if advanced retrieval fails
            hits = retrieve.search(question)
            diags = retrieve.diagrams_for(hits)
            method = "basic_bm25"
            loops = 1
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

        send("context", {"passages": hits, "diagrams": diags,
                         "method": method, "loops": loops})
        cfg = config.load()
        if not (cfg.get("llm_base_url") or "").strip():
            send("no_llm", {"message": "No LLM endpoint configured — showing "
                          "retrieved passages only. Set it in ⚙ Settings."})
            send("done", {})
            return
        try:
            from jeerag.llm.manager import llm_manager
            ctx, _ = advanced_retrieve.build_advanced_context(
                question, k=len(hits), cfg=cfg)
            diag_note = ("\n".join(f"diagram for p.{d['page']} of {d['file']}: {d['url']}"
                                   for d in diags))
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-6:]:
                if h.get("role") in ("user", "assistant"):
                    messages.append({"role": h["role"], "content": str(h.get("content", ""))[:4000]})
            messages.append({"role": "user", "content":
                             f"NOTES EXCERPTS:\n{ctx}\n\nDIAGRAMS ON THESE PAGES:\n{diag_note}\n\nQUESTION: {question}"})

            # Stream answer via LLM provider (openai, anthropic, opencode2api, ...)
            temp = float(cfg.get("llm_temperature", 0.2))
            mgr = llm_manager(cfg)
            for delta in mgr.chat(messages, temp, None):
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
