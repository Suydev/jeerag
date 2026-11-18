"""Localhost web chat UI over the notes RAG. Run: python -m jeerag.webapp"""
import collections
import json
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__
from . import config, retrieve, advanced_retrieve

STATIC = Path(__file__).parent / "static"
DATA_DIR = Path(config.load()["data_dir"])
mimetypes.add_type("application/manifest+json", ".webmanifest")

_STARTED = time.time()

# --- rendered PDF pages: bounded in-memory LRU (never touches disk) ----------
PAGE_DPI = 120
PAGE_CACHE_MAX = 50
_page_cache = collections.OrderedDict()
_page_lock = threading.Lock()

# --- query result cache (exact-match layer, TTL 24h) ------------------------
# Research (RAG-Cache, GPTCache): multi-layer cache with exact match first,
# then semantic match. We implement the exact-match layer (no embeddings).
# Hit rates for internal-doc QA: 15-30%. TTL 24h for factual content.
QUERY_CACHE_TTL = 86400  # 24 hours
QUERY_CACHE_MAX = 500
_query_cache = collections.OrderedDict()  # key -> (timestamp, result)
_query_cache_lock = threading.Lock()


def _query_cache_key(question: str, kind: str | None) -> str:
    """Normalized cache key for a query."""
    import re as _re
    norm = _re.sub(r"\s+", " ", question.strip().lower())
    norm = _re.sub(r"[^\w\s]", "", norm)
    return f"{kind or 'all'}:{norm}"


def _query_cache_get(key: str) -> dict | None:
    """Return cached result if present and not expired."""
    with _query_cache_lock:
        entry = _query_cache.get(key)
        if entry is None:
            return None
        ts, result = entry
        if time.time() - ts > QUERY_CACHE_TTL:
            _query_cache.pop(key, None)
            return None
        _query_cache.move_to_end(key)
        return result


def _query_cache_put(key: str, result: dict) -> None:
    """Store a query result in the cache."""
    with _query_cache_lock:
        _query_cache[key] = (time.time(), result)
        while len(_query_cache) > QUERY_CACHE_MAX:
            _query_cache.popitem(last=False)


# --- Metrics / observability ------------------------------------------------
# Counters and timers for observability. These are in-memory only and reset
# on restart; for production they should be pushed to Prometheus/Promscale.
_METRICS = {
    "requests_total": 0,
    "requests_errors": 0,
    "llm_requests_total": 0,
    "llm_requests_errors": 0,
    "llm_latency_ms": [],
    "retrieval_latency_ms": [],
    "cache_hits": 0,
    "cache_misses": 0,
}
_metrics_lock = threading.Lock()


def _metrics_inc(key: str, value: float = 1.0) -> None:
    with _metrics_lock:
        if key in ("requests_total", "requests_errors", "llm_requests_total", "llm_requests_errors",
                   "cache_hits", "cache_misses"):
            _METRICS[key] += value
        elif key in ("llm_latency_ms", "retrieval_latency_ms"):
            _METRICS[key].append(value)
            # keep only last 1000 samples
            if len(_METRICS[key]) > 1000:
                _METRICS[key] = _METRICS[key][-1000:]


def metrics_snapshot() -> dict:
    with _metrics_lock:
        llm_lat = _METRICS["llm_latency_ms"]
        ret_lat = _METRICS["retrieval_latency_ms"]
        def pct(vals, p):
            if not vals:
                return 0
            s = sorted(vals)
            idx = int(p * (len(s) - 1))
            return s[idx]
        total_req = _METRICS["requests_total"]
        err_req = _METRICS["requests_errors"]
        total_llm = _METRICS["llm_requests_total"]
        err_llm = _METRICS["llm_requests_errors"]
        cache_hits = _METRICS["cache_hits"]
        cache_misses = _METRICS["cache_misses"]
        cache_total = cache_hits + cache_misses
        return {
            "requests_total": total_req,
            "requests_errors": err_req,
            "error_rate": round(err_req / total_req, 4) if total_req else 0.0,
            "llm_requests_total": total_llm,
            "llm_requests_errors": err_llm,
            "llm_error_rate": round(err_llm / total_llm, 4) if total_llm else 0.0,
            "llm_latency_ms": {
                "p50": pct(llm_lat, 0.5),
                "p95": pct(llm_lat, 0.95),
                "p99": pct(llm_lat, 0.99),
                "mean": round(sum(llm_lat) / len(llm_lat), 2) if llm_lat else 0,
            },
            "retrieval_latency_ms": {
                "p50": pct(ret_lat, 0.5),
                "p95": pct(ret_lat, 0.95),
                "p99": pct(ret_lat, 0.99),
                "mean": round(sum(ret_lat) / len(ret_lat), 2) if ret_lat else 0,
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": round(cache_hits / cache_total, 4) if cache_total else 0.0,
            },
        }


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


# ---------------------------------------------------------------------------
# Shared RAG pipeline pieces (used by SSE /api/chat and JSON /api/answer)
# ---------------------------------------------------------------------------

def retrieve_for_question(question: str, cfg: dict | None = None):
    """Retrieval layer shared by every answering route.

    Advanced agentic retrieval (query expansion, RAG-Fusion, HyDE, CRAG
    re-grading) behind the exact-match query cache; falls back to plain BM25
    if the advanced path fails. Returns ``(passages, diagrams, method, loops)``.
    """
    cfg = cfg or config.load()
    top_k = int(cfg.get("top_k", 8))
    cache_key = _query_cache_key(question, None)
    cached = _query_cache_get(cache_key)
    if cached:
        return (cached["passages"], cached["diagrams"],
                cached.get("method", "cached") + " (cached)",
                cached.get("loops", 1))
    try:
        ares = advanced_retrieve.advanced_search(
            question, k=top_k, kind=None, mode="auto", use_loop=True)
        hits, diags = ares["passages"], ares["diagrams"]
        method, loops = ares.get("method", "advanced"), ares.get("loops", 1)
        _query_cache_put(cache_key, {
            "passages": hits, "diagrams": diags, "method": method, "loops": loops,
        })
        return hits, diags, method, loops
    except Exception:
        hits = retrieve.search(question)
        return hits, retrieve.diagrams_for(hits), "basic_bm25", 1


def _related_pyq(question: str) -> list:
    """PYQ/official-key enrichment for chat + answer (never fatal)."""
    try:
        return retrieve.related_pyq(question)
    except Exception:
        return []


def build_messages(question: str, history, cfg: dict, ctx_text: str, diags,
                   pyq=None) -> list[dict]:
    """System prompt (``llm_system_prompt`` override) + trimmed history + RAG turn.

    ``pyq`` (optional) is the related past-year question / official answer-key
    chunk list from :func:`retrieve.related_pyq`; it is appended as its own
    section so the model can cite official answers when they exist.
    """
    messages = [{"role": "system",
                 "content": (str(cfg.get("llm_system_prompt") or "").strip()
                             or SYSTEM_PROMPT)}]
    for h in (history or [])[-6:]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"],
                             "content": str(h.get("content", ""))[:4000]})
    diag_note = "\n".join(f"diagram for p.{d['page']} of {d['file']}: {d['url']}"
                          for d in diags)
    user = (f"NOTES EXCERPTS:\n{ctx_text}\n\n"
            f"DIAGRAMS ON THESE PAGES:\n{diag_note}\n\n")
    if pyq:
        lines = [f"[{p['file']} | p.{p['page']}] {p['text']}" for p in pyq]
        user += ("PAST-YEAR QUESTIONS AND OFFICIAL ANSWER KEYS (kind=pyq — "
                 "official JEE papers; 'Answer Qn: XYZ' lines are the official "
                 "final answers — cite them as (filename, p.N)):\n"
                 + "\n".join(lines) + "\n\n")
    messages.append({"role": "user", "content": user + f"QUESTION: {question}"})
    return messages


def _max_tokens(cfg: dict):
    """``llm_max_tokens`` from config as int | None."""
    mt = cfg.get("llm_max_tokens")
    try:
        return int(mt) if mt else None
    except (TypeError, ValueError):
        return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
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
            cfg = dict(config.load())
            keys = cfg.pop("llm_keys", []) or []
            had_key = bool(str(cfg.pop("llm_api_key", "") or "").strip())
            if isinstance(keys, str):
                keys = keys.replace("\n", ",").split(",")
            n_keys = len([k for k in keys if str(k).strip()])
            # secrets never leave the server — only their presence
            cfg["llm_api_key_set"] = bool(had_key or n_keys)
            cfg["llm_keys_set"] = n_keys
            self._json(cfg)
        elif u.path == "/api/search":
            qs = parse_qs(u.query)
            q = qs.get("q", [""])[0]
            kind = qs.get("kind", [None])[0]
            try:
                k = max(1, min(int(qs.get("k", ["8"])[0]), 24))
            except ValueError:
                k = 8
            rerank = qs.get("rerank", ["false"])[0].lower() == "true"
            fusion = qs.get("fusion", ["rrf"])[0]
            contextual = qs.get("contextual", ["false"])[0].lower() == "true"
            try:
                from . import advanced_retrieve
                res = advanced_retrieve.advanced_search(
                    q, k=k, kind=kind, mode="auto",
                    use_loop=False, rerank=rerank, fusion=fusion,
                    contextual=contextual)
                hits = res["passages"]
                diags = res["diagrams"]
                method = res.get("method", "advanced")
                reranked = res.get("reranked", False)
            except Exception:
                hits = retrieve.search(q, kind=kind)
                diags = retrieve.diagrams_for(hits)
                method = "basic_bm25"
                reranked = False
            from .tracker import log_query
            log_query(q, hits, "search")
            self._json({"passages": hits, "diagrams": diags,
                        "method": method, "reranked": reranked})
        elif u.path == "/api/pyq":
            # past-year questions + official answer keys (kind='pyq' corpus)
            qs = parse_qs(u.query)
            q = qs.get("q", [""])[0]
            try:
                k = max(1, min(int(qs.get("k", ["10"])[0]), 24))
            except ValueError:
                k = 10
            if not q.strip():
                self._json({"passages": [], "diagrams": [], "years": [],
                            "hint": "add ?q=<topic>, e.g. ?q=rotation&k=10"})
                return
            hits = retrieve.search(q, k=k, kind="pyq")
            years: dict = {}
            con = retrieve.connect()
            try:
                for r in con.execute(
                        "SELECT chapter, COUNT(*) n FROM files "
                        "WHERE kind='pyq' GROUP BY chapter ORDER BY chapter"):
                    if r["chapter"]:
                        years[r["chapter"]] = r["n"]
            finally:
                con.close()
            self._json({"passages": hits,
                        "diagrams": retrieve.diagrams_for(hits),
                        "years": years})
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
        elif u.path == "/api/metrics":
            self._json(metrics_snapshot())
        elif u.path == "/api/tracker":
            from .tracker import summary
            self._json(summary(int(parse_qs(u.query).get("days", ["7"])[0])))
        elif u.path == "/api/health":
            cfg = config.load()
            try:
                idx, idx_ok = retrieve.stats(), True
            except Exception as e:
                idx, idx_ok = {"error": f"{type(e).__name__}: {e}"}, False
            self._json({
                "ok": True,
                "service": "jeerag-web",
                "version": __version__,
                "uptime_s": round(time.time() - _STARTED, 1),
                "index_ok": idx_ok,
                "index": idx,
                "llm": {
                    "configured": bool((cfg.get("llm_base_url") or "").strip()),
                    "provider": cfg.get("llm_provider") or "openai",
                    "model": cfg.get("llm_model"),
                },
            })
        elif u.path == "/api/formulas":
            # formula harvest over the best-matching pages of a topic
            qs = parse_qs(u.query)
            q = qs.get("q", [""])[0]
            try:
                k = max(1, min(int(qs.get("k", ["6"])[0]), 12))
            except ValueError:
                k = 6
            if not q.strip():
                self._json({"topic": "", "sections": [],
                            "hint": "add ?q=<topic>, e.g. ?q=friction&k=6"})
                return
            try:
                from . import formulas
                sections = formulas.collect(q, k=k)
                for s in sections:
                    s["images"] = [{"path": p, "url": "/" + p}
                                   for p in s.get("images", [])]
                self._json({"topic": q, "sections": sections})
            except Exception as e:
                self._json({"topic": q, "sections": [],
                            "error": f"{type(e).__name__}: {e}"})
        elif u.path == "/api/recent":
            from .tracker import recent
            try:
                lim = int(parse_qs(u.query).get("limit", ["10"])[0])
            except ValueError:
                lim = 10
            self._json({"queries": recent(lim)})
        elif u.path == "/api/page":
            # OCR text of one page (the text behind every /pdfpage/ image)
            qs = parse_qs(u.query)
            file = qs.get("file", [""])[0]
            try:
                page = int(qs.get("page", ["0"])[0])
            except ValueError:
                page = 0
            if not file or page < 1:
                self._json({"error": "file and page required (?file=F&page=N)"}, 400)
                return
            out = retrieve.page_text(file, page)
            if not out:
                self.send_error(404)
                return
            self._json(out)
        elif u.path == "/api/providers":
            # live health of every configured LLM endpoint (ensemble-aware)
            try:
                from jeerag.llm.manager import llm_manager
                self._json(llm_manager(config.load()).health())
            except Exception as e:
                self._json({"ok": False, "providers": [],
                            "error": f"{type(e).__name__}: {e}"})
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
            self.send_header("Cache-Control", "public, max-age=86400")
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

    @staticmethod
    def _cache_control(p) -> str:
        """Cache class by content stability: vendored assets immutable,
        app code/html must revalidate, derived images 1 day."""
        try:
            rel = p.resolve().relative_to(STATIC.resolve()).as_posix()
        except Exception:
            return "public, max-age=86400"  # /diagrams/, /pdfpage/ renders
        if rel.startswith(("vendor/", "icons/")) or rel == "manifest.webmanifest":
            return "public, max-age=31536000, immutable"
        return "no-cache"

    def _file(self, p, ctype):
        try:
            body = p.read_bytes()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", self._cache_control(Path(p)))
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
                       "llm_vision_model", "top_k", "port", "spaced_repeat",
                       "llm_temperature", "llm_top_p", "llm_max_tokens",
                       "llm_system_prompt", "llm_provider", "llm_providers",
                       "rerank_enabled", "rerank_recall_pool", "rerank_top_n",
                       "contextual_retrieval", "fusion_method", "crag_enabled")
            updates = {k: v for k, v in body.items()
                       if k in allowed
                       and not (k in ("llm_api_key", "llm_keys") and "…" in str(v))}
            for numk, cast in (("llm_temperature", float), ("llm_top_p", float),
                               ("llm_max_tokens", int), ("top_k", int)):
                if numk in updates:
                    try:
                        updates[numk] = cast(updates[numk])
                    except (TypeError, ValueError):
                        updates.pop(numk)
            if updates:
                try:
                    from jeerag.llm.manager import reset_manager
                    reset_manager()  # drop the cached manager so new settings bind
                except Exception:
                    pass
            self._json({"ok": True, **config.save(updates)})
        elif u.path == "/api/cards":
            from . import review
            try:
                if str(body.get("action", "add")) == "grade":
                    res = review.grade(
                        int(body["id"]), int(body.get("score", 4)))
                    self._json({"ok": True,
                                "interval": (res or {}).get("interval"),
                                "card": res})
                elif str(body.get("action")) == "delete":
                    ok = review.delete(body.get("id"))
                    self._json({"ok": ok, "deleted": bool(ok)}, 200 if ok else 404)
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
        elif u.path == "/api/answer":
            self.answer_json(body)
        elif u.path == "/api/contextualize_index":
            lim = body.get("limit")
            try:
                from . import contextual_retrieval
                result = contextual_retrieval.contextualize_index(
                    limit=int(lim) if lim is not None else None)
                self._json(result)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        else:
            self.send_error(404)

    def do_DELETE(self):
        """DELETE /api/cards?id=N (id also accepted as a JSON body field)."""
        u = urlparse(self.path)
        if u.path == "/api/cards":
            from . import review
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                body = {}
            cid = body.get("id") or parse_qs(u.query).get("id", [None])[0]
            ok = review.delete(cid) if cid is not None else False
            self._json({"ok": ok, "deleted": bool(ok)}, 200 if ok else 404)
        else:
            self.send_error(404)

    def answer_json(self, body):
        """POST /api/answer — one-shot JSON RAG answer (curl / scripting / agents).

        Same retrieval + LLM path as /api/chat, without the SSE stream.
        """
        question = str(body.get("question", ""))[:2000]
        if not question.strip():
            self._json({"error": "question required"}, 400)
            return
        cfg = config.load()
        rerank = bool(body.get("rerank", cfg.get("rerank_enabled", False)))
        fusion = body.get("fusion", cfg.get("fusion_method", "rrf"))
        contextual = bool(body.get("contextual", cfg.get("contextual_retrieval", False)))
        try:
            from . import advanced_retrieve
            res = advanced_retrieve.advanced_search(
                question, k=cfg.get("top_k", 8), kind=None, mode="auto",
                use_loop=True, rerank=rerank, fusion=fusion,
                contextual=contextual)
            hits = res["passages"]
            diags = res["diagrams"]
            method = res.get("method", "advanced")
            loops = res.get("loops", 1)
            reranked = res.get("reranked", False)
        except Exception:
            hits = retrieve.search(question)
            diags = retrieve.diagrams_for(hits)
            method = "basic_bm25"
            loops = 1
            reranked = False
        pyq = _related_pyq(question)
        from .tracker import log_query
        log_query(question, hits, "answer")
        out = {"question": question, "method": method, "loops": loops,
               "passages": hits, "diagrams": diags, "pyq": pyq,
               "reranked": reranked, "answer": None}
        if not (cfg.get("llm_base_url") or "").strip():
            out["note"] = ("no LLM endpoint configured — passages only; set "
                           "llm_base_url via POST /api/config or config.json")
            self._json(out)
            return
        try:
            from jeerag.llm.manager import llm_manager
            ctx, _ = advanced_retrieve.build_advanced_context(
                question, k=len(hits), cfg=cfg)
            messages = build_messages(question, body.get("history") or [],
                                      cfg, ctx, diags, pyq=pyq)
            temp = float(cfg.get("llm_temperature", 0.2))
            out["answer"] = "".join(
                llm_manager(cfg).chat(messages, temp, _max_tokens(cfg))).strip()
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        self._json(out)

    def chat_stream(self, body):
        question = str(body.get("question", ""))[:2000]
        history = body.get("history") or []
        cfg = config.load()
        rerank = bool(body.get("rerank", cfg.get("rerank_enabled", False)))
        fusion = body.get("fusion", cfg.get("fusion_method", "rrf"))
        contextual = bool(body.get("contextual", cfg.get("contextual_retrieval", False)))
        # --- Advanced retrieval with agentic follow-loop (cached) ---
        t0 = time.time()
        try:
            from . import advanced_retrieve
            res = advanced_retrieve.advanced_search(
                question, k=cfg.get("top_k", 8), kind=None, mode="auto",
                use_loop=True, rerank=rerank, fusion=fusion,
                contextual=contextual)
            hits = res["passages"]
            diags = res["diagrams"]
            method = res.get("method", "advanced")
            loops = res.get("loops", 1)
            reranked = res.get("reranked", False)
        except Exception:
            hits = retrieve.search(question)
            diags = retrieve.diagrams_for(hits)
            method = "basic_bm25"
            loops = 1
            reranked = False
        _metrics_inc("retrieval_latency_ms", (time.time() - t0) * 1000)
        _metrics_inc("requests_total")
        pyq = _related_pyq(question)
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
                         "pyq": pyq, "method": method, "loops": loops,
                         "reranked": reranked})
        from .tracker import log_query
        log_query(question, hits, "chat")
        if not (cfg.get("llm_base_url") or "").strip():
            send("no_llm", {"message": "No LLM endpoint configured — showing "
                          "retrieved passages only. Set it in ⚙ Settings."})
            send("done", {})
            return
        try:
            from jeerag.llm.manager import llm_manager
            ctx, _ = advanced_retrieve.build_advanced_context(
                question, k=len(hits), cfg=cfg)
            messages = build_messages(question, history, cfg, ctx, diags,
                                      pyq=pyq)

            # Stream answer via LLM provider (openai, anthropic, opencode2api, ...)
            temp = float(cfg.get("llm_temperature", 0.2))
            mgr = llm_manager(cfg)
            _metrics_inc("llm_requests_total")
            llm_t0 = time.time()

            # --- optional in-stream tool loop ({"tools": true}) -----------
            use_tools = bool(body.get("tools", False))
            _tools = None
            if use_tools:
                try:
                    from . import tools as _tools_mod
                    _tools = _tools_mod
                    block = _tools.SYSTEM_PROMPT_BLOCK()
                    if block and messages and messages[0].get("role") == "system":
                        messages[0]["content"] += "\n\n" + block
                except Exception:
                    use_tools = False  # degrade silently to plain RAG chat

            tool_re = re.compile(r"^TOOL\s+([a-z_]+)\s+(\{.*\})\s*$")
            MAX_TOOL_ROUNDS = 4

            def _run_marker(name, args_json):
                """Execute one TOOL marker: emit tool + tool_result, feed back."""
                nonlocal tool_rounds
                tool_rounds += 1
                tool_id = f"t{tool_rounds}"
                try:
                    args = json.loads(args_json)
                    if not isinstance(args, dict):
                        args = {}
                except Exception:
                    args = {}
                send("tool", {"id": tool_id, "name": name, "args": args})
                tt = time.time()
                try:
                    res_tool = _tools.execute(name, args, timeout_s=8)
                except Exception as te:
                    res_tool = {"ok": False, "error": f"{type(te).__name__}: {te}"}
                if not isinstance(res_tool, dict):
                    res_tool = {"ok": False, "error": "bad tool payload"}
                send("tool_result", {"id": tool_id, "name": name,
                                     "ok": bool(res_tool.get("ok", False)),
                                     "payload": res_tool,
                                     "ms": round((time.time() - tt) * 1000, 1)})
                try:
                    payload_s = json.dumps(res_tool, ensure_ascii=False)
                except Exception:
                    payload_s = "{}"
                messages.append({"role": "system",
                                 "content": (f"TOOL_RESULT {name}: {payload_s[:2048]}"
                                             "\nUse these results when answering. Do not "
                                             "call more tools unless another one is "
                                             "really needed; then give your final answer.")})

            tool_rounds = 0
            idle_rounds = 0
            last_tool_key = None
            final_turn = False
            final_done = False
            delivered = False
            while True:
                buf = ""
                saw_marker = False
                had_output = False
                try:
                    for delta in mgr.chat(messages, temp, _max_tokens(cfg)):
                        had_output = True
                        delivered = True
                        buf += delta
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            m = tool_re.match(line) if (use_tools and not final_turn) else None
                            if m and tool_rounds < MAX_TOOL_ROUNDS:
                                key = (m.group(1), m.group(2))
                                if key == last_tool_key and tool_rounds > 0:
                                    idle_rounds += 1
                                else:
                                    last_tool_key = key
                                    saw_marker = True
                                    _run_marker(m.group(1), m.group(2))
                            elif m:
                                pass  # suppress stray markers in the final turn
                            else:
                                send("delta", {"t": line + "\n"})
                    if buf:
                        m = tool_re.match(buf) if (use_tools and not final_turn) else None
                        if m and tool_rounds < MAX_TOOL_ROUNDS:
                            key = (m.group(1), m.group(2))
                            if key == last_tool_key and tool_rounds > 0:
                                idle_rounds += 1
                            else:
                                last_tool_key = key
                                saw_marker = True
                                _run_marker(m.group(1), m.group(2))
                        elif m:
                            pass
                        else:
                            send("delta", {"t": buf})
                except SystemExit:
                    raise
                except Exception:
                    if not had_output and tool_rounds == 0:
                        raise
                    break
                if not use_tools:
                    break
                if saw_marker:
                    idle_rounds = 0
                    if tool_rounds >= MAX_TOOL_ROUNDS:
                        final_turn = True
                else:
                    idle_rounds += 1
                if final_turn and not final_done:
                    final_done = True
                    if not saw_marker:
                        break
                if idle_rounds >= 2:
                    break
            if use_tools and tool_rounds > 0 and not delivered:
                send("delta", {"t": "(tool results could not be turned into an answer by the model)\n"})

            _metrics_inc("llm_latency_ms", (time.time() - llm_t0) * 1000)
            send("done", {})
        except SystemExit:
            raise
        except Exception as e:
            _metrics_inc("requests_errors")
            _metrics_inc("llm_requests_errors")
            send("error", {"message": f"{type(e).__name__}: {e}"})


def serve(port=None):
    port = port or config.load()["port"]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"jeerag web UI: http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    serve()
