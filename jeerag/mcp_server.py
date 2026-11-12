"""MCP server exposing the notes index. Run: python -m jeerag.mcp_server

Deliberately dependency-free: the Model Context Protocol over stdio is
newline-delimited JSON-RPC 2.0, so this speaks it directly instead of pulling
in the ``mcp`` SDK (whose pydantic/cryptography deps do not build on Termux).
Any MCP client — Claude Desktop, an agent SDK, ``mcp-cli`` — can talk to it.

Tools
   search_notes          BM25 over pages, with diagram urls
   search_notes_advanced agentic loop: expansion, RAG-Fusion, HyDE, CRAG
   search_pyq            search restricted to previous-year questions
   search_diagrams       caption/BM25 search over the figure crops
   get_page              one page's OCR text + its diagrams
   get_page_context      one page plus `radius` neighbouring pages
   get_diagram           a diagram crop as a real PNG image
   harvest_formulas      relation lines + recaps + diagram URIs for a topic
   list_topics           every PDF with topic/subject/page counts
   index_stats           file/page/chunk/diagram counts
   add_study_card        add a card to the SM-2 Doubt Deck
   delete_study_card     remove a card from the Doubt Deck
   study_stats           weak-topic usage summary over N days
   recent_queries        most recent logged queries
   evaluate_answer       RAGAS-style grading of one Q/A pair
   rag_answer            one-shot retrieve + LLM answer (non-streaming)
   llm_health            health of every configured LLM endpoint
Resources
   jeerag://notes          index of every indexed PDF
   jeerag://notes/{file}/p{page}   one page as markdown (text + diagram uris)
   jeerag://diagram/{path} one crop as an image resource
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

from . import retrieve, advanced_retrieve
from .llm import health as llm_health

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "jeerag", "version": "1.2.0"}


# ---------------------------------------------------------------------------
# Audit logging (structured JSON on stderr, one line per event)
# ---------------------------------------------------------------------------

def audit_log(event: str, fields: dict) -> None:
    """Write a structured audit record to stderr.

    Each record is one JSON line with a timestamp, event name, and fields.
    Used for tool call tracking, debugging, and cost attribution.
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": event,
        **fields,
    }
    try:
        sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
        sys.stderr.flush()
    except Exception:
        pass  # never break the server on audit failure


INSTRUCTIONS = (
    "RAG over scanned JEE physics/maths notes (Ambarish Sir). Text was "
    "machine-OCR'd from page scans, so formulas may be garbled — always fetch "
    "the diagram image for any page that references a figure, and cite "
    "(file, p.N) in answers. Notes credit: Ambarish Sir Physics, "
    "youtube.com/@ambarishphysics."
)


# ---------------------------------------------------------------- tools
def _diagram_abs(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(retrieve.config.load()["data_dir"]) / path
    return p


def t_search_notes(query: str, k: int = 8, kind: str | None = None):
    hits = retrieve.search(query, k, kind=kind)
    return {"passages": hits, "diagrams": retrieve.diagrams_for(hits)}


def t_search_diagrams(query: str, k: int = 8):
    from . import caption
    return caption.search(query, k)


def t_get_page(file: str, page: int):
    out = retrieve.page_text(file, page)
    return out or {"error": f"page not found: {file} p{page}"}


def t_get_diagram(path: str):
    p = _diagram_abs(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return {"__image__": p}


def t_list_topics():
    return retrieve.topics()


def t_index_stats():
    return retrieve.stats()


def t_search_notes_advanced(query: str, k: int = 8, kind: str | None = None,
                            mode: str = "auto"):
    """Advanced search with agentic follow-loop (query expansion, RAG-Fusion,
    HyDE, CRAG re-grading). Falls back to basic BM25 on error."""
    try:
        res = advanced_retrieve.advanced_search(
            query, k=k, kind=kind, mode=mode, use_loop=True)
        return {
            "passages": res["passages"],
            "diagrams": res["diagrams"],
            "method": res.get("method", "advanced"),
            "loops": res.get("loops", 1),
            "query_variants": res.get("query_variants", []),
            "reranked": res.get("reranked", False),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def t_llm_health():
    """Health check for every configured LLM endpoint (multi-LLM ensemble)."""
    return llm_health()


def t_search_pyq(query: str, k: int = 8):
    """Search restricted to previous-year questions (kind='pyq')."""
    hits = retrieve.search(query, k, kind="pyq")
    return {"passages": hits, "diagrams": retrieve.diagrams_for(hits)}


def t_get_page_context(file: str, page: int, radius: int = 1):
    """One page's OCR text plus `radius` neighbouring pages.

    Handwritten notes continue across pages, so agents usually want the
    surrounding context before answering from a single page.
    """
    radius = max(0, min(int(radius), 3))
    pages = []
    for p in range(max(1, int(page) - radius), int(page) + radius + 1):
        t = retrieve.page_text(file, p)
        if t:
            pages.append(t)
    if not pages:
        return {"error": f"no pages found near {file} p{page}"}
    return {"file": file, "center": int(page), "pages": pages}


def t_harvest_formulas(topic: str, k: int = 6):
    """Harvest relation lines + recap sentences + diagram URIs for a topic.

    Formulas are machine-read from scans and folded to readable Unicode, so
    ``\\frac{\\mu N}{m}`` comes back as ``μN/m``. Check the cited page image
    before trusting a sign or exponent.
    """
    from . import formulas
    sections = formulas.collect(topic, k=max(1, min(int(k), 12)))
    for s in sections:
        s["images"] = [f"jeerag://diagram/{quote(p)}" for p in s.pop("images", [])]
    return {"topic": topic, "sections": sections}


def t_add_study_card(question: str, answer: str, file: str | None = None,
                     page: int | None = None, topic: str | None = None):
    """Add a card to the SM-2 Doubt Deck; review it from the web UI."""
    from . import review
    cid = review.add(str(question)[:500], str(answer)[:4000],
                     file, int(page) if page else None, None, topic)
    return {"id": cid, "scheduled": "tomorrow",
            "note": "visible in the web UI Doubt Deck (SM-2 spaced repetition)"}


def t_delete_study_card(id: int):
    """Remove one card from the Doubt Deck."""
    from . import review
    ok = review.delete(id)
    return {"deleted": bool(ok), "id": id}


def t_study_stats(days: int = 7):
    """Weak-topic usage summary (which chapters get asked about)."""
    from . import tracker
    return tracker.summary(max(1, min(int(days), 365)))


def t_recent_queries(limit: int = 10):
    """Most recent logged queries with their topics and files."""
    from . import tracker
    return {"queries": tracker.recent(max(1, min(int(limit), 100)))}


def t_evaluate_answer(question: str, answer: str,
                      contexts: list[str] | None = None):
    """RAGAS-style grading of one Q/A pair (faithfulness, relevancy,
    context precision/recall). Falls back to retrieving contexts for the
    question when none are supplied."""
    from . import eval as _eval
    ctxs = contexts
    if not ctxs:
        hits = retrieve.search(question, 6)
        ctxs = [h.get("text", "") for h in hits]
    return _eval.evaluate_answer(question, answer, ctxs)


def t_rag_answer(question: str, k: int = 8):
    """One-shot RAG answer: retrieve passages, then answer with the
    configured LLM. Without an LLM endpoint it returns passages + a note."""
    cfg = retrieve.config.load()
    try:
        res = advanced_retrieve.advanced_search(
            question, k=int(k), kind=None, mode="auto", use_loop=True)
        hits, diags = res["passages"], res["diagrams"]
        method = res.get("method", "advanced")
    except Exception:
        hits = retrieve.search(question, int(k))
        diags, method = retrieve.diagrams_for(hits), "basic_bm25"
    out = {"question": question, "method": method,
           "passages": hits, "diagrams": diags, "answer": None}
    if not (cfg.get("llm_base_url") or "").strip():
        out["note"] = "no LLM configured — passages only"
        return out
    try:
        from .llm.manager import llm_manager
        ctx, _ = advanced_retrieve.build_advanced_context(
            question, k=len(hits), cfg=cfg)
        sys_prompt = (str(cfg.get("llm_system_prompt") or "").strip()
                      or ("You are a JEE tutor. Answer ONLY from the provided "
                          "excerpts from the student's scanned notes. Cite "
                          "sources as (filename, p.N). Write math in LaTeX "
                          "($...$, $$...$$). If the excerpts don't cover the "
                          "question, say so."))
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content":
                     f"NOTES EXCERPTS:\n{ctx}\n\nQUESTION: {question}"}]
        mt = cfg.get("llm_max_tokens")
        out["answer"] = "".join(llm_manager(cfg).chat(
            messages, float(cfg.get("llm_temperature", 0.2)),
            int(mt) if mt else None)).strip()
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


TOOLS = {
    "search_notes": {
        "fn": t_search_notes,
        "description": "BM25 search over all indexed note pages. Returns passages "
                       "with file, topic, subject, page, score, plus diagram urls "
                       "for those pages.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "default": 8, "description": "how many passages"},
            "kind": {"type": "string", "description": "filter by kind: 'notes' or 'pyq'"}},
            "required": ["query"]},
    },
    "search_diagrams": {
        "fn": t_search_diagrams,
        "description": "Search the caption index over every diagram crop and get "
                       "image urls back — use for 'show me the figure of …'.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 8}},
            "required": ["query"]},
    },
    "get_page": {
        "fn": t_get_page,
        "description": "Full OCR'd text of one page plus its diagram image paths.",
        "schema": {"type": "object", "properties": {
            "file": {"type": "string", "description": "file name as returned by search"},
            "page": {"type": "integer", "description": "1-based page number"}},
            "required": ["file", "page"]},
    },
    "get_diagram": {
        "fn": t_get_diagram,
        "description": "Return a diagram crop as a PNG image (path from "
                       "search_notes / search_diagrams / get_page).",
        "schema": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]},
    },
    "list_topics": {
        "fn": t_list_topics,
        "description": "Every indexed PDF with chapter, topic, subject and page counts.",
        "schema": {"type": "object", "properties": {}},
    },
    "index_stats": {
        "fn": t_index_stats,
        "description": "Counts of files/pages/chunks/diagrams and the notes directory.",
        "schema": {"type": "object", "properties": {}},
    },
    "search_notes_advanced": {
        "fn": t_search_notes_advanced,
        "description": "Advanced search with agentic follow-loop: query expansion, "
                       "RAG-Fusion multi-query + RRF, BM25-HyDE, CRAG relevance "
                       "re-grading, context distillation, and query routing "
                       "(notes vs pyq). Loops up to 3 times to improve results.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "default": 8, "description": "how many passages"},
            "kind": {"type": "string", "description": "filter by kind: 'notes' or 'pyq'"}},
            "required": ["query"]},
    },
    "llm_health": {
        "fn": t_llm_health,
        "description": "Health check for every configured LLM endpoint "
                       "(multi-LLM ensemble / 'RAG of LLMs'). Returns ok/model/reason "
                       "for each endpoint so you can see which providers are alive.",
        "schema": {"type": "object", "properties": {}},
    },
    "search_pyq": {
        "fn": t_search_pyq,
        "description": "Search restricted to previous-year questions (kind='pyq') "
                       "— past JEE questions from the question bank.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "default": 8}},
            "required": ["query"]},
    },
    "get_page_context": {
        "fn": t_get_page_context,
        "description": "OCR text of one page PLUS `radius` neighbouring pages — "
                       "handwritten notes continue across pages, so fetch the "
                       "surrounding context before answering from a single page.",
        "schema": {"type": "object", "properties": {
            "file": {"type": "string"},
            "page": {"type": "integer"},
            "radius": {"type": "integer", "default": 1,
                       "description": "pages on each side (0-3)"}},
            "required": ["file", "page"]},
    },
    "harvest_formulas": {
        "fn": t_harvest_formulas,
        "description": "Harvest formula/relation lines (LaTeX folded to readable "
                       "Unicode) + recap sentences + diagram URIs for a topic — "
                       "builds a revision-sheet style summary from the scans.",
        "schema": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "e.g. 'friction', 'rotational motion'"},
            "k": {"type": "integer", "default": 6, "description": "pages to draw from"}},
            "required": ["topic"]},
    },
    "add_study_card": {
        "fn": t_add_study_card,
        "description": "Add a question/answer card to the SM-2 spaced-repetition "
                       "Doubt Deck; the student reviews it in the web UI.",
        "schema": {"type": "object", "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "file": {"type": "string", "description": "source PDF name"},
            "page": {"type": "integer", "description": "source page"},
            "topic": {"type": "string"}},
            "required": ["question", "answer"]},
    },
    "delete_study_card": {
        "fn": t_delete_study_card,
        "description": "Remove one card from the Doubt Deck by id.",
        "schema": {"type": "object", "properties": {
            "id": {"type": "integer"}}, "required": ["id"]},
    },
    "study_stats": {
        "fn": t_study_stats,
        "description": "Weak-topic usage summary: which chapters the student has "
                       "been asking about over the last N days.",
        "schema": {"type": "object", "properties": {
            "days": {"type": "integer", "default": 7}},
        },
    },
    "recent_queries": {
        "fn": t_recent_queries,
        "description": "Most recent logged queries (newest first) with their "
                       "topics and files — good for 'what was I studying?'",
        "schema": {"type": "object", "properties": {
            "limit": {"type": "integer", "default": 10}},
        },
    },
    "evaluate_answer": {
        "fn": t_evaluate_answer,
        "description": "RAGAS-style grading of one Q/A pair: faithfulness, answer "
                       "relevancy, context precision and recall. Retrieval-only "
                       "scores when no LLM is configured.",
        "schema": {"type": "object", "properties": {
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "contexts": {"type": "array", "items": {"type": "string"},
                         "description": "optional contexts; retrieved if omitted"}},
            "required": ["question", "answer"]},
    },
    "rag_answer": {
        "fn": t_rag_answer,
        "description": "One-shot RAG: retrieve passages then answer with the "
                       "configured LLM (non-streaming). Use for pipeline-style "
                       "integrations; without an LLM it returns passages + note.",
        "schema": {"type": "object", "properties": {
            "question": {"type": "string"},
            "k": {"type": "integer", "default": 8}},
            "required": ["question"]},
    },
}


def call_tool(name, args):
    """Run one tool, return MCP content list. Logs every call to the audit log."""
    spec = TOOLS.get(name)
    if not spec:
        return [{"type": "text", "text": f"unknown tool: {name}"}], True
    t0 = time.time()
    try:
        out = spec["fn"](**(args or {}))
        audit_log("tool_call", {
            "tool": name,
            "args_keys": list((args or {}).keys()),
            "ok": True,
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })
    except Exception as e:
        audit_log("tool_call", {
            "tool": name,
            "args_keys": list((args or {}).keys()),
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "duration_ms": round((time.time() - t0) * 1000, 1),
        })
        return [{"type": "text", "text": f"{type(e).__name__}: {e}"}], True
    if isinstance(out, dict) and "__image__" in out:
        p = out["__image__"]
        return [{"type": "image", "data": base64.b64encode(p.read_bytes()).decode(),
                 "mimeType": "image/png"}], False
    return [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=str)}], False


# ---------------------------------------------------------------- resources
def resource_list():
    res = [{"uri": "jeerag://notes", "name": "Indexed PDFs",
            "mimeType": "text/markdown",
            "description": "Every indexed note PDF with its topic and page count."}]
    con = retrieve.connect()
    rows = con.execute("SELECT DISTINCT file FROM diagrams WHERE caption IS NOT NULL "
                       "ORDER BY file").fetchall()
    con.close()
    for r in rows:
        res.append({"uri": f"jeerag://diagram/{quote(r[0])}", "name": r[0],
                    "mimeType": "image/png"})
    return res


def resource_read(uri: str):
    if uri == "jeerag://notes":
        rows = retrieve.topics()
        md = ["# jeerag notes index", "",
              f"{len(rows)} PDFs. Page links: `jeerag://notes/<file>/p<N>`.", ""]
        for r in rows:
            md.append(f"- **{r['file']}** — {(r.get('topic') or '?')} "
                      f"({r.get('subject') or '?'}), {r['pages']} pages")
        return [{"uri": uri, "mimeType": "text/markdown", "text": "\n".join(md)}]
    if uri.startswith("jeerag://diagram/"):
        path = unquote(uri[len("jeerag://diagram/"):])
        p = _diagram_abs(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return [{"uri": uri, "mimeType": "image/png",
                 "blob": base64.b64encode(p.read_bytes()).decode()}]
    if uri.startswith("jeerag://notes/"):
        rest = unquote(uri[len("jeerag://notes/"):])
        file, _, page = rest.rpartition("/p")
        if not file or not page.isdigit():
            raise ValueError(f"bad uri: {uri}")
        out = retrieve.page_text(file, int(page))
        if not out:
            raise FileNotFoundError(uri)
        lines = [f"# {file} — p.{page}", "", out.get("text", ""), ""]
        if out.get("diagrams"):
            lines.append("## diagrams")
            for d in out["diagrams"]:
                lines.append(f"- jeerag://diagram/{quote(d['path'])}")
        return [{"uri": uri, "mimeType": "text/markdown", "text": "\n".join(lines)}]
    raise FileNotFoundError(uri)


# ---------------------------------------------------------------- JSON-RPC
def handle(req):
    """One JSON-RPC request → response dict, or None for notifications."""
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None,
                "error": {"code": -32600, "message": "invalid request"}}
    mid = req.get("id")
    method = req.get("method", "")
    params = req.get("params") or {}
    if mid is None:                      # notification: never answer
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, msg):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}

    try:
        if method == "initialize":
            return ok({"protocolVersion": PROTOCOL_VERSION,
                       "capabilities": {"tools": {"listChanged": False},
                                        "resources": {"subscribe": False,
                                                      "listChanged": False}},
                       "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS})
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": [{"name": n, "description": s["description"],
                                  "inputSchema": s["schema"]}
                                 for n, s in TOOLS.items()]})
        if method == "tools/call":
            content, is_error = call_tool(params.get("name", ""),
                                          params.get("arguments") or {})
            return ok({"content": content, "isError": is_error})
        if method == "resources/list":
            return ok({"resources": resource_list()})
        if method == "resources/read":
            return ok({"contents": resource_read(params.get("uri", ""))})
        return err(-32601, f"method not found: {method}")
    except FileNotFoundError as e:
        return err(-32002, f"resource not found: {e}")
    except Exception as e:
        return err(-32603, f"{type(e).__name__}: {e}")


def run_stdio():
    """Serve newline-delimited JSON-RPC on stdin/stdout until EOF."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}
        else:
            if isinstance(req, list):                 # batch
                resps = [r for r in (handle(x) for x in req) if r]
                resp = resps or None
            else:
                resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def run_sdk():
    """Prefer the official SDK when it is installed (nicer client metadata)."""
    try:
        from mcp.server.mcpserver import MCPServer as _Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as _Server
        except ImportError:
            return run_stdio()
    server = _Server("jeerag", instructions=INSTRUCTIONS)
    for name, spec in TOOLS.items():
        def _wrap(_fn=spec["fn"], **_kw):
            out = _fn(**_kw)
            if isinstance(out, dict) and "__image__" in out:
                from mcp.types import ImageContent
                return ImageContent(type="image",
                                    data=base64.b64encode(out["__image__"].read_bytes()).decode(),
                                    mimeType="image/png")
            return out

        _wrap.__name__ = name
        _wrap.__doc__ = spec["description"]
        server.tool(name=name)(_wrap)
    server.run()


if __name__ == "__main__":
    run_sdk()
