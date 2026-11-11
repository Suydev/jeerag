"""MCP server exposing the notes index. Run: python -m jeerag.mcp_server

Deliberately dependency-free: the Model Context Protocol over stdio is
newline-delimited JSON-RPC 2.0, so this speaks it directly instead of pulling
in the ``mcp`` SDK (whose pydantic/cryptography deps do not build on Termux).
Any MCP client — Claude Desktop, an agent SDK, ``mcp-cli`` — can talk to it.

Tools
  search_notes      BM25 over pages, with diagram urls
  search_diagrams   caption/BM25 search over the 2,458 figure crops
  get_page          one page's OCR text + its diagrams
  get_diagram       a diagram crop as a real PNG image
  list_topics       every PDF with topic/subject/page counts
  index_stats       file/page/chunk/diagram counts
Resources
  jeerag://notes          index of every indexed PDF
  jeerag://notes/{file}/p{page}   one page as markdown (text + diagram uris)
  jeerag://diagram/{path} one crop as an image resource
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from urllib.parse import quote, unquote

from . import retrieve

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "jeerag", "version": "1.1.0"}
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


def t_search_notes(query: str, k: int = 8):
    hits = retrieve.search(query, k)
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


TOOLS = {
    "search_notes": {
        "fn": t_search_notes,
        "description": "BM25 search over all indexed note pages. Returns passages "
                       "with file, topic, subject, page, score, plus diagram urls "
                       "for those pages.",
        "schema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search terms"},
            "k": {"type": "integer", "default": 8, "description": "how many passages"}},
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
}


def call_tool(name, args):
    """Run one tool, return MCP content list."""
    spec = TOOLS.get(name)
    if not spec:
        return [{"type": "text", "text": f"unknown tool: {name}"}], True
    try:
        out = spec["fn"](**(args or {}))
    except Exception as e:
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
