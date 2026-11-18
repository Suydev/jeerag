"""Tool registry for the in-stream agentic chat layer.

Each tool wraps an existing jeerag backend (retrieve / pyq_sources / review /
tracker / formulas / caption) so the LLM can call it mid-stream by emitting a
marker line::

    TOOL search_pyqs {"query": "rotation"}

The webapp chat handler scans the model's stream for that line, executes the
tool (with a timeout), feeds the result back as context, and emits
``tool`` / ``tool_result`` SSE events; the frontend renders each result as a
card.  This module holds the registry (``SPEC``), the runner (``execute``)
and the system-prompt block that teaches the marker format.

Payload contract: every ``run()`` returns a dict starting with ``"ok"`` and
never raises; results are size-capped (~6 KB) — citations and URLs, not full
texts.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from urllib.parse import quote

from . import retrieve

MAX_PAYLOAD = 6000  # bytes of JSON
ROOT = Path(__file__).resolve().parent.parent


# --- helpers ----------------------------------------------------------------

def page_url(file: str, page: int) -> str:
    """Viewer URL for a scanned page (1-based, same shape as webapp /pdfpage/)."""
    return f"/pdfpage/{quote(str(file))}/{int(page)}.png"


def _cap_str(v: str, n: int) -> str:
    return v if len(v) <= n else v[:n] + "…"


def _cap(obj, limit=MAX_PAYLOAD):
    """Shrink a payload to <= ``limit`` JSON bytes: trim long strings, then
    drop tail items of the biggest lists."""
    def trim(o):
        if isinstance(o, str):
            return _cap_str(o, 300)
        if isinstance(o, list):
            return [trim(x) for x in o[:20]]
        if isinstance(o, dict):
            return {k: trim(v) for k, v in o.items()}
        return o

    obj = trim(obj)
    while len(json.dumps(obj, ensure_ascii=False, default=str)) > limit:
        lists = [(len(v), k) for k, v in obj.items()
                 if isinstance(v, list) and v]
        if not lists:
            obj = _cap_str(json.dumps(obj, ensure_ascii=False, default=str), limit)
            break
        n, k = max(lists)
        obj[k] = obj[k][: max(1, n // 2)]
    return obj


def _passage(h: dict) -> dict:
    return {"file": h["file"], "page": h["page"],
            "topic": h.get("topic") or "", "text": _cap_str(h.get("text", ""), 280),
            "url": page_url(h["file"], h["page"])}


# --- tools ------------------------------------------------------------------

def _t_search_notes(query: str, k: int = 5) -> dict:
    k = max(1, min(int(k), 6))
    hits = retrieve.search(str(query), k=k)
    return {"ok": True, "passages": [_passage(h) for h in hits]}


def _t_search_pyqs(query: str, k: int = 8) -> dict:
    k = max(1, min(int(k), 12))
    hits = retrieve.search(str(query), k=k, kind="pyq")
    years: dict = {}
    con = retrieve.connect()
    try:
        for r in con.execute(
                "SELECT chapter, COUNT(*) n FROM files WHERE kind='pyq' "
                "GROUP BY chapter ORDER BY chapter"):
            if r["chapter"]:
                years[r["chapter"]] = r["n"]
    finally:
        con.close()
    return {"ok": True, "passages": [_passage(h) for h in hits],
            "diagrams": retrieve.diagrams_for(hits, limit=4),
            "years": years}


def _t_pyq_solution(qno, key_pdf: str = "", page: int = 0) -> dict:
    """Official answer for one question number from the keys sidecar JSONs."""
    qno = str(qno).strip().lstrip("Qq. ")
    keys_dir = ROOT / "papers"
    out = []
    for side in sorted(keys_dir.glob("**/keys/*.answers.json")):
        if key_pdf and key_pdf.lower() not in side.name.lower():
            continue
        try:
            data = json.loads(side.read_text())
        except Exception:
            continue
        if qno in (data.get("answers") or {}):
            out.append({"file": data.get("file", side.stem), "qno": qno,
                        "answer": data["answers"][qno]})
    res = {"ok": bool(out), "answers": out}
    if out and page:
        pt = retrieve.page_text(out[0]["file"], int(page))
        if pt:
            res["ocr"] = _cap_str(pt["text"], 1200)
    if not out:
        res["error"] = f"no official key found for Q{qno}"
    return res


_QNO_RE = re.compile(r"(?:^|\n)\s*(?:Q\.?\s*)?(\d{1,3})[\.\)\s]", re.M)

def _t_build_paper(topic: str, n: int = 8, years=None) -> dict:
    """Mixed PYQ paper: n questions spread across years, official keys attached."""
    n = max(1, min(int(n), 10))
    hits = retrieve.search(str(topic), k=min(60, n * 6), kind="pyq")
    by_year: dict = {}
    for h in hits:
        by_year.setdefault(h.get("topic") or "misc", []).append(h)
    want = [str(y) for y in (years or [])]
    picked, seen = [], set()
    order = sorted(by_year, key=lambda t: -len(by_year[t]))
    for t in order:  # round-robin across years/chapters for the "mix"
        for h in by_year[t]:
            key = (h["file"], h["page"])
            if key in seen or len(picked) >= n:
                continue
            if want and not any(w in (h["file"] + t) for w in want):
                continue
            seen.add(key)
            m = _QNO_RE.search((h.get("text") or "")[:200])
            q = _passage(h)
            q["qno"] = m.group(1) if m else None
            picked.append(q)
            if len(picked) >= n:
                break
        if len(picked) >= n:
            break
    # attach official answers where a qno is parseable
    answers: dict = {}
    for side in sorted((ROOT / "papers").glob("**/keys/*.answers.json")):
        try:
            answers.update(json.loads(side.read_text()).get("answers") or {})
        except Exception:
            pass
    for q in picked:
        if q["qno"] and q["qno"] in answers:
            q["official_answer"] = answers[q["qno"]]
    return {"ok": True, "topic": str(topic), "questions": picked,
            "note": ("questions mixed across papers; official answers attached "
                     "where a question number was found" if picked
                     else "no PYQ passages matched this topic — try search_pyqs "
                          "first or a different topic")}


def _t_diagrams(query: str, n: int = 4) -> dict:
    from . import caption
    rows = caption.search(str(query), k=max(1, min(int(n), 6)))
    out = [{"file": r.get("file"), "page": r.get("page"),
            "path": r.get("path"), "url": "/" + r.get("path", "").lstrip("/")}
           for r in rows]
    return {"ok": bool(out), "diagrams": out}


def _t_open_page(file: str, page: int) -> dict:
    pt = retrieve.page_text(str(file), int(page))
    if not pt:
        return {"ok": False, "error": f"page not found: {file} p.{page}"}
    return {"ok": True, "file": file, "page": int(page),
            "text": _cap_str(pt["text"], 2000), "url": page_url(file, page),
            "diagrams": pt["diagrams"]}


def _t_solve(question: str) -> dict:
    """Context grounding for the main LLM loop — solving stays with the model."""
    hits = retrieve.search(str(question), k=3)
    return {"ok": True, "note": "handled by main loop",
            "context": [_passage(h) for h in hits]}


def _t_study_stats(days: int = 7) -> dict:
    from .tracker import recent, summary
    days = max(1, min(int(days), 365))
    s = summary(days)
    return {"ok": True, "summary": s, "recent": recent(5)}


def _t_formulas(topic: str, k: int = 6) -> dict:
    from .formulas import collect
    sections = collect(str(topic), k=max(1, min(int(k), 8)))
    for s in sections:
        s["images"] = [{"path": p, "url": "/" + p} for p in s.get("images", [])]
        s["latex"] = [_cap_str(x, 160) for x in s.get("latex", [])[:8]]
    return {"ok": bool(sections), "topic": str(topic), "sections": sections}


def _t_due_cards() -> dict:
    from .review import due, stats
    cards = due()
    for c in cards:
        c["question"] = _cap_str(c.get("question", ""), 200)
        c["answer"] = _cap_str(c.get("answer", ""), 160)
    return {"ok": True, "due": cards, "stats": stats()}


SPEC = {
    "search_notes": {
        "desc": "Search the notes corpus (BM25) for passages on a topic.",
        "args_schema": {"query": "str (required)", "k": "int <=6 (default 5)"},
        "run": _t_search_notes,
    },
    "search_pyqs": {
        "desc": "Search past-year JEE questions (PYQ corpus) for a topic; returns passages, diagram crops and per-year counts.",
        "args_schema": {"query": "str (required)", "k": "int <=12 (default 8)"},
        "run": _t_search_pyqs,
    },
    "pyq_solution": {
        "desc": "Official answer for one question number from the answer-key PDFs.",
        "args_schema": {"qno": "str|int (required)", "key_pdf": "str filter (optional)",
                        "page": "int (optional, adds OCR text)"},
        "run": _t_pyq_solution,
    },
    "build_paper": {
        "desc": "Build a mixed PYQ question paper on a topic (n questions across years, official answers attached).",
        "args_schema": {"topic": "str (required)", "n": "int <=10 (default 8)",
                        "years": "list[str] (optional filter)"},
        "run": _t_build_paper,
    },
    "diagrams": {
        "desc": "Find figure/diagram crops matching a caption query.",
        "args_schema": {"query": "str (required)", "n": "int <=6 (default 4)"},
        "run": _t_diagrams,
    },
    "open_page": {
        "desc": "Open one scanned page: OCR text + viewer URL + its diagrams.",
        "args_schema": {"file": "str (required)", "page": "int 1-based (required)"},
        "run": _t_open_page,
    },
    "solve": {
        "desc": "Ground a question with 3 passages for the main answer loop.",
        "args_schema": {"question": "str (required)"},
        "run": _t_solve,
    },
    "study_stats": {
        "desc": "Study tracker summary (topics/queries over N days) + recent queries.",
        "args_schema": {"days": "int (default 7)"},
        "run": _t_study_stats,
    },
    "formulas": {
        "desc": "Harvest formula sections (LaTeX + images) for a topic.",
        "args_schema": {"topic": "str (required)", "k": "int <=8 (default 6)"},
        "run": _t_formulas,
    },
    "due_cards": {
        "desc": "Doubt-deck cards due for review right now.",
        "args_schema": {},
        "run": _t_due_cards,
    },
}


# --- execution ---------------------------------------------------------------

def execute(name: str, args: dict | None = None, timeout_s: float = 8.0) -> dict:
    """Run a tool by name in a thread with a hard timeout; never raises."""
    entry = SPEC.get(name)
    if not entry:
        return {"ok": False, "error": f"unknown tool: {name}",
                "available": sorted(SPEC)}
    args = args or {}
    out: list = []

    def _go():
        try:
            out.append(entry["run"](**args) or {"ok": True})
        except Exception as e:  # noqa: BLE001 — tool boundary must not raise
            out.append({"ok": False, "error": f"{type(e).__name__}: {e}"})

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    th.join(timeout_s)
    if not out:
        return {"ok": False, "error": f"tool timed out after {timeout_s}s"}
    return _cap(out[0])


# --- system prompt -----------------------------------------------------------

def SYSTEM_PROMPT_BLOCK() -> str:
    names = ", ".join(SPEC)
    lines = [
        "TOOLS: you may call tools by emitting, on its own line, exactly:",
        'TOOL <name> {"arg": "value"}',
        f"Available: {names}.",
        "Examples:",
        'TOOL search_pyqs {"query": "rotational motion"}',
        'TOOL pyq_solution {"qno": 12}',
        'TOOL build_paper {"topic": "electrostatics", "n": 6}',
        "Rules: call a tool only when it helps; one per line; wait for its "
        "RESULT (inserted above) before continuing; at most 3 tools per "
        "answer; never mention the TOOL syntax to the user.",
    ]
    return "\n".join(lines)
