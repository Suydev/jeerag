"""Shared retrieval core over the SQLite FTS5 index."""
import os
import re
import sqlite3
from pathlib import Path

from . import config


def db_path():
    return Path(config.load()["data_dir"]) / "index.db"


def connect():
    con = sqlite3.connect(db_path(), timeout=30)
    con.row_factory = sqlite3.Row
    # Migration: add kind column to existing files tables that lack it
    cols = {r[1] for r in con.execute("PRAGMA table_info(files)")}
    if "kind" not in cols:
        con.execute("ALTER TABLE files ADD COLUMN kind TEXT DEFAULT 'notes'")
        con.commit()
    return con


def sanitize_query(q: str) -> str:
    terms = re.findall(r"[A-Za-z0-9_\-\u0080-\u024F]+(?:\.\d+)?", q)
    stop = {"what", "is", "the", "a", "an", "of", "in", "on", "how", "does", "do",
            "explain", "tell", "me", "and", "to", "for", "with", "why", "when",
            "find", "derive", "prove", "can", "you"}
    terms = [t for t in terms if t.lower() not in stop and len(t) > 1]
    if not terms:
        terms = re.findall(r"[A-Za-z0-9_\-\u0080-\u024F]+", q)
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def search(question: str, k: int | None = None, kind: str | None = None):
    cfg = config.load()
    k = k or cfg["top_k"]
    con = connect()
    q = sanitize_query(question)
    hits = []
    kind_clause = " AND f.kind = ?" if kind else ""
    kind_params = ([kind] if kind else [])
    try:
        hits = [dict(r) for r in con.execute(
            f"""SELECT c.text, c.topic, c.subject, c.file, c.page, bm25(chunks) AS score
               FROM chunks c JOIN files f ON c.file = f.file
               WHERE chunks MATCH ?{kind_clause} ORDER BY score LIMIT ?""",
            (q, *kind_params, k))]
    except sqlite3.OperationalError:
        pass
    if len(hits) < 3:  # topic-name fallback
        first = (re.findall(r"[A-Za-z]{4,}", question) or [""])[0]
        params = [f"%{first}%"] + kind_params + [k]
        extra = [dict(r) for r in con.execute(
            f"""SELECT c.text, c.topic, c.subject, c.file, c.page, -9.0 AS score
               FROM chunks c JOIN files f ON c.file = f.file
               WHERE f.topic LIKE ?{kind_clause} LIMIT ?""",
            tuple(params))]
        seen = {(h["file"], h["page"]) for h in hits}
        hits += [e for e in extra if (e["file"], e["page"]) not in seen]
    con.close()
    return hits


def related_pyq(question: str, k: int = 4, limit: int = 3):
    """Past-year-question / official-key chunks that clearly match a question.

    Searches ``kind='pyq'`` only — official JEE papers and answer keys fetched
    via ``python -m jeerag.pyq_sources`` and indexed with ``--kind pyq``.  fts5
    bm25 scores are lower-is-better (negative); rows at the constant fallback
    score -9.0 are topic-name guesses and are used only when nothing stronger
    matched.  Returns ``[]`` when nothing plausibly matches so note-only
    questions don't get PYQ noise injected.
    """
    hits = [h for h in search(question, k=k, kind="pyq") if h["score"] < 0]
    strong = [h for h in hits if h["score"] != -9.0] or hits
    strong.sort(key=lambda h: h["score"])
    return [{"file": h["file"], "page": h["page"], "topic": h["topic"],
             "score": h["score"], "text": h["text"][:280]}
            for h in strong[:limit]]


def build_context(hits, max_chars=9000):
    parts = []
    used = 0
    for h in hits:
        block = f"[{h['file']} | {h['topic']} | p.{h['page']}]\n{h['text'][:1200]}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def diagrams_for(hits, limit=6):
    con = connect()
    out = []
    for h in hits:
        rows = con.execute(
            "SELECT path, page FROM diagrams WHERE file=? AND page=? LIMIT 3",
            (h["file"], h["page"] - 1)).fetchall()
        for r in rows:
            out.append({"file": h["file"], "page": r["page"] + 1,
                        "path": r["path"],
                        "url": "/" + _rel(r["path"])})
        if len(out) >= limit:
            break
    con.close()
    return out[:limit]


def _rel(path):
    """Normalise a stored diagram path to one relative to data_dir.

    Indexes built before the fix store absolute paths; newer ones already
    store ``diagrams/...``.  Be idempotent so both work."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    data_dir = Path(config.load()["data_dir"]).resolve()
    try:
        return p.resolve().relative_to(data_dir).as_posix()
    except ValueError:
        # foreign absolute path: fall back to the part after .../data/
        parts = p.parts
        if "data" in parts:
            return "/".join(parts[parts.index("data") + 1:])
        return p.name


def page_text(file: str, page: int):
    con = connect()
    row = con.execute(
        "SELECT text, source FROM pages WHERE file=? AND page=?",
        (file, page - 1)).fetchone()
    diags = [dict(r) for r in con.execute(
        "SELECT path, page FROM diagrams WHERE file=? AND page=?",
        (file, page - 1))]
    con.close()
    if not row:
        return None
    return {"file": file, "page": page, "text": row["text"], "ocr_source": row["source"],
            "diagrams": [{"path": d["path"], "url": "/" + _rel(d["path"])} for d in diags]}


def topics(kind: str | None = None):
    con = connect()
    kind_clause = " WHERE kind = ?" if kind else ""
    params = (kind,) if kind else ()
    rows = [dict(r) for r in con.execute(
        f"SELECT file, chapter, topic, subject, kind, pages, text_pages, ocr_pages FROM files {kind_clause} "
        "ORDER BY CAST(chapter AS INT)", params)]
    con.close()
    return rows


def stats():
    con = connect()
    out = {
        "files": con.execute("SELECT COUNT(*) c FROM files").fetchone()["c"],
        "pages": con.execute("SELECT COUNT(*) c FROM pages").fetchone()["c"],
        "chunks": con.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"],
        "diagrams": con.execute("SELECT COUNT(*) c FROM diagrams").fetchone()["c"],
        "notes_dir": config.load()["notes_dir"],
        "by_kind": [dict(r) for r in con.execute(
            "SELECT kind, COUNT(*) c FROM files GROUP BY kind")],
    }
    con.close()
    return out
