"""Weak-topic tracker: log queries + retrieved topics, weekly summary.

Every /api/search and /api/chat logs its question and the topics of the
retrieved passages into data/querylog.db. `jeerag tracker --days N` prints
"you asked 14 questions from Rotation, 2 from Fluids".
"""
import json
import sqlite3
import time
from pathlib import Path

from . import config

DB_PATH = Path(config.load()["data_dir"]) / "querylog.db"

SCHEMA = """CREATE TABLE IF NOT EXISTS queries(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, question TEXT, topics TEXT, files TEXT, source TEXT);"""


def log_query(question: str, hits, source: str = "chat") -> None:
    """Record one query with the topics/files it retrieved. Never raises."""
    try:
        topics = sorted({(h.get("topic") or "?") for h in hits})[:8]
        files = sorted({(h.get("file") or "?") for h in hits})[:8]
        con = sqlite3.connect(DB_PATH, timeout=30)
        con.execute(SCHEMA)
        con.execute(
            "INSERT INTO queries(ts,question,topics,files,source) VALUES(?,?,?,?,?)",
            (time.time(), question[:500], json.dumps(topics), json.dumps(files), source))
        con.commit()
        con.close()
    except Exception:
        pass  # logging must never break answering


def summary(days: int = 7, limit: int = 15) -> dict:
    """Topic usage counts over the last `days` days, weakest-usage first view."""
    since = time.time() - days * 86400
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute(SCHEMA)
    rows = con.execute("SELECT topics FROM queries WHERE ts>=?", (since,)).fetchall()
    con.close()
    counts: dict = {}
    for (tj,) in rows:
        for t in json.loads(tj):
            counts[t] = counts.get(t, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return {"days": days, "queries": len(rows),
            "topics": [{"topic": t, "count": c} for t, c in ranked]}


def recent(limit: int = 10) -> list[dict]:
    """Most recent logged queries (newest first) with their topics and files."""
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute(SCHEMA)
    rows = con.execute(
        "SELECT ts, question, topics, files, source FROM queries "
        "ORDER BY ts DESC LIMIT ?", (max(1, min(int(limit), 100)),)).fetchall()
    con.close()
    return [{"ts": r[0], "question": r[1], "topics": json.loads(r[2]),
             "files": json.loads(r[3]), "source": r[4]} for r in rows]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="jeerag tracker")
    ap.add_argument("--days", type=int, default=7,
                    help="look-back window (default 7)")
    a = ap.parse_args(argv)
    s = summary(a.days)
    if not s["queries"]:
        print(f"no queries logged in the last {a.days} days")
        return 0
    print(f"{s['queries']} queries in the last {s['days']} days:")
    for row in s["topics"]:
        print(f"  {row['count']:3d}  {row['topic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
