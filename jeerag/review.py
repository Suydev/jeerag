"""Doubt Deck — the invented feature: turn any chat answer into a spaced-repetition card.

Every answer in the chat UI can be saved as a card ("Doubt Deck").  Cards are
scheduled with a small SM-2 variant (ease + interval growth, lapse handling)
and always keep a link back to the note page and diagram they came from, so
review is never detached from the source.

    python -m jeerag.review              # today's queue
    python -m jeerag.review --stats      # counts, streaks
    python -m jeerag.review --add "q" "a" --file F --page 4
    python -m jeerag.review --grade 12 5 # grade card 12 (0-5, <3 = lapse)

Grading: 0-2 fail (due tomorrow, ease down), 3 hard (+1 day), 4 good, 5 easy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

from . import retrieve

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    file TEXT,
    page INTEGER,
    diagram TEXT,
    topic TEXT,
    due TEXT NOT NULL,
    interval REAL NOT NULL DEFAULT 0,
    ease REAL NOT NULL DEFAULT 2.5,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS cards_due ON cards(due);
"""


def connect():
    con = retrieve.connect()
    con.executescript(SCHEMA)
    return con


def today():
    return dt.date.today().isoformat()


def add(question, answer, file=None, page=None, diagram=None, topic=None):
    con = connect()
    cur = con.execute(
        "INSERT INTO cards(question, answer, file, page, diagram, topic, due, created) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (question.strip(), answer.strip(), file, page, diagram, topic,
         today(), dt.datetime.now().isoformat(timespec="seconds")))
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def due(limit=20):
    con = connect()
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM cards WHERE due <= ? ORDER BY due, id LIMIT ?", (today(), limit))]
    con.close()
    return rows


def grade(card_id, score):
    """SM-2-lite. score 0-5."""
    score = max(0, min(5, int(score)))
    con = connect()
    row = con.execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
    if not row:
        con.close()
        return None
    interval, ease = row["interval"], row["ease"]
    if score < 3:
        lapses = row["lapses"] + 1
        interval = 1
        ease = max(1.3, ease - 0.2)
    else:
        lapses = row["lapses"]
        ease = min(3.0, ease + (0.1 if score == 5 else 0.0) - (0.15 if score == 3 else 0.0))
        if row["reps"] == 0:
            interval = 1
        elif row["reps"] == 1:
            interval = 3 if score >= 4 else 2
        else:
            interval = max(1, round(interval * ease))
    nxt = (dt.date.today() + dt.timedelta(days=int(interval))).isoformat()
    con.execute("UPDATE cards SET due=?, interval=?, ease=?, reps=?, lapses=? WHERE id=?",
                (nxt, float(interval), float(ease), row["reps"] + 1, lapses, card_id))
    con.commit()
    con.close()
    return {"id": card_id, "due": nxt, "interval": interval, "ease": round(ease, 2)}


def delete(card_id) -> bool:
    """Remove one card from the deck. True if it existed."""
    try:
        cid = int(card_id)
    except (TypeError, ValueError):
        return False
    con = connect()
    cur = con.execute("DELETE FROM cards WHERE id=?", (cid,))
    con.commit()
    con.close()
    return cur.rowcount > 0


def stats():
    con = connect()
    total = con.execute("SELECT COUNT(*) c FROM cards").fetchone()["c"]
    today_n = con.execute("SELECT COUNT(*) c FROM cards WHERE due <= ?",
                          (today(),)).fetchone()["c"]
    mature = con.execute("SELECT COUNT(*) c FROM cards WHERE interval >= 21").fetchone()["c"]
    lapses = con.execute("SELECT COALESCE(SUM(lapses),0) c FROM cards").fetchone()["c"]
    con.close()
    return {"total": total, "due_today": today_n, "mature": mature, "lapses": lapses}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Doubt Deck — spaced repetition over the notes")
    ap.add_argument("--add", nargs=2, metavar=("QUESTION", "ANSWER"))
    ap.add_argument("--file")
    ap.add_argument("--page", type=int)
    ap.add_argument("--diagram")
    ap.add_argument("--topic")
    ap.add_argument("--grade", nargs=2, metavar=("CARD_ID", "SCORE"), type=int)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)
    if args.add:
        cid = add(*args.add, file=args.file, page=args.page,
                  diagram=args.diagram, topic=args.topic)
        print(f"card {cid} saved, due today")
        return 0
    if args.grade:
        print(grade(*args.grade))
        return 0
    if args.stats:
        print(stats())
        return 0
    queue = due(args.limit)
    if not queue:
        print("nothing due — add cards from the chat UI or --add")
        return 0
    for c in queue:
        print(f"#{c['id']}  [{c['topic'] or c['file'] or 'general'}]")
        print(f"  Q: {c['question']}")
        print(f"  A: {c['answer'][:300]}")
        if c["file"]:
            print(f"  src: {c['file']} p.{c['page']}")
        print()
    print(f"{len(queue)} due · grade with --grade ID SCORE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
