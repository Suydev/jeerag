"""Give every diagram crop a searchable caption.

Two modes, same output shape (``diagrams.caption`` + ``diagram_fts``):

* **vision** (``--vision`` with ``llm_vision_model`` configured) — the
  multimodal model looks at the crop and writes a <=25 word, LaTeX-aware
  caption.  Best quality; needs an endpoint.
* **offline** (default) — deterministic and free: the diagram inherits the
  distinctive keywords of its own page (that page was OCR'd at index time),
  plus chapter/topic, a coarse shape class, and optionally tesseract labels
  from the crop itself (``--ocr``).  This is what makes "cone of static
  friction" find the cone figure with no model at all.

Run::

    python -m jeerag.caption                # caption everything not yet done
    python -m jeerag.caption --limit 50     # sample first
    python -m jeerag.caption --force        # redo all
    python -m jeerag.caption --vision       # use the vision model
    python -m jeerag.caption --ocr          # also OCR the crops (slow, ~1.2s/crop)
    python -m jeerag.caption --search "cone"
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from . import config, retrieve

CAPTION_MAX_WORDS = 25
SHAPE_WORDS = {
    "tall": "vertical figure", "wide": "wide plot or graph",
    "square": "square figure", "strip": "short horizontal sketch",
}
STOP = {
    "a", "an", "the", "of", "on", "in", "to", "and", "or", "is", "are", "was",
    "for", "with", "by", "at", "as", "it", "this", "that", "then", "than",
    "from", "be", "can", "will", "we", "so", "if", "not", "no", "also", "thus",
    "hence", "therefore", "now", "here", "there", "which", "when", "where",
    "what", "how", "why", "who", "will", "shall", "may", "must", "should",
    "ambarish", "sir", "notes", "youtube", "https", "www", "com", "physics",
    "maths", "page", "pdf", "min", "t25j1", "thephysicsshelter",
    "ambarishphysics", "tuition", "unacademy",
}


def ensure_schema(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(diagrams)")}
    if "caption" not in cols:
        con.execute("ALTER TABLE diagrams ADD COLUMN caption TEXT")
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS diagram_fts USING fts5("
        "caption, file UNINDEXED, page UNINDEXED, path UNINDEXED, "
        "tokenize='unicode61')")
    con.commit()


def shape_word(w, h):
    if not w or not h:
        return ""
    ar = w / h
    if ar > 1.9:
        return SHAPE_WORDS["wide"]
    if ar < 0.6:
        return SHAPE_WORDS["tall"]
    if h < 120:
        return SHAPE_WORDS["strip"]
    return SHAPE_WORDS["square"]


def page_keywords(text, limit=10):
    """Distinctive words of a page: rare-ish, meaningful, order-preserving."""
    if not text:
        return []
    counts = Counter()
    first_pos = {}
    for i, tok in enumerate(re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)):
        t = tok.lower().strip("-")
        if t in STOP or len(t) < 4:
            continue
        counts[t] += 1
        first_pos.setdefault(t, i)
    ranked = sorted(counts, key=lambda w: (-min(counts[w], 4), first_pos[w]))
    return ranked[:limit]


def crop_words(path: Path, limit=8):
    """OCR the crop itself (opt-in: ~1.2s per crop on this device)."""
    try:
        import numpy as np
        import pytesseract
        from PIL import Image
    except Exception:
        return []
    try:
        img = Image.open(path).convert("L")
        if img.width < 400:
            img = img.resize((img.width * 2, img.height * 2))
        a = np.asarray(img)
        if a.size and float((a < 200).mean()) < 0.004:  # blank crop
            return []
        txt = pytesseract.image_to_string(img, config="--psm 11")
    except Exception:
        return []
    words, seen = [], set()
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", txt):
        t = tok.lower().strip("-")
        if len(t) < 3 or t in STOP or t in seen:
            continue
        seen.add(t)
        words.append(t)
        if len(words) >= limit:
            break
    return words


def offline_caption(row, topic="", subject="", use_ocr=False):
    path, w, h = row["path"], row["w"], row["h"]
    topic = topic or Path(row["file"]).stem
    bits = [f"{topic} diagram"]
    kws = page_keywords(row.get("page_text", ""))
    if kws:
        bits.append("keywords: " + " ".join(kws))
    if use_ocr:
        labels = crop_words(Path(config.load()["data_dir"]) / path)
        if labels:
            bits.append("labels: " + " ".join(labels))
    shape = shape_word(w, h)
    if shape:
        bits.append(shape)
    bits.append(f"p.{row['page'] + 1}")
    return " ".join(", ".join(bits).split()[:CAPTION_MAX_WORDS * 3])


VISION_PROMPT = (
    "You caption figures from scanned IIT-JEE physics/maths notes for a search "
    "index. Describe what the figure SHOWS in at most 25 words: the physical "
    "setup, the labelled quantities and the relationship plotted. Keep LaTeX "
    "symbols as plain text (e.g. 'coefficient of friction mu'). No preamble."
)


def vision_caption(png: Path) -> str:
    from . import llm
    cfg = config.load()
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": VISION_PROMPT},
        llm.image_part(png.read_bytes()),
    ]}]
    model_cfg = dict(cfg)
    model_cfg["llm_model"] = cfg.get("llm_vision_model") or cfg.get("llm_model")
    txt = llm.complete(msgs, model_cfg, 0.1, 120)
    return " ".join(txt.replace("\n", " ").split()[:CAPTION_MAX_WORDS])


def index_caption(con, row_id, caption, file, page, path):
    con.execute("UPDATE diagrams SET caption=? WHERE id=?", (caption, row_id))
    con.execute("DELETE FROM diagram_fts WHERE rowid=?", (row_id,))
    con.execute("INSERT INTO diagram_fts(rowid, caption, file, page, path) "
                "VALUES(?,?,?,?,?)", (row_id, caption, file, page, path))


def _caption_job(row):
    try:
        return offline_caption(row, row.get("topic", ""), row.get("subject", ""),
                               row.get("use_ocr", False))
    except Exception:
        return f"{row.get('topic') or 'notes'} diagram, p.{row['page'] + 1}"


def run(limit=None, force=False, vision=False, use_ocr=False, quiet=False):
    con = retrieve.connect()
    ensure_schema(con)
    cfg = config.load()
    data_dir = Path(cfg["data_dir"])
    if vision and not cfg.get("llm_vision_model"):
        msg = "vision mode needs llm_vision_model in config.json — using offline captions"
        if not quiet:
            print(msg)
        vision = False
    where = "" if force else "WHERE caption IS NULL OR caption=''"
    rows = [dict(r) for r in con.execute(
        f"SELECT id, file, page, path, w, h, caption FROM diagrams {where} ORDER BY id")]
    if limit:
        rows = rows[:limit]
    total = len(rows)
    if not quiet:
        print(f"captioning {total} diagrams "
              f"({'vision' if vision else 'offline'}{', +crop OCR' if use_ocr else ''})",
              flush=True)
    meta = {r["file"]: (r["topic"] or "", r["subject"] or "")
            for r in con.execute("SELECT file, topic, subject FROM files")}
    page_text = {(r["file"], r["page"]): r["text"]
                 for r in con.execute("SELECT file, page, text FROM pages")}
    for row in rows:
        t, s = meta.get(row["file"], ("", ""))
        row["topic"], row["subject"] = t, s
        row["page_text"] = page_text.get((row["file"], row["page"]), "")
        row["use_ocr"] = use_ocr
    done = 0
    if vision:
        for row in rows:
            png = data_dir / row["path"]
            try:
                cap = vision_caption(png) if png.exists() else _caption_job(row)
            except Exception as e:
                cap = _caption_job(row)
                if not quiet:
                    print(f"  fallback {row['path']}: {type(e).__name__}: {e}", flush=True)
            index_caption(con, row["id"], cap, row["file"], row["page"], row["path"])
            done += 1
            if done % 25 == 0:
                con.commit()
                if not quiet:
                    print(f"  {done}/{total}", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        workers = int(os.environ.get("JEERAG_WORKERS", "4"))
        chunks = max(1, total // (workers * 8) or 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row, cap in zip(rows, ex.map(_caption_job, rows, chunksize=chunks)):
                index_caption(con, row["id"], cap, row["file"], row["page"], row["path"])
                done += 1
                if done % 500 == 0:
                    con.commit()
                    if not quiet:
                        print(f"  {done}/{total}", flush=True)
    con.commit()
    con.close()
    if not quiet:
        print(f"done: {done} captions")
    return done


def search(query, k=8):
    con = retrieve.connect()
    ensure_schema(con)
    q = retrieve.sanitize_query(query)
    out = []
    if q:
        try:
            out = [dict(r) for r in con.execute(
                """SELECT d.path, d.file, d.page, d.caption, d.w, d.h,
                          bm25(diagram_fts) AS score
                   FROM diagram_fts JOIN diagrams d ON d.id = diagram_fts.rowid
                   WHERE diagram_fts MATCH ? ORDER BY score LIMIT ?""", (q, k))]
        except sqlite3.OperationalError:
            out = []
    for r in out:
        r["page"] = r["page"] + 1
        r["url"] = "/" + retrieve._rel(r["path"])
    con.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="caption diagram crops for search")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--ocr", action="store_true", help="also OCR each crop (slow)")
    ap.add_argument("--search", help="query the caption index and exit")
    args = ap.parse_args(argv)
    if args.search:
        import json
        print(json.dumps(search(args.search), indent=1))
        return 0
    run(args.limit, args.force, args.vision, args.ocr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
