#!/usr/bin/env python3
"""Build the JEE-notes RAG index: text-layer extraction + tesseract OCR fallback,
diagram detection, chunking with topic/page tags into SQLite (FTS5)."""
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

from . import config as _config

_cfg = _config.load()
NOTES_DIR = Path(_cfg["notes_dir"])
DATA_DIR = Path(_cfg["data_dir"])
DB_PATH = DATA_DIR / "index.db"
DIAGRAM_DIR = DATA_DIR / "diagrams"
OCR_DPI = int(os.environ.get("JEERAG_DPI", "200"))
WORKERS = int(os.environ.get("JEERAG_WORKERS", "4"))
MIN_CHARS_FOR_TEXT_LAYER = 30  # below this a page counts as a scan
DEFAULT_KIND = os.environ.get("JEERAG_DEFAULT_KIND", "notes")

PHYSICS_WORDS = {
    "PRE CALCULUS": "maths", "DIFFERENTIATION": "maths", "APPLICATION OF DERIVATIVES": "maths",
    "CONSTRAINT": "physics", "FRICTION": "physics", "CIRCULAR": "physics", "WORK POWER": "physics",
    "EQUILIBRIUM": "physics", "CENTRE OF MASS": "physics", "IMPULSE": "physics",
    "VARIABLE MASS": "physics", "ROTATIONAL": "physics", "MOMENT OF INERTIA": "physics",
    "INSTANTANEOUS AXIS": "physics", "ROLLING": "physics", "ANGULAR MOMENTUM": "physics",
    "FLUID": "physics", "MECHANICAL PROPERTIES OF SOLIDS": "physics", "VISCOSITY": "physics",
    "SURFACE TENSION": "physics", "SIMPLE HARMONIC": "physics", "OSCILLATION": "physics",
    "ERROR ANALYSIS": "physics", "UNITS AND DIMENSIONS": "physics", "VERNIER": "physics",
    "SCREW GAUGE": "physics", "MICROMETER": "physics", "THERMOMETRY": "physics",
    "CALORIMETRY": "physics", "THERMAL EXPANSION": "physics", "KINEMATICS": "physics",
    "GRAVITATION": "physics", "THERMODYNAMIC": "physics", "WAVE": "physics",
    "OSCILLATION AND WAVES": "physics", "ELECTRO": "physics", "CURRENT": "physics",
    "MAGNETISM": "physics", "OPTICS": "physics", "RAY": "physics", "WAVE OPTICS": "physics",
    "MODERN PHYSICS": "physics", "SEMICONDUCTOR": "physics", "COMMUNICATION": "physics",
}


def tags_for(pdf_path: Path, kind: str = DEFAULT_KIND):
    name = pdf_path.stem.upper()
    stem = pdf_path.stem
    m = re.match(r"^(\d+)[_\s]+(.+)", stem, re.IGNORECASE)
    if m:
        chapter, topic = m.group(1), m.group(2)
    elif kind == "pyq":
        # PYQ filenames like "JEE Advanced Question Paper 2007 Paper 1.pdf"
        ym = re.search(r"(19|20)\d{2}", stem)
        chapter = ym.group(0) if ym else ""
        topic = re.sub(r"\s*(19|20)\d{2}\s*", " ", stem, flags=re.I).strip()
        topic = re.sub(r"\b(Paper|Shift|Set|Mock|Previous|Year|Question)\b", "", topic, flags=re.I).strip()
        if not topic:
            topic = stem
    else:
        chapter, topic = "", stem
    topic_clean = re.sub(r"\s*T\d{2}J\d+\s*(-min)?", "", topic, flags=re.I).strip()
    topic_clean = re.sub(r"[_\s]+", " ", topic_clean)
    subject = "physics"
    for word, subj in PHYSICS_WORDS.items():
        if word in name:
            subject = subj
            break
    if kind == "pyq" and subject == "physics":
        # PYQ papers span physics+chemistry+maths; mark as mixed unless a subject word wins
        subject = "mixed"
    return chapter, topic_clean, subject


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS files(
        file TEXT PRIMARY KEY, path TEXT, chapter TEXT, topic TEXT, subject TEXT,
        kind TEXT DEFAULT 'notes', pages INTEGER, ocr_pages INTEGER, text_pages INTEGER);
    CREATE TABLE IF NOT EXISTS done(file TEXT, page INTEGER, PRIMARY KEY(file, page));
    CREATE TABLE IF NOT EXISTS pages(
        file TEXT, page INTEGER, source TEXT, text TEXT, n_diagrams INTEGER,
        PRIMARY KEY(file, page));
    CREATE TABLE IF NOT EXISTS diagrams(
        id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT, page INTEGER,
        path TEXT, w INTEGER, h INTEGER);
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
        text, topic, subject UNINDEXED, file UNINDEXED, page UNINDEXED,
        tokenize='unicode61');
    """)
    # Migration: add kind column to existing files tables that lack it
    cols = {r[1] for r in con.execute("PRAGMA table_info(files)")}
    if "kind" not in cols:
        con.execute("ALTER TABLE files ADD COLUMN kind TEXT DEFAULT 'notes'")
    con.commit()
    return con


def ocr_page(args):
    """Renders one page in memory and OCRs it. Worker-safe: opens its own doc."""
    pdf_path, pageno = args
    doc = fitz.open(pdf_path)
    page = doc[pageno]
    pix = page.get_pixmap(dpi=OCR_DPI)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    text = pytesseract.image_to_string(img, config="--psm 3").strip()
    return pdf_path, pageno, text


def _pix_gray(pix):
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return a[:, :, :3].mean(axis=2).astype(np.uint8)
    return a[:, :, 0]


def _is_blank(pix, max_dark_frac=0.004):
    g = _pix_gray(pix)
    return float((g < 200).mean()) < max_dark_frac


def _ink_and_words(pix):
    """Return (ink_fraction, word_count). Inverts dark-background crops first.
    Low ink => line drawing/diagram; high ink + many words => text block."""
    g = _pix_gray(pix)
    dark_bg = float((g < 128).mean()) > 0.5
    if dark_bg:
        g = 255 - g
    ink = float((g < 200).mean())
    words = None
    if 0.06 <= ink <= 0.95:
        try:
            txt = pytesseract.image_to_string(Image.fromarray(g), config="--psm 11")
            words = len([w for w in txt.split() if any(c.isalpha() for c in w)])
        except Exception:
            pass
    return ink, words


def extract_diagrams(page, slug, con, file, pageno):
    """Detect content images on a page (skipping scanner background tiles and
    blank crops), save them, return marker text."""
    out = []
    try:
        infos = page.get_image_info()
    except Exception:
        infos = []
    pr = page.rect
    page_area = pr.get_area()
    boxes_in_page = []
    for info in infos:
        b = fitz.Rect(info["bbox"])
        # must lie on the page (tiles spill past the page edges)
        if b.x0 < pr.x0 - 5 or b.y0 < pr.y0 - 5 or b.x1 > pr.x1 + 5 or b.y1 > pr.y1 + 5:
            continue
        if b.is_empty:
            continue
        w_frac, h_frac = b.width / pr.width, b.height / pr.height
        # scanner tiles / page backgrounds: big grid blocks
        if (w_frac > 0.6 and h_frac > 0.45) or b.get_area() > 0.35 * page_area:
            continue
        # too small to be a meaningful diagram
        if b.width < 70 or b.height < 55:
            continue
        dup = False
        for sb in boxes_in_page:
            inter = sb & b
            if not inter.is_empty and inter.get_area() > 0.7 * min(sb.get_area(), b.get_area()):
                dup = True
                break
        if dup:
            continue
        boxes_in_page.append(b)
        try:
            pix = page.get_pixmap(clip=b & pr, dpi=150)
        except Exception:
            continue
        if _is_blank(pix):
            continue
        ink, words = _ink_and_words(pix)
        if words is not None and words >= 15:
            continue  # text strip, not a diagram
        idx = len(out) + 1
        d = DIAGRAM_DIR / slug
        d.mkdir(exist_ok=True)
        path = d / f"p{pageno + 1}_{idx}.png"
        rel = path.relative_to(DATA_DIR)  # store portable paths, never absolute
        try:
            pix.save(path)
            con.execute(
                "INSERT INTO diagrams(file, page, path, w, h) VALUES(?,?,?,?,?)",
                (file, pageno, str(rel), pix.width, pix.height))
            out.append(f"[diagram on page {pageno + 1}: {rel}]")
        except Exception:
            continue
    return out


def slug_for(pdf_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_")[:60]


def chunk_text(text, max_chars=2500):
    parts, buf = [], []
    size = 0
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if size + len(block) > max_chars and buf:
            parts.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(block)
        size += len(block) + 2
    if buf:
        parts.append("\n\n".join(buf))
    return parts


def process_pdf(pdf_path, kind: str = DEFAULT_KIND):
    con = init_db_conn()
    file = pdf_path.name
    done = {r[0] for r in con.execute("SELECT page FROM done WHERE file=?", (file,))}
    chapter, topic, subject = tags_for(pdf_path, kind)
    slug = slug_for(pdf_path)
    doc = fitz.open(pdf_path)
    n = len(doc)
    todo = [p for p in range(n) if p not in done]
    if not todo:
        doc.close()
        return file, n, 0
    # text-layer pages processed here, OCR pages fanned out to workers
    ocr_targets = []
    for pageno in todo:
        page = doc[pageno]
        text = page.get_text().strip()
        if len(text) < MIN_CHARS_FOR_TEXT_LAYER:
            ocr_targets.append(pageno)
            continue
        diags = extract_diagrams(page, slug, con, file, pageno)
        full = text + ("\n" + "\n".join(diags) if diags else "")
        con.execute(
            "INSERT OR REPLACE INTO pages(file, page, source, text, n_diagrams) VALUES(?,?,?,?,?)",
            (file, pageno, "text", full, len(diags)))
        for chunk in chunk_text(full):
            con.execute(
                "INSERT INTO chunks(text, topic, subject, file, page) VALUES(?,?,?,?,?)",
                (chunk, topic, subject, file, pageno + 1))
        con.execute("INSERT OR REPLACE INTO done(file, page) VALUES(?,?)", (file, pageno))
    con.commit()
    n_ocr = len(ocr_targets)
    con.close()
    doc.close()
    if ocr_targets:
        return file, n, n_ocr  # OCR handled by parent via pool
    return file, n, 0


def init_db_conn():
    return sqlite3.connect(DB_PATH, timeout=60)


def ingest_ocr_pages(pdf_path, ocr_pages, kind: str = DEFAULT_KIND):
    con = init_db_conn()
    file = pdf_path.name
    chapter, topic, subject = tags_for(pdf_path, kind)
    slug = slug_for(pdf_path)
    doc = fitz.open(pdf_path)
    for pageno, text in ocr_pages:
        page = doc[pageno]
        try:
            diags = extract_diagrams(page, slug, con, file, pageno)
        except Exception:
            diags = []
        full = text + ("\n" + "\n".join(diags) if diags else "")
        con.execute(
            "INSERT OR REPLACE INTO pages(file, page, source, text, n_diagrams) VALUES(?,?,?,?,?)",
            (file, pageno, "ocr", full, len(diags)))
        for chunk in chunk_text(full):
            con.execute(
                "INSERT INTO chunks(text, topic, subject, file, page) VALUES(?,?,?,?,?)",
                (chunk, topic, subject, file, pageno + 1))
        con.execute("INSERT OR REPLACE INTO done(file, page) VALUES(?,?)", (file, pageno))
    con.commit()
    con.execute(
        "INSERT OR REPLACE INTO files(file, path, chapter, topic, subject, kind, pages, ocr_pages, text_pages) "
        "SELECT ?,?,?,?,?,?, "
        "(SELECT COUNT(*) FROM done WHERE file=?), (SELECT COUNT(*) FROM pages WHERE file=? AND source='ocr'), "
        "(SELECT COUNT(*) FROM pages WHERE file=? AND source='text')",
        (file, str(pdf_path), chapter, topic, subject, kind, file, file, file))
    con.commit()
    con.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build/resume the jeerag FTS5 index")
    ap.add_argument("substring", nargs="?", help="optional substring filter on PDF file names")
    ap.add_argument("--kind", default=DEFAULT_KIND, help="kind tag for indexed files (notes|pyq)")
    ap.add_argument("--notes-dir", default=None, help="override notes directory")
    ap.add_argument("--recursive", action="store_true", help="search subdirectories for PDFs")
    args = ap.parse_args()

    notes_dir = Path(args.notes_dir) if args.notes_dir else NOTES_DIR
    kind = args.kind
    globber = notes_dir.rglob if args.recursive else notes_dir.glob
    con = init_db()
    files = sorted(
        (p for p in globber("*.pdf") if not args.substring or args.substring.lower() in p.name.lower()),
        key=lambda p: (int(m.group(1)) if (m := re.match(r"^(\d+)", p.stem)) else 999, p.name))
    print(f"{len(files)} PDFs (kind={kind}) from {notes_dir}, OCR dpi={OCR_DPI}, workers={WORKERS}", flush=True)
    for pdf in files:
        done = {r[0] for r in con.execute("SELECT page FROM done WHERE file=?", (pdf.name,))}
        doc = fitz.open(pdf)
        todo = [p for p in range(len(doc)) if p not in done]
        n_total = len(doc)
        doc.close()
        if not todo:
            print(f"[skip] {pdf.name}: already indexed", flush=True)
            continue
        print(f"[start] {pdf.name}: {len(todo)}/{n_total} pages pending", flush=True)
        result = process_pdf(pdf, kind)
        _, n, n_ocr = result
        if n_ocr:
            args2 = [(str(pdf), p) for p in todo if True]
            # recompute OCR set: pages still not marked done after text pass
            con2 = init_db()
            still = {r[0] for r in con2.execute("SELECT page FROM done WHERE file=?", (pdf.name,))}
            con2.close()
            args2 = [(str(pdf), p) for p in todo if p not in still]
            print(f"  ocr: {len(args2)} pages", flush=True)
            ocr_out = []
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for i, res in enumerate(ex.map(ocr_page, args2)):
                    ocr_out.append(res)
                    if (i + 1) % 10 == 0 or i + 1 == len(args2):
                        print(f"  ocr {i + 1}/{len(args2)}", flush=True)
            ingest_ocr_pages(pdf, [(r[1], r[2]) for r in ocr_out], kind)
        else:
            chapter, topic, subject = tags_for(pdf, kind)
            con3 = init_db()
            con3.execute(
                "INSERT OR REPLACE INTO files(file, path, chapter, topic, subject, kind, pages, ocr_pages, text_pages) "
                "SELECT ?,?,?,?,?,?, (SELECT COUNT(*) FROM done WHERE file=?), "
                "(SELECT COUNT(*) FROM pages WHERE file=? AND source='ocr'), "
                "(SELECT COUNT(*) FROM pages WHERE file=? AND source='text')",
                (pdf.name, str(pdf), chapter, topic, subject, kind, pdf.name, pdf.name, pdf.name))
            con3.commit()
            con3.close()
        print(f"[done] {pdf.name}", flush=True)
    con.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
