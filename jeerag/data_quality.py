"""Data quality validation for the jeerag index.

Runs checks on data/index.db to detect:
- Pages with missing/garbled OCR text
- Diagrams with missing image files
- Duplicate chunks
- Empty or near-empty pages
- Files with zero indexed pages

Usage::

    python -m jeerag.data_quality          # full health report
    python -m jeerag.data_quality --json   # JSON output for CI/CD
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from . import config


def _connect():
    import sqlite3 as _sqlite3
    return _sqlite3.connect(config.load()["data_dir"] + "/index.db")


def _check_files(con) -> list[dict]:
    """Return files with zero indexed pages."""
    rows = con.execute(
        "SELECT file, pages FROM files WHERE pages = 0 OR pages IS NULL"
    ).fetchall()
    return [{"file": r[0], "issue": "zero indexed pages", "pages": r[1]} for r in rows]


def _check_blank_pages(con) -> list[dict]:
    """Pages with essentially no text (< 5 chars)."""
    rows = con.execute(
        "SELECT file, page, text FROM pages WHERE length(trim(text)) < 5"
    ).fetchall()
    return [
        {"file": r[0], "page": r[1], "issue": "blank or near-blank page", "chars": len(r[2] or "")}
        for r in rows
    ]


def _check_diagram_files(con) -> list[dict]:
    """Diagrams whose PNG file no longer exists."""
    from pathlib import Path as _Path

    data_dir = config.load()["data_dir"]
    rows = con.execute("SELECT path FROM diagrams").fetchall()
    missing = []
    for (p,) in rows:
        if not (_Path(data_dir) / p).exists():
            missing.append({"file": p, "issue": "diagram file missing on disk"})
    return missing


def _check_duplicate_chunks(con) -> list[dict]:
    """Chunks with identical text (exact duplicates)."""
    rows = con.execute(
        "SELECT text, COUNT(*) c FROM chunks GROUP BY text HAVING c > 1 LIMIT 20"
    ).fetchall()
    return [{"text": r[0][:80] + "...", "count": r[1]} for r in rows]


def _check_ocr_quality(con) -> list[dict]:
    """Flag pages whose text looks like garbled OCR (too many non-alpha chars)."""
    rows = con.execute(
        "SELECT file, page, text FROM pages WHERE length(text) > 50"
    ).fetchall()
    bad = []
    for file, page, text in rows:
        alpha = sum(1 for c in text if c.isalpha())
        total = len(text)
        if total > 0 and alpha / total < 0.3:
            bad.append(
                {
                    "file": file,
                    "page": page,
                    "issue": "high OCR garble ratio",
                    "alpha_ratio": round(alpha / total, 2),
                }
            )
    return bad


def run_all() -> dict:
    """Run every check and return a combined report."""
    con = _connect()
    report = {
        "files": _check_files(con),
        "blank_pages": _check_blank_pages(con),
        "missing_diagrams": _check_diagram_files(con),
        "duplicate_chunks": _check_duplicate_chunks(con),
        "garbled_ocr": _check_ocr_quality(con),
    }
    con.close()
    total_issues = sum(len(v) for v in report.values())
    report["summary"] = {
        "total_issues": total_issues,
        "healthy": total_issues == 0,
    }
    return report


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="jeerag data quality checker")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args(argv)

    report = run_all()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"jeerag data quality: {report['summary']['total_issues']} issues found")
        for category, items in report.items():
            if category == "summary":
                continue
            if items:
                print(f"\n  {category} ({len(items)}):")
                for item in items[:5]:
                    print(f"    - {item}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
    return 0 if report["summary"]["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())