"""Official JEE question papers and answer keys: registry, fetcher, parser.

Everything in the registry points at **official** sources only:

* JEE (Advanced) — ``jeeadv.ac.in``, the organising IITs' official site.
  Question papers live under ``past_qps/`` (past years) and ``documents/``
  (current year).  Final answer keys are published as PDFs that contain both
  the questions and their correct answers, marked as ``Answer Q8: ABC`` lines,
  so they can be parsed into a machine-readable sidecar.
* JEE (Main) — ``jeemain.nta.nic.in`` (NTA) and its official document CDN
  (``cdnbbsr.s3waas.gov.in``).  NTA publishes final answer keys as
  Question-ID / Correct-Option-ID tables whose layout varies per session, so
  these are indexed as searchable text rather than parsed structurally.

Coaching-institute "solutions" PDFs are copyrighted and deliberately excluded.

Dead URLs (the site is re-hosted every year by the next organising IIT) fall
back to the Internet Archive automatically, so keys for past years can still
be recovered offline-first: fetch once, keep in ``papers/``, no need again.

CLI (all network access is explicit and opt-in)::

    python -m jeerag.pyq_sources --list [--check]
    python -m jeerag.pyq_sources --fetch adv-2026-final-key-1 adv-2026-final-key-2
    python -m jeerag.pyq_sources --fetch-verified
    python -m jeerag.pyq_sources --url <official-pdf-url> --as my-key --exam adv --kind final_key
    python -m jeerag.pyq_sources --parse "papers/jee-adv/keys/JEE Advanced 2026 Final Answer Key Paper 1.pdf"
    python -m jeerag.pyq_sources --index papers/jee-adv/keys

Fetched files are saved under ``papers/`` (``--out`` to override) and indexed
with ``--kind pyq`` so they show up in ``/api/pyq``, the PYQ search tab and the
chat pipeline's related-PYQ enrichment.
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

ADV_HOST = "https://jeeadv.ac.in"
MAIN_DOC_PAGE = "https://jeemain.nta.nic.in/documents/"


def _adv_paper(year: int, paper: int) -> dict:
    """Question paper entry: past_qps/<Y>_<n>.pdf / <Y>_<n>_English.pdf."""
    return {
        "id": f"adv-{year}-paper-{paper}",
        "exam": "adv", "kind": "paper", "year": year, "paper": paper,
        "title": f"JEE Advanced {year} Question Paper {paper}",
        "urls": [
            f"{ADV_HOST}/past_qps/{year}_{paper}.pdf",
            f"{ADV_HOST}/past_qps/{year}_{paper}_English.pdf",
        ],
        "verified": False,
    }


def _adv_year_key(year: int) -> dict:
    """Past-year final answer keys (probed + Wayback fallback)."""
    return {
        "id": f"adv-{year}-final-keys",
        "exam": "adv", "kind": "final_key", "year": year, "paper": None,
        "title": f"JEE Advanced {year} Final Answer Keys (both papers)",
        "urls": [
            f"{ADV_HOST}/documents/jeeadv-{year}-final-answer-keys.pdf",
            f"{ADV_HOST}/pdf/{year}p1key.pdf",
            f"{ADV_HOST}/pdf/{year}p2key.pdf",
        ],
        "verified": year == 2022,  # probed via Wayback snapshot 20221107
    }


def _registry() -> list[dict]:
    items = []
    # question papers 2013-2026 (2007-2012 IIT-JEE single papers exist in the
    # local corpus already but the archive's 2007-2012 entries follow the
    # same past_qps pattern, so generate them too)
    for year in range(2007, 2027):
        for paper in (1, 2):
            if year == 2026:  # current-year layout: documents/p<n>_english.pdf
                items.append({
                    "id": f"adv-{year}-paper-{paper}",
                    "exam": "adv", "kind": "paper", "year": year, "paper": paper,
                    "title": f"JEE Advanced {year} Question Paper {paper}",
                    "urls": [f"{ADV_HOST}/documents/p{paper}_english.pdf"],
                    "verified": True,  # probed live (p1: 200 OK, 3.2 MB)
                })
            else:
                items.append(_adv_paper(year, paper))
    # current-year (2026) official keys — live today
    for paper in (1, 2):
        items.append({
            "id": f"adv-2026-final-key-{paper}",
            "exam": "adv", "kind": "final_key", "year": 2026, "paper": paper,
            "title": f"JEE Advanced 2026 Final Answer Key Paper {paper}",
            "urls": [f"{ADV_HOST}/documents/p{paper}_solutions_final.pdf"],
            "verified": True,  # probed live (2.3 / 2.5 MB, parseable)
        })
        items.append({
            "id": f"adv-2026-provisional-key-{paper}",
            "exam": "adv", "kind": "provisional_key", "year": 2026,
            "paper": paper,
            "title": f"JEE Advanced 2026 Provisional Answer Key Paper {paper}",
            "urls": [f"{ADV_HOST}/documents/p{paper}_provisional_keys.pdf"],
            "verified": True,  # p1 probed live (1.4 MB)
        })
    # past-year final keys 2013-2025 (live URLs die when the site is re-hosted;
    # the fetcher falls back to Internet Archive snapshots automatically)
    for year in range(2013, 2026):
        items.append(_adv_year_key(year))
    # JEE (Main) official keys — NTA document CDN (URLs are per-session; the
    # registry keeps verified ones, others can be added via --url)
    items.append({
        "id": "main-2025-s2-final-key-p1",
        "exam": "main", "kind": "final_key", "year": 2025, "paper": 1,
        "title": "JEE Main 2025 Session 2 Final Answer Key (Paper 1, B.E./B.Tech)",
        "urls": ["https://cdnbbsr.s3waas.gov.in/s3f8e59f4b2fe7c5705bf878bbd494ccdf/uploads/2025/04/2025041892.pdf"],
        "verified": True,  # downloaded + inspected (286 KB, 18 pages)
        "note": ("more sessions/shifts: see " + MAIN_DOC_PAGE +
                 " and the document-category/archive pages"),
    })
    return items


SOURCES = _registry()


def _safe_name(name: str) -> str:
    """Strip path-hostile characters from a registry title."""
    return re.sub(r'[\\/:*?"<>|]+', " ", name).strip()


def dest_path(item: dict, out_root: Path) -> Path:
    """Where a fetched PDF belongs, mirroring the existing papers/ layout."""
    exam, kind = item["exam"], item["kind"]
    if kind == "paper":
        sub = out_root / ("jee-adv" if exam == "adv" else "jee-main") / str(item["year"])
        return sub / f"{_safe_name(item['title'])}.pdf"
    sub = out_root / ("jee-adv" if exam == "adv" else "jee-main") / "keys"
    return sub / f"{_safe_name(item['title'])}.pdf"


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (jeerag pyq_sources)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _is_pdf(data: bytes) -> bool:
    return data[:1024].lstrip().startswith(b"%PDF") and len(data) > 10_000


def wayback_snapshot(url: str) -> str | None:
    """Closest Internet Archive snapshot (im_/ variant for raw bytes)."""
    api = ("https://archive.org/wayback/available?url="
           + urllib.parse.quote(url, safe=""))
    try:
        data = json.loads(_http_get(api, timeout=30))
        snap = (data.get("archived_snapshots") or {}).get("closest") or {}
        if snap.get("available") and snap.get("url"):
            # .../web/<ts>/<orig> -> .../web/<ts>im_/<orig>  (raw content)
            return re.sub(r"/web/(\d+)/", r"/web/\1im_/", snap["url"])
    except Exception:
        pass
    # last-resort shortcut that resolves to the closest snapshot
    return f"https://web.archive.org/web/2im_/{url}"


def fetch_url(url: str, timeout: int = 120) -> tuple[bytes | None, str]:
    """Download one URL, falling back to the Internet Archive.

    Returns ``(data, effective_url)``; data is None when nothing worked.
    """
    try:
        data = _http_get(url, timeout=timeout)
        if _is_pdf(data):
            return data, url
    except Exception:
        pass
    wb = wayback_snapshot(url)
    if wb:
        try:
            data = _http_get(wb, timeout=timeout)
            if _is_pdf(data):
                return data, wb
        except Exception:
            pass
    return None, ""


def fetch_item(item: dict, out_root: Path, force: bool = False) -> dict:
    """Fetch one registry item into papers/, returning a status dict."""
    out = dest_path(item, out_root)
    if out.exists() and not force:
        return {"id": item["id"], "ok": True, "path": str(out),
                "source": "already present"}
    for url in item["urls"]:
        data, effective = fetch_url(url)
        if data is None:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        return {"id": item["id"], "ok": True, "path": str(out),
                "source": effective, "bytes": len(data)}
    return {"id": item["id"], "ok": False, "path": str(out),
            "error": "no candidate URL (live or Wayback) returned a PDF"}


def check_item(item: dict) -> dict:
    """HEAD-style probe: does any candidate URL (or its Wayback copy) resolve?"""
    for url in item["urls"]:
        try:
            req = urllib.request.Request(
                url, method="HEAD",
                headers={"User-Agent": "Mozilla/5.0 (jeerag pyq_sources)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status == 200:
                    return {"id": item["id"], "ok": True, "source": url}
        except Exception:
            pass
    api = ("https://archive.org/wayback/available?url="
           + urllib.parse.quote(item["urls"][0], safe=""))
    try:
        snap = json.loads(_http_get(api, timeout=20)).get(
            "archived_snapshots", {}).get("closest") or {}
        if snap.get("available"):
            return {"id": item["id"], "ok": True, "source": snap["url"],
                    "wayback": True}
    except Exception:
        pass
    return {"id": item["id"], "ok": False, "source": ""}


# ---------------------------------------------------------------------------
# answer-key parsing (JEE Advanced: "Answer Q8: ABC" lines)
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(
    r"Answer\s*Q\.?\s*(\d+)\s*[:.\-–]?\s*((?:[A-D][ \t,]*)+|\d+(?:\.\d+)?)",
    re.IGNORECASE)


def parse_adv_key_text(text: str) -> dict:
    """Parse ``Answer Qn: XXX`` lines -> ``{"8": "ABC", "12": 14.75, ...}``.

    Option answers are normalised to sorted letters ("CBA" -> "ABC");
    numerical answers become floats.
    """
    answers: dict = {}
    for m in ANSWER_RE.finditer(text):
        qn, val = int(m.group(1)), m.group(2).strip()
        letters = re.findall(r"[A-D]", val.upper())
        answers[str(qn)] = ("".join(sorted(letters)) if letters
                            else float(val))
    return answers


def parse_adv_key_pdf(path: Path) -> dict:
    """Extract the answer table from an official JEE Advanced key PDF."""
    try:
        import fitz
    except ImportError as e:
        raise SystemExit(
            f"{e}: PyMuPDF is required for --parse (pip install pymupdf)")
    chunks = []
    with fitz.open(path) as doc:
        for page in doc:
            chunks.append(page.get_text())
    answers = parse_adv_key_text("\n".join(chunks))
    return answers


def save_sidecar(pdf: Path, answers: dict, source_url: str = "") -> Path:
    side = pdf.with_suffix(pdf.suffix + ".answers.json")
    side.write_text(json.dumps({
        "file": pdf.name,
        "source": source_url,
        "n_answers": len(answers),
        "answers": answers,
    }, indent=2))
    return side


# ---------------------------------------------------------------------------
# indexing (delegates to jeerag.index so OCR/files-row handling stays in one place)
# ---------------------------------------------------------------------------

def index_dir(target: Path) -> None:
    """Index everything under ``target`` with kind=pyq (subprocess, resumable)."""
    import subprocess
    cmd = [sys.executable, "-m", "jeerag.index", "--kind", "pyq",
           "--notes-dir", str(target), "--recursive"]
    print("indexing:", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fetch official JEE question papers / answer keys")
    ap.add_argument("--list", action="store_true", help="show the registry")
    ap.add_argument("--check", action="store_true",
                    help="with --list: probe each entry (live + Wayback)")
    ap.add_argument("--fetch", nargs="+", metavar="ID",
                    help="download specific registry items")
    ap.add_argument("--fetch-verified", action="store_true",
                    help="download every entry marked verified")
    ap.add_argument("--url", help="fetch any official PDF by URL")
    ap.add_argument("--as", dest="as_name", help="name for --url downloads")
    ap.add_argument("--exam", default="adv", choices=("adv", "main"))
    ap.add_argument("--kind", default="final_key",
                    choices=("paper", "final_key", "provisional_key"))
    ap.add_argument("--parse", nargs="+", metavar="PDF",
                    help="extract Answer Qn lines into .answers.json sidecars")
    ap.add_argument("--index", nargs="?", const="papers", metavar="DIR",
                    help="index fetched PDFs into the search index (kind=pyq)")
    ap.add_argument("--out", default="papers", help="papers root (default ./papers)")
    ap.add_argument("--force", action="store_true", help="re-download")
    args = ap.parse_args()
    out_root = Path(args.out)

    if args.list or args.check:
        print(f"{'id':34} {'kind':16} {'verified':8} title")
        for it in SOURCES:
            line = f"{it['id']:34} {it['kind']:16} {str(it['verified']):8} {it['title']}"
            print(line)
            if args.check:
                r = check_item(it)
                print(f"    -> {'OK ' if r['ok'] else 'DEAD'} {r['source']}")
        print("\nfetch:  python -m jeerag.pyq_sources --fetch <id...>")
        return
    if args.parse:
        for pdf in args.parse:
            p = Path(pdf)
            answers = parse_adv_key_pdf(p)
            side = save_sidecar(p, answers)
            print(f"{p.name}: {len(answers)} answers -> {side.name}")
        return
    if args.url:
        data, effective = fetch_url(args.url)
        if data is None:
            raise SystemExit(f"could not fetch a valid PDF from {args.url}")
        name = args.as_name or Path(args.url).stem or "official-key"
        out = out_root / ("jee-adv" if args.exam == "adv" else "jee-main") / "keys" / f"{name}.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        print(f"saved {len(data):,} bytes -> {out}  (from {effective})")
        return
    if args.index:
        index_dir(Path(args.index))
        return
    if args.fetch_verified or args.fetch:
        want = set(args.fetch or [])
        items = [it for it in SOURCES
                 if (it["id"] in want or (args.fetch_verified and it["verified"]))]
        missing = want - {it["id"] for it in items}
        if missing:
            raise SystemExit(f"unknown id(s): {', '.join(sorted(missing))}")
        for it in items:
            r = fetch_item(it, out_root, force=args.force)
            if r["ok"]:
                print(f"[ok] {r['id']}: {r['path']}"
                      + (f"  ({r.get('bytes', 0):,} bytes)" if r.get("bytes")
                         else f"  {r.get('source', '')}"))
                if (it["exam"] == "adv" and it["kind"] == "final_key"
                        and "already" not in r.get("source", "")):
                    try:
                        answers = parse_adv_key_pdf(Path(r["path"]))
                        side = save_sidecar(Path(r["path"]), answers, r["source"])
                        print(f"     parsed {len(answers)} answers -> {side.name}")
                    except SystemExit:
                        pass
            else:
                print(f"[!!] {r['id']}: {r['error']}", file=sys.stderr)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
