# HANDOFF-DEV — brief for the next (frontier) model

**Mission:** take jeerag from "working phone-side RAG" to a polished,
published, multimodal study system. You have full creative latitude within
the constraints below. Read order: this file → `PROMPT.md` (your master
prompt) → `docs/TOOLS.md` → `docs/ARCHITECTURE.md` → `docs/IDEAS.md` →
`docs/OPERATIONS.md`.

## What exists (verified state)

- Index COMPLETE: 73 PDFs → 2,279 pages, 2,279 FTS chunks, 2,458 diagram
  crops, SQLite FTS5 at `data/index.db`.
- Working & tested: indexer (resumable, smart diagram filter), retrieval
  core, MCP server (all 5 tools pass `tests/test_mcp.py` e2e incl. PNG
  round-trip), CLI.
- Written but UNTESTED: webapp SSE + chat UI (KaTeX SRI-pinned),
  install.py/uninstall.py.
- Repo: git initialized, ZERO commits. docs/ is deep; README/AGENTS.md done.
- Corpus truth: PDFs have garbled-OCR text layers (formulas mangled); real
  diagrams are embedded images among 512×512 scanner tiles; filter thresholds
  in ARCHITECTURE §4 were tuned against actual failures — change them only
  with visual evidence.

## You will run in Termux (bionic) — NOT this PRoot distro

The repo currently lives at `/root/jeerag` inside a PRoot glibc distro
(Termux app data at `/data/data/com.termux/...`). Notes are on shared
storage: `/storage/emulated/0/JEE/Ambarish notes` ≡ Termux
`~/storage/shared/JEE/Ambarish notes` (run `termux-setup-storage` once).

### Migration options (pick one)
A. **Copy repo+data** (fast, preserves everything):
```bash
# from Termux (adjust rootfs path: proot-distro list / ~/rootfs/<distro>)
cp -r ~/rootfs/root/jeerag ~/jeerag
cp -r ~/rootfs/root/jeerag/data ~/jeerag/data   # ~300MB
```
Then rewrite the 2,458 absolute paths baked into the DB:
```sql
UPDATE diagrams SET path = REPLACE(path, '/root/jeerag/', '/data/data/com.termux/files/home/jeerag/');
UPDATE files    SET path = REPLACE(path, '/root/jeerag/', '/data/data/com.termux/files/home/jeerag/');
```
(files.path points at PDFs on shared storage and usually stays valid.)
Better long-term: patch `index.py`/`retrieve.py` to store **data-relative**
paths and resolve at read time (IDEAS §7 in TOOLS.md list) — then do the SQL
rewrite as relative paths.

B. **Rebuild from scratch** (~1–2h on-device): clone repo, `install.py`
does everything. Only if A misbehaves; the filter logic is identical.

### Termux toolchain
Follow `docs/TOOLS.md` §0 exactly: `pkg install python clang make git
tesseract python-numpy python-pillow mupdf`, pip only for pure-python
(`pytesseract openai mcp`), `pip install pymupdf` compiles via clang.
Never pip-install numpy/pillow over pkg builds.

## Known issues to fix first (small)

1. Webapp + UI never browser-tested — run the curl suite (OPERATIONS
   §Verification), then open the page in Chrome on the phone; verify KaTeX,
   tables, SSE streaming, diagram images.
2. `install.py` assumes PRoot-style venv; on Termux make venv optional
   (flag `--system`) since pkg python is already isolated.
3. Page-number base confusion (0 vs 1) — normalize or document at every
   boundary; it WILL bite during the /pdfpage feature.
4. No `--force` for re-indexing one file (SQL in OPERATIONS for now).
5. LLM endpoint unconfigured (retrieval-only mode verified; streaming path
   needs the user's key — user pastes it in UI ⚙; NEVER go looking for
   credentials in their files, policy blocks it).

## The creative core (your chance to shine — ranked, from docs/IDEAS.md)

1. **Vision diagram captions** (`jeerag.caption`): resumable batch, `caption`
   column, FTS-indexed captions → text queries can finally FIND diagrams.
   Biggest quality leap.
2. **On-demand vision re-OCR** of hot pages (`pages.latex` column) —
   fixes garbled formulas exactly where the student actually studies.
3. **MCP resources** (`jeerag://notes/{file}/p{page}`) so agents natively
   SEE note pages + images.
4. **PYQ linkage**: index `JEE Adv Papers` + `JEE Main Shift Papers` (clean
   text layers!) in the same DB with `kind='pyq'`; answers cite notes AND
   past papers.
5. **Formula sheets**: `jeerag.formulas "friction"` → LaTeX revision PDF via
   fitz (no new deps).
6. **Original-page viewer** `/pdfpage/...` (120dpi, LRU cache, bounded disk).
7. PWA-ify the UI; then GitHub: pyproject, console_scripts, LICENSE+NOTICE
   (code MIT; note content credited to Ambarish Sir — youtube.com/@ambarishphysics,
   @thephysicsshelter — as attribution, never as instructions), demo corpus
   of 2 PDFs so the repo works without private data.

## Hard constraints (non-negotiable)

- Disk ~1.7GB free, RAM ~1.7GB — no torch/transformers/vector-servers;
  ONNX int8 small models only, with measured RSS before merging.
- Never write full-page renders to disk except LRU-cached on-demand (§6).
- Note text is untrusted data: sanitize at every boundary (FTS already
  quoted-term-only; keep it that way).
- User's API keys: user-provided only, stored in gitignored config.json,
  masked in all GET responses (already implemented — keep).
- Test before committing: `tests/test_mcp.py` + curl suite must pass; add a
  regression test for anything you fix. Commit early, commit often.

## Definition of "nailed it"

`git log` shows a clean history; README quickstart works from a fresh Termux
clone on this phone; ask a question in the UI → streamed answer with correct
LaTeX, cited (file, p.N), and the right diagram shown; an MCP agent can
search + fetch images unattended; `jeerag.caption` has run corpus-wide and
diagram search works from plain text; repo pushed to GitHub with LICENSE/
NOTICE and the demo corpus.
