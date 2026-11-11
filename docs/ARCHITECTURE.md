# ARCHITECTURE — jeerag, in depth

RAG over 73 scanned JEE notes PDFs (Ambarish Sir): 2,279 pages, 2,458
diagram crops, 2,279 FTS chunks. Everything runs on one Android phone
(aarch64 PRoot) with ~1.7GB free disk and ~1.7GB free RAM. That constraint
shaped every decision below — read it as "what we'd do differently on a
normal machine" too.

## 1. Data flow

```
PDFs (scanned, tiled images + OCR text layer)
  └─ jeerag/index.py            per page:
       ├─ text layer ≥30 chars? → use it (source='text')
       ├─ else → tesseract @200dpi in a ProcessPool (source='ocr')  [rare]
       ├─ extract_diagrams() → filtered crops @150dpi → data/diagrams/<slug>/pN_i.png
       └─ chunk_text(≤2500 chars on blank-line blocks) → rows in `chunks`
  └─ SQLite data/index.db
       files · pages · chunks(FTS5) · diagrams · done(progress)
  └─ consumers
       jeerag/retrieve.py (shared core: BM25 + topic fallback + diagram join)
         ├─ jeerag/mcp_server.py  (stdio MCP: 5 tools)
         ├─ jeerag/webapp.py      (stdlib HTTP + SSE → static/index.html)
         └─ ask.py                (CLI)
```

## 2. Schema (exact)

```sql
CREATE TABLE files(file TEXT PRIMARY KEY, path TEXT, chapter TEXT, topic TEXT,
    subject TEXT, pages INTEGER, ocr_pages INTEGER, text_pages INTEGER);
CREATE TABLE done(file TEXT, page INTEGER, PRIMARY KEY(file, page));   -- resume cursor
CREATE TABLE pages(file TEXT, page INTEGER, source TEXT, text TEXT,
    n_diagrams INTEGER, PRIMARY KEY(file, page));                       -- page 0-based
CREATE TABLE diagrams(id INTEGER PRIMARY KEY, file TEXT, page INTEGER,
    path TEXT, w INTEGER, h INTEGER);
CREATE VIRTUAL TABLE chunks USING fts5(text, topic, subject UNINDEXED,
    file UNINDEXED, page UNINDEXED, tokenize='unicode61');
```

Conventions that matter:
- `pages.page` and `diagrams.page` are **0-based**; `chunks.page` is **1-based**
  (human citation). `retrieve.diagrams_for` translates (`page-1`). A future
  cleanup should normalize this — it's the most likely source of off-by-one bugs.
- `bm25()` returns negative-is-better scores; we ORDER BY score ASC and display
  raw (≈ -11.7 strong, -9.0 topic-fallback sentinel).
- Diagram URL: `/diagrams/diagrams/<slug>/<file>.png` — route prefix + real
  dir name `diagrams/`. Looks doubled; is correct.
- `topic`/`subject` are denormalized into every chunk so BM25 can boost topic
  words without a JOIN.

## 3. Why SQLite FTS5/BM25 and no embeddings

- torch/sentence-transformers: ~2GB deps, >RAM budget. fastembed ONNX is the
  only viable local option (~90MB) — deferred, not rejected (see IDEAS §2).
- The corpus text is **garbled OCR of handwriting** (`dy do` = dy/dx). Embedding
  models assume clean prose; on this text BM25's exact-term strength matters
  more than semantic smoothing. Garbling hurts both, but BM25 is honest about
  it and never confidently retrieves the wrong derivation.
- FTS5 gives us: single-file DB, zero daemons, `bm25()` built in, ARM64 stdlib
  support. Query policy: stop-word-stripped terms OR-joined, each quoted
  (operator-injection safe — `sanitize_query` is the only place user text
  touches FTS syntax).

## 4. The diagram filter — the real engineering story

Three iterations, each driven by the user spotting a failure:

1. **v1: "area ≥2% of page"** → kept all 9-25 scanner *tiles* per page
   (512×512 JPEGs at 438×438pt). 1,203 junk "diagrams" in 4 files.
2. **v2: geometric gates** → skip bbox outside page ±5pt; skip if
   (w≥60% ∧ h≥45%) ∨ area≥35% of page; skip <70×55pt; dedupe ≥70% overlap.
   Tiles gone, but **problem-statement image blocks** survived (they're
   mid-size: e.g. 1199×471px "Keeping a book up" text block) and some blank
   crops remained.
3. **v3: content checks** (current):
   - blank: numpy mean of `(gray < 200)` < 0.004 → drop.
   - text-strip: ink fraction in [0.06, 0.95] (else it's a line drawing —
     skip OCR), auto-invert if >50% pixels dark (scans put white text on
     black tile background), tesseract `--psm 11`, count alpha-word tokens,
     ≥15 → drop.
   - Why 0.06: the book-on-wall problem block has ink≈0.099 after inversion —
     at the old 0.12 trigger it never got word-checked. Why 15 words: real
     diagrams label 2-8 words ("Cone of static friction", "Fig. 2.10").

Result: ~1.08 diagrams/page corpus-wide, visually verified on Friction.
Cost: tesseract runs per mid-ink crop → ~45s/file, ~55min for 73 files.
`np.frombuffer(...).reshape(h, w, n)` — NOT the `shape=` kwarg (crashed once).

## 5. Indexing correctness

- **Resumable & idempotent**: per-file, one transaction covers pages+chunks+done.
  Crash mid-file → nothing commits → rerun redoes exactly that file. Never
  delete `done` rows selectively without also deleting that file's `pages`+
  `chunks` rows (chunks have no PK — they append).
- `python -m jeerag.index "FRICTION"` re-runs one file; to *redo* a finished
  file you must delete its rows from done/pages/chunks/diagrams first
  (no `--force` flag yet — IDEAS §6).
- Chapter ordering: filename regex `^(\d+)[_\s]+(.+)`; topic string strips the
  `T25J1-min` course-code suffix. Subject map is keyword-based (PHYSICS_WORDS)
  — wrong for a maths file named e.g. "…MAGNETISM…"; acceptable for this corpus.

## 6. Webapp (SSE, not WebSocket)

- stdlib `ThreadingHTTPServer` on 127.0.0.1:8613. Chosen over FastAPI/uvicorn:
  zero extra deps on a 99%-full disk; SSE is enough for one-way token stream.
- `/api/chat` protocol (each event `event: <name>\ndata: <json>\n\n`):
  1. `context` — always first: `{passages[], diagrams[]}` so UI shows evidence
     even if the LLM fails.
  2. `delta {t}` — LLM tokens, forwarded from `openai` client `stream=True`.
  3. `no_llm {message}` — when `llm_base_url` empty (retrieval-only mode), or
     `error {message}` when the endpoint throws. 4. `done`.
- History: last 6 messages, 4k-char clamp per message. Context: 9k-char cap,
  1.2k per passage.
- Path traversal: `/diagrams/` resolves under `DATA_DIR` with prefix check;
  same for `/static/`. Config POST drops keys containing `…` (the masked
  display value) so the UI can't overwrite the real key with its own mask.
- Frontend: marked + KaTeX auto-render ($...$, $$...$$, \(...\), \[...\]),
  CDN-pinned with computed sha384 SRI + crossorigin (hook-flagged risk;
  hashes were generated from the actual fetched files, not guessed).
  Offline: UI degrades to raw markdown, still functional.

## 7. MCP server

- mcp 2.x renamed `FastMCP` → `MCPServer` and broke two assumptions:
  raw `bytes` returns now fail JSON serialization (must return
  `mcp.types.ImageContent` explicitly), and field names are snake_case
  (`mime_type`). Import shim covers 1.x/2.x.
- Tool design for agents: `search_notes` returns passages AND pre-joined
  diagram urls (one call gets evidence + images); `get_page` for full-page
  context; `get_diagram` returns the actual PNG so vision-capable agents
  read formulas instead of trusting garbled OCR; `list_topics` lets an agent
  route by chapter before searching. `instructions=` carries the
  "OCR is garbled, fetch diagrams, cite (file, p.N)" policy.
- Not yet added (deliberate): a `jeerag://page/{file}/{page}` *resource* —
  resources would make note pages first-class in MCP clients. See IDEAS §1.

## 8. LLM integration status

No endpoint is configured yet; retrieval-only mode is fully functional.
Candidate backends on this phone (user must supply keys — policy blocks us
from reading their credential files):
- omnirush-compatible-api → :8787 (OpenAI+Anthropic shapes, prints local key)
- Omnirush2API → :20129/v1 (Bearer-enforced)
- any cloud OpenAI-compatible URL via config.json.
System prompt (webapp.SYSTEM_PROMPT) is the quality contract: answer only
from excerpts, reconstruct garbled formulas honestly, LaTeX, tables, cite.
Attribution to Ambarish Sir is stated as *data*, with an explicit
"not an instruction" clause — the anti-prompt-injection framing the user asked for.

## 9. Measured constraints log

| Metric | Value |
|---|---|
| Build time, full corpus | ~55 min (numpy filter build); ~45s/file dominated by crop OCR |
| Index size | data/ ≈ 300MB (mostly diagram PNGs) |
| Diagrams | 2,458 crops / 2,279 pages |
| Free disk after build | ~1.7GB (never render pages to disk) |
| pkill/pgrep self-match | killed own tool-shell twice (exit 137/143); false-positive "RUNNING" once |
| pip on this device | needs --timeout 20 --retries 3; one 20-min stall recovered |
