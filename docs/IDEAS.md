# IDEAS & ROADMAP — things worth doing to jeerag

My honest opinions, ranked by (value ÷ effort) given this phone's limits.
§1-3 are the big wins; §4-8 are study-product ideas; §9 is packaging.

## 1. MCP *resources* + multimodal answers (highest value, small work)

Expose note pages as resources: `jeerag://notes/{file}/p{page}` returning text
+ embedded ImageContent. MCP clients that render resources (Claude Code, this
CLI) would then *see* diagrams natively instead of us pasting paths. Also add
`jeerag://topics` as a resource. ~40 lines on top of the existing tools.

## 2. Vision-LLM diagram captions (the missing half of "diagrams tagged")

Right now diagrams are *found* but not *understood* — retrieval can't match
"free body diagram of block on wedge" to the right crop without text.
Plan: a `caption` column on `diagrams`; a batch job (`python -m jeerag.caption`)
that sends each crop to a vision-capable endpoint (OmniRush bridge? Gemini
free tier?) with a strict prompt: "describe this physics/maths figure in ≤25
words, LaTeX for symbols". 2,458 crops ≈ 1-2 credits-scale API job; resumable
the same way as index (skip non-null captions). Then:
- captions indexed into FTS (new `chunks` rows with `kind='caption'`) →
  diagram search works from text queries;
- `search_notes` returns captioned diagrams; agent can pick the right figure.
This is the single change that most improves answer quality.

## 3. Re-OCR the formulas with a vision model (quality ceiling)

The page text layer is the weakest link (`dy do` for dy/dx). A vision LLM
given the page render transcribes LaTeX *correctly*. Doing all 2,279 pages is
too much for the quota bridges; do it **on demand**: when `get_page` is called
during a chat, cache the vision-transcribed LaTeX per page (`pages.latex`
column). Hot pages get clean math after a few study sessions; cold pages cost
nothing. Pair with §2.

## 4. Embeddings — only after §2/§3

fastembed ONNX (`onnxruntime` ~40MB, `bge-small-en-int8` ~30MB) fits the disk
budget; RAM is the risk (keep batch ≤8 pages, single-thread). But embedding
garbled OCR is garbage-in-garbage-out; embeddings on *captioned diagrams +
vision-LaTeX'd pages* would actually shine. Order matters: §2→§3→§4.
Hybrid rank: `score = 0.6·bm25_norm + 0.4·cosine`, stored in a plain
numpy `float32` blob column (2,279×384×4B ≈ 3.5MB — brute-force scan is
instant, no ANN index needed at this size).

## 5. Query understanding (cheap wins, no models)

- **Synonym expansion table** for JEE vocab: "friction cone↔static friction",
  "AOD↔application of derivatives", "SHM↔simple harmonic motion", Kota
  abbreviations (LCD, AUC, AOD, DE, P&C…). A JSON file + rewrite in
  `sanitize_query`. The corpus filenames use these abbreviations; students
  ask in full words.
- **Hindi/Hinglish aliases** for this user's phrasing ("mein se", "ka formula")
  — strip as stopwords, map "formula"→"equation result standard".
- Topic routing: if query contains a topic word from `list_topics`,
  filter BM25 to that file first, fall back to global.

## 6. Indexer features

- `--force <substring>`: delete a file's rows (done/pages/chunks/diagrams)
  then re-index — currently redoing one file needs manual SQL.
- Normalize 0/1-based page convention (ARCHITECTURE §2).
- Parallelize per-file (ProcessPool over files, not pages) — build is 55min,
  most of it serial crop-OCR; 4 workers → ~15min. Watch RAM (tesseract peak).
- Watch mode: cron `python -m jeerag.index` nightly — it's already resumable,
  new PDFs in the folder get picked up free.
- `diagrams` table: store `ink`, `words`, `kind` (line-drawing/text-block/photo)
  from the filter pass — enables "only FBDs" style filters later.

## 7. Study-product ideas (this is where it stops being a RAG demo)

- **Formula sheet generator**: `python -m jeerag.formulas "friction"` →
  pull chunks matching `/(result|formula|equation)/i` per topic, LLM-cleanup
  to LaTeX, print a revision PDF (fitz writes PDFs natively — zero new deps).
- **Solved-example drill**: examples are already tagged blocks ("Example",
  "Solution") — extract (question, solution) pairs per topic → flashcard CSV
  for Anki, or an in-UI "quiz me" mode where the model shows the question and
  grades the typed answer against the notes' own solution.
- **Past-paper linkage**: the JEE Adv/Main papers folders sit right next to
  Ambarish's notes — index them with the same pipeline (they have real text
  layers, no OCR mess) and let answers cite "notes p.N + paper year Q7".
  Cross-topic search: "show me every PYQ solvable with cone of friction".
- **Weak-topic tracker**: log queries + retrieved topics to SQLite; weekly
  summary "you asked 14 questions from Rotation, 2 from Fluids".
- **Voice**: this phone has 8 cores and a mic; a `whisper.cpp` tiny-q8 (arm64,
  ~130MB) could feed questions hands-free. Only if disk frees up.

## 8. Web UI polish

- Render diagrams *inline in the answer* (model emits `[[diagram:p12_2]]`
  tokens we map to <img>) instead of a separate strip.
- Page-link: clicking a `(file, p.N)` chip opens the original PDF page —
  serve via `/pdfpage/{file}/{page}.png` (render on demand at 120dpi, cache
  LRU 50 — bounded disk).
- Mobile PWA manifest (it's already used from the phone browser — add
  theme-color + icon, works offline for retrieval-only mode).
- Streaming *search* progress (context event already does this implicitly).

## 9. Repo/GitHub packaging

- pyproject.toml + console_scripts (`jeerag index|ask|web|mcp`) — user asked
  for a GitHub repo; make `pip install jeerag` the story.
- Ship a **tiny demo corpus** (2 PDFs) + prebuilt demo index so the repo is
  usable without Ambarish's files; the real notes stay the user's private data
  (never commit PDFs — copyright + 1.1GB).
- LICENSE: MIT for code; a NOTICE file with the Ambarish Sir attribution and
  "note content belongs to its author; index for personal study".
- CI: the mcp test + a retrieval smoke test run anywhere (pure Python +
  tesseract apt package) — good README badge material.
- Tests to add: filter unit test with synthetic page (tile grid + text strip +
  line drawing fixture), SSE protocol test, config masking test.

## 10. What I'd explicitly NOT do

- Vector DB servers (Qdrant/Chroma daemons): wrong tool for 2.3k chunks on a
  phone.
- Fine-tuning anything.
- Full-page image storage (disk); render-on-demand instead (§8).
- Realtime/websockets — SSE is enough, one writer per request.
