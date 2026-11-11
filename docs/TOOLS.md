# TOOLS.md — every tool in jeerag, its contract, and how to install them

Reference for humans and for the next AI developer. Two environments appear
throughout: **PRoot** (this build, glibc, repo at `/root/jeerag`) and
**Termux** (bionic, repo wherever you clone it, notes at
`~/storage/shared/JEE/Ambarish notes` after `termux-setup-storage`).

## 0. System toolchain

### Termux bootstrap (run FIRST, before any Python)
```bash
pkg update
pkg install -y python clang make git tesseract termux-api
pkg install -y python-numpy python-pillow mupdf    # native builds — do NOT pip these
pip install pytesseract openai mcp                  # pure-python, safe on Termux
pip install pymupdf                                  # builds against pkg mupdf via clang
# verify:
python -c "import fitz, numpy, PIL, pytesseract, openai, mcp; print('toolchain OK')"
tesseract --list-langs    # must include eng
```
Gotchas: `pip` on Termux installs into `$PREFIX`; no PEP-668 lock here (venv
optional). If `import fitz` fails after pip build: `pip install --no-cache-dir
--force-reinstall pymupdf` with `CC=clang` exported. If numpy/PIL were
accidentally pip-installed over pkg versions: `pip uninstall -y numpy pillow`
and reinstall from pkg.

### PRoot (current build — already done, for reference)
`python3 -m venv .venv` + `requirements.txt` (system pip is PEP-668 locked).
tesseract 5.5.3 already on PATH.

## 1. MCP tools (`python -m jeerag.mcp_server`, stdio)

| Tool | Input | Output | Notes |
|---|---|---|---|
| `search_notes` | `query: str`, `k: int=8` | `{passages:[{text,topic,subject,file,page,score}], diagrams:[{file,page,path,url}]}` | BM25 via FTS5; quoted-term OR query, stop-word stripped, injection-safe. `score` = bm25() (more negative = better; -9.0 = topic fallback). |
| `get_page` | `file: str`, `page: int (1-based)` | `{file,page,text,ocr_source,diagrams[]}` | Full page; `ocr_source` ∈ text\|ocr. |
| `get_diagram` | `path: str (abs or data-relative)` | `ImageContent` PNG | mcp 2.x: must return `mcp.types.ImageContent`, raw bytes fail serialization. |
| `list_topics` | — | `[{file,chapter,topic,subject,pages,text_pages,ocr_pages}]` | Route before searching. |
| `index_stats` | — | `{files,pages,chunks,diagrams,notes_dir}` | Health check. |

Registration: see README §MCP. Client test: `tests/test_mcp.py`.

## 2. HTTP API (`python -m jeerag.webapp`, 127.0.0.1:8613)

| Endpoint | Method | Contract |
|---|---|---|
| `/` | GET | chat UI (KaTeX+marked, SRI-pinned CDNs) |
| `/api/search?q=` | GET | same JSON as search_notes |
| `/api/chat` | POST `{question, history[]}` | SSE: `context` → (`delta`* \| `no_llm` \| `error`) → `done` |
| `/api/topics`, `/api/stats` | GET | as MCP |
| `/api/config` | GET/POST | GET masks key (`abcd••••`); POST ignores mask values; keys: llm_base_url, llm_model, llm_api_key, top_k, port |
| `/diagrams/diagrams/<slug>/<name>.png` | GET | crop images; prefix-checked, no traversal |

## 3. CLI

```bash
.venv/bin/python -m jeerag.index ["substring"]   # build/resume index
.venv/bin/python ask.py "question"               # retrieval demo
python3 install.py [--no-build|--rebuild]        # one-shot setup
python3 uninstall.py [--purge]                   # teardown (keeps data by default)
```

## 4. Internal Python API (`jeerag/retrieve.py`)

`search(q,k) · build_context(hits) · diagrams_for(hits,limit) · page_text(file,page)
· topics() · stats() · sanitize_query(q) · connect() · db_path()` — all
config-driven via `jeerag/config.py` (defaults < config.json < `JEERAG_*` env).

## 5. Data contracts

- SQLite `data/index.db` — schema + quirks in docs/ARCHITECTURE.md §2
  (0-based `pages/diagrams.page` vs 1-based `chunks.page`).
- **`diagrams.path` and `files.path` are ABSOLUTE** — migration to a new
  machine/path needs SQL rewrite (see HANDOFF-DEV.md §Migration).
- Diagram crops: `data/diagrams/<slug>/p<1-based-page>_<n>.png` @150dpi.

## 6. Tools that DON'T exist yet (build order — details in docs/IDEAS.md)

1. `jeerag.caption` — vision-LLM diagram captioner (resumable, `caption` column).
2. `jeerag.formulas` — formula-sheet generator (LaTeX → PDF via fitz).
3. `jeerag.resources` — MCP resources (`jeerag://notes/{file}/p{page}`).
4. `index --force <substring>` — clean redo of one file.
5. `/pdfpage/{file}/{page}.png` on-demand original-page render (LRU, 120dpi).
6. PYQ indexer for `JEE Adv Papers`/`JEE Main Shift Papers` folders (same
   pipeline, they have clean text layers).
7. Relative-path storage for diagrams (kills the migration footgun).
