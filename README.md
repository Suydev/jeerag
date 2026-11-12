# jeerag — RAG for scanned JEE notes

[![CI](https://github.com/suydev/jeerag/actions/workflows/ci.yml/badge.svg)](https://github.com/suydev/jeerag/actions/workflows/ci.yml)

Ask questions over 73 scanned PDFs of **Ambarish Sir**'s IIT-JEE physics &
maths notes (2,279 pages). Works fully offline for retrieval; answers stream
from any OpenAI-compatible endpoint you configure.

```
┌────────────┐   MCP tools    ┌──────────────┐
│ any agent  │◄──────────────►│ jeerag.index │  SQLite FTS5 + diagram crops
└────────────┘                └──────┬───────┘
┌────────────┐   SSE + LaTeX        │
│ web chat   │◄─────────────────────►│
└────────────┘   http://127.0.0.1:8613
```

## What it does

- **Indexing** (`python -m jeerag.index`) — reads each PDF's existing OCR text
  layer (falls back to tesseract for blank pages), tags every chunk with
  file/chapter/topic/subject/page, and extracts **only real diagrams**:
  scanner background tiles, blank crops and text strips are rejected
  (word-count check via tesseract).
- **Retrieval** — SQLite FTS5 BM25, topic-name fallback, diagram lookup per hit.
- **Web chat UI** — localhost page with KaTeX-rendered math, Markdown tables,
  inline diagram images, source citations, streaming answers, settings panel
  for your LLM endpoint.
- **MCP server** — `search_notes`, `get_page`, `get_diagram`, `list_topics`,
  `index_stats` for any MCP-capable agent (see `AGENTS.md`).

## Install

```bash
pip install jeerag           # or: python3 install.py (venv + deps + index)
jeerag web                   # → open http://127.0.0.1:8613
jeerag ask "cone of static friction"
jeerag index                 # first run indexes ~73 PDFs (resumable)
```

CLI subcommands: `index` `ask` `web` `mcp` `review` `formulas` `caption` `eval`.

From source (this repo):

```bash
python3 install.py           # venv + deps + index
python3 -m jeerag.webapp     # → open http://127.0.0.1:8613
python3 ask.py "cone of static friction"
```

Requires Python 3.12+, and `tesseract` on PATH (only used for blank-page OCR
fallback and diagram text-strip rejection).

## LLM endpoint

Open the web UI → ⚙ Settings, or edit `config.json`:

```json
{"llm_base_url": "http://127.0.0.1:8787/v1", "llm_model": "some-model", "llm_api_key": "..."}
```

Any OpenAI-compatible `/v1/chat/completions` works (local proxies, Groq,
OpenAI, Gemini bridges…). With no endpoint set, the UI still shows the top
passages and diagrams — retrieval-only mode.

## MCP registration

```json
{"mcpServers": {"jeerag": {
  "command": ".venv/bin/python",
  "args": ["-m", "jeerag.mcp_server"],
  "cwd": "/path/to/jeerag"}}}
```

## Uninstall

```bash
python3 uninstall.py          # keeps your index
python3 uninstall.py --purge  # deletes index + diagrams too
```

## Notes on quality

The PDFs carry a machine-OCR text layer of handwritten notes: formulas often
arrive garbled (`dy do` → dy/dx). The pipeline therefore pairs every answer
with the page diagram crops and citations `(file, p.N)` so you can always
verify against the original scan.

## Documentation

- `docs/ARCHITECTURE.md` — schema, the diagram-filter story, SSE/MCP design, tradeoffs
- `docs/OPERATIONS.md` — runbook, device hazards, verification suite, incident log
- `docs/IDEAS.md` — ranked roadmap: vision captions, on-demand LaTeX re-OCR, study tools
- `AGENTS.md` — how an AI agent should drive this system (MCP tools + answer policy)
- `HANDOFF.md` — current session state for whoever picks this up next

## Credit

All note content is by **Ambarish Sir** — youtube.com/@ambarishphysics and
youtube.com/@thephysicsshelter. This tool indexes his public notes for
personal study; it is not affiliated with him.
