"""jeerag command line: jeerag index|ask|search|web|mcp|review|formulas|caption|eval"""
import sys

USAGE = """usage: jeerag <command> [args]

commands:
  index [substring]    build/resume the index (resumable; redo matching files)
  ask "question"       retrieval demo: passages + diagrams, no LLM
  web [--port N]       web chat UI on 127.0.0.1:8613 (default)
  mcp                  MCP server over stdio
  review ...           spaced-repetition Doubt Deck (--add/--grade/--stats)
  formulas "topic"     formula sheet + revision PDF for a topic
  caption ...          caption diagram crops (offline OCR or vision)
  eval ...             retrieval/answer evaluation against golden_v1.jsonl
  tracker [--days N]   weak-topic summary of logged queries
"""


def _ask(rest):
    from . import retrieve
    q = " ".join(rest).strip()
    if not q:
        print('usage: jeerag ask "your question"')
        return 1
    hits = retrieve.search(q)
    if not hits:
        print("no matches")
        return 0
    for h in hits:
        print(f"\n--- {h['file']} ({h['topic']}, {h['subject']}) p.{h['page']} "
              f"[bm25 {h['score']:.2f}] ---")
        print(h["text"][:500])
    diags = retrieve.diagrams_for(hits)
    if diags:
        print("\ndiagrams:")
        for d in diags:
            print(f"  {d['file']} p.{d['page']}: {d['path']}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd == "index":
            from .index import main as m
            m()
            return 0
        if cmd in ("ask", "search"):
            return _ask(rest)
        if cmd == "web":
            port = None
            if "--port" in rest:
                port = int(rest[rest.index("--port") + 1])
            from .webapp import serve
            serve(port)
            return 0
        if cmd == "mcp":
            from .mcp_server import run_stdio
            run_stdio()
            return 0
        if cmd == "review":
            from .review import main as m
            return m(rest) or 0
        if cmd == "formulas":
            from .formulas import main as m
            return m(rest) or 0
        if cmd == "caption":
            from .caption import main as m
            return m(rest) or 0
        if cmd == "eval":
            from .eval import main as m
            return m(rest) or 0
        if cmd == "tracker":
            from .tracker import main as m
            return m(rest) or 0
        print(f"unknown command: {cmd}\n{USAGE}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
