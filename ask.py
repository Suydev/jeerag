#!/usr/bin/env python3
"""CLI demo of the retrieval core: python ask.py "question" """
import sys

from jeerag import retrieve


def main():
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        print('usage: ask.py "your question"')
        sys.exit(1)
    hits = retrieve.search(q)
    if not hits:
        print("no matches")
        return
    for h in hits:
        print(f"\n--- {h['file']} ({h['topic']}, {h['subject']}) p.{h['page']} "
              f"[bm25 {h['score']:.2f}] ---")
        print(h["text"][:500])
    diags = retrieve.diagrams_for(hits)
    if diags:
        print("\ndiagrams:")
        for d in diags:
            print(f"  {d['file']} p.{d['page']}: {d['path']}")


if __name__ == "__main__":
    main()
