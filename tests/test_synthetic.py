"""Retrieval smoke test on a synthetic index — runs anywhere, no notes data.

Builds a one-file index.db with the real schema in a temp dir, points the
package at it via JEERAG_DATA_DIR, and exercises retrieval plus a full MCP
stdio round trip. Run: python tests/test_synthetic.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA = """
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
"""


def make_index(data_dir: Path):
    import sqlite3
    con = sqlite3.connect(data_dir / "index.db")
    con.executescript(SCHEMA)
    con.execute("INSERT INTO files(file,path,chapter,topic,subject,kind,pages,"
                "ocr_pages,text_pages) VALUES(?,?,?,?,?,?,?,?,?)",
                ("rotation.pdf", "notes/rotation.pdf", "Rotation",
                 "Rigid body rotation", "physics", "notes", 3, 3, 3))
    con.execute("INSERT INTO pages(file,page,source,text,n_diagrams) "
                "VALUES('rotation.pdf',2,'ocr',?,0)",
                ("moment of inertia I = sum m r^2 about the axis of rotation "
                 "parallel axis theorem shifts I by m d^2",))
    con.execute("INSERT INTO done(file,page) VALUES('rotation.pdf',2)")
    con.execute("INSERT INTO chunks(text,topic,subject,file,page) VALUES(?,?,?,?,?)",
                ("moment of inertia I = sum m r^2 about the axis; parallel axis "
                 "theorem shifts I by m d^2", "Rigid body rotation", "physics",
                 "rotation.pdf", 2))
    # PYQ corpus: official question paper + official final answer key (kind='pyq')
    for f, chapter, topic, path in (
        ("JEE Advanced 2024 Paper 1.pdf", "2024", "JEE Advanced Question",
         "papers/jee-adv/2024/JEE Advanced 2024 Paper 1.pdf"),
        ("JEE Advanced 2024 Final Answer Key Paper 1.pdf", "2024",
         "JEE Advanced Answer Key",
         "papers/jee-adv/keys/JEE Advanced 2024 Final Answer Key Paper 1.pdf"),
    ):
        con.execute("INSERT INTO files(file,path,chapter,topic,subject,kind,pages,"
                    "ocr_pages,text_pages) VALUES(?,?,?,?,?,?,?,?,?)",
                    (f, path, chapter, topic, "mixed", "pyq", 2, 2, 0))
    con.execute("INSERT INTO chunks(text,topic,subject,file,page) VALUES(?,?,?,?,?)",
                ("A solid cylinder rolling down an inclined plane; JEE Advanced "
                 "2024 Paper 1 question on rotational dynamics", "JEE Advanced Question",
                 "mixed", "JEE Advanced 2024 Paper 1.pdf", 2))
    con.execute("INSERT INTO chunks(text,topic,subject,file,page) VALUES(?,?,?,?,?)",
                ("Answer Q8: ABC Answer Q9: 6.25 official final answer key "
                 "JEE Advanced 2024", "JEE Advanced Answer Key", "mixed",
                 "JEE Advanced 2024 Final Answer Key Paper 1.pdf", 2))
    con.commit()
    con.close()


class McpClient:
    """Minimal stdio JSON-RPC client against python -m jeerag.mcp_server."""

    def __init__(self, env):
        self.p = subprocess.Popen(
            [sys.executable, "-m", "jeerag.mcp_server"], cwd=str(ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def call(self, method, params=None, mid=1):
        self.p.stdin.write(
            __import__("json").dumps(
                {"jsonrpc": "2.0", "id": mid, "method": method,
                 "params": params or {}}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        assert line, "MCP server closed stdout"
        resp = __import__("json").loads(line)
        assert "error" not in resp, resp["error"]
        return resp["result"]

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def main():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        make_index(data_dir)
        os.environ["JEERAG_DATA_DIR"] = str(data_dir)
        from jeerag import retrieve

        hits = retrieve.search("moment of inertia")
        assert hits, "bm25 search returned nothing"
        h = hits[0]
        assert h["file"] == "rotation.pdf" and h["page"] == 2, h
        assert "inertia" in h["text"], h["text"]

        topics = retrieve.topics()
        assert topics and topics[0]["file"] == "rotation.pdf", topics
        stats = retrieve.stats()
        assert isinstance(stats, dict) and stats, stats

        # PYQ enrichment: a question touching the PYQ corpus gets related
        # past-year chunks; a notes-only question gets silence.
        pyq = retrieve.related_pyq("rolling cylinder on an inclined plane")
        assert pyq and pyq[0]["file"] == "JEE Advanced 2024 Paper 1.pdf", pyq
        assert pyq[0]["score"] < 0, pyq
        assert retrieve.related_pyq("moment of inertia") == [], "pyq noise"
        print("synthetic retrieval: PASS")

        # --- New retrieval features (fallback mode, no LLM) ---
        from jeerag import rerank, contextual_retrieval, advanced_retrieve
        cfg = {}

        # rerank_passages with fallback scoring
        rr = rerank.rerank_passages("moment of inertia", hits, cfg, top_n=2)
        assert len(rr) <= 2, rr
        assert all("rerank_score" in h for h in rr), rr
        assert all("rerank_source" in h for h in rr), rr
        assert all(h["rerank_source"] == "fallback" for h in rr), rr
        print("rerank_passages (fallback): PASS")

        # two_stage_retrieve with fallback scoring
        ts = rerank.two_stage_retrieve("moment of inertia", k=2, cfg=cfg)
        assert ts["reranked"] is True, ts
        assert ts["recall_pool"] >= 1, ts
        assert len(ts["passages"]) <= 2, ts
        print("two_stage_retrieve (fallback): PASS")

        # contextual_retrieval.search_with_context falls back to plain search
        # when no contexts table exists
        ctx_hits = contextual_retrieval.search_with_context(
            "moment of inertia", k=2, cfg=cfg)
        assert len(ctx_hits) <= 2, ctx_hits
        print("search_with_context (no contexts table): PASS")

        # advanced_search with various options
        res = advanced_retrieve.advanced_search(
            "moment of inertia", k=2, mode="direct", cfg=cfg, use_loop=False)
        assert res["method"] == "direct_bm25_domain", res
        assert len(res["passages"]) <= 2, res
        print("advanced_search (direct): PASS")

        res = advanced_retrieve.advanced_search(
            "moment of inertia", k=2, mode="rag_fusion", fusion="rrf",
            cfg=cfg, use_loop=False)
        assert res["method"] == "rag_fusion_rrf", res
        print("advanced_search (rag_fusion rrf): PASS")

        res = advanced_retrieve.advanced_search(
            "moment of inertia", k=2, mode="rag_fusion", fusion="convex_combination",
            cfg=cfg, use_loop=False)
        assert res["method"] == "rag_fusion_convex_combination", res
        print("advanced_search (rag_fusion convex_combination): PASS")

        # convex_combination_fusion directly
        from jeerag.rerank import convex_combination_fusion
        fused = convex_combination_fusion(hits[:2], hits[:2], alpha=0.5, top_n=2)
        assert len(fused) <= 2, fused
        assert all("fusion" in h for h in fused), fused
        print("convex_combination_fusion: PASS")

        env = dict(os.environ, JEERAG_DATA_DIR=str(data_dir))
        c = McpClient(env)
        try:
            init = c.call("initialize", {"protocolVersion": "2025-06-18",
                                         "capabilities": {}})
            assert init["serverInfo"]["name"] == "jeerag", init
            tools = c.call("tools/list")["tools"]
            names = {t["name"] for t in tools}
            assert "search_notes" in names and "list_topics" in names, names
            for new_tool in ("search_pyq", "get_page_context",
                             "harvest_formulas", "add_study_card",
                             "delete_study_card", "study_stats",
                             "recent_queries", "evaluate_answer",
                             "rag_answer"):
                assert new_tool in names, f"missing new tool {new_tool}"
            for newer_tool in ("search_notes_reranked", "contextualize_index"):
                assert newer_tool in names, f"missing newer tool {newer_tool}"
            res = c.call("tools/call",
                         {"name": "search_notes",
                          "arguments": {"query": "moment of inertia"}})
            text = "".join(x.get("text", "") for x in res["content"])
            assert "rotation.pdf" in text, text
            print("synthetic MCP stdio: PASS")

            # --- new tools ---
            res = c.call("tools/call", {"name": "get_page_context",
                                        "arguments": {"file": "rotation.pdf",
                                                      "page": 3, "radius": 1}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert out.get("pages") and "inertia" in out["pages"][0]["text"], out
            print("get_page_context: PASS")

            res = c.call("tools/call", {"name": "harvest_formulas",
                                        "arguments": {"topic": "rotation"}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert out.get("topic") == "rotation", out
            assert "sections" in out, out
            print("harvest_formulas: PASS")

            res = c.call("tools/call", {"name": "study_stats",
                                        "arguments": {"days": 7}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert "days" in out and "topics" in out, out
            print("study_stats: PASS")

            res = c.call("tools/call", {"name": "recent_queries",
                                        "arguments": {"limit": 5}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert "queries" in out, out
            print("recent_queries: PASS")

            res = c.call("tools/call", {"name": "search_pyq",
                                        "arguments": {"query": "inertia"}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert "passages" in out and "diagrams" in out, out
            print("search_pyq: PASS")

            res = c.call("tools/call", {"name": "rag_answer",
                                        "arguments": {"question": "moment of inertia"}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert out.get("answer") is None and out.get("note"), out
            assert out.get("passages"), out
            print("rag_answer (no LLM): PASS")

            res = c.call("tools/call", {"name": "add_study_card",
                                        "arguments": {"question": "test card q",
                                                      "answer": "test card a",
                                                      "topic": "Rotation"}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert out.get("id"), out
            cid = out["id"]
            res = c.call("tools/call", {"name": "delete_study_card",
                                        "arguments": {"id": cid}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            assert out.get("deleted") is True, out
            print("add/delete_study_card: PASS")

            res = c.call("tools/call", {"name": "evaluate_answer",
                                        "arguments": {"question": "moment of inertia",
                                                      "answer": "I = sum m r^2"}})
            out = json.loads("".join(x.get("text", "") for x in res["content"]))
            for metric in ("faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"):
                assert metric in out, (metric, out)
            print("evaluate_answer: PASS", out)
        finally:
            c.close()


if __name__ == "__main__":
    main()
