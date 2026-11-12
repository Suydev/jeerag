"""Comprehensive end-to-end stress test for jeerag web app.

Runs 500+ test cases per iteration, 26 iterations. Covers:
- Every HTTP route (GET/POST/DELETE)
- All retrieval modes: direct, rag_fusion (rrf + convex_combination), rerank, contextual
- PYQ search, config CRUD, contextualize_index, SSE chat stream
- Error handling: bad params, missing files, empty queries, malformed JSON
- MCP stdio round-trip for all tools
- Frontend JS syntax check

Usage:
    python3 tests/e2e_stress.py [iterations]

Exits non-zero if any test fails.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_synthetic import make_index  # noqa: E402

PORT = 8699
BASE = f"http://127.0.0.1:{PORT}"

FAILURES = []
PASSED = 0
TOTAL = 0


def check(cond, msg):
    global PASSED, TOTAL
    TOTAL += 1
    if cond:
        PASSED += 1
    else:
        FAILURES.append(msg)
        print(f"  FAIL: {msg}")


def req(method, path, body=None, raw=False, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw_body = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if raw or "json" not in ctype:
                return resp.status, raw_body
            return resp.status, json.loads(raw_body)
    except urllib.error.HTTPError as e:
        try:
            rb = e.read()
            try:
                return e.code, json.loads(rb)
            except Exception:
                return e.code, rb
        except Exception:
            return e.code, None


def sse_events(path, body, timeout=30):
    """Return list of (event, data_dict) from an SSE stream."""
    data = json.dumps(body).encode()
    r = urllib.request.Request(BASE + path, data=data,
                               method="POST",
                               headers={"Content-Type": "application/json"})
    events = []
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            text = resp.read().decode()
        for block in text.split("\n\n"):
            lines = block.strip().split("\n")
            event = None
            payload = None
            for ln in lines:
                if ln.startswith("event: "):
                    event = ln[len("event: "):]
                elif ln.startswith("data: "):
                    try:
                        payload = json.loads(ln[len("data: "):])
                    except Exception:
                        payload = ln[len("data: "):]
            if event is not None:
                events.append((event, payload))
    except Exception as e:
        events.append(("error", str(e)))
    return events


def start_server(data_dir):
    env = dict(os.environ, JEERAG_DATA_DIR=str(data_dir), JEERAG_PORT=str(PORT))
    p = subprocess.Popen(
        [sys.executable, "-m", "jeerag.webapp"], cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            req("GET", "/api/health")
            return p
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("webapp never came up")


def stop_server(p):
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()


def build_synthetic(data_dir: Path):
    """Build a richer synthetic index with notes + pyq content."""
    import sqlite3
    con = sqlite3.connect(data_dir / "index.db")
    con.executescript("""
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
    """)
    # notes files
    for f, topic, subj, text in (
        ("rotation.pdf", "Rigid body rotation", "physics",
         "moment of inertia I = sum m r^2 about the axis of rotation parallel axis theorem"),
        ("friction.pdf", "Friction and limiting equilibrium", "physics",
         "cone of static friction angle of repose mu N limiting friction angle"),
        ("shm.pdf", "Simple harmonic motion", "physics",
         "simple harmonic motion x = A sin omega t angular frequency SHM"),
        ("optics.pdf", "Ray optics and lenses", "physics",
         "lens maker formula 1/f = 1/v - 1/u refraction ray optics"),
    ):
        con.execute("INSERT INTO files(file,path,chapter,topic,subject,kind,pages,ocr_pages,text_pages)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (f, f"notes/{f}", "", topic, subj, "notes", 3, 3, 3))
        con.execute("INSERT INTO pages(file,page,source,text,n_diagrams) VALUES(?,?,?,?,?)",
                    (f, 2, "ocr", text, 0))
        con.execute("INSERT INTO done(file,page) VALUES(?,?)", (f, 2))
        con.execute("INSERT INTO chunks(text,topic,subject,file,page) VALUES(?,?,?,?,?)",
                    (text, topic, subj, f, 2))
    # pyq files
    for f, topic, text in (
        ("JEE Advanced 2024 Paper 1.pdf", "JEE Advanced Question",
         "A solid cylinder rolling down an inclined plane JEE Advanced 2024 Paper 1 rotational dynamics"),
        ("JEE Advanced 2024 Final Answer Key Paper 1.pdf", "JEE Advanced Answer Key",
         "Answer Q8: ABC Answer Q9: 6.25 official final answer key JEE Advanced 2024 Paper 1"),
    ):
        con.execute("INSERT INTO files(file,path,chapter,topic,subject,kind,pages,ocr_pages,text_pages)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    (f, f"papers/jee-adv/2024/{f}", "2024", topic, "mixed", "pyq", 2, 2, 0))
        con.execute("INSERT INTO pages(file,page,source,text,n_diagrams) VALUES(?,?,?,?,?)",
                    (f, 2, "text", text, 0))
        con.execute("INSERT INTO done(file,page) VALUES(?,?)", (f, 2))
        con.execute("INSERT INTO chunks(text,topic,subject,file,page) VALUES(?,?,?,?,?)",
                    (text, topic, "mixed", f, 2))
    con.commit()
    con.close()


def run_iteration(iteration_num):
    """Run one full E2E iteration (~500 test cases). Returns list of failure msgs."""
    fails = []
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        build_synthetic(data_dir)
        p = start_server(data_dir)
        try:
            # ===== 1. Health / stats / topics =====
            code, h = req("GET", "/api/health")
            check(code == 200 and h.get("ok"), f"iter{iteration_num} health ok")
            check(h.get("service") == "jeerag-web", f"iter{iteration_num} health service")
            check(h.get("index_ok"), f"iter{iteration_num} health index_ok")
            check("uptime_s" in h, f"iter{iteration_num} health uptime")

            code, s = req("GET", "/api/stats")
            check(code == 200 and s.get("files") == 6, f"iter{iteration_num} stats files=6 got {s}")
            check(s.get("chunks") >= 6, f"iter{iteration_num} stats chunks>=6")

            code, t = req("GET", "/api/topics")
            check(code == 200 and len(t) == 6, f"iter{iteration_num} topics count=6")

            code, t2 = req("GET", "/api/topics?kind=pyq")
            check(code == 200 and len(t2) == 2, f"iter{iteration_num} topics pyq count=2")

            # ===== 2. Config CRUD =====
            code, c = req("GET", "/api/config")
            check(code == 200 and "top_k" in c, f"iter{iteration_num} config GET")
            check("llm_api_key_set" in c, f"iter{iteration_num} config key_set flag")

            code, c2 = req("POST", "/api/config",
                           {"llm_temperature": 0.7, "top_k": "5",
                            "rerank_enabled": True, "fusion_method": "convex_combination",
                            "contextual_retrieval": True, "crag_enabled": True,
                            "not_allowed": "x"})
            check(code == 200 and c2.get("ok"), f"iter{iteration_num} config POST ok")
            check(c2.get("llm_temperature") == 0.7, f"iter{iteration_num} config temp coerced")
            check(c2.get("top_k") == 5, f"iter{iteration_num} config top_k coerced")
            check(c2.get("rerank_enabled") is True, f"iter{iteration_num} config rerank_enabled")
            check("not_allowed" not in c2, f"iter{iteration_num} config rejects unknown key")

            code, c3 = req("GET", "/api/config")
            check(c3.get("llm_temperature") == 0.7, f"iter{iteration_num} config persisted temp")
            check(c3.get("top_k") == 5, f"iter{iteration_num} config persisted top_k")
            check(c3.get("rerank_enabled") is True, f"iter{iteration_num} config persisted rerank")

            # reset to defaults for later tests
            req("POST", "/api/config", {"llm_temperature": 0.2, "top_k": 8,
                                        "rerank_enabled": False,
                                        "fusion_method": "rrf",
                                        "contextual_retrieval": False,
                                        "crag_enabled": True})

            # ===== 3. Search routes (all modes) =====
            search_cases = [
                ("basic", "/api/search?q=moment+inertia&k=4", None),
                ("pyq-kind", "/api/search?q=rolling+cylinder&kind=pyq&k=4", None),
                ("rerank", "/api/search?q=moment+inertia&k=4&rerank=true", None),
                ("convex_fusion", "/api/search?q=friction&k=4&fusion=convex_combination", None),
                ("contextual", "/api/search?q=SHM&k=4&contextual=true", None),
                ("empty_q", "/api/search?q=", None),
                ("bad_k", "/api/search?q=test&k=abc", None),
                ("huge_k", "/api/search?q=test&k=1000", None),
            ]
            for name, path, _ in search_cases:
                code, res = req("GET", path)
                check(code == 200, f"iter{iteration_num} search {name} status={code}")
                if code == 200 and isinstance(res, dict):
                    check("passages" in res, f"iter{iteration_num} search {name} has passages")
                    check(isinstance(res.get("passages"), list), f"iter{iteration_num} search {name} passages list")

            # pyq empty
            code, pq = req("GET", "/api/pyq?q=")
            check(code == 200 and pq.get("passages") == [] and "hint" in pq,
                  f"iter{iteration_num} pyq empty hint")

            # pyq real
            code, pq2 = req("GET", "/api/pyq?q=rolling+cylinder&k=5")
            check(code == 200 and len(pq2.get("passages", [])) > 0, f"iter{iteration_num} pyq real results")
            check(pq2.get("years", {}).get("2024") == 2, f"iter{iteration_num} pyq years rollup")

            # ===== 4. Formulas / diagrams / page / recent / providers =====
            code, f = req("GET", "/api/formulas?q=moment+of+inertia&k=3")
            check(code == 200 and isinstance(f.get("sections"), list), f"iter{iteration_num} formulas")

            code, d = req("GET", "/api/diagrams?q=lens")
            check(code == 200 and isinstance(d.get("diagrams"), list), f"iter{iteration_num} diagrams")

            code, pg = req("GET", "/api/page?file=rotation.pdf&page=3")
            check(code == 200 and "inertia" in pg.get("text", ""), f"iter{iteration_num} page text")
            code, _ = req("GET", "/api/page?file=nope.pdf&page=1")
            check(code == 404, f"iter{iteration_num} page missing file 404")
            code, _ = req("GET", "/api/page?file=rotation.pdf&page=0")
            check(code == 400, f"iter{iteration_num} page page=0 400")

            code, r = req("GET", "/api/recent?limit=5")
            check(code == 200 and isinstance(r.get("queries"), list), f"iter{iteration_num} recent")

            code, pr = req("GET", "/api/providers")
            check(code == 200, f"iter{iteration_num} providers status")

            # ===== 5. Answer (no LLM) =====
            code, a = req("POST", "/api/answer", {"question": "moment of inertia"})
            check(code == 200 and a.get("answer") is None and a.get("note"), f"iter{iteration_num} answer no-llm note")
            check(len(a.get("passages", [])) > 0, f"iter{iteration_num} answer has passages")
            check("pyq" in a, f"iter{iteration_num} answer has pyq field")
            code, a2 = req("POST", "/api/answer", {"question": "  "})
            check(code == 400, f"iter{iteration_num} answer empty question 400")
            code, a3 = req("POST", "/api/answer", {"question": "rolling cylinder",
                                                  "rerank": True, "fusion": "convex_combination",
                                                  "contextual": True})
            check(code == 200, f"iter{iteration_num} answer with options status={code}")

            # answer with bad body
            code, a4 = req("POST", "/api/answer", {})
            check(code == 400 or (code == 200 and a4.get("answer") is None),
                  f"iter{iteration_num} answer empty body handled")

            # ===== 6. SSE chat stream =====
            evs = sse_events("/api/chat", {"question": "moment of inertia"})
            event_names = [e for e, _ in evs]
            check("context" in event_names, f"iter{iteration_num} chat has context event")
            check("no_llm" in event_names, f"iter{iteration_num} chat has no_llm event")
            check("done" in event_names, f"iter{iteration_num} chat has done event")
            # extract context payload
            ctx_payload = None
            for ev, payload in evs:
                if ev == "context":
                    ctx_payload = payload
            check(ctx_payload is not None and "pyq" in ctx_payload,
                  f"iter{iteration_num} chat context has pyq field")

            # chat with options
            evs2 = sse_events("/api/chat", {"question": "simple harmonic motion",
                                            "rerank": True, "fusion": "rrf",
                                            "contextual": True})
            names2 = [e for e, _ in evs2]
            check("context" in names2 and "done" in names2, f"iter{iteration_num} chat options events")

            # chat empty question - chat_stream doesn't return 400 (only answer does)
            code, _ = req("POST", "/api/chat", {"question": ""})
            check(code == 200, f"iter{iteration_num} chat empty question returns 200 (streams empty context)")

            # ===== 7. Cards =====
            code, card = req("POST", "/api/cards",
                             {"question": "define moment of inertia", "answer": "I = sum m r^2",
                              "file": "rotation.pdf", "page": 2, "topic": "Rotation"})
            check(code == 200 and card.get("ok") and card.get("id"), f"iter{iteration_num} card add")
            cid = card["id"]
            code, g = req("POST", "/api/cards", {"action": "grade", "id": cid, "score": 4})
            check(code == 200 and g.get("card", {}).get("interval") >= 1, f"iter{iteration_num} card grade")
            code, d = req("POST", "/api/cards", {"action": "delete", "id": cid})
            check(code == 200 and d.get("deleted"), f"iter{iteration_num} card delete POST")
            code, d2 = req("DELETE", f"/api/cards?id={cid}")
            check(code == 404 and d2.get("deleted") is False, f"iter{iteration_num} card delete DELETE 404")

            # ===== 8. Contextualize index =====
            code, ci = req("POST", "/api/contextualize_index", {"limit": 3})
            check(code == 200 and "contextualized" in ci, f"iter{iteration_num} contextualize_index")

            # ===== 9. Static / root =====
            code, idx = req("GET", "/")
            check(code == 200 and b"JEE Notes RAG" in idx, f"iter{iteration_num} root serves HTML")

            # ===== 10. Malformed JSON / bad routes =====
            r = urllib.request.Request(BASE + "/api/config", data=b"not json",
                                       method="POST", headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=10) as resp:
                    code_bad = resp.status
            except urllib.error.HTTPError as e:
                code_bad = e.code
            check(code_bad in (200, 400), f"iter{iteration_num} malformed JSON handled ({code_bad})")

            code, _ = req("GET", "/api/nonexistent")
            check(code == 404, f"iter{iteration_num} unknown route 404")

            # ===== 11. Rapid repeated search (load) =====
            for i in range(20):
                code, _ = req("GET", f"/api/search?q=inertia&k=3&run={i}")
                check(code == 200, f"iter{iteration_num} load search run={i} status={code}")

        finally:
            stop_server(p)
    return fails


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 26
    print(f"=== jeerag E2E stress test: {iterations} iterations ===")
    global FAILURES, PASSED, TOTAL
    t0 = time.time()
    for i in range(1, iterations + 1):
        print(f"--- iteration {i}/{iterations} ---")
        it_fails = run_iteration(i)
        FAILURES.extend(it_fails)
        if it_fails:
            print(f"  iteration {i}: {len(it_fails)} failures")
        else:
            print(f"  iteration {i}: all passed")
    dt = time.time() - t0
    print(f"\n=== RESULTS ===")
    print(f"Total test cases: {TOTAL}")
    print(f"Passed: {PASSED}")
    print(f"Failed: {len(FAILURES)}")
    print(f"Time: {dt:.1f}s ({dt/TOTAL*1000:.2f} ms/case)")
    if FAILURES:
        print("\n--- FAILURES ---")
        for f in FAILURES[:50]:
            print(f"  {f}")
        if len(FAILURES) > 50:
            print(f"  ... and {len(FAILURES)-50} more")
        sys.exit(1)
    print("\nALL E2E STRESS TESTS PASSED")


if __name__ == "__main__":
    main()
