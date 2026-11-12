"""Route-level smoke test for jeerag.webapp — runs anywhere, no notes data.

Builds the one-file synthetic index (from test_synthetic), starts the real
webapp subprocess pointed at it via JEERAG_DATA_DIR, and exercises every
route: health, stats, formulas, recent, page, providers, config, answer,
cards (add/grade/delete), the SSE chat stream, and static serving::

    python tests/test_webapp_routes.py
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

PORT = 8619
BASE = f"http://127.0.0.1:{PORT}"


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            if "json" in ctype:
                return resp.status, json.loads(raw)
            return resp.status, raw
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def main():
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        make_index(data_dir)
        env = dict(os.environ, JEERAG_DATA_DIR=str(data_dir),
                   JEERAG_PORT=str(PORT))
        p = subprocess.Popen(
            [sys.executable, "-m", "jeerag.webapp"], cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            # wait for the server to bind
            for _ in range(50):
                try:
                    req("GET", "/api/health")
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("webapp never came up")

            # --- health ---
            code, h = req("GET", "/api/health")
            assert code == 200 and h["ok"] and h["service"] == "jeerag-web", h
            assert h["index"]["files"] == 1 and h["index_ok"], h
            assert h["llm"]["configured"] is False, h
            assert "uptime_s" in h, h
            print("health: PASS")

            # --- stats / topics still fine ---
            code, s = req("GET", "/api/stats")
            assert code == 200 and s["files"] == 1, s
            print("stats: PASS")

            # --- page text (pages table stores 0-based; API is 1-based) ---
            code, pg = req("GET", "/api/page?file=rotation.pdf&page=3")
            assert code == 200 and "inertia" in pg["text"], pg
            code, _ = req("GET", "/api/page?file=nope.pdf&page=1")
            assert code == 404
            code, _ = req("GET", "/api/page?file=rotation.pdf&page=0")
            assert code == 400
            print("page: PASS")

            # --- formulas harvest (unrouted before; now live) ---
            code, f = req("GET", "/api/formulas?q=moment%20of%20inertia&k=3")
            assert code == 200 and f["topic"] == "moment of inertia", f
            assert isinstance(f["sections"], list), f
            print("formulas: PASS (sections:", len(f["sections"]), ")")

            # --- config: GET + POST with new keys + bogus key rejected ---
            code, c = req("GET", "/api/config")
            assert code == 200 and "top_k" in c, c
            code, c2 = req("POST", "/api/config",
                           {"llm_temperature": 0.7, "top_k": "5",
                            "not_allowed": "x", "llm_max_tokens": "512"})
            assert code == 200 and c2["ok"], c2
            code, c3 = req("GET", "/api/config")
            assert c3["llm_temperature"] == 0.7, c3          # coerced float
            assert c3["top_k"] == 5 and c3["llm_max_tokens"] == 512, c3
            assert "not_allowed" not in c3, c3
            print("config: PASS")

            # --- providers (no LLM configured) ---
            code, pr = req("GET", "/api/providers")
            assert code == 200 and pr["ok"] is False, pr
            print("providers: PASS")

            # --- one-shot JSON answer (no LLM → note + passages) ---
            code, a = req("POST", "/api/answer", {"question": "moment of inertia"})
            assert code == 200 and a["answer"] is None and a["note"], a
            assert a["passages"], a
            code, a2 = req("POST", "/api/answer", {"question": "  "})
            assert code == 400, a2
            print("answer: PASS")

            # --- SSE chat (no LLM → context + no_llm + done) ---
            r = urllib.request.Request(
                BASE + "/api/chat",
                data=json.dumps({"question": "moment of inertia"}).encode(),
                method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=30) as resp:
                body = resp.read().decode()
            events = [ln.split(": ", 1)[1] for ln in body.splitlines()
                      if ln.startswith("event: ")]
            assert events == ["context", "no_llm", "done"], events
            print("chat SSE (no_llm): PASS", events)

            # --- cards: add → grade → delete (POST + DELETE) ---
            code, card = req("POST", "/api/cards",
                             {"question": "define moment of inertia",
                              "answer": "I = sum m r^2", "file": "rotation.pdf",
                              "page": 2, "topic": "Rotation"})
            assert code == 200 and card["ok"] and card["id"], card
            cid = card["id"]
            code, g = req("POST", "/api/cards",
                          {"action": "grade", "id": cid, "score": 4})
            assert code == 200 and g["card"]["interval"] >= 1, g
            code, d = req("POST", "/api/cards", {"action": "delete", "id": cid})
            assert code == 200 and d["deleted"], d
            code, d2 = req("DELETE", f"/api/cards?id={cid}")
            assert code == 404 and d2["deleted"] is False, d2
            print("cards add/grade/delete: PASS")

            # --- recent queries (chat + answer logged above) ---
            code, rc = req("GET", "/api/recent?limit=5")
            assert code == 200 and rc["queries"], rc
            assert any("moment of inertia" in q["question"]
                       for q in rc["queries"]), rc
            print("recent: PASS", len(rc["queries"]), "queries")

            # --- static + index ---
            code, idx = req("GET", "/")
            assert code == 200 and b"JEE Notes RAG" in idx, "index.html broken"
            print("static: PASS")

            print("\nALL WEBAPP ROUTE TESTS PASS")
        finally:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
