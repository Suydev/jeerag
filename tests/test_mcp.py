"""End-to-end MCP smoke test: spawn the server over stdio, call every tool.

Uses only the stdlib (the server itself does too), so it runs unchanged in a
venv, a system python, or Termux::

    python tests/test_mcp.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Client:
    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-m", "jeerag.mcp_server"],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.n = 0

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            self.n += 1
            msg["id"] = self.n
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("server closed stdout")
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if resp.get("id") == msg["id"]:
                if "error" in resp:
                    raise RuntimeError(resp["error"])
                return resp["result"]

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()


def content_text(res):
    return "".join(c.get("text", "") for c in res["content"] if c["type"] == "text")


def main():
    c = Client()
    try:
        init = c.send("initialize", {"protocolVersion": "2025-06-18",
                                     "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "0"}})
        print("SERVER:", init["serverInfo"]["name"], init["protocolVersion"])
        c.send("notifications/initialized", notify=True)

        tools = c.send("tools/list")["tools"]
        names = [t["name"] for t in tools]
        print("TOOLS:", names)
        for want in ("search_notes", "search_diagrams", "get_page",
                     "get_diagram", "list_topics", "index_stats"):
            assert want in names, f"missing tool {want}"

        stats = json.loads(content_text(c.send("tools/call", {"name": "index_stats",
                                                               "arguments": {}})))
        print("STATS:", stats["files"], "files", stats["pages"], "pages",
              stats["diagrams"], "diagrams")
        assert stats["diagrams"] > 2000

        res = c.send("tools/call", {"name": "search_notes",
                                    "arguments": {"query": "cone of static friction", "k": 2}})
        data = json.loads(content_text(res))
        p0 = data["passages"][0]
        print("SEARCH:", p0["file"], "p" + str(p0["page"]),
              "| diagrams:", len(data["diagrams"]))
        assert data["diagrams"], "no diagrams returned"

        res = c.send("tools/call", {"name": "get_diagram",
                                    "arguments": {"path": data["diagrams"][0]["path"]}})
        img = res["content"][0]
        print("DIAGRAM:", img["mimeType"], len(img["data"]), "b64 chars")
        assert img["mimeType"].startswith("image/") and len(img["data"]) > 1000

        res = c.send("tools/call", {"name": "get_page",
                                    "arguments": {"file": p0["file"], "page": p0["page"]}})
        print("PAGE:", content_text(res)[:100].replace("\n", " "))

        res = c.send("tools/call", {"name": "search_diagrams",
                                    "arguments": {"query": "cone of static friction", "k": 3}})
        diags = json.loads(content_text(res))
        print("DIAGRAM SEARCH:", len(diags), "hits", diags[0]["path"] if diags else "-")
        assert diags, "caption index empty — run python -m jeerag.caption"

        res = c.send("tools/call", {"name": "list_topics", "arguments": {}})
        print("TOPICS rows:", len(json.loads(content_text(res))))

        res = c.send("resources/list")
        print("RESOURCES:", len(res["resources"]))
        assert res["resources"], "no resources"
        uri = f"jeerag://notes/{p0['file']}/p{p0['page']}"
        res = c.send("resources/read", {"uri": uri})
        txt = res["contents"][0]["text"]
        print("RESOURCE READ:", uri, "→", len(txt), "chars")
        assert txt.startswith("# ")
        print("OK")
    finally:
        c.close()


if __name__ == "__main__":
    main()
