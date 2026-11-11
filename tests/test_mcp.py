"""End-to-end MCP smoke test: spawn the server over stdio, call every tool.

Portable: uses the interpreter running this test and the repo root as cwd,
so it works from a venv, a system python, or Termux without edits.
"""
import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "jeerag.mcp_server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            res = await s.call_tool("index_stats", {})
            print("STATS:", res.content[0].text[:150])
            res = await s.call_tool("search_notes", {"query": "cone of static friction", "k": 2})
            data = json.loads(res.content[0].text)
            p0 = data["passages"][0]
            print("SEARCH:", p0["file"], "p" + str(p0["page"]), "| diagrams:", len(data["diagrams"]))
            assert data["diagrams"], "no diagrams returned"
            res = await s.call_tool("get_diagram", {"path": data["diagrams"][0]["path"]})
            img = res.content[0]
            print("DIAGRAM:", getattr(img, "mime_type", None) or getattr(img, "mimeType", None),
                  len(img.data), "bytes")
            assert str(getattr(img, "mime_type", None) or getattr(img, "mimeType", "")).startswith("image/")
            res = await s.call_tool("get_page", {"file": p0["file"], "page": p0["page"]})
            print("PAGE:", res.content[0].text[:120].replace(chr(10), " "))
            res = await s.call_tool("list_topics", {})
            print("TOPICS rows:", len(json.loads(res.content[0].text)))
            print("OK")


asyncio.run(main())
