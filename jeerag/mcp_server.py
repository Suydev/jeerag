"""MCP server exposing the notes index as tools. Run: python -m jeerag.mcp_server"""
import base64
from pathlib import Path

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import retrieve

mcp = _Server(
    "jeerag",
    instructions=(
        "RAG over scanned JEE physics/maths notes (Ambarish Sir). "
        "Text was machine-OCR'd from page scans, so formulas may be garbled — "
        "always fetch the diagram image for any page that references a figure, "
        "and cite (file, p.N) in answers. Notes credit: Ambarish Sir Physics, "
        "youtube.com/@ambarishphysics."),
)


@mcp.tool()
def search_notes(query: str, k: int = 8) -> list:
    """BM25 search over all indexed note pages. Returns passages with file,
    topic, subject, page, score, plus diagram urls for those pages."""
    hits = retrieve.search(query, k)
    diags = retrieve.diagrams_for(hits)
    return {"passages": hits, "diagrams": diags}


@mcp.tool()
def get_page(file: str, page: int) -> dict:
    """Full OCR'd text of one page plus its diagram image paths."""
    out = retrieve.page_text(file, page)
    return out or {"error": f"page not found: {file} p{page}"}


@mcp.tool()
def get_diagram(path: str):
    """Return a diagram crop as a PNG image (from search_notes/get_page results)."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(retrieve.config.load()["data_dir"]) / path
    if not p.exists():
        raise FileNotFoundError(path)
    try:
        from mcp.types import ImageContent
        return ImageContent(type="image",
                            data=base64.b64encode(p.read_bytes()).decode(),
                            mimeType="image/png")
    except ImportError:
        return p.read_bytes()


@mcp.tool()
def list_topics() -> list:
    """Every indexed PDF with chapter, topic, subject and page counts."""
    return retrieve.topics()


@mcp.tool()
def index_stats() -> dict:
    """Counts of files/pages/chunks/diagrams and the notes directory."""
    return retrieve.stats()


if __name__ == "__main__":
    mcp.run()
