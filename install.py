#!/usr/bin/env python3
"""One-shot installer for jeerag.

  python3 install.py            # venv + deps + build index if absent
  python3 install.py --no-build # skip indexing (do it later: python -m jeerag.index)
  python3 install.py --rebuild  # force full re-index
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "bin" / "python"


def run(cmd, **kw):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not PY.exists():
        run([sys.executable, "-m", "venv", ".venv"])
    run([str(PY), "-m", "pip", "install", "--quiet", "--timeout", "30", "-r", "requirements.txt"])

    (ROOT / "config.json").touch(exist_ok=True)

    db = ROOT / "data" / "index.db"
    if not args.no_build and (args.rebuild or not db.exists()):
        if args.rebuild:
            print("rebuilding index from scratch")
            subprocess.run(["rm", "-rf", str(ROOT / "data")])
        run([str(PY), "-m", "jeerag.index"])

    print(
        "\nInstalled. Next:\n"
        f"  web UI   : {PY} -m jeerag.webapp   → http://127.0.0.1:8613\n"
        f"  CLI ask  : {PY} ask.py \"your question\"\n"
        "  MCP      : add to your agent's mcp config:\n"
        '    {"mcpServers": {"jeerag": {"command": "' + str(PY) + '", '
        '"args": ["-m", "jeerag.mcp_server"], "cwd": "' + str(ROOT) + '"}}}\n'
    )


if __name__ == "__main__":
    sys.exit(main())
