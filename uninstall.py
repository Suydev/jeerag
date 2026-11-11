#!/usr/bin/env python3
"""Remove jeerag venv and code artifacts. The index (data/) is KEPT unless --purge.

  python3 uninstall.py [--purge]
"""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true",
                    help="also delete data/ (index.db + diagram crops)")
    args = ap.parse_args()

    for d in (ROOT / ".venv", ROOT / "__pycache__", ROOT / "jeerag" / "__pycache__"):
        if d.exists():
            shutil.rmtree(d)
            print("removed", d)
    if args.purge and (ROOT / "data").exists():
        shutil.rmtree(ROOT / "data")
        print("removed data/ (index + diagrams)")
    else:
        print("kept data/ — re-run install.py to reuse the index")
    print("\nUninstalled. To finish: remove the 'jeerag' entry from your MCP config.")


if __name__ == "__main__":
    main()
