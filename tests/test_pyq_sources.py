"""Offline tests for jeerag.pyq_sources — registry + answer-key parser.

No network, no PyMuPDF: covers everything that must hold everywhere. The
live fetch path is exercised manually with --list --check / --fetch (and was
verified end-to-end against jeeadv.ac.in + the Wayback Machine).
Run: python tests/test_pyq_sources.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jeerag import pyq_sources  # noqa: E402


def main():
    # --- parser: official "Answer Qn: ..." line formats -------------------
    text = ("Answer Q8: ABC\nAnswer Q.12: B D\nAnswer Q15: 14.75\nAnswer Q3: C\n")
    out = pyq_sources.parse_adv_key_text(text)
    assert out == {"8": "ABC", "12": "BD", "15": 14.75, "3": "C"}, out
    print("parse_adv_key_text: PASS")

    # options sorted ("C A" -> "AC"), must not bleed across line boundaries
    assert pyq_sources.parse_adv_key_text("Answer Q1: C A") == {"1": "AC"}
    assert pyq_sources.parse_adv_key_text("Answer Q2: B\nAnswer Q3: D") == \
        {"2": "B", "3": "D"}
    print("parse_adv_key_text edge cases: PASS")

    # --- registry --------------------------------------------------------
    ids = [it["id"] for it in pyq_sources.SOURCES]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for need in ("adv-2026-final-key-1", "adv-2026-final-key-2",
                 "adv-2026-paper-1", "main-2025-s2-final-key-p1"):
        assert need in ids, need
    for it in pyq_sources.SOURCES:
        assert it["urls"] and all(u.startswith("http") for u in it["urls"]), it
        assert 2007 <= it["year"] <= 2026, it
    verified = {it["id"] for it in pyq_sources.SOURCES if it["verified"]}
    assert {"adv-2026-final-key-1", "adv-2026-final-key-2",
            "adv-2026-paper-1", "main-2025-s2-final-key-p1"} <= verified, verified
    print(f"registry: PASS ({len(ids)} entries, {len(verified)} verified)")

    # --- destination paths mirror the papers/ layout ---------------------
    out_root = Path("/tmp/papers")
    paper = next(it for it in pyq_sources.SOURCES if it["id"] == "adv-2024-paper-1")
    p = pyq_sources.dest_path(paper, out_root)
    assert p == out_root / "jee-adv" / "2024" / "JEE Advanced 2024 Question Paper 1.pdf", p
    key = next(it for it in pyq_sources.SOURCES if it["id"] == "adv-2026-final-key-1")
    kp = pyq_sources.dest_path(key, out_root)
    assert kp == out_root / "jee-adv" / "keys" / \
        "JEE Advanced 2026 Final Answer Key Paper 1.pdf", kp
    main_key = next(it for it in pyq_sources.SOURCES
                    if it["id"] == "main-2025-s2-final-key-p1")
    mp = pyq_sources.dest_path(main_key, out_root)
    assert mp.parent == out_root / "jee-main" / "keys", mp
    print("dest_path: PASS")

    print("\nALL PYQ SOURCES TESTS PASS")


if __name__ == "__main__":
    main()
