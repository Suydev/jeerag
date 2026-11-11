"""Turn a topic's notes into a printable one-page-per-topic revision sheet.

    python -m jeerag.formulas "friction"
    python -m jeerag.formulas "rotational motion" --out ~/rot.pdf --k 8

The corpus is scanned, so the formulas are recovered from OCR text: every line
that looks like a relation (contains ``=`` or ``→`` and reads like an
expression) is harvested from the best-matching pages, de-duplicated, LaTeX
commands are folded to Unicode (so ``\\frac{\\mu N}{m}`` prints as ``μN/m``),
and the page's diagram crops are embedded beside them.  Output is a PDF via
PyMuPDF — no LaTeX installation, no network.
"""
from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

from . import config, retrieve

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ",
    "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
SYMBOLS = {
    "times": "×", "cdot": "·", "pm": "±", "mp": "∓", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈",
    "equiv": "≡", "propto": "∝", "infty": "∞", "partial": "∂",
    "nabla": "∇", "sum": "∑", "int": "∫", "sqrt": "√", "rightarrow": "→",
    "to": "→", "Rightarrow": "⇒", "leftarrow": "←", "leftrightarrow": "↔",
    "degree": "°", "circ": "°", "angle": "∠", "perp": "⊥", "parallel": "∥",
    "therefore": "∴", "because": "∵", "in": "∈", "ldots": "…", "cdots": "⋯",
}
SUPER = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
SUB = str.maketrans("0123456789+-=()aeijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
CMD = re.compile(r"\\([A-Za-z]+)")


def latex_to_unicode(s: str) -> str:
    """Best-effort readable form of a LaTeX fragment (no math layout)."""
    for _ in range(3):                      # nested fractions
        s = FRAC.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)
    s = SQRT.sub(lambda m: "√(" + m.group(1) + ")", s)

    def sym(m):
        name = m.group(1)
        return GREEK.get(name) or SYMBOLS.get(name) or name
    s = CMD.sub(sym, s)
    s = s.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\ ", " ")
    s = s.replace("{", "").replace("}", "").replace("$", "")
    s = re.sub(r"_\{([^{}]*)\}", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: m.group(1).translate(SUPER), s)
    s = re.sub(r"_([A-Za-z0-9])", lambda m: m.group(1).translate(SUB), s)
    s = re.sub(r"\^([A-Za-z0-9])", lambda m: m.group(1).translate(SUPER), s)
    return re.sub(r"\s+", " ", s).strip()


FORMULAISH = re.compile(r"[=→∝≈<>]|\\[a-zA-Z]+")


SENT = re.compile(r"[^.!?\n]{35,200}[.!?]")


def harvest(text, limit=5, topic_words=()):
    """Pull relation lines and concept sentences out of one page's OCR text.

    Scans give few clean ``=`` lines (the corpus has ~60 in 128k lines), so the
    sheet also carries short explanatory sentences that mention the topic — the
    readable part of the OCR — instead of inventing formulas.
    """
    relations, recaps = [], []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not (4 <= len(line) <= 90) or not FORMULAISH.search(line):
            continue
        if sum(ch.isalpha() for ch in line) > len(line) * 0.8:   # prose
            continue
        clean = latex_to_unicode(line)
        if len(clean) >= 3 and clean.lower() not in {c.lower() for c in relations}:
            relations.append(clean)
        if len(relations) >= limit:
            break
    flat = " ".join((text or "").split())
    for m in SENT.finditer(flat):
        sent = " ".join(m.group(0).split())
        low = sent.lower()
        if sum(w in low for w in topic_words) < (1 if len(topic_words) <= 2 else 2):
            continue
        digits = sum(ch.isdigit() for ch in sent)
        if digits > len(sent) * 0.10:              # mostly OCR'd numbers
            continue
        if any(j in low for j in ("youtube", "http", "ambarish", "thephysicsshelter", "[dia", "com/")):
            continue
        if sum(ch.isalpha() for ch in sent) < len(sent) * 0.6:
            continue
        if low not in {c.lower() for c in recaps}:
            recaps.append(sent if len(sent) <= 170 else sent[:167].rsplit(" ", 1)[0] + "…")
        if len(recaps) >= min(limit, 3):
            break
    return relations, recaps


def collect(topic, k=6):
    """Search the topic, harvest formulas and diagrams per hit page."""
    hits = retrieve.search(topic, k=k)
    con = retrieve.connect()
    data_dir = Path(config.load()["data_dir"])
    sections, seen, seen_recaps = [], set(), set()
    topic_words = tuple(w for w in re.findall(r"[a-z]{4,}", topic.lower()))
    for h in hits:
        txt = con.execute("SELECT text FROM pages WHERE file=? AND page=?",
                          (h["file"], h["page"] - 1)).fetchone()
        relations, recaps = harvest(txt["text"] if txt else "", topic_words=topic_words) if txt else ([], [])
        relations = [f for f in relations if f.lower() not in seen]
        for f in relations:
            seen.add(f.lower())
        recaps = [r for r in recaps if r.lower() not in seen_recaps]
        for r in recaps:
            seen_recaps.add(r.lower())
        rows = con.execute("SELECT path FROM diagrams WHERE file=? AND page=? LIMIT 2",
                           (h["file"], h["page"] - 1)).fetchall()
        imgs = [r["path"] for r in rows
                if (data_dir / r["path"]).exists()]
        if relations or recaps or imgs:
            sections.append({"file": h["file"], "page": h["page"],
                             "topic": h.get("topic") or "", "formulas": relations,
                             "recaps": recaps, "images": imgs})
    con.close()
    return sections


CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; color: #7a1f1f; }
h2 { font-size: 12pt; margin: 14pt 0 4pt 0; color: #24406e;
     border-bottom: 1px solid #ccd; padding-bottom: 2pt; }
.meta { color: #666; font-size: 8.5pt; }
pre { font-family: monospace; font-size: 10.5pt; margin: 3pt 0 3pt 10pt;
      white-space: pre-wrap; }
.cite { color: #888; font-size: 8pt; }
img { margin: 4pt 6pt 4pt 0; }
.note { color: #444; font-size: 9pt; margin-top: 10pt; }
.lbl { color: #7a1f1f; font-size: 8.5pt; text-transform: uppercase;
       letter-spacing: .5pt; margin: 4pt 0 0 0; }
ul { margin: 2pt 0 2pt 14pt; padding: 0; }
li { margin: 1pt 0; }
"""


def build_html(topic, sections, notes_dir=""):
    esc = html.escape
    parts = [f"<h1>{esc(topic.title())} — revision sheet</h1>",
             f'<p class="meta">Recovered from {len(sections)} note pages · '
             f'Ambarish Sir Physics · jeerag</p>']
    for s in sections:
        parts.append(f'<h2>{esc(s["topic"] or Path(s["file"]).stem)}</h2>')
        parts.append(f'<p class="cite">{esc(s["file"])} · p.{s["page"]}</p>')
        if s["formulas"]:
            parts.append('<p class="lbl">relations found</p>')
            for f in s["formulas"]:
                parts.append(f"<pre>{esc(f)}</pre>")
        if s.get("recaps"):
            parts.append('<p class="lbl">what the notes say</p><ul>')
            for r in s["recaps"]:
                parts.append(f"<li>{esc(r)}</li>")
            parts.append("</ul>")
        if s["images"]:
            imgs = "".join(
                f'<img src="{esc(p)}" width="240">' for p in s["images"])
            parts.append(f"<p>{imgs}</p>")
    parts.append('<p class="note">Formulas were machine-read from scanned '
                 'handwriting; check the cited page image before trusting a'
                 ' sign or exponent.</p>')
    return "<html><body>" + "".join(parts) + "</body></html>"


def make_pdf(topic, k=6, out=None, diagrams=True):
    import pymupdf
    cfg = config.load()
    data_dir = Path(cfg["data_dir"])
    sections = collect(topic, k=k)
    if not sections:
        raise SystemExit(f"no note pages matched {topic!r}")
    if not diagrams:
        for s in sections:
            s["images"] = []
    out = Path(out) if out else data_dir / "exports" / (re.sub(r"\W+", "-", topic.lower()).strip("-") + "-revision.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    story = pymupdf.Story(html=build_html(topic, sections), user_css=CSS,
                          archive=str(data_dir))
    writer = pymupdf.DocumentWriter(str(out))
    mb = pymupdf.paper_rect("a4")
    where = mb + (44, 44, -44, -50)
    more = 1
    pages = 0
    while more:
        dev = writer.begin_page(mb)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
        pages += 1
    writer.close()
    return out, pages, len(sections)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LaTeX-aware revision PDF from the notes")
    ap.add_argument("topic")
    ap.add_argument("--k", type=int, default=6, help="pages to draw from")
    ap.add_argument("--out", help="output .pdf path")
    ap.add_argument("--no-diagrams", action="store_true")
    args = ap.parse_args(argv)
    out, pages, sections = make_pdf(args.topic, args.k, args.out,
                                    diagrams=not args.no_diagrams)
    print(f"wrote {out} ({pages} pages, {sections} note pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
