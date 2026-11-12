"""Advanced retrieval pipeline for jeerag.

All techniques are dependency-free: they layer on top of the existing
BM25/FTS5 core (``jeerag.retrieve``) and the LLM client (``jeerag.llm``).
No torch, no embeddings, no vector DB — pure Python + BM25 + your LLM
endpoint.

Techniques implemented (researched from GitHub advanced-RAG repos)
------------------------------------------------------------------
1. **Query expansion (JEE domain)** — synonym/abbreviation table
   (IDEAS §5) plus LLM-generated paraphrases. Bridges the gap between
   how students ask ("cone of friction") and how the notes are OCR'd
   ("cone of static friction", "limiting friction angle").

2. **Multi-Query RAG (RAG-Fusion)** — the LLM rewrites the question into
   complementary phrasings; each is run through BM25; results are fused
   with Reciprocal Rank Fusion (RRF). Surfaces material a single
   phrasing misses.

3. **BM25-HyDE** — the LLM writes a hypothetical answer; that text is
   used as the BM25 query. Matches answer-shaped document language.

4. **Self-Corrective RAG (CRAG-lite)** — a relevance grader scores the
   top passages against the question; if all are weak, the pipeline
   retries with a broader / rephrased query (the follow-loop rethink step).

5. **Context distillation** — long retrieved passages are trimmed to the
   most relevant sentences before being sent to the LLM.

6. **Query routing** — detects PYQ-style questions and routes to
   ``kind='pyq'``; otherwise searches notes (falls back to pyq).

7. **Agentic follow-loop** — ``AgentLoop`` orchestrates
   finish → rethink → creativity → patch → research → apply → test e2e →
   finish with a bounded retry budget.

Usage::

    from jeerag.advanced_retrieve import advanced_search, AgentLoop
    result = advanced_search("cone of static friction", k=8)
    # result: {"passages": [...], "diagrams": [...], "method": "rag_fusion",
    #          "query_variants": [...], "reranked": True, "loops": 1}

All functions return the same shapes as ``jeerag.retrieve`` so they drop
in to the MCP server and webapp unchanged.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from typing import Iterable

from . import config, retrieve, llm, rerank, contextual_retrieval
from .rerank import convex_combination_fusion

RRF_K = 60  # RRF constant; higher = less influence from top rank
MAX_LOOPS = 3  # agentic follow-loop budget
CONVEX_ALPHA = 0.5  # convex combination: equal weight to BM25 + dense


# ---------------------------------------------------------------------------
# 1. JEE domain synonym / abbreviation expansion  (IDEAS §5)
# ---------------------------------------------------------------------------

SYNONYM_TABLE: dict[str, list[str]] = {
    "friction cone": [
        "static friction", "limiting friction", "angle of friction",
        "cone of static friction", "friction angle", "μN", "mu N",
    ],
    "static friction": [
        "friction cone", "limiting friction", "angle of friction",
        "limiting friction angle", "μN", "mu N",
    ],
    "limiting friction": ["static friction", "friction cone", "f_max", "fmax"],
    "aod": ["application of derivatives"],
    "application of derivatives": ["aod", "derivatives", "differentiation", "dy/dx"],
    "shm": ["simple harmonic motion"],
    "simple harmonic motion": ["shm", "harmonic motion", "oscillation", "simple harmonic"],
    "com": ["centre of mass", "center of mass", "centre-of-mass"],
    "centre of mass": ["com", "center of mass", "momentum", "centre"],
    "i": ["moment of inertia", "mo i"],
    "p&c": ["permutation and combination"],
    "lcd": ["least common divisor"],
    "auc": ["area under curve"],
    "de": ["differential equation"],
    "pyq": ["previous year question", "jee advanced", "jee main", "past paper"],
    "jee main": ["jee-main", "nta jee"],
    "jee advanced": ["jee-advanced", "iit jee"],
    "work energy": ["work power energy", "work-energy theorem"],
    "work power and energy": ["work energy", "power", "energy"],
    "rotational motion": ["rotation", "angular momentum", "moment of inertia"],
    "simple harmonic": ["simple harmonic motion", "shm", "oscillation"],
    "gravitation": ["gravity", "gravitational", "g"],
    "current electricity": ["current electric", "circuit", "ohm", "resistance"],
    "electromagnetic induction": ["emi", "induction", "faraday", "lenz"],
    "ray optics": ["optics", "refraction", "reflection", "lens", "mirror"],
    "wave optics": ["diffraction", "interference", "polarisation", "polarization"],
    "modern physics": ["photoelectric", "bohr", "radioactivity", "nuclear"],
    "semiconductor": ["diode", "transistor", "junction", "p-n"],
    "communication": ["modulation", "signal", "bandwidth"],
    "error analysis": ["least count", "zero error", "significant figures"],
    "vernier": ["vernier calipers", "vernier scale"],
    "screw gauge": ["micrometer", "screw gauge", "least count"],
    "elasticity": ["youngs modulus", "young modulus", "stress", "strain"],
    "surface tension": ["capillarity", "angle of contact", "excess pressure"],
    "viscosity": ["poiseuille", "stokes", "terminal velocity"],
    "fluid": ["fluid mechanics", "bernoulli", "continuity", "pressure"],
    "thermodynamics": ["heat", "entropy", "carnot", "first law", "second law"],
    "kinetic theory": ["gas laws", "rms speed", "maxwell"],
    "oscillation": ["shm", "damped", "forced", "resonance"],
    "waves": ["wave motion", "transverse", "longitudinal", "superposition"],
    "electrostatics": ["coulomb", "electric field", "gauss", "potential"],
    "capacitance": ["capacitor", "dielectric", "energy stored"],
    "current": ["current electricity", "drift", "mobility", "resistivity"],
    "magnetic field": ["biot savart", "ampere", "lorentz", "solenoid"],
    "magnetism": ["magnetic moment", "torque", "hysteresis"],
    "alternating current": ["ac", "rms", "impedance", "resonance"],
    "electromagnetic waves": ["em waves", "spectrum", "maxwell"],
    "dual nature": ["photoelectric", "de broglie", "compton", "matter wave"],
    "atomic physics": ["bohr", "spectrum", "hydrogen", "spectral"],
    "nuclear": ["radioactivity", "fission", "fusion", "half-life"],
    "semiconductors": ["diode", "transistor", "logic gate", "junction"],
    "communication systems": ["modulation", "bandwidth", "signal"],
}

NOISE_TERMS = {"jee", "notes", "physics", "maths", "question", "answer", "formula"}


def _synonym_terms(query: str) -> list[str]:
    low = query.lower()
    extra: list[str] = []
    for key, syns in SYNONYM_TABLE.items():
        if key in low:
            for s in syns:
                if s not in NOISE_TERMS and s not in extra:
                    extra.append(s)
    return extra


def expand_query_domain(query: str) -> list[str]:
    """Return [original, +domain synonyms] as separate query strings."""
    variants = [query]
    extra = _synonym_terms(query)
    if extra:
        variants.append(query + " " + " ".join(extra))
    return variants


# ---------------------------------------------------------------------------
# 2. LLM-based query rewriting
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = """You are a query-rewriting assistant for a JEE (IIT-JEE) notes search index.
The index contains scanned IIT-JEE physics/maths notes (machine-OCR'd, formulas garbled)
and JEE Main/Advanced previous-year question papers.

Given the user's question, write {n} DIFFERENT search queries that would retrieve
relevant passages. Use different terminology, abbreviations, and phrasings that a
student or the notes might use. Include both broad and specific variants.

Rules:
- Output exactly {n} lines, one query per line.
- No numbering, no bullets, no preamble, no explanations.
- Each line must be a standalone search query (3–12 words).
- Prefer JEE terminology: "friction cone", "limiting friction", "angle of friction",
  "application of derivatives", "simple harmonic motion", "centre of mass",
  "work energy theorem", "conservation of momentum".

User question: {question}

Queries:"""


def _llm_chat(messages: list[dict], cfg: dict, temperature: float = 0.3) -> str:
    """Call the LLM and return concatenated text."""
    try:
        return "".join(llm.stream_chat(messages, cfg, temperature)).strip()
    except Exception:
        return ""


def generate_query_variants(query: str, cfg: dict, n: int = 3) -> list[str]:
    """Ask the LLM to produce {n} alternative search queries."""
    if not llm.configured(cfg):
        return []
    prompt = MULTI_QUERY_PROMPT.format(n=n, question=query)
    text = _llm_chat(
        [{"role": "user", "content": prompt}], cfg, temperature=0.3
    )
    variants = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip("-• ").strip()
        if 3 <= len(line) <= 80 and line.lower() != query.lower():
            variants.append(line)
    seen = {query.lower()}
    uniq = []
    for v in variants:
        if v.lower() not in seen:
            seen.add(v.lower())
            uniq.append(v)
    return uniq[:n]


# ---------------------------------------------------------------------------
# 2b. Step-Back Prompting (research: RAG_Techniques §6, StackOps AI)
# ---------------------------------------------------------------------------

STEP_BACK_PROMPT = """You are a query-rewriting assistant for a JEE (IIT-JEE) notes search index.

Given the user's question, write a MORE ABSTRACT, GENERAL version of it that
captures the underlying concept. This "step-back" query helps retrieve
foundational passages when the original question is too specific for any
single chunk to contain the answer.

Rules:
- Output ONLY the step-back query, no preamble, no explanation.
- Keep it 4–12 words.
- Generalize: "What is the minimum force to push a block up a plane?" →
  "forces on an inclined plane"
- Preserve the core subject; drop specific numbers / scenario details.

User question: {question}

Step-back query:"""


def step_back_query(query: str, cfg: dict) -> str:
    """Generate a more abstract, general version of the query."""
    if not llm.configured(cfg):
        return query
    prompt = STEP_BACK_PROMPT.format(question=query)
    text = _llm_chat(
        [{"role": "user", "content": prompt}], cfg, temperature=0.2
    )
    text = text.strip()
    text = re.sub(r"^(step-back query|query):\s*", "", text, flags=re.I)
    return text or query


# ---------------------------------------------------------------------------
# 2c. Query Decomposition (research: RAG_Techniques §6)
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """You are a query decomposition assistant for a JEE (IIT-JEE)
notes search index.

Given a complex question, break it into {n} simpler SUB-QUESTIONS that each
target one specific aspect. Each sub-question should be independently
answerable from a single passage.

Rules:
- Output exactly {n} lines, one sub-question per line.
- No numbering, no bullets, no preamble.
- Each sub-question must be self-contained (3–15 words).
- If the question is already simple, output it unchanged on one line.

User question: {question}

Sub-questions:"""


def decompose_query(query: str, cfg: dict, n: int = 3) -> list[str]:
    """Break a complex question into simpler sub-questions."""
    if not llm.configured(cfg):
        return []
    prompt = DECOMPOSE_PROMPT.format(n=n, question=query)
    text = _llm_chat(
        [{"role": "user", "content": prompt}], cfg, temperature=0.2
    )
    subs = []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = line.strip("-• ").strip()
        if 3 <= len(line) <= 80:
            subs.append(line)
    return subs[:n]


# ---------------------------------------------------------------------------
# 3. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    lists: list[list[dict]], k: int = RRF_K, top_n: int = 8
) -> list[dict]:
    """Fuse multiple ranked result lists using RRF."""
    scores: dict[tuple[str, int], float] = {}
    best: dict[tuple[str, int], dict] = {}
    for lst in lists:
        for rank, hit in enumerate(lst):
            key = (hit["file"], hit["page"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in best:
                best[key] = dict(hit)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for key, score in fused[:top_n]:
        hit = best[key]
        hit = dict(hit)
        hit["rrf_score"] = round(score, 4)
        hit.setdefault("bm25_score", hit.pop("score", None))
        hit["score"] = score
        out.append(hit)
    return out


# ---------------------------------------------------------------------------
# 4. BM25-HyDE
# ---------------------------------------------------------------------------

HYDE_PROMPT = """Write a concise hypothetical answer (2–4 sentences) to the following
JEE physics/maths question, as if it appeared in Ambarish Sir's notes. Use the
terminology and notation the notes would use (e.g. "limiting friction", "μN",
"angle of friction", "application of derivatives", "dy/dx"). Do NOT say you are
an AI; just write the answer content.

Question: {question}

Hypothetical answer:"""


def hyde_query(query: str, cfg: dict) -> str:
    """Generate a hypothetical answer to use as a BM25 query."""
    if not llm.configured(cfg):
        return query
    text = _llm_chat(
        [{"role": "user", "content": HYDE_PROMPT.format(question=query)}],
        cfg,
        temperature=0.2,
    )
    text = text.strip()
    text = re.sub(r"^(hypothetical answer|answer):\s*", "", text, flags=re.I)
    return text or query


# ---------------------------------------------------------------------------
# 5. Self-Corrective RAG (CRAG-lite) relevance grader
# ---------------------------------------------------------------------------

GRADER_PROMPT = """You are a relevance grader for a JEE notes search system.

Question: {question}

Retrieved passage (from the student's scanned notes):
{passage}

Rate relevance on a scale of 0–5:
- 5: The passage directly answers or strongly supports the question.
- 3: The passage is on the same topic and partially useful.
- 0: The passage is irrelevant, garbled beyond use, or off-topic.

Output ONLY a single integer 0–5, nothing else."""


def grade_relevance(question: str, passage: str, cfg: dict) -> int:
    """Return 0-5 relevance grade for one passage."""
    if not llm.configured(cfg):
        return 3
    text = _llm_chat(
        [{"role": "user", "content": GRADER_PROMPT.format(
            question=question, passage=passage[:1200]
        )}],
        cfg,
        temperature=0,
    )
    m = re.search(r"([0-5])", text)
    return int(m.group(1)) if m else 3


# ---------------------------------------------------------------------------
# 5b. Enhanced CRAG: 3-class relevance grading (RELEVANT / AMBIGUOUS / IRRELEVANT)
# ---------------------------------------------------------------------------

CRAG_EVAL_PROMPT = """You are a relevance evaluator for a JEE notes search system.

Question: {question}

Retrieved document:
{document}

Classify the document's relevance to answering the question.

Respond with exactly one of:
- RELEVANT: The document contains information that directly helps answer the question.
- AMBIGUOUS: The document is partially relevant or tangentially related but may not fully answer.
- IRRELEVANT: The document does not contain useful information for answering the question.

Classification:"""


CRAG_REWRITE_PROMPT = """The following question was used to search a notes corpus, but the retrieved results were not sufficiently relevant.

Original question: {question}

Rewrite this question to be more specific and likely to retrieve the correct notes. Focus on including specific terms, chapter names, or concepts that would appear in the target material.

Rewritten question:"""


def crag_grade_passages(question: str, passages: list[dict], cfg: dict) -> dict:
    """CRAG-style 3-class grading of retrieved passages.

    Returns {"label": "RELEVANT"/"AMBIGUOUS"/"IRRELEVANT",
             "grades": [...], "should_rewrite": bool}
    """
    if not llm.configured(cfg) or not passages:
        return {"label": "RELEVANT", "grades": [], "should_rewrite": False}

    grades = []
    for h in passages[:5]:
        text = _llm_chat(
            [{"role": "user", "content": CRAG_EVAL_PROMPT.format(
                question=question, document=h.get("text", "")[:1200])}],
            cfg, temperature=0)
        label = "IRRELEVANT"
        for cand in ("RELEVANT", "AMBIGUOUS", "IRRELEVANT"):
            if cand in text.upper():
                label = cand
                break
        grades.append({"file": h.get("file"), "page": h.get("page"), "label": label})

    relevant = sum(1 for g in grades if g["label"] == "RELEVANT")
    ambiguous = sum(1 for g in grades if g["label"] == "AMBIGUOUS")
    irrelevant = sum(1 for g in grades if g["label"] == "IRRELEVANT")

    if relevant > 0:
        label = "RELEVANT"
        should_rewrite = False
    elif ambiguous > 0 and irrelevant == len(grades):
        label = "AMBIGUOUS"
        should_rewrite = True
    else:
        label = "IRRELEVANT"
        should_rewrite = True

    return {"label": label, "grades": grades, "should_rewrite": should_rewrite,
            "counts": {"relevant": relevant, "ambiguous": ambiguous, "irrelevant": irrelevant}}


def crag_rewrite_query(question: str, cfg: dict) -> str:
    """Rewrite a question when CRAG marks retrieval as AMBIGUOUS/IRRELEVANT."""
    if not llm.configured(cfg):
        return question
    text = _llm_chat(
        [{"role": "user", "content": CRAG_REWRITE_PROMPT.format(question=question)}],
        cfg, temperature=0.5)
    text = text.strip()
    text = re.sub(r"^(rewritten question|question):\s*", "", text, flags=re.I).strip()
    return text or question


# ---------------------------------------------------------------------------
# 6. Query routing (notes vs pyq)
# ---------------------------------------------------------------------------

PYQ_HINTS = [
    "202", "201", "200", "shift", "paper", "previous year", "pyq",
    "jee main", "jee advanced", "january", "april", "session",
]


def _looks_like_pyq(query: str) -> bool:
    low = query.lower()
    return any(h in low for h in PYQ_HINTS)


# ---------------------------------------------------------------------------
# 7. Context distillation
# ---------------------------------------------------------------------------

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def distill_passage(text: str, query: str, max_chars: int = 700) -> str:
    """Keep the most query-relevant sentences of a passage."""
    if len(text) <= max_chars:
        return text
    query_terms = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", query)
                   if t.lower() not in {"what", "how", "why", "when", "where",
                                        "does", "do", "the", "and", "for",
                                        "with", "from", "that", "this", "have"}}
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return text[:max_chars]
    scored = []
    for s in sentences:
        st = {t.lower() for t in re.findall(r"[A-Za-z]{4,}", s)}
        overlap = len(query_terms & st)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    kept = []
    used = 0
    for overlap, s in scored:
        if used + len(s) + 1 > max_chars:
            break
        kept.append(s)
        used += len(s) + 1
    if not kept:
        kept = [sentences[0][:max_chars]]
    return " ".join(kept)


# ---------------------------------------------------------------------------
# 8. Agentic follow-loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Orchestrates the follow-loop: finish → rethink → creativity → patch
    → research → apply → test e2e → finish, with a bounded retry budget.

    Each loop iteration:
      1. finish   — produce an answer from current context
      2. rethink  — grade retrieved passages; if weak, mark for retry
      3. creativity — generate a new query variant (HyDE / paraphrase / route switch)
      4. patch    — re-run retrieval with the new variant
      5. research — (no-op locally; the LLM has internal knowledge)
      6. apply    — merge new results into context
      7. test e2e — re-grade; if still weak and budget remains, loop again
      8. finish   — return best answer found
    """

    def __init__(self, cfg: dict, max_loops: int = MAX_LOOPS,
                 crag_enabled: bool = True,
                 rerank: bool = False,
                 fusion: str = "rrf",
                 contextual: bool = False):
        self.cfg = cfg
        self.max_loops = max_loops
        self.loop_count = 0
        self.trace: list[dict] = []
        self.crag_enabled = crag_enabled
        self.rerank = rerank
        self.fusion = fusion
        self.contextual = contextual

    def run(self, query: str, k: int = 8, kind: str | None = None,
            mode: str = "auto") -> dict:
        """Run the full agentic follow-loop.

        Returns the best result dict found across all loop iterations.
        """
        best: dict | None = None
        best_grade = -1
        self.loop_count = 0
        self.trace = []

        current_kind = kind
        current_mode = mode

        while self.loop_count < self.max_loops:
            self.loop_count += 1
            loop_start = time.time()

            # --- finish: retrieve with current strategy (no nested loop) ---
            result = advanced_search(
                query, k=k, kind=current_kind, mode=current_mode,
                cfg=self.cfg, use_loop=False,
                rerank=self.rerank, fusion=self.fusion,
                contextual=self.contextual)
            elapsed = time.time() - loop_start

            # --- rethink: grade top passages (enhanced CRAG 3-class) ---
            if self.crag_enabled:
                crag = crag_grade_passages(query, result["passages"], self.cfg)
                grade_map = {"RELEVANT": 5, "AMBIGUOUS": 3, "IRRELEVANT": 0}
                grade = grade_map.get(crag["label"], 0)
                crag_info = crag
            else:
                grade = self._grade_result(query, result["passages"])
                crag_info = {"label": "SKIPPED", "grades": []}

            self.trace.append({
                "loop": self.loop_count,
                "method": result["method"],
                "kind": current_kind,
                "mode": current_mode,
                "grade": grade,
                "crag": crag_info,
                "elapsed": round(elapsed, 2),
                "n_passages": len(result["passages"]),
            })

            # Keep the best result so far.
            if grade > best_grade:
                best_grade = grade
                best = result

            # If good enough, finish early.
            if grade >= 4 or len(result["passages"]) == 0:
                best = result
                break

            # --- creativity: decide next strategy ---
            if self.crag_enabled and crag_info.get("should_rewrite"):
                # CRAG says results are weak → rewrite the query
                query = crag_rewrite_query(query, self.cfg)
                current_kind, current_mode = kind, "direct"
            else:
                current_kind, current_mode = self._choose_next_strategy(
                    query, result, current_kind, current_mode
                )

        # If loop exhausted without a good grade, return best found.
        if best is None:
            best = result
        best["loops"] = self.loop_count
        best["trace"] = self.trace
        best["crag_enabled"] = self.crag_enabled
        return best

    def _grade_result(self, query: str, passages: list[dict]) -> int:
        """Grade the overall result quality (0–5)."""
        if not passages:
            return 0
        if not llm.configured(self.cfg):
            return 3  # neutral when no LLM
        # Grade the top passage; use average if multiple.
        grades = []
        for h in passages[:3]:
            grades.append(grade_relevance(query, h.get("text", ""), self.cfg))
        return sum(grades) // len(grades) if grades else 0

    def _choose_next_strategy(
        self, query: str, result: dict,
        current_kind: str | None, current_mode: str
    ) -> tuple[str | None, str]:
        """Creativity step: pick a different strategy for the next loop."""
        method = result["method"]
        variants = result.get("query_variants", [])

        # Creativity: switch strategy based on what failed.
        if current_mode == "direct":
            # Try HyDE next (answer-shaped matching).
            return current_kind, "hyde"
        if current_mode == "hyde":
            # Try RAG-Fusion next (multi-query + RRF).
            return current_kind, "rag_fusion"
        if current_mode == "rag_fusion":
            # Try switching kind (notes <-> pyq).
            new_kind = "pyq" if current_kind != "pyq" else None
            return new_kind, "direct"
        # Fallback: toggle kind.
        new_kind = "pyq" if current_kind != "pyq" else None
        return new_kind, "direct"


# ---------------------------------------------------------------------------
# 9. Main advanced search entry point (with optional follow-loop)
# ---------------------------------------------------------------------------

def advanced_search(
    query: str,
    k: int = 8,
    kind: str | None = None,
    mode: str = "auto",
    cfg: dict | None = None,
    use_loop: bool = True,
    rerank: bool | None = None,
    fusion: str | None = None,
    contextual: bool | None = None,
) -> dict:
    """Advanced retrieval: query expansion + multi-query RAG + RRF (+ HyDE + CRAG).

    Research-backed improvements (arxiv:2604.01733):
      - Cross-encoder reranking (two-stage: BM25 recall pool → LLM rerank)
        is the single most impactful component (+17pp MRR@3).
      - Convex Combination fusion as alternative to RRF.
      - Contextual retrieval: LLM context prepended to chunks at query time.
      - Enhanced CRAG with 3-class relevance grading (RELEVANT/AMBIGUOUS/IRRELEVANT).

    Parameters
    ----------
    query : str
        User's question.
    k : int
        Final number of passages to return.
    kind : str | None
        Force ``"notes"`` or ``"pyq"``; None = auto-route.
    mode : str
        - ``"auto"``: choose technique by query characteristics.
        - ``"rag_fusion"``: always multi-query + RRF.
        - ``"hyde"``: always BM25-HyDE.
        - ``"direct"``: plain BM25 with domain expansion only.
        - ``"rerank"``: two-stage BM25 recall → LLM cross-encoder rerank.
    cfg : dict | None
        Runtime config.
    use_loop : bool
        If True, wrap in the agentic follow-loop (rethink/creativity/patch).
    rerank : bool | None
        Override config ``rerank_enabled``. Force two-stage reranking on/off.
    fusion : str | None
        Override config ``fusion_method``: "rrf" or "convex_combination".
    contextual : bool | None
        Override config ``contextual_retrieval``. Use context-augmented search.

    Returns
    -------
    dict with keys: ``passages``, ``diagrams``, ``method``, ``query_variants``,
    ``reranked``, ``loops``, ``trace``.
    """
    cfg = cfg or config.load()
    cfg = dict(cfg)

    # Resolve overrides against config defaults
    do_rerank = cfg.get("rerank_enabled", False) if rerank is None else rerank
    do_fusion = cfg.get("fusion_method", "rrf") if fusion is None else fusion
    do_contextual = cfg.get("contextual_retrieval", False) if contextual is None else contextual
    do_crag = cfg.get("crag_enabled", True)

    # --- Step 1: domain expansion (always, cheap) ---
    domain_variants = expand_query_domain(query)

    # --- Step 2: route kind if not forced ---
    if kind is None:
        if _looks_like_pyq(query):
            kind = "pyq"

    # --- Step 3: choose technique ---
    if mode == "rerank" or do_rerank:
        # Two-stage: BM25 recall pool → LLM cross-encoder rerank
        result = rerank.two_stage_retrieve(
            query, k=k, kind=kind, cfg=cfg,
            recall_pool=cfg.get("rerank_recall_pool", 50))
        result["query_variants"] = domain_variants
    elif mode == "direct":
        result = _direct_search(query, k, kind, cfg)
    elif mode == "hyde":
        result = _hyde_search(query, k, kind, cfg)
    elif mode == "rag_fusion":
        result = _rag_fusion_search(query, k, kind, cfg, domain_variants, fusion=do_fusion)
    else:  # auto
        if len(query.split()) <= 4 or any(
            t in query.lower() for t in
            ["formula", "derive", "what is", "define"]
        ):
            result = _hyde_search(query, k, kind, cfg)
        else:
            result = _rag_fusion_search(query, k, kind, cfg, domain_variants, fusion=do_fusion)

    # --- Step 3b: contextual retrieval augmentation (prepend context to chunks) ---
    if do_contextual:
        ctx_hits = contextual_retrieval.search_with_context(
            question, k=k, kind=kind, cfg=cfg)
        if ctx_hits:
            # Merge: contextual hits take priority, fill remaining with BM25
            seen = {(h["file"], h["page"]) for h in ctx_hits}
            for h in result["passages"]:
                if (h["file"], h["page"]) not in seen:
                    ctx_hits.append(h)
                    seen.add((h["file"], h["page"]))
            result["passages"] = ctx_hits[:k]
            result["diagrams"] = retrieve.diagrams_for(result["passages"], limit=k)
            result["method"] += "_contextual"
            result["contextual"] = True

    # --- Step 4: wrap in agentic follow-loop if requested ---
    if use_loop and llm.configured(cfg):
        loop = AgentLoop(cfg, max_loops=MAX_LOOPS, crag_enabled=do_crag)
        return loop.run(query, k=k, kind=kind, mode=mode,
                        rerank=do_rerank, fusion=do_fusion,
                        contextual=do_contextual)

    result.setdefault("loops", 1)
    result.setdefault("trace", [])
    result["crag"] = do_crag
    return result


# ---------------------------------------------------------------------------
# Internal per-technique search implementations
# ---------------------------------------------------------------------------

def _search_variant(q: str, k: int, kind: str | None, cfg: dict) -> list[dict]:
    return retrieve.search(q, k=k, kind=kind)


def _direct_search(query: str, k: int, kind: str | None, cfg: dict) -> dict:
    variants = expand_query_domain(query)
    all_hits: list[dict] = []
    seen = set()
    for v in variants:
        for h in _search_variant(v, k, kind, cfg):
            key = (h["file"], h["page"])
            if key not in seen:
                seen.add(key)
                all_hits.append(h)
    all_hits.sort(key=lambda h: h.get("score", 0))
    hits = all_hits[:k]
    diags = retrieve.diagrams_for(hits, limit=k)
    return {
        "passages": hits,
        "diagrams": diags,
        "method": "direct_bm25_domain",
        "query_variants": variants,
        "reranked": False,
    }


def _hyde_search(query: str, k: int, kind: str | None, cfg: dict) -> dict:
    hypothetical = hyde_query(query, cfg)
    hits = _search_variant(query, k, kind, cfg)
    if hypothetical != query:
        h2 = _search_variant(hypothetical, k, kind, cfg)
        seen = {(h["file"], h["page"]) for h in hits}
        for h in h2:
            key = (h["file"], h["page"])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    hits = hits[:k]
    if llm.configured(cfg) and len(hits) > 2:
        graded = []
        for h in hits:
            grade = grade_relevance(query, h.get("text", ""), cfg)
            if grade >= 2 or len(graded) < 2:
                graded.append(h)
        hits = graded[:k]
    diags = retrieve.diagrams_for(hits, limit=k)
    return {
        "passages": hits,
        "diagrams": diags,
        "method": "hyde_bm25",
        "query_variants": [query, hypothetical],
        "reranked": llm.configured(cfg),
    }


def _rag_fusion_search(query: str, k: int, kind: str | None, cfg: dict,
                       domain_variants: list[str], fusion: str = "rrf") -> dict:
    """Multi-query RAG + RRF fusion.

    Query variants come from:
      1. domain synonym expansion (cheap, always)
      2. LLM paraphrases (multi-query)
      3. step-back abstraction (for multi-hop / too-specific questions)
      4. query decomposition (for complex multi-part questions)

    Fusion method:
      - ``rrf``: Reciprocal Rank Fusion (default, research-proven)
      - ``convex_combination``: weighted score combination (α=0.5, can outperform RRF)
    """
    variants = list(domain_variants)
    llm_variants = generate_query_variants(query, cfg, n=3)
    variants.extend(v for v in llm_variants if v not in variants)

    # Step-back: only for longer / multi-hop questions
    if len(query.split()) >= 6:
        sb = step_back_query(query, cfg)
        if sb and sb.lower() != query.lower() and sb not in variants:
            variants.append(sb)

    # Decomposition: only for complex questions with multiple parts
    if len(query.split()) >= 8 or " and " in query.lower() or " or " in query.lower():
        subs = decompose_query(query, cfg, n=3)
        for s in subs:
            if s.lower() != query.lower() and s not in variants:
                variants.append(s)

    per_query: list[list[dict]] = []
    for v in variants:
        per_query.append(_search_variant(v, max(k * 2, 12), kind, cfg))

    if fusion == "convex_combination":
        # Convex Combination: need BM25 + dense, but we only have BM25.
        # Use domain-expanded variants as the "dense" proxy (different phrasings).
        bm25_hits = per_query[0] if per_query else []
        proxy_hits = per_query[1] if len(per_query) > 1 else []
        fused = convex_combination_fusion(bm25_hits, proxy_hits,
                                          alpha=CONVEX_ALPHA, top_n=k * 2)
    else:
        fused = reciprocal_rank_fusion(per_query, k=RRF_K, top_n=k * 2)

    if llm.configured(cfg):
        kept = []
        for h in fused:
            grade = grade_relevance(query, h.get("text", ""), cfg)
            if grade >= 3 or len(kept) < 3:
                kept.append(h)
            if len(kept) >= k:
                break
        if kept:
            fused = kept
        else:
            fused = fused[:k]
    else:
        fused = fused[:k]

    for h in fused:
        h["text"] = distill_passage(h.get("text", ""), query)

    diags = retrieve.diagrams_for(fused, limit=k)
    return {
        "passages": fused,
        "diagrams": diags,
        "method": f"rag_fusion_{fusion}",
        "query_variants": variants,
        "reranked": llm.configured(cfg),
        "fusion": fusion,
    }


# ---------------------------------------------------------------------------
# 9b. Convenience context builder for webapp / MCP
# ---------------------------------------------------------------------------

def build_advanced_context(query: str, k: int = 8, kind: str | None = None,
                           cfg: dict | None = None) -> tuple[str, list[dict]]:
    """Build ``(context_string, diagrams)`` for the LLM chat prompt."""
    cfg = cfg or config.load()
    result = advanced_search(query, k=k, kind=kind, cfg=cfg)
    passages = result["passages"]
    parts = []
    for h in passages:
        cite = f"[{h['file']} | {h.get('topic', '')} | p.{h['page']}]"
        parts.append(f"{cite}\n{h.get('text', '')[:1200]}")
    context = "\n\n---\n\n".join(parts)
    return context, result["diagrams"]
