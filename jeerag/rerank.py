"""Cross-encoder reranking for retrieval results.

Research (arxiv:2604.01733 "From BM25 to Corrective RAG"):
- Reranking is the single most impactful component: +17.2pp MRR@3, +12.1pp Recall@5.
- Two-stage pipeline: broad recall (BM25, 50 candidates) → neural rerank (top 10).
- Reranker depth matters: need 50+ candidates for effectiveness.
- Cross-encoders process query+document jointly, capturing relationships bi-encoders miss.

This module is dependency-free: it reuses the existing LLM endpoint as a
cross-encoder reranker by scoring each (query, passage) pair and returning
relevance scores. No torch, no vector DB, no embedding model required.

Usage::

    from jeerag.rerank import rerank_passages
    reranked = rerank_passages("cone of static friction", hits, cfg, top_n=8)
"""
from __future__ import annotations

import re
from typing import Optional

from . import config as _config, llm
from . import retrieve


# Prompt asks the LLM to act as a cross-encoder: score relevance 0-5 for a
# query-passage pair, output ONLY the integer.
RERANK_PROMPT = """You are a relevance reranker for a JEE (IIT-JEE) notes search system.

Question: {question}

Retrieved passage (from scanned notes, may be OCR-garbled):
{passage}

Rate relevance on a scale of 0-5:
- 5: The passage directly answers or strongly supports the question.
- 3: The passage is on the same topic and partially useful.
- 0: The passage is irrelevant, garbled beyond use, or off-topic.

Output ONLY a single integer 0-5, nothing else."""


def _llm_rerank_score(question: str, passage: str, cfg: dict,
                      temperature: float = 0.0) -> Optional[int]:
    """Ask the LLM to score one (query, passage) pair 0-5. Returns None on failure."""
    try:
        from .llm.manager import llm_manager
        text = "".join(llm_manager(cfg).chat(
            [{"role": "user", "content":
              RERANK_PROMPT.format(question=question,
                                   passage=passage[:1500])}],
            temperature, None)).strip()
        m = re.search(r"([0-5])", text)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _fallback_rerank_score(question: str, passage: str) -> float:
    """Cheap keyword-overlap score when no LLM is available (0.0 to 1.0)."""
    q_terms = {t.lower() for t in __import__("re").findall(r"[A-Za-z]{3,}", question)
               if t.lower() not in {"the", "and", "for", "with", "from", "that",
                                    "this", "have", "what", "how", "why", "when",
                                    "where", "does", "do", "are", "was", "is"}}
    p_terms = {t.lower() for t in __import__("re").findall(r"[A-Za-z]{3,}", passage)}
    if not q_terms:
        return 0.0
    overlap = len(q_terms & p_terms)
    return min(overlap / len(q_terms), 1.0)


def rerank_passages(question: str, hits: list[dict], cfg: dict,
                    top_n: int = 8) -> list[dict]:
    """Rerank ``hits`` by relevance to ``question``.

    Uses the LLM as a cross-encoder when configured; falls back to keyword
    overlap otherwise. Returns hits sorted by descending relevance with a
    ``rerank_score`` field added (0-5 via LLM, 0.0-1.0 via fallback).
    """
    if not hits:
        return hits

    scored = []
    for h in hits:
        text = h.get("text", "")
        score = _llm_rerank_score(question, text, cfg)
        source = "llm"
        if score is None:
            score = _fallback_rerank_score(question, text)
            source = "fallback"
        scored.append((score, source, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, source, h in scored[:top_n]:
        h = dict(h)
        h["rerank_score"] = score
        h["rerank_source"] = source
        out.append(h)
    return out


def two_stage_retrieve(question: str, k: int = 8, kind: str | None = None,
                       cfg: dict | None = None,
                       recall_pool: int = 50) -> dict:
    """Two-stage retrieval: BM25 recall pool → LLM cross-encoder rerank.

    Research-backed configuration:
      - First stage: BM25 over a large candidate pool (default 50)
      - Second stage: LLM reranks to top-k (default 8)

    Returns a dict shaped like advanced_search results plus ``reranked=True``.
    """
    cfg = cfg or _config.load()
    cfg = dict(cfg)

    # Stage 1: broad BM25 recall over a large pool
    pool = retrieve.search(question, k=recall_pool, kind=kind)
    if not pool:
        return {
            "passages": [], "diagrams": [],
            "method": "two_stage_bm25_rerank",
            "reranked": False,
            "query_variants": [question],
            "recall_pool": 0,
        }

    # Stage 2: LLM rerank to top-k
    reranked = rerank_passages(question, pool, cfg, top_n=k)
    diags = retrieve.diagrams_for(reranked, limit=k)

    return {
        "passages": reranked,
        "diagrams": diags,
        "method": "two_stage_bm25_rerank",
        "reranked": True,
        "query_variants": [question],
        "recall_pool": len(pool),
    }


def convex_combination_fusion(bm25_hits: list[dict], dense_hits: list[dict],
                              alpha: float = 0.5, top_n: int = 8) -> list[dict]:
    """Fuse BM25 and dense hit lists via convex combination (weighted score).

    Research finding: CC with α=0.5 (equal weighting) can outperform RRF.
    Each hit carries a ``score`` field; BM25 scores are negated (lower=better
    in FTS5 bm25) so we normalize both to [0,1] before combining.

    ``dense_hits`` may be empty when no dense retriever is configured — then
    this reduces to BM25-only ranking.
    """
    def norm(scores):
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        span = hi - lo if hi != lo else 1.0
        return {k: (v - lo) / span for k, v in scores.items()}

    bm25_scores = { (h["file"], h["page"]): -h.get("score", 0.0) for h in bm25_hits }
    dense_scores = { (h["file"], h["page"]): h.get("score", 0.0) for h in dense_hits }

    bm25_norm = norm(bm25_scores)
    dense_norm = norm(dense_scores)

    all_keys = set(bm25_norm) | set(dense_norm)
    fused = []
    for key in all_keys:
        b = bm25_norm.get(key, 0.0)
        d = dense_norm.get(key, 0.0)
        combined = alpha * d + (1.0 - alpha) * b
        fused.append((key, combined))

    fused.sort(key=lambda x: x[1], reverse=True)

    best = {}
    for h in bm25_hits + dense_hits:
        key = (h["file"], h["page"])
        if key not in best:
            best[key] = h

    out = []
    for key, score in fused[:top_n]:
        h = dict(best[key])
        h["score"] = score
        h["fusion"] = "convex_combination"
        out.append(h)
    return out
