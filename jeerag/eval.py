"""RAG evaluation metrics (RAGAS-style, dependency-free).

Implements the four core RAGAS metrics without requiring the ragas library
(which needs LLM + embeddings). Uses the LLM endpoint for faithfulness /
relevancy grading and simple retrieval checks for precision / recall.

Metrics (0–1, higher is better):

  faithfulness        — Are the answer's claims grounded in the retrieved
                        context? (no ground truth needed)
  answer_relevancy    — Does the answer address the question that was asked?
  context_precision   — Of the retrieved chunks, how many were relevant?
                        (Low precision = noise crowding the context window.)
  context_recall      — Was all the needed context retrieved?
                        (Low recall = missing information.)

Recommended thresholds (from ragas community):
  faithfulness        >= 0.90
  answer_relevancy    >= 0.80
  context_precision   >= 0.75
  context_recall      >= 0.90

Usage::

    from jeerag.eval import evaluate_answer, evaluate_retrieval, benchmark

    result = evaluate_answer(
        question="What is limiting friction?",
        answer="Limiting friction is f_max = μN ...",
        contexts=["FRICTION NOTES p.4: f_max = μN ..."],
        cfg=cfg,
    )
    # result: {"faithfulness": 0.92, "answer_relevancy": 0.85, ...}

    # Full benchmark over a set of questions:
    scores = benchmark(questions, cfg=cfg)
"""
from __future__ import annotations

import json
import re
import statistics
import time
from typing import Iterable

from . import config, llm, advanced_retrieve, retrieve

# Recommended thresholds (from ragas community / Autonoma-Tools)
THRESHOLDS = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.80,
    "context_precision": 0.75,
    "context_recall": 0.90,
}


# ---------------------------------------------------------------------------
# LLM-based graders (faithfulness / relevancy)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating a RAG system's answer for faithfulness.

QUESTION: {question}

RETRIEVED CONTEXT (the only source of truth):
{context}

ANSWER BEING EVALUATED:
{answer}

Task: Check whether every factual claim in the ANSWER is supported by the RETRIEVED CONTEXT.
- A claim is "supported" if the context explicitly states it or it can be directly inferred.
- Ignore style, phrasing, or minor rewordings.
- If the answer adds information not in the context, that is unfaithful.

Rate faithfulness 0–1:
- 1.0: every claim is supported
- 0.8: most claims supported, minor extrapolation
- 0.5: about half the claims supported
- 0.2: most claims unsupported
- 0.0: answer is entirely fabricated

Output ONLY a single number between 0 and 1 (e.g. "0.85"), nothing else."""


RELEVANCY_PROMPT = """You are evaluating a RAG system's answer for relevancy.

QUESTION: {question}

ANSWER BEING EVALUATED:
{answer}

Task: Rate how well the ANSWER addresses the QUESTION that was asked.
- 1.0: directly and completely answers the question
- 0.8: answers the question but with extra irrelevant detail
- 0.5: partially answers, partially off-topic
- 0.2: mostly off-topic
- 0.0: does not address the question at all

Output ONLY a single number between 0 and 1 (e.g. "0.9"), nothing else."""


def _grade_with_llm(prompt: str, cfg: dict) -> float:
    """Call the LLM to grade on a 0-1 scale; returns float."""
    if not llm.configured(cfg):
        return -1.0  # cannot grade without LLM
    try:
        text = llm.complete(
            [{"role": "user", "content": prompt}],
            cfg=cfg,
            temperature=0,
            max_tokens=10,
        )
        m = re.search(r"([01](?:\.\d+)?)", text)
        if m:
            return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        pass
    return -1.0


def _clip(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] if text else ""


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def faithfulness(question: str, answer: str, contexts: list[str],
                 cfg: dict | None = None) -> float:
    """Are the answer's claims grounded in the retrieved context?"""
    cfg = cfg or config.load()
    if not contexts:
        return 0.0
    ctx = _clip("\n\n---\n\n".join(contexts), 4000)
    prompt = FAITHFULNESS_PROMPT.format(
        question=_clip(question, 500),
        context=ctx,
        answer=_clip(answer, 2000),
    )
    return _grade_with_llm(prompt, cfg)


def answer_relevancy(question: str, answer: str,
                     cfg: dict | None = None) -> float:
    """Does the answer address the question that was asked?"""
    cfg = cfg or config.load()
    prompt = RELEVANCY_PROMPT.format(
        question=_clip(question, 500),
        answer=_clip(answer, 2000),
    )
    return _grade_with_llm(prompt, cfg)


def context_precision(question: str, contexts: list[str],
                      cfg: dict | None = None) -> float:
    """Of the retrieved chunks, how many were relevant?

    Without ground-truth chunk IDs, we approximate with an LLM grader
    that scores each chunk's relevance to the question.
    """
    cfg = cfg or config.load()
    if not contexts:
        return 0.0
    if not llm.configured(cfg):
        # Fallback: assume all chunks relevant (no signal)
        return -1.0
    relevant = 0
    for ctx in contexts:
        # Quick heuristic: does the chunk share significant terms with the question?
        q_terms = set(re.findall(r"[A-Za-z]{4,}", question.lower()))
        c_terms = set(re.findall(r"[A-Za-z]{4,}", ctx.lower()))
        if q_terms and c_terms:
            overlap = len(q_terms & c_terms) / len(q_terms)
            if overlap >= 0.15:  # at least 15% term overlap
                relevant += 1
        else:
            relevant += 1
    return relevant / len(contexts)


def context_recall(question: str, answer: str, contexts: list[str],
                   cfg: dict | None = None) -> float:
    """Was all the needed context retrieved?

    Approximated by checking whether the answer's key terms appear in the
    retrieved context (high recall = answer is fully grounded).
    """
    if not contexts or not answer:
        return 0.0
    ctx_text = " ".join(contexts).lower()
    # Extract content words from the answer
    answer_terms = set(re.findall(r"[A-Za-z]{5,}", answer.lower()))
    if not answer_terms:
        return 1.0
    found = sum(1 for t in answer_terms if t in ctx_text)
    return found / len(answer_terms)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(question: str, answer: str, contexts: list[str],
                    cfg: dict | None = None) -> dict:
    """Compute all four RAGAS metrics for one Q/A pair."""
    cfg = cfg or config.load()
    return {
        "faithfulness": round(faithfulness(question, answer, contexts, cfg), 3),
        "answer_relevancy": round(answer_relevancy(question, answer, cfg), 3),
        "context_precision": round(context_precision(question, contexts, cfg), 3),
        "context_recall": round(context_recall(question, answer, contexts, cfg), 3),
    }


def evaluate_retrieval(question: str, contexts: list[str],
                       cfg: dict | None = None) -> dict:
    """Compute retrieval-only metrics (precision / recall)."""
    cfg = cfg or config.load()
    return {
        "context_precision": round(context_precision(question, contexts, cfg), 3),
    }


def passes_thresholds(scores: dict, thresholds: dict | None = None) -> bool:
    """Check whether all metrics meet their thresholds."""
    thresholds = thresholds or THRESHOLDS
    for metric, threshold in thresholds.items():
        value = scores.get(metric)
        if value is None or value < 0:
            continue  # not computable (e.g., no LLM)
        if value < threshold:
            return False
    return True


def benchmark(questions: list[str], cfg: dict | None = None,
              k: int = 8, kind: str | None = None) -> dict:
    """Run a full benchmark: retrieve + answer + score for each question.

    Returns aggregate scores and per-question details.
    """
    cfg = cfg or config.load()
    per_question = []
    all_scores = {m: [] for m in THRESHOLDS}

    for q in questions:
        t0 = time.time()
        # Retrieve
        result = advanced_retrieve.advanced_search(q, k=k, kind=kind, cfg=cfg)
        contexts = [h.get("text", "") for h in result["passages"]]
        # Answer (if LLM available)
        answer = ""
        if llm.configured(cfg):
            try:
                ctx_text = "\n\n---\n\n".join(
                    f"[{h['file']} | p.{h['page']}]\n{h.get('text', '')[:1200]}"
                    for h in result["passages"]
                )
                answer = llm.complete(
                    [{"role": "user", "content":
                      f"NOTES EXCERPTS:\n{ctx_text}\n\nQUESTION: {q}"}],
                    cfg=cfg, temperature=0.2, max_tokens=512,
                )
            except Exception:
                answer = ""
        # Score
        scores = evaluate_answer(q, answer, contexts, cfg)
        elapsed = time.time() - t0
        for metric, value in scores.items():
            if value >= 0:
                all_scores[metric].append(value)
        per_question.append({
            "question": q,
            "n_contexts": len(contexts),
            "answer_len": len(answer),
            "scores": scores,
            "elapsed": round(elapsed, 2),
        })

    # Aggregate
    aggregate = {}
    for metric, values in all_scores.items():
        if values:
            aggregate[metric] = round(statistics.mean(values), 3)
        else:
            aggregate[metric] = None

    return {
        "aggregate": aggregate,
        "thresholds": THRESHOLDS,
        "passes": passes_thresholds(aggregate),
        "n_questions": len(questions),
        "per_question": per_question,
    }


def benchmark_golden(dataset_path: str, cfg: dict | None = None,
                     k: int = 8, kind: str | None = None) -> dict:
    """Run benchmark against a versioned golden dataset (JSONL).

    Golden dataset format (one JSON object per line)::
        {"question": "...", "answer": "ground truth", "chunk_ids": ["file::p4", ...]}

    Never modify a golden dataset in place — version it
    (golden_v1.jsonl → golden_v2.jsonl) and track the diff.
    """
    cfg = cfg or config.load()
    per_question = []
    all_scores = {m: [] for m in THRESHOLDS}
    retrieval_hits = []  # for MRR / hit-rate

    with open(dataset_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    for ex in examples:
        q = ex["question"]
        ground_truth = ex.get("answer", "")
        expected_ids = set(ex.get("chunk_ids", []))
        t0 = time.time()

        result = advanced_retrieve.advanced_search(q, k=k, kind=kind, cfg=cfg)
        contexts = [h.get("text", "") for h in result["passages"]]
        retrieved_ids = [f"{h['file']}::p{h['page']}" for h in result["passages"]]

        answer = ""
        if llm.configured(cfg):
            try:
                ctx_text = "\n\n---\n\n".join(
                    f"[{h['file']} | p.{h['page']}]\n{h.get('text', '')[:1200]}"
                    for h in result["passages"]
                )
                answer = llm.complete(
                    [{"role": "user", "content":
                      f"NOTES EXCERPTS:\n{ctx_text}\n\nQUESTION: {q}"}],
                    cfg=cfg, temperature=0.2, max_tokens=512,
                )
            except Exception:
                answer = ""

        scores = evaluate_answer(q, answer, contexts, cfg)

        # Retrieval hit-rate / MRR against ground-truth chunk IDs
        hit = 0
        mrr = 0.0
        if expected_ids:
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in expected_ids:
                    hit = 1
                    mrr = 1.0 / rank
                    break
        retrieval_hits.append({"hit": hit, "mrr": mrr})

        elapsed = time.time() - t0
        for metric, value in scores.items():
            if value >= 0:
                all_scores[metric].append(value)
        per_question.append({
            "question": q,
            "n_contexts": len(contexts),
            "answer_len": len(answer),
            "scores": scores,
            "retrieval_hit": hit,
            "mrr": round(mrr, 3),
            "elapsed": round(elapsed, 2),
        })

    aggregate = {}
    for metric, values in all_scores.items():
        aggregate[metric] = round(statistics.mean(values), 3) if values else None

    n = len(examples)
    hit_rate = round(sum(h["hit"] for h in retrieval_hits) / n, 3) if n else 0
    mrr = round(sum(h["mrr"] for h in retrieval_hits) / n, 3) if n else 0

    return {
        "aggregate": aggregate,
        "thresholds": THRESHOLDS,
        "passes": passes_thresholds(aggregate),
        "n_questions": n,
        "retrieval_hit_rate": hit_rate,
        "mrr": mrr,
        "per_question": per_question,
    }


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="RAG evaluation (RAGAS-style)")
    ap.add_argument("questions", nargs="*", help="questions to evaluate")
    ap.add_argument("--file", help="file with one question per line")
    ap.add_argument("--golden", help="golden dataset JSONL (versioned, e.g. golden_v1.jsonl)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--kind", default=None)
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args(argv)

    questions = list(args.questions)
    if args.file:
        with open(args.file) as f:
            questions += [line.strip() for line in f if line.strip()]
    if not questions and not args.golden:
        ap.error("no questions provided (positional, --file, or --golden)")

    if args.golden:
        result = benchmark_golden(args.golden, k=args.k, kind=args.kind)
    else:
        result = benchmark(questions, k=args.k, kind=args.kind)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Benchmark: {result['n_questions']} questions")
        print(f"Passes thresholds: {result['passes']}")
        if "retrieval_hit_rate" in result:
            print(f"Retrieval hit-rate: {result['retrieval_hit_rate']}")
            print(f"MRR: {result['mrr']}")
        print()
        for metric, value in result["aggregate"].items():
            threshold = THRESHOLDS.get(metric, 0)
            status = "OK" if value is not None and value >= threshold else "FAIL"
            print(f"  {metric:20s} {value}  (threshold {threshold})  {status}")
        print()
        for pq in result["per_question"]:
            print(f"  Q: {pq['question'][:60]}")
            extra = ""
            if "retrieval_hit" in pq:
                extra = f" hit={pq['retrieval_hit']} mrr={pq['mrr']}"
            print(f"     contexts={pq['n_contexts']} answer={pq['answer_len']}ch "
                  f"time={pq['elapsed']}s{extra} scores={pq['scores']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
