"""Versioned token-decision metrics used by the M1 instrumentation."""

from __future__ import annotations

import math
from typing import Any, Iterable

METRICS_VERSION = "bitplan.m1.metrics.v1"


def _as_floats(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    if not values:
        raise ValueError("logit vectors cannot be empty")
    return values


def softmax(logits: Iterable[float]) -> list[float]:
    values = _as_floats(logits)
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def top_k(logits: Iterable[float], k: int) -> list[dict[str, float | int]]:
    values = _as_floats(logits)
    if k < 1:
        raise ValueError("k must be positive")
    probabilities = softmax(values)
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))[:k]
    return [
        {
            "token_id": index,
            "rank": rank,
            "logit": values[index],
            "probability": probabilities[index],
        }
        for rank, index in enumerate(order, start=1)
    ]


def _rank_movement(reference: list[int], candidate: list[int]) -> float:
    ref_ranks = {token: rank for rank, token in enumerate(reference, start=1)}
    cand_ranks = {token: rank for rank, token in enumerate(candidate, start=1)}
    union = sorted(set(ref_ranks) | set(cand_ranks))
    if not union:
        return 0.0
    missing_rank = max(len(reference), len(candidate)) + 1
    return sum(abs(ref_ranks.get(token, missing_rank) - cand_ranks.get(token, missing_rank)) for token in union) / len(union)


def _jensen_shannon_divergence(reference: list[float], candidate: list[float]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("logit vectors must have equal vocabulary size")
    ref = softmax(reference)
    cand = softmax(candidate)
    midpoint = [(left + right) / 2.0 for left, right in zip(ref, cand)]

    def kl(left: list[float], right: list[float]) -> float:
        # Terms with p=0 contribute zero. The epsilon also makes underflowed
        # probabilities deterministic across Python versions.
        epsilon = 1e-12
        return sum(p * math.log(max(p, epsilon) / max(q, epsilon)) for p, q in zip(left, right) if p > 0.0)

    return max(0.0, (kl(ref, midpoint) + kl(cand, midpoint)) / 2.0)


def compare_logits(
    reference_logits: Iterable[float],
    candidate_logits: Iterable[float],
    *,
    top_k_count: int,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return deterministic token-level metrics and the documented predicate.

    The M1 stability predicate is:
    ``top1_agrees AND top_k_jaccard >= min_topk_jaccard AND
    probability_jsd <= max_probability_jsd AND rank_movement <=
    max_rank_movement``.  A vocabulary smaller than k uses all available
    tokens.  A one-token vocabulary has reference_margin=0.0.
    """
    reference = _as_floats(reference_logits)
    candidate = _as_floats(candidate_logits)
    if len(reference) != len(candidate):
        raise ValueError("logit vectors must have equal vocabulary size")
    reference_top = top_k(reference, top_k_count)
    candidate_top = top_k(candidate, top_k_count)
    reference_ids = [int(item["token_id"]) for item in reference_top]
    candidate_ids = [int(item["token_id"]) for item in candidate_top]
    overlap = len(set(reference_ids) & set(candidate_ids))
    union = len(set(reference_ids) | set(candidate_ids))
    reference_order = sorted(reference, reverse=True)
    margin = reference_order[0] - reference_order[1] if len(reference_order) > 1 else 0.0
    metrics: dict[str, Any] = {
        "implementation_version": METRICS_VERSION,
        "top1_disagreement": int(reference_ids[0] != candidate_ids[0]),
        "top_k": min(top_k_count, len(reference)),
        "top_k_overlap": overlap,
        "top_k_jaccard": overlap / union if union else 1.0,
        "rank_movement": _rank_movement(reference_ids, candidate_ids),
        "logit_mae": sum(abs(left - right) for left, right in zip(reference, candidate)) / len(reference),
        "probability_jsd": _jensen_shannon_divergence(reference, candidate),
        "reference_margin": margin,
        "reference_top_k": reference_top,
        "candidate_top_k": candidate_top,
    }
    thresholds = thresholds or {
        "max_probability_jsd": float("inf"),
        "min_topk_jaccard": 0.0,
        "max_rank_movement": float("inf"),
    }
    metrics["decision_stable"] = bool(
        metrics["top1_disagreement"] == 0
        and metrics["top_k_jaccard"] >= float(thresholds["min_topk_jaccard"])
        and metrics["probability_jsd"] <= float(thresholds["max_probability_jsd"])
        and metrics["rank_movement"] <= float(thresholds["max_rank_movement"])
    )
    return metrics
