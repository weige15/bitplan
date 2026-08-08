"""Paired-prefix generation and raw-result serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .metrics import compare_logits, top_k
from .models import BoundaryModel, argmax_token, stable_tokenize
from .oracle import oracle_frontier


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    if not observations:
        return {
            "steps": 0,
            "top1_disagreement_rate": 0.0,
            "unsafe_decision_rate": 0.0,
            "mean_top_k_jaccard": 0.0,
            "mean_probability_jsd": 0.0,
            "mean_reference_margin": 0.0,
        }
    metrics = [item["metrics"] for item in observations]
    return {
        "steps": len(metrics),
        "top1_disagreement_rate": _mean([float(item["top1_disagreement"]) for item in metrics]),
        "unsafe_decision_rate": _mean([float(not item["decision_stable"]) for item in metrics]),
        "mean_top_k_jaccard": _mean([float(item["top_k_jaccard"]) for item in metrics]),
        "mean_rank_movement": _mean([float(item["rank_movement"]) for item in metrics]),
        "mean_logit_mae": _mean([float(item["logit_mae"]) for item in metrics]),
        "mean_probability_jsd": _mean([float(item["probability_jsd"]) for item in metrics]),
        "mean_reference_margin": _mean([float(item["reference_margin"]) for item in metrics]),
    }


def run_paired_prefix(
    *,
    reference: BoundaryModel,
    conditions: dict[str, BoundaryModel],
    prompts: list[dict[str, Any]],
    boundaries: list[int],
    max_new_tokens: int,
    top_k_count: int,
    thresholds: dict[str, float],
    vocab_size: int,
) -> dict[str, Any]:
    """Evaluate every condition on the reference prefix at every step.

    The only prefix advancement is the reference greedy token. Candidate
    next-token choices are recorded but never fed into a later step.
    """
    condition_observations: dict[str, list[dict[str, Any]]] = {name: [] for name in conditions}
    raw_prompts: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        prefix = list(prompt.get("token_ids", stable_tokenize(str(prompt["text"]), vocab_size=vocab_size)))
        prompt_record: dict[str, Any] = {"id": prompt_id, "prefix_tokens": list(prefix), "steps": []}
        for step in range(max_new_tokens):
            reference_output = reference.forward(prefix, boundaries)
            reference_next = argmax_token(reference_output.logits)
            step_record: dict[str, Any] = {
                "step": step,
                "prefix_tokens": list(prefix),
                "reference_next_token": reference_next,
                "reference_top_k": top_k(reference_output.logits, top_k_count),
                "reference_logits": list(reference_output.logits),
                "reference_hidden_by_boundary": reference_output.hidden_by_boundary,
                "conditions": {},
            }
            for name, model in conditions.items():
                candidate_output = model.forward(prefix, boundaries)
                metrics = compare_logits(
                    reference_output.logits,
                    candidate_output.logits,
                    top_k_count=top_k_count,
                    thresholds=thresholds,
                )
                frontier = oracle_frontier(
                    reference_logits=reference_output.logits,
                    reference_model=reference,
                    candidate_hidden_by_boundary=candidate_output.hidden_by_boundary,
                    boundaries=boundaries,
                    top_k_count=top_k_count,
                    thresholds=thresholds,
                )
                observation = {
                    "prompt_id": prompt_id,
                    "step": step,
                    "prefix_tokens": list(prefix),
                    "candidate_next_token": argmax_token(candidate_output.logits),
                    "metrics": metrics,
                    "oracle": frontier,
                }
                condition_observations[name].append(observation)
                step_record["conditions"][name] = {
                    "logits": list(candidate_output.logits),
                    "top_k": top_k(candidate_output.logits, top_k_count),
                    "metrics": metrics,
                    "hidden_by_boundary": candidate_output.hidden_by_boundary,
                    "oracle": frontier,
                }
            prompt_record["steps"].append(step_record)
            # All conditions use this exact reference prefix at the next step.
            prefix = prefix + [reference_next]
        raw_prompts.append(prompt_record)

    return {
        "schema_version": "bitplan.m1.paired-prefix.v1",
        "prefix_policy": "reference-greedy-prefix-only",
        "boundaries": boundaries,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k_count,
        "thresholds": thresholds,
        "prompts": raw_prompts,
        "summary": {name: _summarize_observations(items) for name, items in condition_observations.items()},
        "observations": condition_observations,
    }


def write_raw_result(result: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "paired-prefix.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path
