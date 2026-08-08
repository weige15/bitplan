"""Offline cost-quality frontier and minimum-cost selective replay oracle."""

from __future__ import annotations

from typing import Any

from .metrics import compare_logits
from .models import BoundaryModel


def replay_cost(*, num_layers: int, boundary: int) -> dict[str, int | float]:
    """Count suffix layers plus one output-head unit; deeper is cheaper."""
    suffix_layers = num_layers - boundary
    units = suffix_layers + 1
    return {
        "suffix_layers": suffix_layers,
        "output_head_units": 1,
        "cost_units": units,
        "relative_cost": units / (num_layers + 1),
    }


def oracle_frontier(
    *,
    reference_logits: list[float],
    reference_model: BoundaryModel,
    candidate_hidden_by_boundary: dict[int, list[list[float]]],
    boundaries: list[int],
    top_k_count: int,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Enumerate every eligible boundary, then select the deterministic minimum.

    Tie-breaking is ``(cost_units, -boundary)``: the cheapest suffix wins, and
    if two implementations report equal cost the deeper boundary wins.  The
    full frontier remains in the returned object even when no repair satisfies
    the predicate.
    """
    frontier: list[dict[str, Any]] = []
    for boundary in boundaries:
        if boundary not in candidate_hidden_by_boundary:
            raise ValueError(f"missing captured hidden state for boundary {boundary}")
        replayed = reference_model.replay_suffix(boundary, candidate_hidden_by_boundary[boundary])
        metrics = compare_logits(
            reference_logits,
            replayed,
            top_k_count=top_k_count,
            thresholds=thresholds,
        )
        frontier.append(
            {
                "boundary": boundary,
                "cost": replay_cost(num_layers=reference_model.num_layers, boundary=boundary),
                "quality": metrics,
                "satisfies_threshold": bool(metrics["decision_stable"]),
            }
        )
    eligible = [entry for entry in frontier if entry["satisfies_threshold"]]
    selected = None
    if eligible:
        selected = min(eligible, key=lambda entry: (entry["cost"]["cost_units"], -entry["boundary"]))
    return {
        "implementation_version": "bitplan.m1.oracle.v1",
        "thresholds": thresholds,
        "frontier": frontier,
        "selected": selected,
        "selection_rule": "minimum cost_units, then maximum boundary",
    }
