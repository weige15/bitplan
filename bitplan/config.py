"""Configuration loading and validation for the deliberately small M1 scope."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "m1.json"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CONDITIONS = ("bf16", "w8", "w4", "w3")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_json(path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "bitplan.m1.config.v1":
        raise ValueError("unsupported M1 configuration schema")

    models = config.get("models")
    if not isinstance(models, dict) or set(models) != {"development", "primary"}:
        raise ValueError("M1 must define exactly development and primary models")
    for key, model in models.items():
        if not isinstance(model, dict):
            raise ValueError(f"model {key} must be an object")
        for field in ("name", "revision", "tokenizer_revision", "license", "rationale"):
            if not model.get(field):
                raise ValueError(f"model {key} is missing {field}")
        if not REVISION_RE.fullmatch(model["revision"]):
            raise ValueError(f"model {key} revision must be a full immutable commit")
        if not REVISION_RE.fullmatch(model["tokenizer_revision"]):
            raise ValueError(f"model {key} tokenizer revision must be a full immutable commit")
    if models["primary"].get("parameter_scale") != "7B-8B":
        raise ValueError("primary model must be the one configured 7B-8B model")

    conditions = config.get("quantization", {}).get("conditions")
    if not isinstance(conditions, list) or tuple(c.get("name") for c in conditions) != REQUIRED_CONDITIONS:
        raise ValueError("conditions must be exactly BF16, W8, W4, and W3 in that order")
    for condition in conditions:
        if condition.get("name") == "bf16":
            if condition.get("bits") != 16:
                raise ValueError("BF16 must be represented with bits=16")
        elif condition.get("bits") not in (8, 4, 3):
            raise ValueError(f"invalid bits for {condition.get('name')}")
        if not condition.get("backend") or "support_scope" not in condition:
            raise ValueError(f"condition {condition.get('name')} needs backend and support scope")

    boundary_sets = config.get("boundaries_by_model")
    if not isinstance(boundary_sets, dict):
        raise ValueError("boundaries_by_model is required")
    for model_key, model in models.items():
        boundaries = boundary_sets.get(model_key)
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError(f"missing boundaries for {model_key}")
        if boundaries[0] != 0 or boundaries != sorted(set(boundaries)):
            raise ValueError(f"boundaries for {model_key} must be sorted and start at zero")
        if boundaries[-1] != model.get("transformer_layers"):
            raise ValueError(f"boundaries for {model_key} must include its final boundary")

    decoding = config.get("decoding", {})
    if decoding.get("strategy") != "greedy" or decoding.get("seed") != 0:
        raise ValueError("M1 decoding must be deterministic greedy decoding with seed 0")
    if int(decoding.get("top_k", 0)) < 2 or int(decoding.get("max_new_tokens", 0)) < 1:
        raise ValueError("decoding top_k and max_new_tokens must be positive")

    thresholds = config.get("oracle", {}).get("stability_thresholds", {})
    for field in ("max_probability_jsd", "min_topk_jaccard", "max_rank_movement"):
        if field not in thresholds:
            raise ValueError(f"missing oracle threshold {field}")
    if config.get("oracle", {}).get("threshold_source_split") != "validation":
        raise ValueError("oracle thresholds must be sourced from validation, never final")

    data = config.get("data", {})
    for split in ("calibration", "validation", "final", "smoke"):
        if split not in data.get("splits", {}):
            raise ValueError(f"missing data split {split}")
    if data.get("provenance", {}).get("revision") != "bitplan-m1-eval-v1":
        raise ValueError("unexpected evaluation-manifest revision")
