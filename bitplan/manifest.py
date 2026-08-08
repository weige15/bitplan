"""Validation for committed M1 manifests and result indexes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import REVISION_RE


def validate_run_manifest(manifest: dict[str, Any]) -> None:
    required = (
        "schema_version",
        "run_id",
        "repository",
        "model",
        "data",
        "environment_lock",
        "hardware",
        "quantization",
        "decoding",
        "seed",
        "command",
        "configuration",
        "raw_output_location",
        "metrics",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"run manifest missing {missing}")
    if manifest["schema_version"] != "bitplan.m1.run-manifest.v1":
        raise ValueError("unsupported run manifest schema")
    if not REVISION_RE.fullmatch(manifest["model"]["revision"]):
        raise ValueError("run manifest model revision is not immutable")
    if not REVISION_RE.fullmatch(manifest["model"]["tokenizer"]["revision"]):
        raise ValueError("run manifest tokenizer revision is not immutable")
    if manifest["data"].get("threshold_source_split") == "final":
        raise ValueError("final data cannot source thresholds")
    if manifest["decoding"].get("strategy") != "greedy" or manifest["seed"] != 0:
        raise ValueError("run manifest is not deterministic greedy decoding")
    if not manifest["raw_output_location"].startswith("results/raw/"):
        raise ValueError("raw output must stay under ignored results/raw/")
    if manifest["metrics"].get("implementation_version") != "bitplan.m1.metrics.v1":
        raise ValueError("unknown metric implementation version")


def validate_result_index(index: dict[str, Any]) -> None:
    if index.get("schema_version") != "bitplan.m1.result-index.v1":
        raise ValueError("unsupported result index schema")
    runs = index.get("runs")
    if not isinstance(runs, list):
        raise ValueError("result index runs must be a list")
    seen: set[str] = set()
    for run in runs:
        for field in ("run_id", "manifest", "raw_output_location", "summary"):
            if field not in run:
                raise ValueError(f"result index entry missing {field}")
        if run["run_id"] in seen:
            raise ValueError("duplicate run id in result index")
        seen.add(run["run_id"])
        if not run["raw_output_location"].startswith("results/raw/"):
            raise ValueError("result index raw output is not ignored")
        if not isinstance(run["summary"], dict):
            raise ValueError("result index summary must be an object")


def load_and_validate_manifest(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("manifest must contain an object")
    validate_run_manifest(value)
    return value


def load_and_validate_index(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("index must contain an object")
    validate_result_index(value)
    return value
