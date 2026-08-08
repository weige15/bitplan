"""Run provenance and manifest helpers; no credentials or large outputs are recorded."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .metrics import METRICS_VERSION


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import torch  # type: ignore

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snapshot["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda"] = False
    return snapshot


def build_run_manifest(
    *,
    root: Path,
    config: dict[str, Any],
    model_key: str,
    split: str,
    run_id: str,
    raw_output_location: str,
    command: list[str],
    execution: dict[str, Any],
) -> dict[str, Any]:
    model = config["models"][model_key]
    data_manifest = root / config["data"]["manifest_path"]
    environment_lock = root / config["environment"]["lockfile"]
    return {
        "schema_version": "bitplan.m1.run-manifest.v1",
        "run_id": run_id,
        "repository": {
            "commit": repository_commit(root),
            "root": ".",
        },
        "model": {
            "key": model_key,
            "name": model["name"],
            "revision": model["revision"],
            "license": model["license"],
            "rationale": model["rationale"],
            "tokenizer": {
                "revision": model["tokenizer_revision"],
            },
        },
        "data": {
            "manifest": config["data"]["manifest_path"],
            "revision": config["data"]["provenance"]["revision"],
            "sha256": sha256_file(data_manifest),
            "split": split,
            "threshold_source_split": config["oracle"]["threshold_source_split"],
        },
        "environment_lock": {
            "path": config["environment"]["lockfile"],
            "sha256": sha256_file(environment_lock),
        },
        "hardware": hardware_snapshot(),
        "quantization": {
            "reference": config["quantization"]["reference"],
            "conditions": config["quantization"]["conditions"],
            "boundaries": config["boundaries_by_model"][model_key],
        },
        "decoding": config["decoding"],
        "seed": config["decoding"]["seed"],
        "command": command,
        "configuration": "configs/m1.json",
        "raw_output_location": raw_output_location,
        "metrics": {
            "implementation_version": METRICS_VERSION,
            "tool_version": __version__,
        },
        "execution": execution,
    }
