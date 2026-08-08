"""Executable network-free M1 smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, load_config, load_json
from .manifest import validate_run_manifest
from .models import ToyTransformer
from .provenance import build_run_manifest
from .runner import run_paired_prefix, write_raw_result

RUN_ID = "m1-smoke-fixture-v1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _replay_equivalence(reference: ToyTransformer, result: dict, boundaries: list[int], tolerance: float) -> dict:
    checked = 0
    maximum_error = 0.0
    for prompt in result["prompts"]:
        for step in prompt["steps"]:
            prefix = step["prefix_tokens"]
            full = reference.forward(prefix, boundaries).logits
            for boundary in boundaries:
                replayed = reference.replay_suffix(boundary, step["reference_hidden_by_boundary"][boundary])
                error = max(abs(left - right) for left, right in zip(full, replayed))
                maximum_error = max(maximum_error, error)
                checked += 1
                if error > tolerance:
                    raise AssertionError(f"replay mismatch at boundary {boundary}: {error} > {tolerance}")
    return {"checked": checked, "max_abs_error": maximum_error, "tolerance": tolerance, "passed": True}


def run(config_path: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    evaluation = load_json(ROOT / config["data"]["manifest_path"])
    prompts = evaluation["splits"]["smoke"]
    boundaries = config["boundaries_by_model"]["development"]
    reference = ToyTransformer(num_layers=2)
    conditions = {
        "bf16": reference.copy_for_bits(16),
        "w8": reference.copy_for_bits(8),
        "w4": reference.copy_for_bits(4),
        "w3": reference.copy_for_bits(3),
    }
    thresholds = config["oracle"]["stability_thresholds"]
    result = run_paired_prefix(
        reference=reference,
        conditions=conditions,
        prompts=prompts,
        boundaries=boundaries,
        max_new_tokens=config["decoding"]["max_new_tokens"],
        top_k_count=config["decoding"]["top_k"],
        thresholds=thresholds,
        vocab_size=reference.vocab_size,
    )
    replay = _replay_equivalence(reference, result, boundaries, tolerance=1e-12)
    result["replay_smoke"] = replay
    output = Path(output_dir)
    write_raw_result(result, output)
    manifest = build_run_manifest(
        root=ROOT,
        config=config,
        model_key="development",
        split="smoke",
        run_id=RUN_ID,
        raw_output_location="results/raw/m1-smoke-fixture-v1/",
        command=[sys.executable, "-m", "bitplan.smoke", "--config", str(config_path), "--output", str(output_dir)],
        execution={
            "mode": "dependency-free-fixture",
            "fixture": "bitplan.models.ToyTransformer",
            "network_access": False,
            "note": "Smoke validates contracts, not the pinned model's quality or hardware behavior.",
        },
    )
    validate_run_manifest(manifest)
    _write_json(output / "run-manifest.json", manifest)
    summary = {
        "schema_version": "bitplan.m1.summary.v1",
        "run_id": RUN_ID,
        "config": "configs/m1.json",
        "raw_output_location": "results/raw/m1-smoke-fixture-v1/",
        "replay_smoke": replay,
        "conditions": result["summary"],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1.json")
    parser.add_argument("--output", default="results/raw/m1-smoke-fixture-v1")
    args = parser.parse_args(argv)
    summary = run(args.config, args.output)
    print("M1 SMOKE PASS")
    print(f"run_id={summary['run_id']}")
    print(f"replay_checked={summary['replay_smoke']['checked']}")
    for name, condition in summary["conditions"].items():
        print(f"{name}: unsafe_rate={condition['unsafe_decision_rate']:.6f} top1_disagreement_rate={condition['top1_disagreement_rate']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
