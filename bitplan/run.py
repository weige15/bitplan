"""Run the optional pinned Transformers model or the dependency-free smoke fixture."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import ROOT, load_config, load_json
from .hf import load_transformers_conditions
from .manifest import validate_run_manifest
from .provenance import build_run_manifest
from .runner import run_paired_prefix, write_raw_result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(
    config_path: str | Path,
    model_key: str,
    split: str,
    output_dir: str | Path,
    devices: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    evaluation = load_json(ROOT / config["data"]["manifest_path"])
    prompts = evaluation["splits"].get(split)
    if prompts is None:
        raise ValueError(f"unknown split {split}")
    tokenizer, loaded = load_transformers_conditions(config, model_key, devices=devices)
    for prompt in prompts:
        prompt["token_ids"] = tokenizer(str(prompt["text"]), add_special_tokens=True)["input_ids"]
    reference = loaded["bf16"]
    result = run_paired_prefix(
        reference=reference,
        conditions=loaded,
        prompts=prompts,
        boundaries=config["boundaries_by_model"][model_key],
        max_new_tokens=config["decoding"]["max_new_tokens"],
        top_k_count=config["decoding"]["top_k"],
        thresholds=config["oracle"]["stability_thresholds"],
        vocab_size=reference.vocab_size,
    )
    output = Path(output_dir)
    write_raw_result(result, output)
    run_id = output.name
    manifest = build_run_manifest(
        root=ROOT,
        config=config,
        model_key=model_key,
        split=split,
        run_id=run_id,
        raw_output_location=f"results/raw/{run_id}/",
        command=[
            sys.executable,
            "-m",
            "bitplan.run",
            "--config",
            str(config_path),
            "--model",
            model_key,
            "--split",
            split,
            "--output",
            str(output_dir),
            "--devices",
            ",".join(devices or []),
        ],
        execution={
            "mode": "transformers",
            "network_access": True,
            "devices": devices or ["auto"],
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "note": "Native fake-dequantized weights; not a packed-kernel systems measurement.",
        },
    )
    validate_run_manifest(manifest)
    _write_json(output / "run-manifest.json", manifest)
    summary = {
        "schema_version": "bitplan.m1.summary.v1",
        "run_id": run_id,
        "config": "configs/m1.json",
        "raw_output_location": f"results/raw/{run_id}/",
        "conditions": result["summary"],
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1.json")
    parser.add_argument("--model", choices=("development", "primary"), default="development")
    parser.add_argument("--split", choices=("calibration", "validation", "final"), default="validation")
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", default=None, help="comma-separated devices, one per condition")
    args = parser.parse_args(argv)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()] if args.devices else None
    summary = run(args.config, args.model, args.split, args.output, devices=devices)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
