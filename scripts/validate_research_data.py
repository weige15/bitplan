#!/usr/bin/env python3
"""Validate the committed M1 configuration, schemas, manifests, and index."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bitplan.config import load_config
from bitplan.manifest import validate_result_index


def main() -> int:
    config = load_config(ROOT / "configs/m1.json")
    with (ROOT / config["data"]["manifest_path"]).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("revision") != config["data"]["provenance"]["revision"]:
        raise ValueError("evaluation manifest revision does not match config")
    with (ROOT / "results/index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    validate_result_index(index)
    print("research data validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
