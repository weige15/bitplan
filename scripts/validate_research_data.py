#!/usr/bin/env python3
"""Validate BitPlan's dependency-free literature and result registries."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "research/literature/papers.yaml"
MATRIX = ROOT / "research/literature/claim-matrix.csv"
RESULTS = ROOT / "results/index.jsonl"
PAPER_FIELDS = {"id", "title", "authors", "year", "venue", "identifiers", "url", "status", "notes"}
RESULT_FIELDS = {"id", "status", "claim_ids", "config", "raw_output", "repository_commit", "model", "dataset", "environment_lock", "hardware", "quantization", "seed", "command", "metric_implementation"}
STATUSES = {"planned", "running", "completed", "failed"}

def fail(errors):
    if errors:
        raise ValueError("\n".join(f"- {e}" for e in errors))

def load_papers():
    try:
        data = json.loads(PAPERS.read_text())  # JSON is a valid YAML subset.
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"papers.yaml is not valid JSON-compatible YAML: {e}") from e
    errors, ids = [], set()
    if not isinstance(data, list): errors.append("papers.yaml must contain a list")
    for i, p in enumerate(data if isinstance(data, list) else []):
        if not isinstance(p, dict): errors.append(f"paper {i} must be an object"); continue
        missing = PAPER_FIELDS - p.keys()
        if missing: errors.append(f"paper {i} missing required fields: {sorted(missing)}")
        pid = p.get("id")
        if not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]+", pid or ""):
            errors.append(f"paper {i} has invalid identifier")
        elif pid in ids: errors.append(f"duplicate paper identifier: {pid}")
        else: ids.add(pid)
        if not isinstance(p.get("authors"), list) or not p.get("authors"): errors.append(f"paper {pid or i} authors must be a non-empty list")
        if not isinstance(p.get("identifiers"), dict): errors.append(f"paper {pid or i} identifiers must be an object")
        if p.get("status") not in {"candidate", "verified", "rejected"}: errors.append(f"paper {pid or i} has invalid status")
    fail(errors)
    return ids

def load_matrix(paper_ids):
    errors = []
    with MATRIX.open(newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"claim_id", "paper_id", "closest_prior_work", "exact_overlap", "proposed_differentiator", "needed_experiment", "falsifier", "status"}
    for i, row in enumerate(rows, 2):
        if set(row) != required: errors.append(f"claim matrix row {i} has wrong columns")
        if row.get("paper_id") not in paper_ids: errors.append(f"claim matrix row {i} references unknown paper: {row.get('paper_id')}")
        if row.get("status") not in {"open", "supported", "killed"}: errors.append(f"claim matrix row {i} has invalid status")
    fail(errors)

def load_results():
    errors, ids = [], set()
    for line_no, line in enumerate(RESULTS.read_text().splitlines(), 1):
        try: record = json.loads(line)
        except json.JSONDecodeError as e: errors.append(f"result line {line_no} is invalid JSON: {e}"); continue
        if not isinstance(record, dict): errors.append(f"result line {line_no} must be an object"); continue
        missing = RESULT_FIELDS - record.keys()
        if missing: errors.append(f"result line {line_no} missing required fields: {sorted(missing)}")
        rid = record.get("id")
        if rid in ids: errors.append(f"duplicate result identifier: {rid}")
        ids.add(rid)
        if record.get("status") not in STATUSES: errors.append(f"result {rid} has invalid status")
        for key, prefix in (("config", "configs/"), ("raw_output", "results/raw/")):
            value = record.get(key, "")
            path = Path(value) if isinstance(value, str) else Path("__invalid__")
            if not isinstance(value, str) or path.is_absolute() or ".." in path.parts or not value.startswith(prefix): errors.append(f"result {rid} has invalid {key} path: {value}")
    fail(errors)

def validate():
    paper_ids = load_papers(); load_matrix(paper_ids); load_results()
    print("research data valid: papers, claim matrix, and results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try: validate()
    except (OSError, ValueError) as e: parser.error(str(e))
