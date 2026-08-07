# Repository intake

## Snapshot

- **Purpose:** BitPlan is a research repository for reliable, risk-bounded, precision-adaptive LLM inference (see `README.md`).
- **Starting state:** The repository contained only `README.md`, `AGENTS.md`, and `.gitignore`; no implementation, test, build, or CI system was present.
- **Authoritative project contract:** `AGENTS.md` defines evidence, novelty, experiment reproducibility, and artifact restrictions.

## Scaffold decisions

- Structured literature is stored in `research/literature/papers.yaml` as a JSON-compatible YAML document so the standard library validator can run without third-party dependencies.
- Results are append-only JSON Lines in `results/index.jsonl`; raw outputs remain ignored under `results/raw/`.
- `scripts/validate_research_data.py` is the supported offline validation entry point; `tests/test_research_data.py` exercises valid and invalid records.
- `src/bitplan/` is intentionally importable but contains no research algorithm implementation.

## Structured-record schema

- `papers.yaml` is a top-level list of objects with required fields `id`, `title`, `authors` (non-empty list), `year`, `venue`, `identifiers` (object), `url`, `status`, and `notes`. IDs are unique slug-like strings; status is `candidate`, `verified`, or `rejected`.
- `claim-matrix.csv` requires the columns `claim_id`, `paper_id`, `closest_prior_work`, `exact_overlap`, `proposed_differentiator`, `needed_experiment`, `falsifier`, and `status`. Paper references must resolve; status is `open`, `supported`, or `killed`.
- Each `results/index.jsonl` record requires `id`, `status`, `claim_ids`, `config`, `raw_output`, `repository_commit`, `model`, `dataset`, `environment_lock`, `hardware`, `quantization`, `seed`, `command`, and `metric_implementation`. IDs are unique; status is `planned`, `running`, `completed`, or `failed`. Config paths must stay under `configs/`, and raw-output paths under ignored `results/raw/`; absolute and parent-traversal paths are invalid.
- Records are validated by `scripts/validate_research_data.py`; malformed records, duplicate IDs, missing fields, unresolved matrix references, invalid statuses, and invalid paths fail with actionable errors.

## Constraints

Validation and CI must not require models, datasets, GPUs, network access, or services. Do not commit checkpoints, datasets, caches, credentials, raw generated corpora, or large traces. Future research changes must preserve the rules and review gates in `AGENTS.md`.
