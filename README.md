# BitPlan

Research repository for reliable, risk-bounded, precision-adaptive
large-language-model inference.

## M1: paired-prefix instrumentation and offline replay oracle

M1 is internal research tooling, not a product-facing quantizer, calibrated
runtime monitor, safety certificate, or paper result. The scope is fixed in
[`configs/m1.json`](configs/m1.json):

- development model: `sshleifer/tiny-gpt2` at commit
  `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` (Apache-2.0);
- primary model: `Qwen/Qwen2.5-7B-Instruct` at commit
  `a09a35458c702b33eeacc393d103063234e8bc28` (Apache-2.0);
- tokenizer revisions, rationales, layer boundaries, data splits, thresholds,
  and quantization support scopes are recorded in that configuration;
- conditions are exactly BF16, W8, W4, and W3. W8/W4/W3 use native
  fake-dequantized weights for numerical instrumentation, not packed kernels or
  systems measurements;
- decoding is deterministic greedy decoding with seed 0. Every condition is
  evaluated on the reference greedy prefix, never on a diverged candidate
  prefix.

Calibration, validation, final, and smoke prompts are separate in
[`data/m1/evaluation-manifest.json`](data/m1/evaluation-manifest.json).
Thresholds are sourced from validation only. Raw traces, hidden tensors, and
large outputs are written below ignored `results/raw/`; the committed
[`results/index.json`](results/index.json) contains only run IDs, locations, and
small summaries.

### Deterministic CPU smoke test

The smoke path has no third-party dependencies and exercises prefix alignment,
all four conditions, boundary capture at `[0, 1, 2]`, offline suffix replay,
metric calculation, and the cost-quality frontier:

```bash
python3 -m unittest discover -s tests -v
python3 -m bitplan.smoke --config configs/m1.json --output results/raw/m1-smoke-fixture-v1
# or: make m1-smoke
```

Expected output includes `M1 SMOKE PASS`, `run_id=m1-smoke-fixture-v1`, and
`replay_checked=48`. The exact smoke summary is regenerated under the ignored
raw-result path; it is not a claim about either pinned model.

### Optional pinned-model run

The optional Transformers backend uses the exact versions in
[`environment/m1-lock.json`](environment/m1-lock.json):

```bash
python3 -m bitplan.run --model development --split validation \
  --output results/raw/dev-validation-v1
python3 -m bitplan.run --model primary --split validation \
  --output results/raw/primary-validation-v1
```

These commands require the locked optional environment and model access. They
capture top-k/logit summaries plus boundary hidden states and write a run
manifest with repository commit, immutable model/tokenizer/data revisions,
environment lock hash, hardware, quantization, seed, command, metric version,
and raw-output location. They intentionally do not benchmark serving speed or
claim runtime safety. In this environment the optional backend is not assumed
to be installed; the M1 report therefore labels full-model evidence pending.

### Validation and research record

```bash
python3 scripts/validate_research_data.py
```

The decision report is [`research/m1/decision-report.md`](research/m1/decision-report.md).
It answers the five launch questions and labels smoke-only, absent, and
hardware-limited evidence. JSON schemas live in `schemas/`.

The current candidate direction is calibrated selective precision replay:
run a low-bit model by default, detect numerically unsafe token decisions, and
spend higher precision only on the minimum computation needed for repair. This
remains a research hypothesis, not a verified novelty claim.
