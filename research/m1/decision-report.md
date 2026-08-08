# M1 follow-up decision report: pinned-model evaluations

## Scope and evidence key

This remains internal M1 research tooling. It is not a product-facing
quantizer, calibrated runtime monitor, safety certificate, paper result,
novelty claim, or systems benchmark. The exact scope is
`configs/m1.json`; the exact prompt/data provenance is
`data/m1/evaluation-manifest.json`.

- **[Source fact]** The configured models are `sshleifer/tiny-gpt2` at
  `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` and
  `Qwen/Qwen2.5-7B-Instruct` at
  `a09a35458c702b33eeacc393d103063234e8bc28`; tokenizer revisions, Apache-2.0
  licenses, boundaries, and rationales are in `configs/m1.json` and each run
  manifest.
- **[Source fact]** The stability predicate is
  `top1_agrees AND top_k_jaccard >= 0.5 AND probability_jsd <= 0.02 AND
  rank_movement <= 1.0`. “Unsafe” below means this predicate is false; top-1
  disagreement is reported separately.
- **[Source fact]** Thresholds are sourced from validation only. Calibration,
  validation, and final prompts remain separate; final data was not used to
  select thresholds.
- **[Experimental result]** Every configured model/split completed BF16, W8,
  W4, and W3: development runs
  `dev-calibration-gpu4-v1`, `dev-validation-gpu4-v1`,
  `dev-final-gpu4-v1`; primary runs `primary-calibration-gpu4-7-v1`,
  `primary-validation-gpu4-7-v1`, `primary-final-gpu4-7-v1`.
- **[Artifact reference]** Small manifests are in `manifests/`; small summaries
  and raw-result locations are in `results/index.json`; large paired-prefix
  outputs are under ignored `results/raw/`.

The first primary attempt stopped before producing a completed run because the
Qwen2 suffix adapter omitted rotary position embeddings. The adapter was fixed,
committed at `2deeaa208a89545c6219510597acbad4500074b3`, and all six full runs
were rerun from that committed code. The failed attempt is not used as result
evidence.

## Environment and hardware

**[Environment fact]** Before installation, `nvidia-smi` reported eight NVIDIA
GeForce RTX 3090 GPUs, each with 24576 MiB. GPUs 0–3 were already occupied
(about 10.6 GiB and 99–100% utilization); GPUs 4–7 were effectively idle and
were selected. The exact pre-install UUID/inventory and commands are recorded
in `environment/m1-installed.json`.

**[Environment fact]** Driver `580.159.03`, `nvidia-smi` CUDA `13.0`,
`CUDA_HOME=/usr/local/cuda-12.4`, Python `3.12.3`, torch `2.5.1+cu124` (torch
CUDA `12.4`), Transformers `4.48.3`, safetensors `0.4.5`, huggingface-hub
`0.27.0`, and tokenizers `0.21.0` were used. Installation was isolated in the
ignored project `.venv`; no system packages were modified.

**[Source fact]** Development runs used
`CUDA_VISIBLE_DEVICES=4` with all four tiny condition copies on logical
`cuda:0`. Primary runs used `CUDA_VISIBLE_DEVICES=4,5,6,7`, assigning BF16,
W8, W4, W3 respectively to logical `cuda:0,cuda:1,cuda:2,cuda:3`. No
uncontrolled multi-GPU job, custom CUDA kernel, or Triton kernel was used.

The dependency-free smoke replay check still passes: 48 boundary checks with
maximum absolute error `0.0` at tolerance `1e-12` in
`m1-smoke-fixture-v1`. Full runs executed the higher-precision replay oracle;
the explicit full-forward replay equivalence gate is the deterministic CPU
smoke gate, not a claim that the fake-dequantized weights are a production
backend.

## Token-level results

Each run has 16 paired generated steps per condition (two prompts, eight
reference-greedy tokens). Values are rates over those 16 steps, mean top-k
Jaccard, mean probability JSD, and mean reference margin. The raw artifacts
also retain rank movement, logit MAE, top-k summaries, logits, and boundary
hidden states.

### Development model: `sshleifer/tiny-gpt2`

| split | condition | unsafe | top-1 disagreement | mean Jaccard | mean JSD |
|---|---|---:|---:|---:|---:|
| calibration | BF16 | 0/16 | 0/16 | 1.000 | 0.000000000 |
| calibration | W8 | 0/16 | 0/16 | 1.000 | 0.000000013 |
| calibration | W4 | 16/16 | 16/16 | 0.600 | 0.000003816 |
| calibration | W3 | 16/16 | 16/16 | 0.525 | 0.000054172 |
| validation | BF16 | 0/16 | 0/16 | 1.000 | 0.000000000 |
| validation | W8 | 0/16 | 0/16 | 1.000 | 0.000000012 |
| validation | W4 | 16/16 | 8/16 | 0.600 | 0.000003734 |
| validation | W3 | 16/16 | 16/16 | 0.467 | 0.000020531 |
| final | BF16 | 0/16 | 0/16 | 1.000 | 0.000000000 |
| final | W8 | 0/16 | 0/16 | 1.000 | 0.000000012 |
| final | W4 | 16/16 | 0/16 | 0.600 | 0.000003700 |
| final | W3 | 16/16 | 16/16 | 0.333 | 0.000020497 |

**[Experimental result: development model]** The mean reference margins were
`0.0021362305` (calibration), `0.0016174316` (validation), and `0.0009765625`
(final). The fixture/development model is useful for exercising the harness,
but does not establish primary-model behavior.

### Primary model: `Qwen/Qwen2.5-7B-Instruct`

| split | condition | unsafe | top-1 disagreement | mean Jaccard | mean JSD | mean margin |
|---|---|---:|---:|---:|---:|---:|
| calibration | BF16 | 0/16 | 0/16 | 1.000 | 0.000000 | 4.546875 |
| calibration | W8 | 7/16 | 2/16 | 0.702 | 0.015556 | 4.546875 |
| calibration | W4 | 16/16 | 16/16 | 0.000 | 0.691900 | 4.546875 |
| calibration | W3 | 16/16 | 16/16 | 0.000 | 0.693135 | 4.546875 |
| validation | BF16 | 0/16 | 0/16 | 1.000 | 0.000000 | 3.742188 |
| validation | W8 | 11/16 | 2/16 | 0.772 | 0.015938 | 3.742188 |
| validation | W4 | 16/16 | 16/16 | 0.000 | 0.691665 | 3.742188 |
| validation | W3 | 16/16 | 16/16 | 0.000 | 0.693131 | 3.742188 |
| final | BF16 | 0/16 | 0/16 | 1.000 | 0.000000 | 1.070313 |
| final | W8 | 14/16 | 7/16 | 0.637 | 0.041472 | 1.070313 |
| final | W4 | 16/16 | 16/16 | 0.000 | 0.685285 | 1.070313 |
| final | W3 | 16/16 | 16/16 | 0.000 | 0.693100 | 1.070313 |

**[Experimental result: primary model]** W3 and W4 were technically supported
by the configured native symmetric fake-dequantized backend, but both were
unstable on every primary step in every split. W8 degraded from 7/16 unsafe on
calibration to 14/16 on final under this fixed prompt manifest. These are
paired numerical results, not packed-weight quality or runtime results.

## 1. Are unsafe token decisions sparse?

**[Experimental result: primary model]** No for W3/W4: 48/48 primary steps were
unsafe for each condition. W8 was unsafe on 7/16 calibration, 11/16
validation, and 14/16 final; it was not sparse on the final split. BF16 was
stable by construction in these comparisons. Top-1 disagreement understates
W4/W3 damage because the broader predicate also catches rank movement,
top-k movement, and probability divergence.

**[Inference]** The development model gives a different, less severe pattern
(W8 stable on all 48 steps; W4/W3 mostly unstable), so sparsity is model/data
conditional rather than a general M1 observation.

## 2. Are they localized by replay depth?

**[Source fact]** The primary boundary set is `[0, 7, 14, 21, 28]`. The explicit
costs are respectively `29, 22, 15, 8, 1` units, where cost is replayed suffix
layers plus one output-head unit. The oracle enumerates all five boundaries.

**[Experimental result: primary model]** Stable counts at each boundary are
shown in boundary order `b0/b7/b14/b21/b28`; selected counts use the same order.

| split | condition | stable frontier counts | selected counts | no repair |
|---|---|---|---|---:|
| calibration | BF16 | 16/16/16/16/16 | 0/0/0/0/16 | 0 |
| calibration | W8 | 15/14/15/13/8 | 0/0/3/5/8 | 0 |
| calibration | W4 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |
| calibration | W3 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |
| validation | BF16 | 16/16/16/16/16 | 0/0/0/0/16 | 0 |
| validation | W8 | 13/12/11/10/7 | 2/1/1/3/7 | 2 |
| validation | W4 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |
| validation | W3 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |
| final | BF16 | 16/16/16/16/16 | 0/0/0/0/16 | 0 |
| final | W8 | 16/13/8/8/6 | 3/4/0/3/6 | 0 |
| final | W4 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |
| final | W3 | 0/0/0/0/0 | 0/0/0/0/0 | 16 |

**[Inference, primary model]** W8 repair outcomes are depth-dependent: the
oracle selected costs from 1 to 29 units and selected the shortest boundary 28
in 8/16 calibration, 7/16 validation, and 6/16 final steps. W3/W4 had no
eligible boundary, so the oracle cannot localize a repair for those conditions.
This is localization evidence for this fixed run set, not a general runtime
monitor claim.

## 3. What fraction can short suffix replay repair?

**[Experimental result: primary model]** Define short suffix as selected
boundary greater than zero:

| primary condition | calibration | validation | final | all splits |
|---|---:|---:|---:|---:|
| W8 | 16/16 (100%) | 12/16 (75%) | 13/16 (81.25%) | 41/48 (85.42%) |
| W4 | 0/16 | 0/16 | 0/16 | 0/48 |
| W3 | 0/16 | 0/16 | 0/16 | 0/48 |

W8 had two validation steps with no repair; all other W8 steps were repaired,
but not always by the shortest possible suffix. BF16 is excluded from this
low-bit repair fraction because it is the higher-precision reference condition.
The raw frontier retains the complete per-step quality values underlying these
counts.

## 4. What is the oracle cost–quality frontier?

**[Experimental result]** The complete frontier is present in each paired-prefix
artifact at:

- `results/raw/primary-calibration-gpu4-7-v1/paired-prefix.json`
- `results/raw/primary-validation-gpu4-7-v1/paired-prefix.json`
- `results/raw/primary-final-gpu4-7-v1/paired-prefix.json`

The frontier table in section 2 reports every boundary's stable count and every
selected point. Its quality dimensions are the per-boundary top-1 disagreement,
top-k overlap/rank movement, logit MAE, probability JSD, and reference margin.
The primary W8 mean JSD for the un-replayed paired condition was `0.015556`
(calibration), `0.015938` (validation), and `0.041472` (final); W3/W4 were about
`0.69` JSD and had zero stable frontier entries. Thus the frontier contains
substantial repair headroom only for W8 on these runs; it contains no eligible
repair for W3/W4.

**[Source fact]** Costs here are deterministic offline accounting units, not
measured latency, FLOPs, memory, or device time. Nominal bit-width and these
cost units are not systems results.

## 5. Does the oracle leave enough headroom to justify a learned monitor?

**[Inference, bounded by primary runs]** W8 has enough empirical variation to
motivate a narrower follow-up question: whether a monitor can predict which
W8 tokens need one of the enumerated replay depths. However, W3/W4 have no
repairable primary steps, final W8 degradation is high, and no learned monitor,
calibration study, generalization test, or runtime measurement was run. The
current evidence does not justify claiming monitor headroom or starting a
learned routing implementation. Learned routing remains out of M1 scope.

The validation commands after the code/report changes were the existing unit
tests, the dependency-free CPU smoke command, and
`scripts/validate_research_data.py`; all passed. The recommendation therefore
remains deliberately limited to the evidence collected here.

**Recommendation: narrow**
