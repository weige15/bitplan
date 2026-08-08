# M1 decision report: paired-prefix instrumentation and offline replay oracle

## Scope and evidence key

This is an internal M1 tooling record. It is not a paper result, novelty claim,
calibrated-safety claim, runtime-monitor claim, or systems benchmark.

- **[Source fact]** The exact models, tokenizer revisions, licenses, boundaries,
  conditions, thresholds, split separation, and cost definition are in
  `configs/m1.json`.
- **[Source fact]** The fixed prompt manifest is
  `data/m1/evaluation-manifest.json`; calibration, validation, final, and smoke
  sets are separate. No threshold was tuned on final data.
- **[Experimental result: smoke-only]** `run_id=m1-smoke-fixture-v1`, using
  `configs/m1.json`, generated raw artifacts at
  `results/raw/m1-smoke-fixture-v1/`. The committed run manifest is
  `manifests/m1-smoke-fixture-v1.json`; the compact index is
  `results/index.json`.
- **[Environment fact]** The available environment has Python 3.12.3 but no
  installed `torch` or `transformers`, so neither pinned-model run was executed.
  The smoke result uses the dependency-free `ToyTransformer` fixture and is not
  evidence about either pinned model. Full-model evidence is pending; this is
  dependency- and hardware-limited rather than a substituted benchmark.

The configured stability predicate is `top1_agrees AND top_k_jaccard >= 0.5 AND
probability_jsd <= 0.02 AND rank_movement <= 1.0`. In this report, “unsafe” means
that predicate is false. Top-1 disagreement is reported separately because a
stable top-1 choice can still fail the broader predicate.

## Instrumentation checks

**[Experimental result: smoke-only]** The smoke command passed with
`replay_checked=48` and maximum absolute replay error `0.0` against tolerance
`1e-12`. It captured boundaries `[0, 1, 2]`, including final boundary `2`, and
ran BF16, W8, W4, and W3 on identical reference-greedy prefixes. The raw result
stores top-k/logit summaries, metrics, boundary hidden states, and provenance;
large/raw artifacts remain under the ignored raw-result path.

## 1. Are unsafe token decisions sparse?

**[Experimental result: smoke-only]** There were 16 generated steps per
condition. The broad-predicate unsafe rates and top-1 disagreement rates were:

| condition | unsafe decisions | top-1 disagreements |
|---|---:|---:|
| BF16 | 0/16 (0.0%) | 0/16 (0.0%) |
| W8 | 2/16 (12.5%) | 0/16 (0.0%) |
| W4 | 15/16 (93.75%) | 0/16 (0.0%) |
| W3 | 16/16 (100.0%) | 1/16 (6.25%) |

Thus sparsity is observed only for W8 in this fixture. The W4 and W3 results
are not sparse under the configured predicate, despite W4 having no top-1
changes. This does not answer the question for the pinned models.

## 2. Are they localized by replay depth?

**[Experimental result: smoke-only]** The oracle selected the following
boundaries; boundary `2` is the shortest suffix and boundary `0` is the full
two-layer suffix:

| condition | boundary 0 | boundary 1 | boundary 2 | no repair |
|---|---:|---:|---:|---:|
| BF16 | 0 | 0 | 16 | 0 |
| W8 | 0 | 0 | 16 | 0 |
| W4 | 0 | 0 | 16 | 0 |
| W3 | 5 | 1 | 10 | 0 |

**[Inference, smoke-only]** The fixture suggests replay depth can localize a
repair: W8/W4 repairs were all at the shortest suffix, while W3 sometimes
needed the full suffix. This is a fixture behavior, not a claim about
Transformer depth localization in either pinned model.

## 3. What fraction can short suffix replay repair?

**[Experimental result: smoke-only]** Define “short suffix” as a selected
boundary greater than zero. All eligible repairs in this fixture were found;
the short-suffix fractions were BF16 `16/16`, W8 `16/16`, W4 `16/16`, and W3
`11/16` (68.75%). Across the three low-bit conditions, this is `43/48`
(89.58%). These denominators count fixture steps, not examples from a final
evaluation set.

## 4. What is the oracle cost–quality frontier?

**[Source fact]** The explicit cost is `cost_units = replayed suffix layers +
one output-head unit`; with two layers, boundaries 0/1/2 cost 3/2/1. The
oracle enumerates every boundary and selects minimum cost, breaking equal-cost
ties toward the deeper boundary. The following is the complete smoke frontier;
`stable` is the number satisfying the predicate, and `mean JSD` is measured
against the higher-precision reference:

| condition | boundary | cost units | stable | mean JSD | top-1 disagreements |
|---|---:|---:|---:|---:|---:|
| BF16 | 0 | 3 | 16/16 | 0.000000017 | 0/16 |
| BF16 | 1 | 2 | 16/16 | 0.000000148 | 0/16 |
| BF16 | 2 | 1 | 16/16 | 0.000000361 | 0/16 |
| W8 | 0 | 3 | 16/16 | 0.000000114 | 0/16 |
| W8 | 1 | 2 | 16/16 | 0.000000366 | 0/16 |
| W8 | 2 | 1 | 16/16 | 0.000000598 | 0/16 |
| W4 | 0 | 3 | 16/16 | 0.000132083 | 0/16 |
| W4 | 1 | 2 | 16/16 | 0.000423079 | 0/16 |
| W4 | 2 | 1 | 16/16 | 0.000894933 | 0/16 |
| W3 | 0 | 3 | 16/16 | 0.000180513 | 0/16 |
| W3 | 1 | 2 | 2/16 | 0.000845727 | 14/16 |
| W3 | 2 | 1 | 10/16 | 0.001800376 | 6/16 |

**[Experimental result: smoke-only]** This frontier is reproducible from the
raw run and shows that the selected point is not the only reported point. It
must not be read as a quality or cost result for the 7B–8B model; fake-
quantized, dequantized fixture weights do not establish hardware cost.

## 5. Does the oracle leave enough headroom to justify a learned monitor?

**[Absent evidence]** No. There is no pinned-model run, no held-out final
result, no calibration study, no monitor, and no hardware measurement. The
fixture shows a deterministic oracle frontier and validates the replay
implementation, but it cannot establish monitor headroom, generalization, or
runtime behavior. A learned monitor remains an untested hypothesis.

**Recommendation: narrow**
