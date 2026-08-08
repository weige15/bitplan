# BitPlan research contract

## Evidence

- Treat novelty as unverified until the claim-collision matrix and adversarial
  novelty review are complete.
- Prefer primary papers, official proceedings, and official code repositories.
- Record exact arXiv, OpenReview, DOI, commit, and paper-version identifiers.
- Label important statements as source fact, inference, hypothesis, or
  experimental result.
- Do not silently convert an abstract claim into a stronger claim about the
  full paper.

## Novelty

Every candidate contribution must record:

- the closest prior work;
- the exact overlap;
- the proposed differentiator;
- the experiment needed to establish that differentiator;
- a result that would falsify or kill the contribution.

Do not implement a quantization method until the survey synthesis and
independent novelty red-team have been reviewed.

## Experiments

Every experiment must record:

- repository commit;
- model name and immutable revision;
- dataset and immutable revision;
- environment lock;
- hardware;
- quantization configuration;
- random seed;
- complete command or configuration;
- raw-output location;
- metric implementation version.

Report paired quality and systems results. Nominal bit-width, parameter count,
or estimated FLOPs alone are not systems results.

Keep calibration, validation, and final evaluation data separate. Do not tune
thresholds on final evaluation sets.

## Artifacts

Do not commit model checkpoints, datasets, caches, credentials, raw generated
corpora, or large profiling traces. Commit manifests, schemas, configurations,
small derived tables, and scripts needed to reproduce them.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
