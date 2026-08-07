# Research decisions

| ID | Decision | Rationale | Reversal trigger |
|---|---|---|---|
| D001 | Keep the scaffold dependency-free and offline-validatable. | Early research infrastructure must run without models, datasets, GPUs, network, or services. | A documented reproducibility need requires a dependency and a replacement lockfile/process is approved. |
