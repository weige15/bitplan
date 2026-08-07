# Experiment protocol

Every run record must include repository commit, immutable model and dataset revisions, environment lock, hardware, quantization configuration, random seed, complete command/configuration, raw-output location, and metric implementation version. Calibration, validation, and final evaluation data remain separate.

Report paired quality and systems results (including measured latency/throughput/memory or cost), not nominal bit-width or parameter count alone. Results are registered in `results/index.jsonl`; raw outputs belong under ignored `results/raw/`.
