# bitplan

Research repository for reliable, risk-bounded, precision-adaptive
large-language-model inference.

The current candidate direction is calibrated selective precision replay:
run a low-bit model by default, detect numerically unsafe token decisions,
and spend higher precision only on the minimum computation needed for repair.

This is a research hypothesis, not yet a verified novelty claim.
