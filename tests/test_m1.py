from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bitplan.config import CONFIG_PATH, load_config
from bitplan.manifest import validate_result_index, validate_run_manifest
from bitplan.metrics import METRICS_VERSION, compare_logits
from bitplan.models import ForwardOutput, ToyTransformer
from bitplan.oracle import oracle_frontier
from bitplan.runner import run_paired_prefix
from bitplan.smoke import run as run_smoke


class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.num_layers = model.num_layers
        self.vocab_size = model.vocab_size
        self.prefixes = []

    def forward(self, token_ids, boundaries):
        self.prefixes.append(tuple(token_ids))
        return self.model.forward(token_ids, boundaries)

    def replay_suffix(self, boundary, hidden_states):
        return self.model.replay_suffix(boundary, hidden_states)


class StableReplay:
    num_layers = 2
    vocab_size = 3

    def forward(self, token_ids, boundaries):
        return ForwardOutput([3.0, 1.0, 0.0], {boundary: [[0.0]] for boundary in boundaries})

    def replay_suffix(self, boundary, hidden_states):
        return [3.0, 1.0, 0.0]


class M1Tests(unittest.TestCase):
    def test_config_is_exactly_the_small_m1_scope(self):
        config = load_config(CONFIG_PATH)
        self.assertEqual(set(config["models"]), {"development", "primary"})
        self.assertEqual([c["name"] for c in config["quantization"]["conditions"]], ["bf16", "w8", "w4", "w3"])
        self.assertEqual(config["boundaries_by_model"]["primary"][-1], 28)

    def test_metrics_and_edge_cases_are_versioned(self):
        metrics = compare_logits([3.0, 1.0, 0.0], [3.0, 1.0, 0.0], top_k_count=8)
        self.assertEqual(metrics["implementation_version"], METRICS_VERSION)
        self.assertEqual(metrics["top_k"], 3)
        self.assertEqual(metrics["top1_disagreement"], 0)
        self.assertEqual(metrics["top_k_jaccard"], 1.0)
        self.assertEqual(metrics["reference_margin"], 2.0)
        self.assertTrue(metrics["decision_stable"])
        one = compare_logits([0.0], [1.0], top_k_count=8)
        self.assertEqual(one["reference_margin"], 0.0)
        self.assertEqual(one["top_k"], 1)

    def test_prefixes_are_identical_and_reference_advances_them(self):
        reference = RecordingModel(ToyTransformer())
        candidate = RecordingModel(ToyTransformer().copy_for_bits(3))
        result = run_paired_prefix(
            reference=reference,
            conditions={"w3": candidate},
            prompts=[{"id": "p", "text": "same prefix"}],
            boundaries=[0, 1, 2],
            max_new_tokens=3,
            top_k_count=4,
            thresholds={"max_probability_jsd": 0.02, "min_topk_jaccard": 0.5, "max_rank_movement": 1.0},
            vocab_size=32,
        )
        self.assertEqual(reference.prefixes, candidate.prefixes)
        self.assertEqual(len(reference.prefixes), 3)
        self.assertEqual(result["prompts"][0]["steps"][1]["prefix_tokens"], result["prompts"][0]["steps"][0]["prefix_tokens"] + [result["prompts"][0]["steps"][0]["reference_next_token"]])

    def test_boundary_capture_and_replay_equivalence(self):
        model = ToyTransformer()
        output = model.forward([1, 2, 3], [0, 1, 2])
        self.assertEqual(set(output.hidden_by_boundary), {0, 1, 2})
        self.assertEqual(len(output.hidden_by_boundary[1]), 3)
        for boundary in (0, 1, 2):
            replayed = model.replay_suffix(boundary, output.hidden_by_boundary[boundary])
            for left, right in zip(output.logits, replayed):
                self.assertAlmostEqual(left, right, places=12)

    def test_oracle_minimum_cost_and_deeper_tie_break(self):
        model = StableReplay()
        thresholds = {"max_probability_jsd": 0.02, "min_topk_jaccard": 0.5, "max_rank_movement": 1.0}
        frontier = oracle_frontier(
            reference_logits=[3.0, 1.0, 0.0],
            reference_model=model,
            candidate_hidden_by_boundary={0: [[0.0]], 1: [[0.0]], 2: [[0.0]]},
            boundaries=[0, 1, 2],
            top_k_count=2,
            thresholds=thresholds,
        )
        self.assertEqual(frontier["selected"]["boundary"], 2)
        self.assertEqual(len(frontier["frontier"]), 3)
        self.assertEqual(frontier["selection_rule"], "minimum cost_units, then maximum boundary")

    def test_manifest_and_cpu_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_smoke(CONFIG_PATH, Path(directory))
            self.assertEqual(summary["run_id"], "m1-smoke-fixture-v1")
            self.assertTrue(summary["replay_smoke"]["passed"])
            import json

            with (Path(directory) / "run-manifest.json").open(encoding="utf-8") as handle:
                validate_run_manifest(json.load(handle))
        validate_result_index(
            {
                "schema_version": "bitplan.m1.result-index.v1",
                "runs": [
                    {
                        "run_id": "x",
                        "manifest": "results/raw/x/run-manifest.json",
                        "raw_output_location": "results/raw/x/",
                        "summary": {"pending": True},
                    }
                ],
            }
        )


if __name__ == "__main__":
    unittest.main()
