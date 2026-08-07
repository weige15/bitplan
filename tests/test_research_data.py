import json, sys, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import validate_research_data as v

class ResearchDataTests(unittest.TestCase):
    def test_committed_records_validate(self):
        v.validate()
    def test_duplicate_papers_are_rejected(self):
        original = v.PAPERS.read_text()
        try:
            data = json.loads(original); data.append(data[0]); v.PAPERS.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "duplicate paper identifier"): v.load_papers()
        finally: v.PAPERS.write_text(original)
    def test_missing_paper_field_is_rejected(self):
        original = v.PAPERS.read_text()
        try:
            data = json.loads(original); del data[0]["title"]; v.PAPERS.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "missing required fields"): v.load_papers()
        finally: v.PAPERS.write_text(original)
    def test_invalid_matrix_reference_is_rejected(self):
        original = v.MATRIX.read_text()
        try:
            v.MATRIX.write_text(original.replace("paper-example-001", "missing-paper"))
            with self.assertRaisesRegex(ValueError, "unknown paper"): v.load_matrix({"paper-example-001"})
        finally: v.MATRIX.write_text(original)
    def test_malformed_and_duplicate_results_are_rejected(self):
        original = v.RESULTS.read_text()
        try:
            v.RESULTS.write_text("not-json\\n")
            with self.assertRaisesRegex(ValueError, "invalid JSON"): v.load_results()
            v.RESULTS.write_text(original + original)
            with self.assertRaisesRegex(ValueError, "duplicate result identifier"): v.load_results()
        finally: v.RESULTS.write_text(original)

    def test_invalid_result_path_and_status_are_rejected(self):
        original = v.RESULTS.read_text()
        try:
            record = json.loads(original); record["status"] = "unknown"; v.RESULTS.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid status"): v.load_results()
            record["status"] = "planned"; record["raw_output"] = "/tmp/raw"; v.RESULTS.write_text(json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid raw_output path"): v.load_results()
        finally: v.RESULTS.write_text(original)

if __name__ == "__main__": unittest.main()
