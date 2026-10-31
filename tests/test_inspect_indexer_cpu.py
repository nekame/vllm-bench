"""Precise selection and explicit missing evidence for the short indexer report."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from vllm_bench.mtp_debug.inspect_indexer import extract, history_locations, main


def fixture():
    records = [{"boundary": "indexer_kernel_input", "field": field, "status": "compared", "exact_equal": True}
               for field in ("query", "query_dequant_scale", "weights", "key_scales")]
    return {"first_boundary": {"rank": 0, "position": 8193, "layer": 0}, "comparisons": [
        {"rank": 0, "position": 8193, "layer": 0, "left_cycle": 0, "right_cycle": 68, "records": records}]}


class IndexerReportTests(unittest.TestCase):
    def test_byte_history_maps_actual_positions_and_respects_offset_sample_limit(self):
        scales = {"compared_positions": list(range(8194)), "left_nbytes": 8194 * 4, "right_nbytes": 8194 * 4,
                  "different_bytes": 16, "first_different_offsets": [17167, 17176, 21653, 22752]}
        mapping = history_locations(scales)
        self.assertEqual(mapping["bytes_per_position"], 4)
        self.assertEqual(mapping["sampled_different_positions"], [4291, 4294, 5413, 5688])
        self.assertFalse(mapping["all_different_positions_known"])
        scales.update(different_bytes=4)
        self.assertTrue(history_locations(scales)["all_different_positions_known"])
        scales.update(compared_positions=[7, 11], left_nbytes=8, right_nbytes=8, different_bytes=1, first_different_offsets=[5])
        self.assertEqual(history_locations(scales)["first_different_position"], 11)
        scales["first_different_offsets"] = [8]
        self.assertEqual(history_locations(scales)["mapping_status"], "unavailable")

    def test_history_summary_includes_operator_and_only_two_cache_fields(self):
        report = fixture()
        records = report["comparisons"][0]["records"]
        scales = records[3]
        scales.update(exact_equal=False, different_bytes=1, first_different_offsets=[5],
                      compared_positions=[3, 7], left_nbytes=8, right_nbytes=8)
        records.append({**scales, "field": "key_rows"})
        parameters = {"operator": "quant_lightning_indexer", "scalars": {"quant_mode": 3}}
        records.append({"boundary": "indexer_kernel_parameters", "field": "parameters", "status": "compared",
                        "exact_equal": True, "left": parameters, "right": parameters})
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "comparison.json"
            path.write_text(json.dumps(report))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main([str(path), "--history-summary"])
        lines = dict(line.split("=", 1) for line in output.getvalue().splitlines())
        self.assertEqual(set(lines), {"context", "key_rows", "key_scales"})
        self.assertEqual(json.loads(lines["context"])["indexer_parameters"]["left"]["operator"], "quant_lightning_indexer")
        self.assertEqual(json.loads(lines["key_scales"])["sampled_different_positions"], [7])
        self.assertNotIn("first_different_offsets", json.loads(lines["key_scales"]))

    def test_defaults_alias_and_cli(self):
        report = fixture()
        report["comparisons"][0]["records"][1]["field"] = "query_scale"
        report["comparisons"][0]["records"][3].update(exact_equal=False, max_abs_finite=0.015625)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "comparison.json"
            path.write_text(json.dumps(report))
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main([str(path)])
        lines = dict(line.split("=", 1) for line in output.getvalue().splitlines())
        self.assertEqual(len(lines), 5)
        self.assertEqual(json.loads(lines["context"])["position"], 8193)
        self.assertTrue(json.loads(lines["query_dequant_scale"])["exact_equal"])
        self.assertFalse(json.loads(lines["key_scales"])["exact_equal"])

    def test_absent_unavailable_and_duplicates_are_never_equal(self):
        report = fixture()
        records = report["comparisons"][0]["records"]
        records.pop(3)
        records[0].update(status="unavailable", reason="missing payload")
        records.append(dict(records[2]))
        fields = extract(report)["fields"]
        self.assertEqual(fields["key_scales"]["status"], "missing")
        self.assertEqual(fields["weights"]["status"], "ambiguous")
        self.assertIsNone(fields["query"]["exact_equal"])
        self.assertEqual(fields["query"]["reason"], "missing payload")

    def test_explicit_selector_and_ambiguous_comparisons(self):
        report = fixture()
        report["first_boundary"]["rank"] = 3
        with self.assertRaises(ValueError):
            extract(report)
        self.assertEqual(extract(report, rank=0)["rank"], 0)
        report["comparisons"].append(report["comparisons"][0])
        with self.assertRaises(ValueError):
            extract(report, rank=0)
        with self.assertRaisesRegex(ValueError, "full attention-comparison"):
            extract({"first_boundary": {"rank": 0, "position": 8193, "layer": 0}})


if __name__ == "__main__":
    unittest.main()
