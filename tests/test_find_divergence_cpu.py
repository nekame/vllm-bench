"""First-divergence search across different acceptance/cycle boundaries."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from vllm_bench.mtp_debug.find_divergence import analyze, main


def item(start, cycle, current, output, label, accepted=0):
    draft = output[0] if accepted else 999
    target = draft if accepted else output[0]
    return {"sources": [f"capture:{cycle}"], "event": {
        "cycle": cycle, "rank": 0, "pid": 1, "label": label,
        "target_forward": {"request_ids": [label], "input_ids": {"values": [current, draft]},
                           "positions": {"values": [start, start + 1]}, "mome_calls": []},
        "verify": {"accepted": [accepted], "cu_num_draft_tokens": [1], "draft_ids": [draft],
                   "target_argmax": [target], "bonus_ids": [output[-1]], "matches_reference": True},
        "sampler_output_with_padding": [output + [-1] * (2 - len(output))]}}


class FirstDifferenceTests(unittest.TestCase):
    def test_find_earliest_difference_not_later_requested_position(self):
        left = [item(100, 0, 50, [10, 20], "a", 1), item(102, 1, 20, [30], "a")]
        right = [item(100, 68, 50, [10], "b"), item(101, 69, 10, [99], "b"), item(102, 70, 99, [88], "b")]
        report = analyze(left, right)
        self.assertEqual(report["first_difference"]["position"], 102)
        self.assertEqual(report["left_producer"]["source"], "bonus")
        self.assertEqual(report["right_producer"]["source"], "target_correction")
        self.assertTrue(report["recorded_prefix_complete_before_difference"])
        self.assertEqual(report["left_producer"]["cycle"], 0)
        self.assertEqual(report["right_producer"]["cycle"], 69)

    def test_first_input_already_different_and_producer_missing(self):
        report = analyze([item(100, 0, 50, [10], "a")], [item(100, 9, 51, [10], "b")])
        self.assertEqual(report["first_difference"]["position"], 100)
        self.assertIsNone(report["left_producer"])
        self.assertIsNone(report["same_recorded_prefix_range"])
        self.assertFalse(report["recorded_prefix_complete_before_difference"])

    def test_gap_does_not_certify_first_global_difference(self):
        left = [item(100, 0, 50, [10], "a"), item(105, 5, 70, [80], "a")]
        right = [item(100, 10, 50, [10], "b"), item(105, 15, 70, [81], "b")]
        report = analyze(left, right)
        self.assertFalse(report["recorded_prefix_complete_before_difference"])
        self.assertEqual(report["unknown_prefix_ranges"], [[102, 104]])

    def test_same_tokens_despite_different_acceptance(self):
        left = [item(100, 0, 50, [10, 20], "a", 1)]
        right = [item(100, 10, 50, [10], "b"), item(101, 11, 10, [20], "b")]
        self.assertEqual(analyze(left, right)["status"], "no_difference_in_shared_recorded_tokens")

    def test_cli_defaults_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "rank-0-pid-1.jsonl", Path(directory) / "first.json"
            events = [item(100, 0, 50, [10], "request-01"), item(100, 68, 50, [20], "request-02")]
            path.write_text("\n".join(json.dumps(entry["event"]) for entry in events))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main([str(path), "--output", str(output)])
            self.assertIn('"position": 101', stream.getvalue())
            self.assertEqual(json.loads(output.read_text())["first_difference"]["left_token_id"], 10)


if __name__ == "__main__":
    unittest.main()
