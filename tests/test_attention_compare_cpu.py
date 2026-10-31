"""Automatic diagnosis must identify the boundary and rank, not require manual JSON searches."""
import contextlib
import io
import json
import copy
from pathlib import Path
import tempfile
import unittest

from attention_fixtures import dataset, payload
from vllm_bench.mtp_debug.compare_attention import compare_capture, main
from vllm_bench.mtp_debug.attention_layout import index_differences, page_coordinates, selected_positions, split_packed


class LayoutTests(unittest.TestCase):
    def test_paging_padding_duplicates_and_packed_components(self):
        self.assertEqual(page_coordinates([0, 3, 4], [[2, 0]], 4, 3), ([2, 2, 0], [0, 3, 0]))
        with self.assertRaises(ValueError):
            page_coordinates([9], [[2, 0]], 4, 3)
        self.assertEqual(selected_positions([[0, 2, 2, -1, 10]], 4), ([0, 2], {"padding_count": 1, "invalid_indices": [10]}))
        self.assertTrue(index_differences([0, 2, -1], [2, 0, -1])["only_order_or_padding_changed"])
        self.assertEqual(index_differences([0, 0], [0, -1])["left_duplicates"], 1)
        self.assertEqual({k: len(v) for k, v in split_packed(bytes(656)).items()}, {"latent": 512, "rotary": 128, "scales": 16})


class DiagnosisTests(unittest.TestCase):
    def test_all_eight_ranks_equal_attention_despite_output_difference_elsewhere(self):
        with tempfile.TemporaryDirectory() as root:
            report = compare_capture(root, dataset(root), "request-01", "request-02")
            self.assertEqual(report["result"], "NO_DIFFERENCE_IN_CAPTURED_ATTENTION")
            self.assertEqual(report["observed_ranks"], list(range(8)))
            self.assertEqual(report["first_token_difference"]["position"], 101)
            self.assertTrue(all(c["exact_equal"] for c in report["tp_gather_checks"]))

    def test_each_earliest_boundary_is_found_on_rank_three(self):
        cases = {"query": "ATTENTION_KERNEL_INPUT_DIFFERS", "indexer_query": "INDEXER_INPUT_OR_PARAMETERS_DIFFER",
                 "indexer_history": "INDEXER_HISTORY_DIFFERS", "indexer_output": "DSA_SELECTED_INDICES_DIFFER",
                 "kv": "ATTENTION_SELECTED_KV_DIFFERS", "kernel_output": "ATTENTION_OUTPUT_DIFFERS_WITH_EQUAL_SELECTED_INPUTS",
                 "projection": "POST_ATTENTION_PROJECTION_DIFFERS"}
        for mutation, expected in cases.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                report = compare_capture(root, dataset(root, mutation), "request-01", "request-02")
                self.assertEqual(report["result"], expected, report["coverage_issues"])
                self.assertEqual(report["first_boundary"]["rank"], 3)
                self.assertEqual(report["first_boundary"]["position"], 100)

    def test_gather_reconstruction_and_missing_rank(self):
        with tempfile.TemporaryDirectory() as root:
            events = dataset(root, "gather")
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertTrue(any(not check["exact_equal"] for check in report["tp_gather_checks"]))
            self.assertEqual(report["result"], "TP_RECONSTRUCTION_MISMATCH")
            events = [e for e in events if e["event"]["rank"] != 7]
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")

    def test_rejected_rows_are_excluded_even_when_their_inputs_differ(self):
        with tempfile.TemporaryDirectory() as root:
            events = dataset(root)
            for item in events:
                stage = item["event"]["target_forward"]["attention_trace"]["layers"][0]["stages"][1]
                rank, side = item["event"]["rank"], item["event"]["label"]
                stage["row_positions"] = [100, 101]
                stage["tensors"]["query"] = payload(root, f"extra-{rank}-{side}", "float32", [2, 1, 2],
                    [3, 4, 100 if side == "request-01" else -100, 900])
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "NO_DIFFERENCE_IN_CAPTURED_ATTENTION")
            self.assertEqual(report["compared_query_contexts"], 8)

    def test_history_gap_and_legacy_capture_are_explicitly_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            events = dataset(root)
            later = copy.deepcopy(events)
            for item in events:
                item["event"]["verify"]["target_argmax"] = [10]
                item["event"]["sampler_output_with_padding"] = [[10, -1]]
            for item in later:
                item["event"]["cycle"] += 1
                item["event"]["target_forward"]["positions"]["values"] = [104, 105]
            report = compare_capture(root, events + later, "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")
            self.assertTrue(any(c["reason"] == "recorded_prefix_gap_before_token_difference" for c in report["coverage_issues"]))
            for item in events:
                del item["event"]["target_forward"]["attention_trace"]
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")

    def test_nonfinite_and_broken_operator_link_are_reported(self):
        with tempfile.TemporaryDirectory() as root:
            events = dataset(root)
            stage = events[7]["event"]["target_forward"]["attention_trace"]["layers"][0]["stages"][1]
            stage["tensors"]["query"] = payload(root, "nan-query", "float32", [1, 1, 2], [float("nan"), 4])
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "NONFINITE_VALUES_OBSERVED")
            projection = events[7]["event"]["target_forward"]["attention_trace"]["layers"][0]["stages"][2]
            projection["tensors"]["input"] = payload(root, "broken-link", "float32", [1, 1, 2], [10, 20])
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "PROTOCOL_INVARIANT_FAILURE")
            self.assertTrue(any(i["code"] == "kernel_output_does_not_match_projection_input" for i in report["protocol_issues"]))

    def test_partial_payload_is_not_equal_and_cli_writes_short_report(self):
        with tempfile.TemporaryDirectory() as root:
            events = dataset(root, "query")
            directory = Path(root)
            capture = directory / "rank-0-pid-1.jsonl"
            capture.write_text("\n".join(json.dumps(e["event"]) for e in events))
            output = directory / "report.json"
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main([str(directory), "--output", str(output)])
            self.assertLess(len(stream.getvalue().splitlines()), 10)
            compact = json.loads(output.with_suffix(".summary.json").read_text())
            self.assertEqual(compact["first_boundary"]["rank"], 3)
            self.assertTrue(output.with_suffix(".summary.txt").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(directory), "--output", str(output)])
            descriptor = events[1]["event"]["target_forward"]["attention_trace"]["layers"][0]["stages"][1]["tensors"]["query"]
            (directory / descriptor["path"]).unlink()
            report = compare_capture(root, events, "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")


if __name__ == "__main__":
    unittest.main()
