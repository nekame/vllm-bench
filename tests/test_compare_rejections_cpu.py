"""Offline alignment tests: committed history, capture gaps and state scope."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from vllm_bench.mtp_debug.compare_rejections import compare, load_events, main, timeline


def snap(values):
    return {"values": values, "truncated": False}


def item(position, cycle, label="a", accepted=1, physical=1, sha="same"):
    top = {"shape": [1, 100], "rows": [{"ids": [10, 11], "logits": [20., 18.], "margin": 2.}],
           "truncated": False}
    meta = {"num_accepted_tokens": snap([2]), "num_computed_tokens": snap([position]),
            "cache_indices": snap([physical]), "query_start_loc": snap([0, 2])}
    state = {"block": physical, "shape": [3, 4], "dtype": "bfloat16", "sha256": sha, "sample": [1.]}
    return {"sources": [f"capture:{cycle}"], "event": {
        "cycle": cycle, "rank": 0, "pid": 12, "label": label,
        "target_forward": {"request_ids": [label], "input_ids": snap([position * 10, (position + 1) * 10]),
                           "positions": snap([position, position + 1]), "is_hybrid": True,
                           "attention": {"layer": {"slot_mapping": snap([physical]), "seq_lens": snap([position + 2])}},
                           "runner_accepted_gpu": snap([2]),
                           "mome_calls": [{"prefix": "layers.0.qa_conv", "metadata": meta,
                                           "states_before": [state], "states_after": [copy.deepcopy(state)]}]},
        "verify": {"cu_num_draft_tokens": [1], "accepted": [accepted]},
        "sampler_output_with_padding": [[(position + 1) * 10, (position + 2) * 10 if accepted else -1]],
        "raw_target_top2": copy.deepcopy(top), "processed_target_top2": copy.deepcopy(top),
        "raw_bonus_top2": copy.deepcopy(top)}}


class CompareTests(unittest.TestCase):
    def test_physical_blocks_and_cycle_numbers_do_not_define_alignment(self):
        report = compare([item(100, 1), item(102, 2)],
                         [item(100, 80, "b", physical=999), item(102, 81, "b", physical=999)])
        self.assertEqual(report["matched_forwards"], 2)
        self.assertEqual(report["differences"], [])
        self.assertEqual(report["fingerprint_pairs_compared"], 4)

    def test_different_acceptance_boundaries_are_not_token_divergence(self):
        report = compare([item(100, 1), item(102, 2)],
                         [item(100, 10, "b", accepted=0), item(101, 11, "b", accepted=0), item(102, 12, "b")])
        self.assertIsNone(report["first_observed_token_difference"])
        self.assertEqual(report["matched_forwards"], 2)
        self.assertEqual(report["unmatched_forward_positions"]["right"], [101])

    def test_no_comparison_after_first_output_divergence(self):
        left = [item(100, 1), item(102, 2), item(104, 3)]
        right = [item(100, 51, "b"), item(102, 52, "b"), item(104, 53, "b")]
        right[1]["event"]["sampler_output_with_padding"][0][0] = 999
        report = compare(left, right)
        self.assertEqual(report["first_observed_token_difference"], {"path": "103", "left": 1030, "right": 999})
        self.assertEqual(report["matched_forwards"], 2)
        self.assertEqual(report["skipped"][0]["position"], 104)

    def test_capture_gap_is_not_silently_crossed(self):
        report = compare([item(100, 1), item(106, 4)], [item(100, 51, "b"), item(106, 54, "b")])
        self.assertEqual(report["matched_forwards"], 1)
        self.assertEqual(report["skipped"][0]["position"], 106)

    def test_different_recording_starts_are_not_certified(self):
        report = compare([item(100, 1), item(102, 2)], [item(102, 50, "b")])
        self.assertEqual(report["matched_forwards"], 0)

    def test_whole_block_hash_difference_is_observation(self):
        report = compare([item(100, 1)], [item(100, 10, "b", sha="different")])
        detail = report["differences"][0]
        self.assertEqual(detail["metadata_differences"], [])
        self.assertEqual(detail["fingerprint_observations"][0]["status"], "whole_block_differs")
        self.assertIn("not proof", detail["fingerprint_observations"][0]["diagnosis"])

    def test_missing_fingerprints_not_reported_as_equal_states(self):
        a, b = item(100, 1), item(100, 10, "b")
        for entry in (a, b):
            call = entry["event"]["target_forward"]["mome_calls"][0]
            del call["states_before"], call["states_after"]
        self.assertEqual(compare([a], [b])["fingerprint_pairs_compared"], 0)

    def test_logit_deltas_are_for_same_token_ids_and_conditioning(self):
        a, b = item(100, 1), item(100, 10, "b")
        b["event"]["target_forward"]["input_ids"]["values"][1] = 999
        b["event"]["raw_target_top2"]["rows"][0] = {"ids": [30, 31], "logits": [158., 150.], "margin": 8.}
        b["event"]["raw_bonus_top2"]["rows"][0]["logits"] = [500., 400.]
        report = compare([a], [b])
        changes = report["differences"][0]["logits_differences"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["common_token_logit_deltas"], {})
        self.assertTrue(changes[0]["top1_changed"])

    def test_tolerance_and_metadata_mismatch(self):
        a, b = item(100, 1), item(100, 10, "b")
        b["event"]["raw_target_top2"]["rows"][0]["logits"][0] += 1e-7
        self.assertIsNone(compare([a], [b])["first_logits_difference"])
        b["event"]["target_forward"]["runner_accepted_gpu"] = snap([1])
        self.assertEqual(compare([a], [b])["first_metadata_difference"]["position"], 100)

    def test_truncated_inputs_and_inconsistent_history_fail(self):
        a = item(100, 1)
        a["event"]["target_forward"]["input_ids"]["truncated"] = True
        with self.assertRaisesRegex(ValueError, "truncated"):
            timeline([a])
        a, b = item(100, 1), item(102, 2)
        b["event"]["target_forward"]["input_ids"]["values"][0] = 999
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            timeline([a, b])

    def test_overlapping_windows_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = item(100, 1)["event"], item(102, 2)["event"]
            window = {"format": "pangu-rejection-window-v1", "previous": a, "rejection": b, "next": None}
            for number in (1, 2):
                (Path(directory) / f"cycle-{number}.json").write_text(json.dumps(window))
            self.assertEqual(len(load_events(directory)), 2)
            window["rejection"] = copy.deepcopy(b)
            window["rejection"]["sampler_output_with_padding"] = [[999, -1]]
            (Path(directory) / "cycle-3.json").write_text(json.dumps(window))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_events(directory)

    def test_cli_two_labels_in_one_jsonl_and_full_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "rank-0-pid-12.jsonl", Path(directory) / "report.json"
            path.write_text("\n".join(json.dumps(item(100, n, label)["event"])
                                      for n, label in ((1, "a"), (20, "b"))))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main([str(path), "--list-requests"])
            self.assertEqual(len(stream.getvalue().splitlines()), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(path), "--left-label", "a", "--right-label", "b", "--output", str(output)])
            self.assertEqual(json.loads(output.read_text())["matched_forwards"], 1)
            before = output.read_text()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(path), "--left-label", "a", "--right-label", "b", "--output", str(output)])
            self.assertEqual(output.read_text(), before)


if __name__ == "__main__":
    unittest.main()
