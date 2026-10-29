"""Locate emitted tokens rather than incidental position/ID occurrences."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from vllm_bench.mtp_debug.compare_rejections import logits_differences
from vllm_bench.mtp_debug.locate_output import locate, main


def item(start=8209, accepted=0, count=1, selected=27991, label="a", cycle=5):
    draft = list(range(100, 100 + count))
    target = draft.copy()
    if accepted < count:
        target[accepted] = selected
    bonus = selected if accepted == count else 999
    output = draft[:accepted] + [selected]
    top = {"rows": [{"ids": [selected, 222], "logits": [100., 100.], "margin": 0.}
                    for _ in range(count)]}
    return {"sources": ["capture:1"], "event": {
        "rank": 0, "pid": 1, "cycle": cycle, "label": label,
        "target_forward": {"request_ids": [label],
                           "input_ids": {"values": [50] + draft, "truncated": False},
                           "positions": {"values": list(range(start, start + count + 1)), "truncated": False},
                           "mome_calls": []},
        "verify": {"accepted": [accepted], "cu_num_draft_tokens": [count],
                   "draft_ids": draft, "target_argmax": target, "bonus_ids": [bonus],
                   "matches_reference": True},
        "sampler_output_with_padding": [output + [-1] * (count + 1 - len(output))],
        "raw_target_top2": copy.deepcopy(top), "processed_target_top2": copy.deepcopy(top),
        "raw_bonus_top2": {"rows": [{"ids": [bonus, 333], "logits": [99., 80.], "margin": 19.}]}}}


class LocateTests(unittest.TestCase):
    def test_correction_and_repeated_incidental_position(self):
        entry = item()
        entry["event"]["target_forward"]["unused_metadata"] = [8210] * 170
        result = locate([entry], 8210)
        self.assertEqual(result["source"], "target_correction")
        self.assertEqual(result["token_id"], 27991)
        self.assertEqual(result["source_row"], 0)
        self.assertTrue(result["output_matches_selection"])

    def test_bonus_and_accepted_draft(self):
        result = locate([item(start=8208, accepted=1, selected=8886)], 8210)
        self.assertEqual(result["source"], "bonus")
        self.assertEqual(result["actual_selected_id"], 8886)
        self.assertIsNone(result["processed_top2"])
        result = locate([item(accepted=1)], 8210)
        self.assertEqual(result["source"], "accepted_draft")
        self.assertEqual(result["token_id"], 100)
        self.assertTrue(result["accepted_draft_matches_target"])

    def test_correction_after_two_accepted_tokens(self):
        result = locate([item(start=8207, accepted=2, count=3)], 8210)
        self.assertEqual(result["source_row"], 2)
        self.assertEqual(result["conditioning_positions"], [8207, 8208, 8209])

    def test_position_seen_only_as_input_is_not_an_emission(self):
        self.assertIsNone(locate([item(start=8210)], 8210))

    def test_tied_topk_order_is_not_actual_choice_change(self):
        a = item(start=8193, selected=676)
        b = copy.deepcopy(a)
        b["event"]["processed_target_top2"]["rows"][0]["ids"] = [222, 676]
        for entry in (a, b):
            entry["input_ids"] = entry["event"]["target_forward"]["input_ids"]["values"]
        diff = logits_differences(a, b, 1e-5, 1e-5)[0]
        self.assertTrue(diff["top1_changed"])
        self.assertFalse(diff["actual_choice_changed"])
        self.assertEqual(diff["left_actual_choice"], 676)
        self.assertEqual(diff["top2_has_tie"], {"left": True, "right": True})

    def test_two_requests_window_dedup_and_cli_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for n, event in enumerate((item()["event"], item(start=8208, accepted=1, selected=8886, label="b")["event"])):
                window = {"format": "pangu-rejection-window-v1", "previous": event, "rejection": event, "next": None}
                (path / f"cycle-{n}.json").write_text(json.dumps(window))
            output = path / "token.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(path), "--position", "8210", "--output", str(output)])
            matches = json.loads(output.read_text())["matches"]
            self.assertEqual(len(matches), 2)
            self.assertEqual({m["token_id"] for m in matches}, {27991, 8886})
            self.assertEqual({m["source"] for m in matches}, {"target_correction", "bonus"})
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main([str(path), "--position", "8210", "--label", "a"])
            self.assertEqual(len(json.loads(stream.getvalue())["matches"]), 1)


if __name__ == "__main__":
    unittest.main()
