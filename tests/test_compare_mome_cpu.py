"""Offline exact MoME comparisons using portable binary fixtures."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from vllm_bench.mtp_debug.compare_mome import build_report, compare_activations, main


def payload(root, name, data):
    path = Path(root) / "tensors" / (name + ".bin")
    path.parent.mkdir(exist_ok=True)
    raw = struct.pack("<" + "f" * sum(map(len, data)), *(x for row in data for x in row))
    path.write_bytes(raw)
    return {"status": "complete", "path": path.relative_to(root).as_posix(), "dtype": "float32",
            "shape": [len(data), len(data[0])], "byteorder": "little", "nbytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def entry(root, name, output, state=None, x=None, y=None):
    state = state or [[1., 2.], [3., 4.], [5., 6.]]
    x = x or [[7., 8.], [9., 10.]]
    y = y or [[11., 12.], [13., 14.]]
    trace = {"format": "pangu-mome-tensors-v1", "status": "complete", "parameters": {"weight": {"sha256": "same"}, "on_ascend950": True,
             "enable_precision_strong_consistency": False, "kernel_width": 3, "inplace": False},
             "row_positions": [100, 101], "candidate_history_rows": [1, 2],
             "x_before": payload(root, name + "-x", x), "y_after": payload(root, name + "-y", y),
             "states_before": [{"block": 0, "tensor": payload(root, name + "-state-before", state)}],
             "states_after": [{"block": 0, "tensor": payload(root, name + "-state-after", state)}]}
    return {"sources": [name], "event": {"cycle": 0 if name == "left" else 10, "rank": 0, "pid": 1, "label": name,
        "target_forward": {"request_ids": [name], "input_ids": {"values": [50, 999]},
            "positions": {"values": [100, 101]},
            "mome_calls": [{"prefix": "layers.0.o_conv", "tensor_capture": trace,
                "metadata": {"num_accepted_tokens": {"values": [2]}, "num_computed_tokens": {"values": [100]},
                             "query_start_loc": {"values": [0, 2]}, "B_size": 1, "pad_slot_id": -1}}]},
        "verify": {"accepted": [0], "cu_num_draft_tokens": [1], "draft_ids": [999],
                   "target_argmax": [output], "bonus_ids": [42], "matches_reference": True},
        "sampler_output_with_padding": [[output, -1]]}}


def trace(entry):
    return entry["event"]["target_forward"]["mome_calls"][0]["tensor_capture"]


class ComparisonTests(unittest.TestCase):
    def test_find_first_divergence_and_single_different_state_row(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20, state=[[1., 2.], [30., 4.], [5., 6.]])
            report = build_report(root, [left], [right])
            self.assertEqual(report["position"], 101)
            self.assertTrue(report["recorded_prefix_complete_and_equal"])
            result = report["calls"][0]
            rows = result["states_before"]["rows"]
            self.assertEqual([row["row"] for row in rows if not row["exact_equal"]], [1])
            self.assertEqual(rows[1]["max_abs_finite"], 27.)
            self.assertEqual(rows[1]["first_differences"][0]["index"], [0])
            self.assertTrue(rows[1]["left_candidate_history"])
            self.assertIn("state_before_differs_active_history_not_proven", result["observations"])

    def test_input_difference_and_equal_state(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20, x=[[17., 8.], [9., 10.]], y=[[21., 12.], [13., 14.]])
            result = build_report(root, [left], [right])["calls"][0]
            self.assertTrue(result["states_before"]["full_block"]["exact_equal"])
            self.assertIn("input_activations_differ_at_shared_positions", result["observations"])
            self.assertFalse(any("equal_captured_arguments" in x for x in result["observations"]))

    def test_output_differs_with_equal_arguments_only_when_parameters_known(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20, y=[[21., 12.], [13., 14.]])
            result = build_report(root, [left], [right])["calls"][0]
            self.assertTrue(any("equal_captured_arguments" in x for x in result["observations"]))
            trace(right)["parameters"]["weight"]["sha256"] = "different"
            result = build_report(root, [left], [right])["calls"][0]
            self.assertFalse(any("equal_captured_arguments" in x for x in result["observations"]))

    def test_metadata_difference_blocks_equal_arguments_observation(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20, y=[[21., 12.], [13., 14.]])
            right["event"]["target_forward"]["mome_calls"][0]["metadata"]["num_accepted_tokens"] = {"values": [1]}
            result = build_report(root, [left], [right])["calls"][0]
            self.assertFalse(result["metadata_equal"])
            self.assertFalse(any("equal_captured_arguments" in x for x in result["observations"]))

    def test_legacy_capture_missing_and_corrupt_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20)
            payload_path = Path(root) / trace(right)["x_before"]["path"]
            payload_path.unlink()
            result = build_report(root, [left], [right])["calls"][0]
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["x_before"]["full_tensor"]["status"], "unavailable")
            del right["event"]["target_forward"]["mome_calls"][0]["tensor_capture"]
            self.assertEqual(build_report(root, [left], [right])["calls"][0]["status"], "unavailable")

    def test_activations_aligned_by_position_across_different_batch_lengths(self):
        with tempfile.TemporaryDirectory() as root:
            a = {"row_positions": [100, 101], "x_before": payload(root, "a", [[1., 2.], [3., 4.]])}
            b = {"row_positions": [101], "x_before": payload(root, "b", [[3., 4.]])}
            result = compare_activations(root, a, b, "x_before", 102)
            self.assertEqual(result["full_tensor"]["status"], "incompatible_shape_or_dtype")
            self.assertEqual(result["rows"][0]["position"], 101)
            self.assertTrue(result["rows"][0]["exact_equal"])

    def test_capture_stopped_is_unavailable(self):
        with tempfile.TemporaryDirectory() as root:
            left = entry(root, "left", 10)
            right = entry(root, "right", 20)
            trace(right).update(status="stopped", reason="run_byte_budget")
            result = build_report(root, [left], [right])["calls"][0]
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["right_reason"], "run_byte_budget")

    def test_cli_full_report_no_overwrite_and_explicit_position(self):
        with tempfile.TemporaryDirectory() as root:
            left, right = entry(root, "left", 10), entry(root, "right", 20)
            path, output = Path(root) / "rank-0-pid-1.jsonl", Path(root) / "comparison.json"
            path.write_text("\n".join(json.dumps(x["event"]) for x in (left, right)))
            command = [str(path), "--left", "left", "--right", "right", "--output", str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                main(command)
            self.assertEqual(json.loads(output.read_text())["calls"][0]["status"], "compared")
            before = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(command)
            self.assertEqual(output.read_bytes(), before)
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main(command[:-2] + ["--position", "100"])
            self.assertIn("producer_not_recorded", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
