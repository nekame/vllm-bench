"""Exact payload I/O and before/after capture without torch or NPU."""
import copy
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from vllm_bench.mtp_debug import spec_rejection
from vllm_bench.mtp_debug import debug_spec
from vllm_bench.mtp_debug.tensor_io import (
    CODES, TensorWriter, WIDTH, compare_bytes, compare_tensors, json_safe, numbers, read_tensor)


class Tensor:
    def __init__(self, data, dtype="float32"):
        self.data = data
        self.dtype = dtype
        self.shape = self._shape(data)

    @staticmethod
    def _shape(data):
        return [len(data), *Tensor._shape(data[0])] if isinstance(data, list) and data else ([0] if isinstance(data, list) else [])

    def flat(self):
        def flatten(data):
            if isinstance(data, list):
                return [x for child in data for x in flatten(child)]
            return [data]
        return flatten(self.data)

    def numel(self):
        return len(self.flat())

    def element_size(self):
        return WIDTH[self.dtype]

    def __getitem__(self, key):
        return Tensor(self.data[key], self.dtype)

    def reshape(self, *shape):
        return Tensor(self.flat(), self.dtype)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)


def encode(tensor):
    if tensor.dtype == "bfloat16":
        return b"".join(struct.pack("<H", struct.unpack("<I", struct.pack("<f", x))[0] >> 16) for x in tensor.flat())
    flat = [int(x) for x in tensor.flat()] if tensor.dtype in ("uint8", "int8", "int32", "int64") else tensor.flat()
    return struct.pack("<" + CODES[tensor.dtype] * tensor.numel(), *flat)


class PayloadTests(unittest.TestCase):
    def test_launcher_opt_in_validation_and_dry_run(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {}, clear=True):
            output = Path(directory) / "new"
            args = ["debug_spec", "--output", str(output), "--mome-tensors", "--rejection-windows", "--", "bash", "launch.sh"]
            with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()) as stream:
                debug_spec.main()
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan["environment"]["PANGU_SPEC_DEBUG_MOME_TENSORS"], "1")
            self.assertEqual(plan["environment"]["PANGU_SPEC_DEBUG_TENSOR_BUDGET_MIB"], "512")
            self.assertFalse(output.exists())
            args.remove("--rejection-windows")
            with patch("sys.argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                debug_spec.main()

    def test_round_trip_all_dtypes_rows_and_integrity(self):
        with tempfile.TemporaryDirectory() as directory, patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode):
            writer = TensorWriter(directory, 1)
            for index, dtype in enumerate(WIDTH):
                second = 2. if dtype == "uint8" else -2.
                tensor = Tensor([[1., second], [3., 4.]], dtype)
                descriptor = writer.save(tensor, index, 0, "value")
                self.assertEqual(numbers(read_tensor(directory, descriptor), dtype), [1., second, 3., 4.])
                self.assertEqual(descriptor["rows"][1]["sample"], [3., 4.])
                self.assertEqual(descriptor["byteorder"], "little")
                path = Path(directory) / descriptor["path"]
                content = path.read_bytes()
                path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
                self.assertEqual(compare_tensors(directory, descriptor, directory, descriptor)["status"], "unavailable")
                path.write_bytes(content[:-1])
                self.assertIn("size", compare_tensors(directory, descriptor, directory, descriptor)["reason"])

    def test_budget_and_individual_limit_stop_without_copying_more(self):
        with tempfile.TemporaryDirectory() as directory, patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode) as copy_bytes:
            writer = TensorWriter(directory, 1, max_tensor_bytes=16, budget_bytes=12)
            self.assertEqual(writer.save(Tensor([1., 2.]), 0, 0, "one")["status"], "complete")
            self.assertEqual(writer.save(Tensor([3., 4.]), 0, 0, "two")["reason"], "run_byte_budget")
            self.assertEqual(writer.save(Tensor([1.]), 0, 0, "three")["status"], "not_recorded")
            self.assertEqual(copy_bytes.call_count, 1)
            self.assertEqual(writer.written, 8)
            self.assertEqual(len(list(Path(directory).rglob("*.bin"))), 1)
            writer = TensorWriter(directory, 2, max_tensor_bytes=4)
            self.assertEqual(writer.save(Tensor([1., 2.]), 0, 0, "big")["reason"], "tensor_size_limit")

    def test_failed_capture_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=OSError("disk error")):
            writer = TensorWriter(directory, 1)
            with self.assertRaises(OSError):
                writer.save(Tensor([1.]), 0, 0, "one")
            self.assertEqual(writer.stopped_reason, "capture_error")
            self.assertFalse(list(Path(directory).rglob("*.bin")))

    def test_bfloat16_8192_channels_difference_outside_old_sample(self):
        with tempfile.TemporaryDirectory() as directory, patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode):
            writer = TensorWriter(directory, 1)
            data = [[0.] * 8192 for _ in range(3)]
            left = writer.save(Tensor(data, "bfloat16"), 0, 0, "left")
            data[1][2048] = 1.
            right = writer.save(Tensor(data, "bfloat16"), 0, 0, "right")
            self.assertEqual(left["rows"][1]["sample"], right["rows"][1]["sample"])
            self.assertEqual(compare_tensors(directory, left, directory, right)["different_elements"], 1)
            row = compare_tensors(directory, left, directory, right, rows=(1, 1))
            self.assertEqual(row["first_differences"][0]["index"], [2048])
            self.assertEqual(row["max_abs_finite"], 1.)

    def test_nonfinite_values_signed_zero_and_coordinates(self):
        a = struct.pack("<4f", 0., float("nan"), float("inf"), 2.)
        b = struct.pack("<4f", -0., float("nan"), float("-inf"), 5.)
        result = compare_bytes(a, b, "float32", [2, 2])
        self.assertEqual(result["different_elements"], 3)
        self.assertEqual(result["max_abs_finite"], 3.)
        self.assertEqual(result["max_abs_index"], [1, 1])
        self.assertEqual(result["nonfinite_elements"], {"left": 2, "right": 2})
        json.dumps(json_safe({"nan": float("nan"), "result": result}), allow_nan=False)

    def test_missing_unsupported_and_escaping_files_are_unavailable(self):
        descriptor = {"status": "complete", "path": "../outside.bin", "dtype": "float32", "shape": [1], "nbytes": 4, "byteorder": "little"}
        with tempfile.TemporaryDirectory() as directory:
            result = compare_tensors(directory, descriptor, directory, descriptor)
            self.assertIn("escapes", result["reason"])
            self.assertEqual(compare_tensors(directory, None, directory, None)["status"], "unavailable")
        with self.assertRaises(ValueError):
            numbers(b"\0\0", "int16")
        with tempfile.TemporaryDirectory() as directory:
            writer = TensorWriter(directory, 1)
            self.assertEqual(writer.save(None, 0, 0, "none")["reason"], "no_tensor_value")
            self.assertEqual(writer.written, 0)


class CaptureTests(unittest.TestCase):
    def test_cycle_limit_updates_tensor_status(self):
        with tempfile.TemporaryDirectory() as directory:
            rec = spec_rejection.Recorder(directory)
            rec.limit = 1
            rec.tensor_writer = TensorWriter(directory, 1)
            rec.record({"rank": 0, "cycle": 0, "target_forward": {"request_ids": ["r1"]},
                        "verify": {"accepted": [1], "cu_num_draft_tokens": [1]},
                        "sampler_output_with_padding": [[10, 20]]})
            self.assertEqual(rec.tensor_writer.stopped_reason, "cycle_limit")

    def run_conv(self, directory, prefill=False, armed=True, return_none=False, inplace=True):
        class Conv:
            weight = Tensor([[1., 0.], [0., 1.], [0., 0.]])
            kernel_width = 3
            on_ascend950 = True

            def forward(self, x, conv_states, mome_metadata, inplace=True):
                x.data[0][0] = 70.
                conv_states.data[0][1][0] = 30.
                return None if return_none else x

        import types
        meta = types.SimpleNamespace(num_prefills=int(prefill), num_decodes=int(not prefill),
             num_actual_tokens=2, cache_indices=Tensor([0]), num_accepted_tokens=Tensor([2]))
        x, cache = Tensor([[7., 8.], [9., 10.]]), Tensor([[[1., 2.], [3., 4.], [5., 6.]]])
        frame = {"forward_index": 0, "positions": {"values": [100, 101]}, "mome_calls": []}
        rec = spec_rejection.Recorder(directory)
        module = Conv()
        with patch.dict("os.environ", {"PANGU_SPEC_DEBUG_MOME_TENSORS": "1", "PANGU_SPEC_DEBUG_STATE_FINGERPRINTS": "0"}), \
                patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode), \
                patch.object(spec_rejection, "_RECORDER", rec):
            spec_rejection.install_conv(module, "layers.0.o_conv")
            token = spec_rejection.TARGET.set(frame if armed else None)
            try:
                self.assertIs(module.forward(x, cache, meta, inplace=inplace), None if return_none else x)
            finally:
                spec_rejection.TARGET.reset(token)
        return frame

    def test_inplace_none_return_captures_mutated_x_and_preserves_none(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = self.run_conv(directory, return_none=True)["mome_calls"][0]["tensor_capture"]
            self.assertEqual(trace["status"], "complete")
            self.assertEqual(trace["return_kind"], "none")
            self.assertEqual(trace["y_source"], "inplace_x")
            self.assertEqual(numbers(read_tensor(directory, trace["y_after"]), "float32"), [70., 8., 9., 10.])
            self.assertEqual(trace["y_after"]["sha256"], trace["x_after"]["sha256"])
            self.assertNotEqual(trace["x_before"]["sha256"], trace["y_after"]["sha256"])

    def test_none_without_inplace_does_not_invent_output(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = self.run_conv(directory, return_none=True, inplace=False)["mome_calls"][0]["tensor_capture"]
            self.assertEqual(trace["y_after"]["status"], "not_recorded")
            self.assertEqual(trace["y_source"], "unavailable")
            self.assertEqual(trace["states_after"][0]["tensor"]["status"], "complete")
            self.assertNotIn("x_after", trace)

    def test_before_inplace_mutation_and_after_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = self.run_conv(directory)["mome_calls"][0]["tensor_capture"]
            self.assertEqual(trace["status"], "complete")
            self.assertEqual(trace["row_positions"], [100, 101])
            self.assertEqual(trace["candidate_history_rows"], [1, 2])
            self.assertEqual(numbers(read_tensor(directory, trace["x_before"]), "float32"), [7., 8., 9., 10.])
            self.assertEqual(numbers(read_tensor(directory, trace["x_after"]), "float32"), [70., 8., 9., 10.])
            self.assertEqual(trace["states_before"][0]["tensor"]["rows"][1]["sample"], [3., 4.])
            self.assertEqual(trace["states_after"][0]["tensor"]["rows"][1]["sample"], [30., 4.])

    def test_prefill_skipped_and_no_capture_outside_armed_forward(self):
        for prefill, armed in ((True, True), (False, False)):
            with tempfile.TemporaryDirectory() as directory:
                frame = self.run_conv(directory, prefill, armed)
                self.assertFalse(list(Path(directory).rglob("*.bin")))
                if armed:
                    self.assertEqual(frame["mome_calls"][0]["tensor_capture"]["status"], "skipped")
                else:
                    self.assertEqual(frame["mome_calls"], [])


if __name__ == "__main__":
    unittest.main()
