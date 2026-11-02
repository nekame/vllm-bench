"""Standalone operator harness correctness without pretending to execute quantization."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from vllm_bench.mtp_debug import test_indexer_quant_cache as tool
from vllm_bench.mtp_debug.tensor_io import TensorWriter


def plan(cases=None):
    return {"rows": 7, "block_size": 4, "page_bytes": None, "seed": 123, "repeats": 3,
            "cases": list(tool.CASES) if cases is None else cases, "chunk_size": 2, "sync_each_call": False}


class Backend:
    """Controlled byte writer tests orchestration; it is not an MXFP8 reference."""
    info = {"kind": "CPU harness fixture; not quantization"}

    def __init__(self, fault=None):
        self.fault, self.calls, self.allocations = fault, 0, 0

    def bind_slots(self, slots):
        self.slots = slots

    def allocate(self, spec, fill):
        self.allocations += 1
        return spec, {name: bytearray([fill]) * size for name, size in spec["buffers"].items()}

    def invoke(self, cache, start, stop):
        self.calls += 1
        spec, buffers = cache
        for row in range(start, stop):
            for name, width in tool.WIDTHS.items():
                buffer, offset = tool.address(spec, name, self.slots[row])
                value = row + (17 if name == "scales" else 0)
                if self.fault == "repeat" and self.calls % 2 == 0 and row == 2:
                    value += 1
                if self.fault == "batch" and start > 0:
                    value += 1
                buffers[buffer][offset:offset + width] = bytes([value]) * width
        if self.fault == "guard":
            buffers[next(iter(buffers))][0] = 1

    def synchronize(self):
        pass

    def snapshot(self, cache):
        return {name: bytes(raw) for name, raw in cache[1].items()}

    def input_unchanged(self):
        return self.fault != "input"


class OperatorHarnessTests(unittest.TestCase):
    def test_layout_permutations_shifts_chunks_and_reuse_preserve_logical_rows(self):
        with tempfile.TemporaryDirectory() as root:
            report = tool.experiment(plan(), Backend(), root)
        self.assertEqual(report["result"], "PASS_IN_TESTED_CASES")
        self.assertEqual(len(report["cases"]), 7)
        self.assertTrue(all(trial["versus_reference"]["key"]["exact_equal"]
                            for case in report["cases"] for trial in case["trials"]))
        backend = Backend()
        with tempfile.TemporaryDirectory() as root:
            tool.experiment(plan(["reuse"]), backend, root)
        self.assertEqual(backend.allocations, 1)
        self.assertEqual(backend.calls, 3)

    def test_repeated_difference_and_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            report = tool.experiment(plan(["shared"]), Backend("repeat"), root)
            self.assertTrue((Path(root) / "first-failure" / "actual-key.bin").exists())
        self.assertEqual(report["result"], "REPEATED_CASE_DIFFERS")
        self.assertEqual(report["first_failure"]["trial"], 1)
        self.assertEqual(report["first_failure"]["sampled_logical_rows"], [2])

    def test_guard_input_and_batch_dependence_are_distinct(self):
        for fault, expected, cases in (("guard", "OUT_OF_SLOT_WRITE", ["shared"]),
                                      ("input", "INPUT_MUTATED", ["shared"]),
                                      ("batch", "SCENARIO_DEPENDENT_OUTPUT", ["shared", "chunked"])):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as root:
                report = tool.experiment(plan(cases), Backend(fault), root)
                self.assertEqual(report["result"], expected)

    def test_layout_guard_spans_and_custom_page_padding(self):
        spec = tool.layout(7, 4, page_bytes=4096)
        self.assertEqual(spec["key"]["offset"], 4 * 656)
        self.assertEqual(spec["scales"]["offset"], 4 * 784)
        self.assertEqual(spec["scales"]["page_stride"], 4096)
        with self.assertRaises(ValueError):
            tool.layout(7, 4, page_bytes=0)
        slots = tool.slots_for("shifted", 7, spec, 1)
        self.assertEqual(slots[0] % 4, 3)
        self.assertEqual(len(set(slots)), 7)

    def test_backend_invokes_real_signature_with_exact_input_slices(self):
        backend = object.__new__(tool.NPUBackend)
        backend.ki, backend.slots = list(range(7)), list(range(10, 17))
        calls = []
        backend.op = lambda *args, **kwargs: calls.append((args, kwargs))
        key, scales = object(), object()
        backend.invoke(({}, {"key": key, "scales": scales}), 2, 5)
        args, kwargs = calls[0]
        self.assertEqual(args, (key, scales, [2, 3, 4], [12, 13, 14]))
        self.assertEqual(kwargs, {"quant_mode": "mxfp8", "round_scale": True})

    def test_backend_views_share_storage_and_match_model_offsets_and_strides(self):
        from types import SimpleNamespace as NS
        class Buffer:
            def __init__(self, dtype="uint8", root=None):
                self.dtype, self.root = dtype, root if root is not None else self

            def view(self, dtype):
                return Buffer(dtype, self.root)

            def as_strided(self, shape, strides, storage_offset):
                return NS(shape=shape, strides=strides, offset=storage_offset, dtype=self.dtype, root=self.root)

        backend = object.__new__(tool.NPUBackend)
        backend.torch = NS(uint8="uint8", float8_e4m3fn="e4m3", full=lambda *args, **kwargs: Buffer())
        backend.device = "npu:0"
        spec = tool.layout(7, 4, page_bytes=4096)
        buffers, views = backend.allocate(spec, 93)
        self.assertIs(views["key"].root, buffers["raw"])
        self.assertIs(views["scales"].root, buffers["raw"])
        self.assertEqual((views["key"].offset, views["scales"].offset), (4 * 656, 4 * 784))
        self.assertEqual(views["key"].strides, (4096, 128, 128, 1))
        self.assertEqual(views["scales"].strides, (4096, 4, 4, 1))
        self.assertEqual((views["key"].dtype, views["scales"].dtype), ("e4m3", "uint8"))

    def test_dry_run_and_runtime_failure_report_without_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "new"
            args = ["--output", str(output), "--tokens", "7", "--block-size", "4"]
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(tool.main(args), 0)
            self.assertEqual(json.loads(stream.getvalue())["plan"]["rows"], 7)
            self.assertFalse(output.exists())
            with patch.object(tool, "NPUBackend", side_effect=RuntimeError("operator unavailable")), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(tool.main([*args, "--execute"]), 1)
            self.assertEqual(json.loads((output / "result.json").read_text())["result"], "ERROR")
            self.assertTrue((output / "summary.txt").exists())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(args)

    def test_recorded_input_integrity_and_planning(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            writer = TensorWriter(directory, 1)
            descriptor = writer.save_bytes(struct.pack("<256f", *([1.] * 256)), "float32", [2, 1, 128], 0, 0, "ki")
            path = directory / "input.json"
            path.write_text(json.dumps({"ki": descriptor}))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                tool.main(["--output", str(directory / "new"), "--input", str(path)])
            result = json.loads(stream.getvalue())["plan"]
            self.assertEqual((result["rows"], result["dtype"], result["pattern"]), (2, "float32", "recorded"))
            (directory / descriptor["path"]).write_bytes(b"bad")
            with self.assertRaises(ValueError):
                tool.input_descriptor(path)


if __name__ == "__main__":
    unittest.main()
