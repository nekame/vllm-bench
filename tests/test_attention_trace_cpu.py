"""Exercise real wrapper argument binding, paged copies and quota without NPU."""
import contextlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from test_mome_tensors_cpu import Tensor, encode
from vllm_bench.mtp_debug import attention_trace as trace, debug_spec
from vllm_bench.mtp_debug.ranks import parse_ranks, rank_selected
from vllm_bench.mtp_debug.tensor_io import TensorWriter, compare_bytes, read_tensor


class Cache(Tensor):
    device = "npu"

    def __init__(self, data, dtype="float32", slices=None):
        super().__init__(data, dtype)
        self.slices = [] if slices is None else slices

    def dim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise AssertionError("NPU advanced indexing must not be used")
        self.slices.append((key.start, key.stop))
        return Cache(self.data[key], self.dtype, self.slices)

    def contiguous(self):
        return self

    def view(self, dtype):
        if dtype != "uint8":
            raise AssertionError("Cache view must preserve bytes")
        return Tensor(list(encode(self)), "uint8")


class RuntimeTests(unittest.TestCase):
    def test_attention_actual_operands_sinks_and_output_are_captured(self):
        query = Tensor([[[0.] * 576], [[1.] * 576]])
        key = Tensor([[[[0] * 656], [[1] * 656]]], "uint8")
        indices = Tensor([[[0] + [-1] * 2047], [[0, 1] + [-1] * 2046]], "int32")
        output = Tensor([[[1., 2.]], [[3., 4.]]])
        arguments = {"query": query, "key": key, "value": Tensor([[[[0]]]], "uint8"),
                     "sparse_indices": indices, "actual_seq_lengths_query": Tensor([2], "int32"),
                     "actual_seq_lengths_kv": Tensor([2], "int32"), "block_table": Tensor([[0]], "int32"),
                     "key_sink": Tensor([[1., 2.]]), "value_sink": Tensor([[3., 4.]]), "attention_mode": 2}
        def original(**kwargs):
            self.assertIs(kwargs["query"], query)
            self.assertIs(kwargs["key"], key)
            self.assertIs(kwargs["sparse_indices"], indices)
            return output
        owner = NS(op=original)
        trace.patch_operator(owner, "op", "attention_kernel")
        layer = {"stages": []}
        frame = {"forward_index": 0, "positions": {"values": [0, 1]}}
        def fake_save(frame, item, name, tensor, opaque=False):
            return {"status": "complete", "shape": tensor.shape}
        with patch.object(trace, "save", side_effect=fake_save), patch.object(trace, "paged", return_value={"status": "complete"}) as paged, patch.object(trace, "writer", return_value=NS(stopped_reason=None)):
            token = trace.OP.set({"frame": frame, "layer": layer, "kind": "attention_kernel"})
            try:
                self.assertIs(owner.op(**arguments), output)
            finally:
                trace.OP.reset(token)
        item = layer["stages"][0]
        self.assertEqual(item["status"], "complete")
        self.assertEqual(item["row_positions"], [0, 1])
        self.assertEqual(paged.call_args.args[4], [0, 1])
        self.assertEqual(item["tensors"]["key_rows"]["semantic"], "hif8_packed_512_128_16")
        self.assertTrue({"query", "key_rows", "key_sink", "value_sink", "output"} <= item["tensors"].keys())

    def test_scope_is_reset_after_operator_failure(self):
        from vllm_bench.mtp_debug.spec_rejection import TARGET
        module = NS(prefix="layers.0.attn", layer_idx=0, _pangu_attention_config={})
        def original():
            self.assertEqual(trace.OP.get()["kind"], "attention_kernel")
            raise RuntimeError("operator failed")
        module.method = original
        trace.scope_method(module, "method", module, "attention_kernel")
        frame = {"attention_trace": {"status": "capturing", "layers": []}}
        token = TARGET.set(frame)
        try:
            with self.assertRaisesRegex(RuntimeError, "operator failed"):
                module.method()
            self.assertIsNone(trace.OP.get())
        finally:
            TARGET.reset(token)

    def test_actual_operator_capture_none_tuple_positional_and_scope(self):
        for result in (None, Tensor([[4., 5.]]), (Tensor([[1, -1]], "int32"), "extra")):
            with self.subTest(result=type(result).__name__):
                owner = NS(op=lambda *args, **kwargs: result)
                frame = {"forward_index": 0, "positions": {"values": [8195]}}
                layer = {"stages": []}
                trace.patch_operator(owner, "op", "indexer_kernel", ("query", "key", "weights", "query_dequant_scale", "key_dequant_scale", "sparse_count", "quant_mode"))
                kwargs = {"cu_seqlens_q": Tensor([0, 1], "int32"), "seqused_k": Tensor([2], "int32"),
                          "block_table": Tensor([[0]], "int32"), "layout_k": "PA_BBND"}
                args = (Tensor([[[1., 2.]]]), Tensor([[[[1., 2.]], [[3., 4.]]]]), Tensor([[1.]]), Tensor([[1.]]), Tensor([[[1.], [1.]]]), 2048, 3)
                self.assertIs(owner.op(*args, **kwargs), result)
                self.assertEqual(layer["stages"], [])
                def fake_save(frame, item, name, tensor, opaque=False):
                    return {"status": "complete"} if tensor is not None else {"status": "not_recorded"}
                with patch.object(trace, "save", side_effect=fake_save), patch.object(trace, "paged", return_value={"status": "complete"}) as paged, patch.object(trace, "writer", return_value=NS(stopped_reason=None)):
                    token = trace.OP.set({"frame": frame, "layer": layer, "kind": "indexer_kernel"})
                    try:
                        self.assertIs(owner.op(*args, **kwargs), result)
                    finally:
                        trace.OP.reset(token)
                item = layer["stages"][0]
                self.assertEqual(item["row_positions"], [8195])
                self.assertEqual(item["scalars"]["quant_mode"], 3)
                self.assertEqual(paged.call_args_list[0].args[4], [0, 1])
                self.assertEqual(item["status"], "partial" if result is None else "complete")
                self.assertIsNone(trace.OP.get())

    def test_invalid_layout_preserves_operator_return(self):
        output = object()
        owner = NS(op=lambda **kwargs: output)
        trace.patch_operator(owner, "op", "attention_kernel")
        layer = {"stages": []}
        frame = {"forward_index": 1, "positions": {"values": [0]}}
        token = trace.OP.set({"frame": frame, "layer": layer, "kind": "attention_kernel"})
        try:
            actual = owner.op(actual_seq_lengths_kv=[1], actual_seq_lengths_query=[1], block_table=[[0]],
                              sparse_indices=Tensor([[[0] * 2048]], "int32"), key=Tensor([[[[0] * 655]]]), query=Tensor([[[0] * 576]]))
        finally:
            trace.OP.reset(token)
        self.assertIs(actual, output)
        self.assertEqual(len(layer["stages"]), 1)
        self.assertEqual(layer["stages"][0]["status"], "unsupported")

    def test_paged_payload_maps_logical_positions_and_checks_budget_before_copy(self):
        cache = Cache([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
        fake_torch = NS(uint8="uint8")
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"torch": fake_torch}), patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode):
            writer = TensorWriter(root, 1)
            with patch.object(trace, "writer", return_value=writer), patch.object(trace, "tensor_bytes", side_effect=encode):
                descriptor = trace.paged({"forward_index": 0}, {"slot": 0}, "selected", cache, [0, 3], [[1, 0]])
            self.assertEqual(struct.unpack("<4f", read_tensor(root, descriptor)), (5., 6., 3., 4.))
            self.assertEqual(descriptor["physical_pages"], [1, 0])
            self.assertEqual(descriptor["page_slots"], [0, 1])
            self.assertEqual(cache.slices, [(1, 2), (0, 1)])
            self.assertEqual(descriptor["gather_method"], "basic_page_slice_to_cpu_then_byte_offsets")
            writer = TensorWriter(root, 2, max_tensor_bytes=1)
            with patch.object(trace, "writer", return_value=writer), patch.object(trace, "tensor_bytes") as copy_bytes:
                descriptor = trace.paged({"forward_index": 0}, {"slot": 0}, "selected", cache, [0], [[0]])
            self.assertEqual(descriptor["reason"], "tensor_size_limit")
            copy_bytes.assert_not_called()
            self.assertEqual(len(cache.slices), 2)

    def test_packed_hif8_bytes_are_preserved_and_each_page_copied_once(self):
        rows = [[(i + offset) % 256 for i in range(656)] for offset in range(4)]
        cache = Cache([[[rows[0]], [rows[1]]], [[rows[2]], [rows[3]]]], "uint8")
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"torch": NS(uint8="uint8")}), patch.object(trace, "tensor_bytes", side_effect=encode):
            writer = TensorWriter(root, 1)
            with patch.object(trace, "writer", return_value=writer):
                descriptor = trace.paged({"forward_index": 0}, {"slot": 0}, "packed", cache, [3, 0, 1], [[1, 0]], opaque=True)
            self.assertEqual(read_tensor(root, descriptor), bytes(rows[1] + rows[2] + rows[3]))
            self.assertEqual(descriptor["shape"], [3, 1, 656])
            self.assertEqual(descriptor["comparison_mode"], "bytes_only")
            self.assertEqual(cache.slices, [(0, 1), (1, 2)])

    def test_empty_paged_selection_creates_valid_empty_payload_without_copy(self):
        cache = Cache([[[1., 2.]]])
        with tempfile.TemporaryDirectory() as root, patch.object(trace, "tensor_bytes") as copy_bytes:
            writer = TensorWriter(root, 1)
            with patch.object(trace, "writer", return_value=writer):
                descriptor = trace.paged({"forward_index": 0}, {"slot": 0}, "empty", cache, [], [[0]])
            self.assertEqual(read_tensor(root, descriptor), b"")
            self.assertEqual(descriptor["shape"], [0, 2])
            self.assertEqual(cache.slices, [])
            copy_bytes.assert_not_called()

    def test_decode_quota_is_per_request_and_prefill_does_not_consume_it(self):
        runner = NS(_pangu_attention_counts={}, _pangu_attention_topology={})
        with patch.dict("os.environ", {"PANGU_SPEC_DEBUG_ATTENTION_STEPS": "1"}):
            for request, prefills, expected in (("a", 1, "skipped"), ("a", 0, "capturing"), ("a", 0, "skipped"), ("b", 0, "capturing")):
                frame = {"request_ids": [request]}
                trace.prepare_frame(runner, frame, {"m": NS(num_prefills=prefills, num_decodes=1-prefills)})
                self.assertEqual(frame["attention_trace"]["status"], expected)
        self.assertEqual(runner._pangu_attention_counts, {("a",): 1, ("b",): 1})

    def test_launcher_all_ranks_and_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {}, clear=True):
            args = ["debug_spec", "--output", str(Path(root) / "new"), "--attention-tensors", "--rejection-windows", "--", "bash", "launch.sh"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()) as stream:
                debug_spec.main()
            env = json.loads(stream.getvalue())["environment"]
            self.assertEqual(env["PANGU_SPEC_DEBUG_RANK"], "all")
            self.assertEqual(env["PANGU_SPEC_DEBUG_MOME_TENSORS"], "1")
            self.assertTrue(all(rank_selected(r, "all") for r in range(8)))
            self.assertEqual(parse_ranks("0,3,7"), {0, 3, 7})
        a, b = struct.pack("<I", 0x3f800000), struct.pack("<I", 0x3f800001)
        metrics = compare_bytes(a, b, "float32", [1])
        self.assertEqual(metrics["max_ulp_finite"], 1)
        self.assertGreater(metrics["relative_l2_finite"], 0)
        a, b = struct.pack("<q", 2**60), struct.pack("<q", 2**60 + 1)
        self.assertEqual(compare_bytes(a, b, "int64", [1])["max_abs_finite"], 1)

    def test_new_modules_import_without_ml_packages(self):
        code = '''
import importlib.abc, importlib, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torch_npu','vllm','omni_npu','numpy'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
for module in ('attention_trace','attention_layout','attention_compare','compare_attention','ranks'):
    importlib.import_module('vllm_bench.mtp_debug.' + module)
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
