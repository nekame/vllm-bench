"""Exact prefill ki comparison, write/read isolation and safe operator instrumentation."""
import contextlib
import copy
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from attention_fixtures import payload
from test_attention_trace_cpu import Cache
from test_mome_tensors_cpu import Tensor, encode
from vllm_bench.mtp_debug import attention_trace, debug_spec, indexer_ki_trace as trace, spec_rejection
from vllm_bench.mtp_debug.compare_indexer_ki import compare, main
from vllm_bench.mtp_debug.tensor_io import TensorWriter, read_tensor
from vllm_bench.mtp_debug.test_indexer_quant_cache import input_descriptor


def frames(root, rank, side, mutation=None, chunked=False):
    affected = rank == 3 and side == "right"
    label = "request-01" if side == "left" else "request-02"
    key_rows = [[p + 1] * 128 for p in range(5)]
    scales = [[p + 17] * 4 for p in range(5)]
    if affected and mutation in ("ki", "cache"):
        key_rows[2][0] = 77
    if affected and mutation == "scales":
        scales[2][0] = 99
    def frame(positions, phase, number):
        def p(name, dtype, shape, data, **kw):
            return payload(root, f"ki-{rank}-{side}-{number}-{name}", dtype, shape, data, **kw)
        n = len(positions)
        raw = [float(pos) + (0.125 if affected and mutation == "ki" and pos == 2 else 0)
               for pos in positions for channel in range(128)]
        after = list(raw)
        if affected and mutation == "mutated" and 2 in positions:
            after[positions.index(2) * 128] += 0.5
        call = {"prefix": "layers.0.attn", "layer": 0, "status": "complete",
                "parameters": {"quant_mode": "mxfp8", "round_scale": True},
                "active_rows": list(range(n)), "active_positions": positions, "row_positions": positions,
                "row_input_ids": [100 + pos for pos in positions], "slots": [10 + pos for pos in positions],
                "ki_before": p("before", "float32", [n, 128], raw),
                "ki_after": p("after", "float32", [n, 128], after),
                "key_after": p("key", "uint8", [n, 1, 128], [x for pos in positions for x in key_rows[pos]], opaque=True),
                "scales_after": p("scales", "uint8", [n, 1, 4], [x for pos in positions for x in scales[pos]], opaque=True)}
        reads = []
        if phase == "decode":
            rows = copy.deepcopy(key_rows)
            if affected and mutation == "retention":
                rows[2][0] = 88
            reads = [{"layer": 0, "status": "complete", "row_positions": [4], "kv_length": 5,
                      "key_rows": p("read-key", "uint8", [5, 1, 128], [x for row in rows for x in row], opaque=True, logical=list(range(5))),
                      "key_scales": p("read-scale", "uint8", [5, 1, 2, 2], [x for row in scales for x in row], opaque=True, logical=list(range(5)))}]
        return {"format": "pangu-indexer-ki-v1", "rank": rank, "pid": rank + 1, "request_ids": [side], "label": label,
                "forward_index": number, "phase": phase, "status": "complete", "calls": [call], "reads": reads}
    result = [frame([0, 1], "prefill", 0), frame([2, 3], "prefill", 1)] if chunked else [frame([0, 1, 2, 3], "prefill", 0)]
    result.append(frame([4], "decode", len(result)))
    return result


def dataset(root, mutation=None, chunked=False):
    return [item for rank in range(8) for side in ("left", "right")
            for item in frames(root, rank, side, mutation, chunked and side == "right")]


class ComparisonTests(unittest.TestCase):
    def test_equal_prefill_across_chunk_boundaries_and_v2_scale_reshape(self):
        with tempfile.TemporaryDirectory() as root:
            report, pair = compare(root, dataset(root, chunked=True), "request-01", "request-02")
        self.assertEqual(report["result"], "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI", report["coverage_issues"])
        self.assertTrue(report["raw_ki_exact_equal"])
        self.assertIsNotNone(pair)

    def test_upstream_cache_write_and_later_retention_are_distinguished(self):
        for mutation, code in (("ki", "KI_INPUT_DIFFERS"), ("cache", "CACHE_OUTPUT_DIFFERS_WITH_EQUAL_RECORDED_KI"),
                               ("scales", "CACHE_OUTPUT_DIFFERS_WITH_EQUAL_RECORDED_KI"),
                               ("retention", "CACHE_CHANGED_AFTER_WRITE"), ("mutated", "KI_MUTATED_BY_CACHE_CALL")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                report, pair = compare(root, dataset(root, mutation), "request-01", "request-02")
                self.assertEqual(report["result"], code, report["coverage_issues"])
                self.assertEqual(report["first_difference"]["rank"], 3)
                self.assertEqual(pair[0]["frame"]["rank"], 3)
                if mutation == "ki":
                    self.assertFalse(report["raw_ki_exact_equal"])
                    self.assertEqual(report["first_difference"]["first"]["position"], 2)

    def test_incomplete_rank_prompt_and_corrupt_payload_never_claim_equal(self):
        with tempfile.TemporaryDirectory() as root:
            data = dataset(root)
            report, _ = compare(root, [f for f in data if f["rank"] != 7], "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")
            right = next(f for f in data if f["rank"] == 0 and f["label"] == "request-02" and f["phase"] == "prefill")
            right["calls"][0]["row_input_ids"][0] = 999
            report, _ = compare(root, data, "request-01", "request-02")
            self.assertTrue(any(i["reason"] == "prompt_input_ids_differ" for i in report["coverage_issues"]))
            right["calls"][0]["row_input_ids"][0] = 100
            source = right["calls"][0]["ki_before"]
            (Path(root) / source["path"]).write_bytes(b"broken")
            report, _ = compare(root, data, "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")
            self.assertIsNone(report["raw_ki_exact_equal"])

    def test_cli_short_summary_and_portable_real_input_replay(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            data = dataset(root, "ki")
            (directory / "ki-rank-0-pid-1.jsonl").write_text("\n".join(json.dumps(f) for f in data))
            output = directory / "report.json"
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main([str(directory), "--output", str(output)])
            self.assertEqual(len(stream.getvalue().splitlines()), 3)
            self.assertEqual(json.loads(output.read_text())["result"], "KI_INPUT_DIFFERS")
            for side in ("left", "right"):
                descriptor = input_descriptor(directory / "report-replays" / side / "input.json")
                self.assertEqual(descriptor["shape"], [4, 128])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([str(directory), "--output", str(output)])

    def test_missing_replay_payload_still_produces_incomplete_report(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            data = dataset(root)
            source = data[0]["calls"][0]["ki_before"]
            (directory / source["path"]).unlink()
            (directory / "ki-rank-0-pid-1.jsonl").write_text("\n".join(json.dumps(f) for f in data))
            output = directory / "missing.json"
            with contextlib.redirect_stdout(io.StringIO()):
                main([str(directory), "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")
            self.assertIsNone(report["raw_ki_exact_equal"])
            self.assertFalse((directory / "missing-replays").exists())


class RuntimeTests(unittest.TestCase):
    def test_unknown_row_alignment_preserves_operator_and_is_explicitly_unsupported(self):
        sentinel = object()
        owner = NS(indexer_quant_cache=lambda *args, **kwargs: sentinel)
        trace.patch_cache_operator(owner)
        frame = {"forward_index": 0, "_indexer_ki": {"calls": [], "positions": [0, 1, 2], "input_ids": [10, 11, 12]}}
        token = trace.UPDATE.set((frame, "layers.0.attn", 0))
        try:
            result = owner.indexer_quant_cache(None, None, Tensor([[1.] * 128] * 2), Tensor([0, 1], "int32"), quant_mode="mxfp8", round_scale=True)
        finally:
            trace.UPDATE.reset(token)
        self.assertIs(result, sentinel)
        call = frame["_indexer_ki"]["calls"][0]
        self.assertEqual(call["status"], "unsupported")
        self.assertEqual(call["forward_row_counts"], {"positions": 3, "input_ids": 3})

    def test_budget_exhaustion_preserves_none_and_marks_partial_without_cpu_copies(self):
        with tempfile.TemporaryDirectory() as root:
            writer = TensorWriter(root, 1, max_tensor_bytes=1, budget_bytes=1)
            owner = NS(indexer_quant_cache=lambda *args, **kwargs: None)
            trace.patch_cache_operator(owner)
            frame = {"forward_index": 0, "_indexer_ki": {"calls": [], "positions": [0], "input_ids": [10]}}
            cache = Cache([[[[0] * 128]]], "uint8")
            scales = Cache([[[[0] * 4]]], "uint8")
            with patch.object(attention_trace, "writer", return_value=writer), patch.object(trace, "writer", return_value=writer), \
                    patch.object(attention_trace, "tensor_bytes") as copy_bytes:
                token = trace.UPDATE.set((frame, "layers.0.attn", 0))
                try:
                    self.assertIsNone(owner.indexer_quant_cache(cache, scales, Tensor([[1.] * 128]), Tensor([0], "int32"), quant_mode="mxfp8", round_scale=True))
                finally:
                    trace.UPDATE.reset(token)
                copy_bytes.assert_not_called()
            call = frame["_indexer_ki"]["calls"][0]
            self.assertEqual(call["status"], "partial")
            self.assertEqual(call["reason"], "tensor_size_limit")

    def test_runner_persists_prefill_without_sampler_and_leaves_old_snapshot_bounded(self):
        meta = NS(num_prefills=1, num_decodes=0)
        forward_context = NS(get_forward_context=lambda: NS(attn_metadata={"a": meta}))
        class Runner:
            model_config = NS(enforce_eager=True, is_hybrid=True)
            cache_config = NS(enable_prefix_caching=False)
            input_batch = NS(num_reqs=1, req_ids=["r"], num_accepted_tokens_cpu=[1], num_computed_tokens_cpu=[0])
            num_accepted_tokens = NS(gpu=[1])
            def get_model(self):
                return NS(named_modules=lambda: [])
            def _model_forward(self, input_ids=None, positions=None):
                frame = spec_rejection.TARGET.get()
                if frame is not None:
                    self.seen = frame["_indexer_ki"]["positions"]
                return "original-result"
        def install_attention(runner, allowed):
            runner._pangu_attention_counts, runner._pangu_attention_topology = {}, {}
            return {}
        def install_ki(runner, allowed):
            runner._pangu_ki_counts = {}
            return {}
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {
                "PANGU_SPEC_DEBUG_ATTENTION": "1", "PANGU_SPEC_DEBUG_INDEXER_KI": "1",
                "PANGU_SPEC_DEBUG_MOME_TENSORS": "0", "PANGU_SPEC_DEBUG_RANK": "0"}), \
                patch.dict(sys.modules, {"vllm.forward_context": forward_context}), \
                patch.object(attention_trace, "install_attention", side_effect=install_attention), \
                patch.object(trace, "install_ki", side_effect=install_ki), contextlib.redirect_stdout(io.StringIO()):
            rec = spec_rejection.Recorder(root)
            with patch.object(spec_rejection, "_RECORDER", rec):
                runner = Runner()
                spec_rejection.install_runner(runner)
                tokens = Tensor(list(range(100)), "int64")
                self.assertEqual(runner._model_forward(input_ids=tokens, positions=tokens), "original-result")
                self.assertFalse(list(Path(root).glob("ki-rank-*.jsonl")))
                (Path(root) / "ARM").touch()
                self.assertEqual(runner._model_forward(input_ids=tokens, positions=tokens), "original-result")
                self.assertEqual(runner.seen, list(range(100)))
                self.assertTrue(rec.latest_target["positions"]["truncated"])
                self.assertNotIn("_indexer_ki", rec.latest_target)
            captured = json.loads(next(Path(root).glob("ki-rank-*.jsonl")).read_text())
            self.assertEqual(captured["phase"], "prefill")
            self.assertEqual(len(captured["positions"]), 100)

    def test_actual_operator_boundary_preserves_none_and_exact_pre_mutation_input(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(sys.modules, {"torch": NS(uint8="uint8")}), \
                patch("vllm_bench.mtp_debug.tensor_io.tensor_bytes", side_effect=encode), \
                patch.object(attention_trace, "tensor_bytes", side_effect=encode):
            writer = TensorWriter(root, 1)
            ki = Tensor([[1.] * 128, [2.] * 128])
            key = Cache([[[[0] * 128] for _ in range(4)]], "uint8")
            scales = Cache([[[[0] * 4] for _ in range(4)]], "uint8")
            slots = Tensor([2, 3], "int32")
            calls = []
            def original(k, s, x, indices, **kwargs):
                calls.append(kwargs)
                for i, slot in enumerate(indices.data):
                    k.data[0][slot][0] = [int(x.data[i][0])] * 128
                    s.data[0][slot][0] = [17 + i] * 4
                x.data[0][0] = 9.
                return None
            owner = NS(indexer_quant_cache=original)
            trace.patch_cache_operator(owner)
            frame = {"forward_index": 0, "_indexer_ki": {"status": "capturing", "positions": [0, 1], "input_ids": [100, 101], "calls": []}}
            with patch.object(attention_trace, "writer", return_value=writer):
                token = trace.UPDATE.set((frame, "layers.0.attn", 0))
                try:
                    self.assertIsNone(owner.indexer_quant_cache(key, scales, ki, slots, quant_mode="mxfp8", round_scale=True))
                finally:
                    trace.UPDATE.reset(token)
            call = frame["_indexer_ki"]["calls"][0]
            self.assertEqual(call["status"], "complete")
            self.assertEqual(call["row_positions"], [0, 1])
            self.assertEqual(struct.unpack("<f", read_tensor(root, call["ki_before"])[:4])[0], 1.)
            self.assertEqual(struct.unpack("<f", read_tensor(root, call["ki_after"])[:4])[0], 9.)
            self.assertEqual(read_tensor(root, call["key_after"]), bytes([1] * 128 + [2] * 128))
            self.assertEqual(calls, [{"quant_mode": "mxfp8", "round_scale": True}])
            self.assertIsNone(trace.UPDATE.get())

    def test_prefill_full_ids_quota_sidecar_and_context_reset(self):
        with tempfile.TemporaryDirectory() as root:
            rec = NS(output=Path(root))
            (rec.output / "LABEL").write_text("request-01")
            runner = NS(_pangu_ki_counts={})
            frame = {"rank": 0, "forward_index": 0, "request_ids": ["r"]}
            bound = {"input_ids": Tensor(list(range(100)), "int64"), "positions": Tensor(list(range(100)), "int64")}
            with patch.object(spec_rejection, "recorder", return_value=rec), patch.dict("os.environ", {"PANGU_SPEC_DEBUG_INDEXER_KI_STEPS": "1"}):
                trace.prepare_ki_frame(runner, frame, bound, {"a": NS(num_prefills=1)})
                self.assertEqual(len(frame["_indexer_ki"]["input_ids"]), 100)
                self.assertEqual(frame["_indexer_ki"]["phase"], "prefill")
                trace.finalize_ki_frame(frame)
                self.assertNotIn("_indexer_ki", frame)
                trace.prepare_ki_frame(runner, frame, bound, {"a": NS(num_prefills=0)})
                self.assertEqual(frame["_indexer_ki"]["reason"], "per_request_step_quota")
                trace.finalize_ki_frame(frame)
            files = list(Path(root).glob("ki-rank-*.jsonl"))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(files[0].read_text().splitlines()), 2)
            def fail():
                self.assertIsNotNone(trace.UPDATE.get())
                raise RuntimeError("failure")
            indexer = NS(_update_cache=fail)
            trace.scope_update(indexer, "layers.0.attn", 0)
            token = spec_rejection.TARGET.set({"_indexer_ki": {"status": "capturing"}})
            try:
                with self.assertRaises(RuntimeError):
                    indexer._update_cache()
            finally:
                spec_rejection.TARGET.reset(token)
            self.assertIsNone(trace.UPDATE.get())

    def test_launcher_implies_attention_and_validates_prerequisite(self):
        with tempfile.TemporaryDirectory() as root, patch.dict("os.environ", {}, clear=True):
            args = ["debug_spec", "--output", str(Path(root) / "new"), "--indexer-ki", "--rejection-windows", "--", "bash", "launch.sh"]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()) as stream:
                debug_spec.main()
            env = json.loads(stream.getvalue())["environment"]
            self.assertEqual(env["PANGU_SPEC_DEBUG_INDEXER_KI"], "1")
            self.assertEqual(env["PANGU_SPEC_DEBUG_ATTENTION"], "1")
            self.assertEqual(env["PANGU_SPEC_DEBUG_RANK"], "all")
            args.remove("--rejection-windows")
            with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                debug_spec.main()


if __name__ == "__main__":
    unittest.main()
