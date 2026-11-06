"""Boundary hook lifecycle and logical state comparison without ML packages."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.boundary_trace import BoundaryRecorder, event_difference, first_difference, logical_blocks
from vllm_bench.pangu_infer import run


class Tensor:
    shape, dtype = (1,), "bf16"

    def __init__(self, value):
        self.value = value

    def numel(self):
        return 1

    def float(self):
        return self

    def __sub__(self, other):
        return Tensor(self.value - other.value)

    def abs(self):
        return Tensor(abs(self.value))

    def max(self):
        return self

    def item(self):
        return self.value


def torch_stub():
    return SimpleNamespace(Tensor=Tensor, equal=lambda a, b: a.value == b.value, save=Mock())


class HookModule:
    def __init__(self):
        self.hooks, self.handles = [], []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        handle = SimpleNamespace(remove=Mock())
        self.handles.append(handle)
        return handle

    def fire(self, output=None):
        for hook in self.hooks:
            hook(self, (object(), object()), output)


class BoundaryTraceTests(unittest.TestCase):
    def test_prefill_and_eager_snapshots_do_not_read_stale_decode_buffers_or_null_blocks(self):
        class Table:
            def detach(self):
                return self
            def cpu(self):
                return self
            def tolist(self):
                return [0] * 65
        class Builder:
            kv_cache_spec = SimpleNamespace(block_size=128)
            decode_cudagraph_max_bs = 8
            def __getattr__(self, name):
                raise AssertionError(f"Stale buffer read: {name}")
        builder = Builder()
        recorder = BoundaryRecorder.__new__(BoundaryRecorder)
        recorder.context, recorder.start, recorder.draft_calls = 8192, 7680, 0
        recorder.active = {"request_id": "r"}
        recorder.layers = {"target": ("target.mome", SimpleNamespace(kv_cache=[]), 0, builder)}
        recorder.runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r": 0}, num_reqs=1,
                block_table=[SimpleNamespace(get_device_tensor=lambda count: [Table()])]),
            requests={"r": SimpleNamespace(block_ids=([0] * 65,))})
        for full_graph, event in ((True, "initial_target"), (False, "first_verification")):
            builder.use_full_cuda_graph = full_graph
            result = recorder.states(event)["target"]
            self.assertFalse(result["persistent_decode_fields_current"])
            self.assertEqual(result["metadata"], {})
            self.assertTrue(all(not block["available"] for block in result["blocks"].values()))

    def test_compute_difference_is_reported_separately_from_unused_state_bytes(self):
        reference = [{"event": "initial_target", "logits": Tensor(13), "argmax": [13], "states": 0},
                     {"event": "first_verification", "logits": Tensor(13), "argmax": [13], "states": 0}]
        candidate = [{**reference[0], "states": 99},
                     {**reference[1], "logits": Tensor(330), "argmax": [330]}]
        with patch.dict("sys.modules", {"torch": torch_stub()}):
            difference = event_difference(reference, candidate, ("event", "logits", "argmax"))
        self.assertEqual(difference["field"], "1.logits")
        self.assertEqual(difference["event"], "first_verification")
        self.assertEqual(difference["reference_argmax"], [13])
        self.assertEqual(difference["candidate_argmax"], [330])

    def test_boundary_columns_and_comparison_ignore_physical_allocation(self):
        self.assertEqual(logical_blocks(7680, 8192, 128), [59, 63, 64])
        self.assertEqual(logical_blocks(0, 8192, 128), [63, 64])
        self.assertEqual(logical_blocks(7680, 8193, 128), [59, 64])
        a = {"physical_block": 10, "allocated_block": 10, "values": [Tensor(3)]}
        b = {**a, "physical_block": 20, "allocated_block": 20}
        with patch.dict("sys.modules", {"torch": torch_stub()}):
            self.assertIsNone(first_difference(a, b))
            b["values"] = [Tensor(5)]
            self.assertEqual(first_difference(a, b), {"field": "values.0", "max_abs_diff": 2.0})
            self.assertEqual(first_difference([1], [1, 2])["candidate_length"], 2)

    def test_hooks_capture_only_first_window_compare_repeats_and_detect_target_state_writes(self):
        target, draft = HookModule(), HookModule()
        names = ["model.layers.0.attn.mome", "draft_model.model.layers.12.attn.mome"]
        backend = SimpleNamespace(get_name=lambda: "NPUPanguMome")
        registry = {name: SimpleNamespace(get_attn_backend=lambda: backend) for name in names}
        builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128))
        group = SimpleNamespace(layer_names=names, get_metadata_builder=lambda: builder)
        batch = SimpleNamespace(req_id_to_index={"r-0": 0}, num_computed_tokens_cpu=[7680])
        runner = SimpleNamespace(get_model=lambda: SimpleNamespace(logits_processor=target),
            get_drafter_model=lambda: SimpleNamespace(model=SimpleNamespace(logits_processor=draft)),
            input_batch=batch, attn_groups=[[group]], query_start_loc=SimpleNamespace(np=[0, 512]),
            vllm_config=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=registry)))
        torch = torch_stub()
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"torch": torch}):
            recorder = BoundaryRecorder(runner, tmp, 0, 8192, 7)
            def snapshot(event, *args):
                record = recorder.active["record"]
                value = 5 if record == 1 and event.startswith("mtp_") else 3
                physical = 10 + record
                return {"event": event, "states": {"target": {"blocks": {"63": {
                    "physical_block": physical, "allocated_block": physical,
                    "available": True, "values": [Tensor(value)]}}}}, "logits": Tensor(13)}
            recorder.snapshot = Mock(side_effect=snapshot)
            for index in range(2):
                batch.req_id_to_index = {f"r-{index}": 0}
                recorder.arm(f"r-{index}", "prompt-0", index)
                draft.fire(Tensor(1))
                target.fire(None)
                self.assertEqual(recorder.target_calls, 0)
                batch.num_computed_tokens_cpu = [0]
                target.fire(Tensor(1))
                self.assertEqual(recorder.target_calls, 0)
                batch.num_computed_tokens_cpu = [7680]
                target.fire(Tensor(1))
                for _ in range(7):
                    draft.fire(Tensor(1))
                target.fire(Tensor(1))
                target.fire(Tensor(1))
                draft.fire(Tensor(1))
                report = recorder.finish()
                self.assertEqual(len(recorder.events), 9)
                if index == 0:
                    self.assertIsNone(report["reference_record"])
                    self.assertEqual(report["target_state_changes_during_mtp"], [])
                else:
                    self.assertIsNone(report["first_compute_difference"])
                    self.assertEqual(report["first_state_difference"]["event"], "mtp_0")
                    self.assertEqual(report["first_state_difference"]["max_abs_diff"], 2)
                    self.assertEqual(len(report["target_state_changes_during_mtp"]), 7)
                    self.assertEqual(report["target_state_changes_during_mtp"][0]["logical_block"], 63)
            self.assertEqual(torch.save.call_count, 2)
            self.assertTrue((Path(tmp) / "rank-0/record-00001.json").is_file())
            recorder.arm("missing", "prompt-0", 2)
            target.fire(Tensor(1))
            with self.assertRaisesRegex(RuntimeError, "Incomplete boundary trace"):
                recorder.finish()
            recorder.close()
            for handle in target.handles + draft.handles:
                handle.remove.assert_called_once()

    def test_cli_allows_apc_async_compile_and_rejects_unsupported_combinations(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"architectures": [run.ULTRA_ARCHITECTURE],
                "num_hidden_layers": 2, "num_nextn_predict_layers": 3, "vocab_size": 32}))
            operator = root / "operator.json"
            operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
            args = ["--model", str(root), "--operator-config", str(operator), "--output", str(root / "out"),
                    "--engine", "omni", "--temperature", "0", "--trace-mtp-boundary"]
            def plan(*flags):
                return run.make_plan(run.parser().parse_args([*args, *flags]))
            reference = plan("--mode", "compile-acl", "--enable-prefix-caching", "--async-scheduling",
                             "--prefill-chunk-size", "2048", "--kv-cache-dtype", "hif8_ds_mla")
            self.assertTrue(reference["options"]["trace_mtp_boundary"])
            for flags in (("--engine", "direct"), ("--no-mtp",), ("--temperature", "1"),
                          ("--trace-target",), ("--profile",)):
                with self.assertRaises(ValueError):
                    plan(*flags)


if __name__ == "__main__":
    unittest.main()
