"""Padded buffers, logical query metadata and graph-task ownership without NPU."""

from collections import namedtuple
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.cache import MetadataLane
from vllm_bench.pangu_infer.execution import Execution
from vllm_bench.pangu_infer import run
from vllm_bench.pangu_infer.runtime import DirectEngine, engine_options


class Buffer:
    """Mutable slice views for checking buffer writes, with no model arithmetic."""
    def __init__(self, values, *, storage=None, indices=None):
        self.storage = deepcopy(values) if storage is None else storage
        self.indices = list(range(len(self.storage))) if indices is None else indices

    @property
    def values(self):
        return [self.storage[index] for index in self.indices]

    @property
    def shape(self):
        values = self.values
        return (len(values), len(values[0])) if values and isinstance(values[0], list) else (len(values),)

    def __getitem__(self, index):
        if not isinstance(index, slice):
            return self.values[index]
        return Buffer([], storage=self.storage, indices=self.indices[index])

    def copy_(self, other):
        values = deepcopy(other.values)
        if len(values) != len(self.indices):
            raise ValueError("Buffer copy size mismatch")
        for index, value in zip(self.indices, values):
            self.storage[index] = value
        return self

    def fill_(self, value):
        for index in self.indices:
            old = self.storage[index]
            self.storage[index] = [value] * len(old) if isinstance(old, list) else value
        return self

    def zero_(self):
        return self.fill_(0)

    def add_(self, value):
        for index in self.indices:
            self.storage[index] += value
        return self

    def to(self, *args, **kwargs):
        return Buffer(self.values)


class Module:
    def eval(self):
        return self


class Builder:
    def __init__(self, spec, names, cfg, device):
        self.calls = []

    def build_for_drafting(self, *, common_attn_metadata, draft_index, **kwargs):
        self.calls.append((draft_index, kwargs))
        return common_attn_metadata


def fixture():
    groups = [SimpleNamespace(name=name, backend=SimpleNamespace(get_builder_cls=lambda: Builder), spec=SimpleNamespace(block_size=128),
                              block_table=Buffer([[1]]), mome=mome)
              for name, mome in (("draft.attn", False), ("draft.mome", True))]
    cfg = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=2), dtype="bf16"))
    cache = SimpleNamespace(cfg=cfg, device="npu:0", groups=groups)
    cache.lane = lambda names, size, **kwargs: MetadataLane(cache, names, size, **kwargs)
    def empty(*shape, **kwargs):
        values = [99] * shape[0] if len(shape) == 1 else [[99] * shape[1] for _ in range(shape[0])]
        return Buffer(values)
    torch = SimpleNamespace(empty=empty, tensor=lambda values, **kwargs: Buffer(values),
                            arange=lambda size, **kwargs: Buffer(list(range(size))),
                            int32="i32", int64="i64", nn=SimpleNamespace(Module=Module))
    modules = {"torch": torch, "vllm.v1.attention.backend": SimpleNamespace(CommonAttentionMetadata=SimpleNamespace)}
    return cache, modules


class PaddingMetadataTests(unittest.TestCase):
    def test_mtp_lane_uses_its_own_parallel_config(self):
        cache, modules = fixture()
        draft_cfg = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2))
        with patch.dict(sys.modules, modules):
            lane = MetadataLane(cache, ["draft.attn"], 8, cfg=draft_cfg)
        self.assertIs(lane.cfg, draft_cfg)
        self.assertIsNot(lane.cfg, cache.cfg)

    def test_single_query_uses_one_position_and_masks_all_tail_slots(self):
        cache, modules = fixture()
        with patch.dict(sys.modules, modules):
            lane = MetadataLane(cache, [group.name for group in cache.groups], 8, query_size=1)
            metadata, slots = lane.prepare(8192, 3, 8192, draft_index=1)
            self.assertEqual(lane.positions.values, [8192, 0, 0, 0, 0, 0, 0, 0])
            self.assertEqual(lane.q_cpu.values, [0, 1])
            self.assertEqual(lane.q.values, [0, 1])
            self.assertEqual(lane.seq.values, [8193])
            self.assertEqual(lane.seq_cpu.values, [8193])
            self.assertEqual(lane.computed_cpu.values, [8192])
            self.assertEqual(slots["draft.attn"].values, [8320, -1, -1, -1, -1, -1, -1, -1])
            self.assertEqual(slots["draft.mome"].values, [0, -1, -1, -1, -1, -1, -1, -1])
            for common in metadata.values():
                self.assertEqual(common.max_query_len, 1)
                self.assertEqual(common.num_actual_tokens, 8)
                self.assertEqual(common.max_seq_len, 8193)
                self.assertEqual(common.num_reqs, 1)
            lane.positions.fill_(42)
            lane.prepare(8193, None, 8192, draft_index=2)
            self.assertEqual(lane.positions.values, [8193, 0, 0, 0, 0, 0, 0, 0])
            self.assertIsNone(lane.entries[1][1].calls[-1][1]["num_accepted_tokens"])

    def test_default_packed_query_keeps_all_positions_and_slots(self):
        cache, modules = fixture()
        with patch.dict(sys.modules, modules):
            lane = MetadataLane(cache, ["draft.attn"], 8)
            metadata, slots = lane.prepare(8192, 8, 8192, draft_index=0)
            self.assertEqual(lane.positions.values, list(range(8192, 8200)))
            self.assertEqual(lane.q_cpu.values, [0, 8])
            self.assertEqual(lane.seq.values, [8200])
            self.assertEqual(metadata["draft.attn"].max_query_len, 8)
            self.assertEqual(slots["draft.attn"].values, list(range(8320, 8328)))


class PaddingExecutionTests(unittest.TestCase):
    def test_buffers_are_zero_padded_and_only_valid_hidden_rows_are_returned(self):
        cache, modules = fixture()
        context = SimpleNamespace(capturing=False)
        descriptors = []
        @contextmanager
        def forward_context(**kwargs):
            descriptors.append(kwargs)
            yield
        modules.update({
            "vllm.config": SimpleNamespace(CUDAGraphMode=SimpleNamespace(NONE="NONE", FULL="FULL")),
            "vllm.forward_context": SimpleNamespace(set_forward_context=forward_context,
                get_forward_context=lambda: context, BatchDescriptor=lambda **kwargs: SimpleNamespace(**kwargs)),
        })
        with patch.dict(sys.modules, modules):
            execution = Execution(object(), cache.cfg, cache, ["draft.attn", "draft.mome"], 8, "npu:0",
                                  head=0, query_size=1)
            outputs = Buffer([[i, i + 100] for i in range(8)])
            logits = object()
            def runnable(ids, positions, hidden, sample_index):
                self.assertEqual(ids.values[1:], [0] * 7)
                self.assertEqual(positions.values[1:], [0] * 7)
                self.assertEqual(hidden.values[1:], [[0, 0]] * 7)
                self.assertEqual(sample_index.values, [0])
                return outputs, logits
            execution.runnable = Mock(side_effect=runnable)
            for token in (71, 280):
                execution.ids.fill_(99)
                execution.hidden.fill_(99)
                hidden, result = execution(Buffer([token]), 8192, 8, 8192,
                                           hidden=Buffer([[1, 2]]), sample_index=Buffer([0]), draft_index=1)
                self.assertEqual(execution.ids.values, [token] + [0] * 7)
                self.assertEqual(execution.hidden.values[0], [1, 2])
                self.assertEqual(hidden.values, [[0, 100]])
                self.assertIs(result, logits)
            execution.ids[:1].copy_(Buffer([17]))
            execution.hidden[:1].copy_(Buffer([[21, 22]]))
            execution(execution.ids[:1], 8193, None, 8192, hidden=execution.hidden[:1],
                      sample_index=Buffer([0]), draft_index=2)
            self.assertEqual(execution.ids.values, [17] + [0] * 7)
            self.assertEqual(execution.hidden.values[0], [21, 22])
            self.assertEqual(execution.calls, 3)
            self.assertTrue(all(item["num_tokens"] == 8 and item["batch_descriptor"].num_tokens == 8
                                for item in descriptors))

    def test_engine_separates_packed_and_padded_lanes_and_reuses_continuations(self):
        engine = DirectEngine.__new__(DirectEngine)
        engine.a = {"depth": 7, "context_length": 8192, "profile": False, "mode": "compile-acl", "mtp_pad_to_8": True}
        engine.plan = {"config": {"num_hidden_layers": 12, "num_nextn_predict_layers": 3}}
        engine.names = {"target": ["model.layers.0.attn"], "draft": ["draft_model.model.layers.12.attn"]}
        engine.executions, engine.model_timings = {}, None
        engine.cfg, engine.caches, engine.device = object(), object(), "npu:0"
        engine.target, engine.draft = object(), object()
        with patch("vllm_bench.pangu_infer.execution.Execution") as execution:
            engine.forward("draft", Buffer(list(range(8))), 8192, 8, draft_index=0)
            engine.forward("draft", Buffer([71]), 8193, 1, draft_index=1)
            engine.forward("draft", Buffer([280]), 8194, None, draft_index=2)
            engine.forward("target", Buffer(list(range(8))), 8192, 8)
            self.assertEqual(execution.call_count, 3)
            self.assertEqual([call.args[4] for call in execution.call_args_list], [8, 8, 8])
            self.assertEqual([call.kwargs["query_size"] for call in execution.call_args_list], [8, 1, 8])
            self.assertTrue(all(call.kwargs["isolate_graph_tasks"] for call in execution.call_args_list))
            engine.a["mtp_pad_to_8"] = False
            engine.forward("draft", Buffer([71]), 8193, 1, draft_index=1)
            self.assertEqual(execution.call_args.args[4], 1)
            engine.a.update(mtp_pad_to_8=True, depth=3)
            engine.forward("draft", Buffer([71]), 8193, 1, head=1, draft_index=1)
            self.assertEqual(execution.call_args.args[4], 1)

    def test_mixed_tp_isolates_both_models_graphs_without_padding(self):
        engine = DirectEngine.__new__(DirectEngine)
        engine.a = {"depth": 7, "context_length": 8192, "profile": False, "mode": "compile-acl"}
        engine.plan = {"config": {"num_hidden_layers": 12, "num_nextn_predict_layers": 3}}
        engine.names = {"target": ["model.layers.0.attn"], "draft": ["draft_model.model.layers.12.attn"]}
        engine.executions, engine.model_timings = {}, None
        engine.cfg, engine.caches, engine.device = object(), object(), "npu:0"
        engine.target, engine.draft = object(), object()
        engine.mtp = SimpleNamespace(cfg=object())
        with patch("vllm_bench.pangu_infer.execution.Execution") as execution:
            engine.forward("target", Buffer(list(range(8))), 8192, 1)
            engine.forward("draft", Buffer(list(range(8))), 8192, 1, draft_index=0)
            self.assertEqual(execution.call_count, 2)
            self.assertTrue(all(call.kwargs["isolate_graph_tasks"] for call in execution.call_args_list))
            self.assertIs(execution.call_args_list[0].args[1], engine.cfg)
            self.assertIs(execution.call_args_list[1].args[1], engine.mtp.cfg)

    def test_cli_only_allows_direct_mtp_and_registers_size_eight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"architectures": ["PanguUltraMoEForCausalLM"],
                "num_hidden_layers": 2, "num_nextn_predict_layers": 1, "vocab_size": 32}))
            operator = root / "operator.json"
            operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
            args = ["--model", str(root), "--operator-config", str(operator), "--output", str(root / "out"),
                    "--depth", "3", "--mtp-pad-to-8"]
            for mode in run.MODES:
                plan = run.make_plan(run.parser().parse_args([*args, "--mode", mode]))
                self.assertTrue(plan["options"]["mtp_pad_to_8"])
                self.assertEqual(engine_options(plan, direct=True)["compilation_config"]["cudagraph_capture_sizes"], [1, 4, 8])
            for flags in (["--engine", "omni"], ["--no-mtp"]):
                with self.assertRaisesRegex(ValueError, "positive depth"):
                    run.make_plan(run.parser().parse_args([*args, *flags]))


class GraphTaskOwnershipTests(unittest.TestCase):
    def test_equal_size_graphs_keep_their_own_handles_workspaces_and_events(self):
        descriptor = namedtuple("Descriptor", "num_tokens")(8)
        context = SimpleNamespace(batch_descriptor=descriptor, cudagraph_runtime_mode="FULL", attn_metadata={}, value=1)
        original_tasks = {"other_model": object()}
        original_workspaces = {"workspace": "other-model-workspace"}
        params = SimpleNamespace(task_entries={8: original_tasks}, workspaces={8: original_workspaces})
        operations = []
        def capture(label):
            op = SimpleNamespace(compute_dynamic_kwargs=lambda context, name, cfg: {"value": context.value},
                                 workspace_fn="workspace", op_out_fn=Mock())
            return SimpleNamespace(op_desc=op, captured_kwargs={"input_layout": "BSND", "actual_seq_lengths": [8]},
                                   out_tensors=[label], handle=label, event=Mock())
        class Base:
            def __init__(self, label):
                self.label, self.runtime_mode, self.vllm_config = label, "FULL", object()
                self.attn_layer_names, self.concrete_aclgraph_entries = ["draft.attn"], {}
            def __call__(self):
                if context.cudagraph_runtime_mode != self.runtime_mode:
                    return self.label
                if descriptor not in self.concrete_aclgraph_entries:
                    params.task_entries[8]["draft.attn"] = capture(self.label)
                    # Match Omni: reuse the first workspace for this op and
                    # token count even if a later graph needs a different size.
                    params.workspaces[8].setdefault("workspace", self.label + "-workspace")
                    if self.label == "fail":
                        raise RuntimeError("capture failed")
                    self.concrete_aclgraph_entries[descriptor] = SimpleNamespace(aclgraph=object())
                else:
                    self._update_graph_tasks("stream", context)
                return self.label
            def reset_captured_graphs(self):
                self.concrete_aclgraph_entries.clear()
        torch = SimpleNamespace(npu=SimpleNamespace(stream=lambda stream: nullcontext(),
            graph_task_update_begin=lambda stream, handle: operations.append(handle), graph_task_update_end=Mock()))
        modules = {"torch": torch, "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: context),
                   "omni_npu.compilation.acl_graph": SimpleNamespace(ACLGraphWrapper=Base, get_graph_params=lambda: params)}
        name = "vllm_bench.pangu_infer.model.acl_graph"
        with patch.dict(sys.modules, modules):
            # Fresh import binds the stand-in base only for this test.
            sys.modules.pop(name, None)
            module = importlib.import_module(name)
            try:
                packed, padded = module.DirectACLGraphWrapper("packed"), module.DirectACLGraphWrapper("padded")
                packed()
                first_task = packed.owned_tasks[descriptor]["draft.attn"]
                padded()
                second_task = padded.owned_tasks[descriptor]["draft.attn"]
                self.assertIs(params.task_entries[8], original_tasks)
                self.assertIs(params.workspaces[8], original_workspaces)
                self.assertIsNot(first_task, second_task)
                context.value = 2
                packed()
                padded()
                self.assertEqual(operations, ["packed", "padded"])
                self.assertEqual(first_task.op_desc.op_out_fn.call_args.kwargs["workspace"], "packed-workspace")
                self.assertEqual(second_task.op_desc.op_out_fn.call_args.kwargs["workspace"], "padded-workspace")
                failed = module.DirectACLGraphWrapper("fail")
                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    failed()
                self.assertEqual(failed.owned_tasks, {})
                self.assertEqual(failed.owned_workspaces, {})
                self.assertIs(params.task_entries[8], original_tasks)
                self.assertIs(params.workspaces[8], original_workspaces)
                self.assertIsNone(first_task.op_desc.op_out_fn.call_args.kwargs["actual_seq_lengths"])
                self.assertEqual(first_task.op_desc.op_out_fn.call_args.kwargs["value"], 2)
                first_task.event.record.assert_called_once_with("stream")
                second_task.event.record.assert_called_once_with("stream")
                packed.reset_captured_graphs()
                self.assertEqual(packed.owned_tasks, {})
                self.assertEqual(packed.owned_workspaces, {})
                packed()
                self.assertIsNot(packed.owned_tasks[descriptor]["draft.attn"], first_task)
                context.cudagraph_runtime_mode = "NONE"
                eager = module.DirectACLGraphWrapper("eager")
                eager()
                self.assertEqual(eager.owned_tasks, {})
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
