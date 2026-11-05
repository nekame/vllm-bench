"""Model-call timing boundaries and decode phases without torch or NPU."""

import contextlib
from copy import deepcopy
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.execution import Execution
from vllm_bench.pangu_infer.runtime import DirectEngine
from vllm_bench.pangu_infer.model_timing import (
    ModelTimings, reduce_model_timings, log_model_timings,
    summarize_model_timings, log_model_timing_summary,
)


class Tensor:
    """Small buffer stand-in for control-flow tests, with no model arithmetic."""
    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return Tensor(self.values[index]) if isinstance(index, slice) else self.values[index]

    def reshape(self, *shape):
        return self

    def fill_(self, value):
        self.values = [value] * len(self.values)
        return self

    def scatter_(self, dim, indices, values):
        for index, value in zip(indices.values, values.values):
            self.values[index] = value
        return self

    def index_select(self, dim, indices):
        return Tensor(self.values[index] for index in indices.values)

    def item(self):
        return self.values[0]

    def tolist(self):
        return self.values.copy()


def fake_torch():
    return SimpleNamespace(
        tensor=lambda values, **kwargs: Tensor(values),
        cat=lambda tensors: Tensor(value for tensor in tensors for value in tensor.values),
        roll=lambda tensor, shift: Tensor(tensor.values[-shift:] + tensor.values[:-shift]),
        zeros_like=lambda tensor: Tensor([0] * len(tensor)),
        int32="int32", int64="int64", float64="float64",
        npu=SimpleNamespace(synchronize=Mock()),
        distributed=SimpleNamespace(all_reduce=Mock(), ReduceOp=SimpleNamespace(MAX="MAX")))


class ModelCallBoundaryTests(unittest.TestCase):
    def test_only_completed_model_step_is_timed_in_eager_and_acl(self):
        for graph in (False, True):
            for head in (None, 1):
                for enabled in (False, True):
                    with self.subTest(graph=graph, head=head, enabled=enabled):
                        events = []
                        torch = fake_torch()
                        torch.npu.synchronize.side_effect = lambda: events.append("sync")
                        @contextlib.contextmanager
                        def context(**kwargs):
                            events.append("context")
                            yield
                            events.append("leave_context")
                        modules = {
                            "torch": torch,
                            "vllm.config": SimpleNamespace(CUDAGraphMode=SimpleNamespace(NONE="NONE", FULL="FULL")),
                            "vllm.forward_context": SimpleNamespace(
                                set_forward_context=context,
                                get_forward_context=lambda: SimpleNamespace(capturing=False),
                                BatchDescriptor=lambda **kwargs: tuple(kwargs.values())),
                        }
                        execution = Execution.__new__(Execution)
                        execution.profile, execution.cfg = False, object()
                        execution.head, execution.size = head, 8
                        execution.query_size = 8
                        execution.ids = SimpleNamespace(copy_=lambda value: events.append("copy_ids"))
                        execution.hidden = (SimpleNamespace(copy_=lambda value: events.append("copy_hidden"))
                                            if head is not None else None)
                        execution.sample_index = SimpleNamespace(copy_=lambda value: events.append("copy_index"))
                        def prepare(*args):
                            events.append("metadata")
                            return {}, {}
                        execution.lane = SimpleNamespace(prepare=prepare, positions=object())
                        execution.calls, execution.captures, execution.replays = 0, 0, 0
                        execution.wrapper = (SimpleNamespace(concrete_aclgraph_entries=SimpleNamespace(
                            get=lambda descriptor: SimpleNamespace(aclgraph=object()))) if graph else None)
                        result = object()
                        def model(*args):
                            events.append("replay_update" if graph else "model")
                            return result
                        execution.runnable = model
                        timing = ModelTimings()
                        timing.cycle = 3 if enabled else None
                        ticks = iter((1_000_000, 3_500_000))
                        def clock():
                            events.append("clock")
                            return next(ticks)
                        output = io.StringIO()
                        with patch.dict(sys.modules, modules), \
                                patch("vllm_bench.pangu_infer.model_timing.time.perf_counter_ns", side_effect=clock), \
                                contextlib.redirect_stdout(output):
                            actual = execution(object(), 8192, 1, 8192, hidden=object(),
                                               sample_index=object(), draft_index=2, model_timing=timing)
                        self.assertIs(actual, result)
                        self.assertEqual(output.getvalue(), "")
                        call = "replay_update" if graph else "model"
                        if enabled:
                            self.assertEqual(events[events.index("context") + 1:],
                                             ["sync", "clock", call, "sync", "clock", "leave_context"])
                            self.assertEqual(timing.records[0]["wall_ms"], 2.5)
                            self.assertEqual(timing.records[0]["input_tokens"], 8)
                            self.assertEqual(timing.records[0]["model"], "target" if head is None else "mtp")
                            if head is not None:
                                self.assertEqual((timing.records[0]["step"], timing.records[0]["head"]), (2, 1))
                        else:
                            self.assertNotIn("sync", events)
                            self.assertEqual(timing.records, [])
                        self.assertLess(events.index("metadata"), events.index(call))

    def test_failed_call_does_not_publish_a_duration(self):
        timing = ModelTimings()
        timing.cycle = 0
        with patch.dict(sys.modules, {"torch": fake_torch()}), self.assertRaisesRegex(RuntimeError, "model failed"):
            timing.measure(Mock(side_effect=RuntimeError("model failed")), (), model="target", input_tokens=1)
        self.assertEqual(timing.records, [])


class DecodeTimingPhaseTests(unittest.TestCase):
    def engine(self, depth):
        engine = DirectEngine.__new__(DirectEngine)
        engine.a = {"context_length": 4, "depth": depth, "temperature": 0, "respect_eos": False}
        engine.plan = {"prompt_ids": [1, 2, 3, 4],
                       "config": {"num_nextn_predict_layers": 3, "eos_token_id": 99}}
        engine.prompt = Tensor([1, 2, 3, 4])
        engine.device = "npu:0"
        engine.caches = SimpleNamespace(reset=Mock())
        engine.sampler = SimpleNamespace(
            reset=Mock(), distribution=lambda logits: logits, sample=lambda scores: Tensor([0]),
            verify=lambda p, q, drafts: (Tensor([0] * (depth + 1)), Tensor([depth])))
        calls = []
        def forward(self, kind, ids, start, accepted, *, hidden=None, index=None, head=0, draft_index=None):
            timing = self.model_timings
            calls.append((kind, len(ids), start, timing.cycle if timing else None))
            def model():
                return Tensor([0] * len(ids)), Tensor([0])
            if timing is not None and timing.cycle is not None:
                return timing.measure(model, (), model="target" if kind == "target" else "mtp",
                                      input_tokens=len(ids), step=draft_index, head=head)
            return model()
        engine.forward = MethodType(forward, engine)
        return engine, calls

    def test_prefill_sampler_count_drives_initial_mtp_and_first_target_verify(self):
        for depth in (0, 3, 7):
            with self.subTest(depth=depth):
                engine, _ = self.engine(depth)
                calls = []
                forward = engine.forward
                def checked_forward(self, kind, ids, start, accepted, **kwargs):
                    calls.append((kind, start, accepted, kwargs.get("draft_index")))
                    return forward(kind, ids, start, accepted, **kwargs)
                engine.forward = MethodType(checked_forward, engine)
                with patch.dict(sys.modules, {"torch": fake_torch()}):
                    engine.generate(2 * (depth + 1) + 1)
                target = [entry for entry in calls if entry[0] == "target"]
                self.assertIsNone(target[0][2])  # No previous sampler before prefill.
                self.assertEqual(target[1][2], 1)  # One output token from prefill.
                self.assertTrue(all(entry[2] == depth + 1 for entry in target[2:]))
                first_verify = calls.index(target[1])
                initial_draft = calls[1:first_verify]
                expected = [] if depth == 0 else [1] * 3 if depth == 3 else [1, 1, None, None, None, None, None]
                self.assertEqual([entry[2] for entry in initial_draft], expected)

    def test_warmup_prefill_and_initial_mtp_are_excluded(self):
        for depth in (0, 3, 7):
            with self.subTest(depth=depth):
                engine, calls = self.engine(depth)
                torch = fake_torch()
                ticks = itertools.count(step=1_000_000)
                console = io.StringIO()
                with patch.dict(sys.modules, {"torch": torch}), \
                        patch("vllm_bench.pangu_infer.model_timing.time.perf_counter_ns", side_effect=lambda: next(ticks)) as clock, \
                        contextlib.redirect_stdout(console):
                    limit = 2 * (depth + 1) + 1
                    warmup = engine.generate(limit)
                    self.assertNotIn("model_timings", warmup)
                    self.assertEqual(clock.call_count, 0)
                    self.assertEqual(torch.npu.synchronize.call_count, 3)
                    calls.clear()
                    torch.npu.synchronize.reset_mock()
                    record = engine.generate(limit, model_timing=True)
                self.assertEqual(console.getvalue(), "")
                rows = record["model_timings"]
                targets = [row for row in rows if row["model"] == "target"]
                mtp = [row for row in rows if row["model"] == "mtp"]
                self.assertEqual([row["input_tokens"] for row in targets], [depth + 1, depth + 1])
                self.assertEqual([row["cycle"] for row in targets], [0, 1])
                self.assertEqual(len(mtp), depth)
                self.assertEqual([row["step"] for row in mtp], list(range(depth)))
                if depth:
                    self.assertTrue(all(row["cycle"] == 1 for row in mtp))
                    self.assertEqual([row["head"] for row in mtp], list(range(3)) if depth == 3 else [0] * 7)
                    self.assertEqual([row["input_tokens"] for row in mtp], [4] * 3 if depth == 3 else [8] + [1] * 6)
                if depth == 7:
                    # These six initial MTP calls look like decode by shape and
                    # position, but still belong to initial prefill.
                    self.assertEqual(sum(kind == "draft" and size == 1 and start >= 4 and cycle is None
                                         for kind, size, start, cycle in calls), 6)
                self.assertEqual(torch.npu.synchronize.call_count, 3 + 2 * len(rows))
                self.assertIsNone(engine.model_timings.cycle)

    def test_initial_eos_has_no_decode_model_calls(self):
        engine, _ = self.engine(7)
        engine.a["respect_eos"] = True
        engine.plan["config"]["eos_token_id"] = 0
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            record = engine.generate(17, model_timing=True)
        self.assertEqual(record["model_timings"], [])
        self.assertEqual(record["decode_tokens"], 0)


class TimingAggregationTests(unittest.TestCase):
    def rows(self):
        return [{"cycle": 0, "model": "target", "input_tokens": 8, "wall_ms": 2.0},
                {"cycle": 1, "model": "mtp", "input_tokens": 1, "step": 1, "head": 0, "wall_ms": 3.0}]

    def test_tp_max_is_reduced_after_counts_check(self):
        rows = self.rows()
        torch = fake_torch()
        reductions = []
        def reduce(tensor, op):
            reductions.append(tensor.tolist())
            tensor.values = [2, -2] if len(reductions) == 1 else [5.0, 4.0]
        torch.distributed.all_reduce.side_effect = reduce
        console = io.StringIO()
        with patch.dict(sys.modules, {"torch": torch}), contextlib.redirect_stdout(console):
            reduce_model_timings(rows, "npu:0", 8)
        self.assertEqual(reductions, [[2, -2], [2.0, 3.0]])
        self.assertEqual([row["wall_ms_max_tp"] for row in rows], [5.0, 4.0])
        self.assertTrue(all("wall_ms" not in row for row in rows))
        self.assertEqual(console.getvalue(), "")

    def test_count_mismatch_fails_before_reducing_durations(self):
        torch = fake_torch()
        torch.distributed.all_reduce.side_effect = lambda tensor, **kwargs: setattr(tensor, "values", [3, -2])
        with patch.dict(sys.modules, {"torch": torch}), self.assertRaisesRegex(RuntimeError, "call counts differ"):
            reduce_model_timings(self.rows(), "npu:0", 2)
        self.assertEqual(torch.distributed.all_reduce.call_count, 1)

    def test_empty_request_does_not_reduce_an_empty_tensor(self):
        torch = fake_torch()
        with patch.dict(sys.modules, {"torch": torch}):
            reduce_model_timings([], "npu:0", 2)
        self.assertEqual(torch.distributed.all_reduce.call_count, 1)

    def test_rank_zero_log_and_grouped_summary_format(self):
        rows = self.rows()
        reduce_model_timings(rows, "npu:0", 1)
        extra = {"cycle": 1, "model": "mtp", "input_tokens": 8,
                 "step": 0, "head": 0, "wall_ms_max_tp": 8.0}
        records = [{"model_timings": [*rows, extra]}, {"model_timings": [deepcopy(rows[0])]}]
        summary = summarize_model_timings(records)
        target = next(group for group in summary if group["model"] == "target")
        self.assertEqual((target["count"], target["median_ms"], target["min_ms"], target["max_ms"]), (2, 2, 2, 2))
        self.assertEqual(len(summary), 3)  # MTP 1 and 8 input tokens stay separate.
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            log_model_timings(rows, 2)
            log_model_timing_summary(summary)
        text = console.getvalue()
        self.assertIn("repeat=2 cycle=0 model=target input_tokens=8 wall_ms_max_tp=2.000", text)
        self.assertIn("model=mtp step=1 head=0 input_tokens=1 wall_ms_max_tp=3.000", text)
        self.assertIn("aggregation=max_tp", text)


class TimingResultTests(unittest.TestCase):
    def test_main_reduces_each_measured_request_and_logs_only_on_rank_zero(self):
        from vllm_bench.pangu_infer import runtime
        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as tmp:
                operations = []
                def generate(*args, **kwargs):
                    timed = kwargs.get("model_timing", False)
                    operations.append("measured" if timed else "warmup")
                    record = {"token_ids": [0, 1], "decode_seconds": .25, "decode_tokens": 1}
                    if timed:
                        record["model_timings"] = [{"cycle": 0, "model": "target",
                                                    "input_tokens": 1, "wall_ms": 2.0}]
                    return record
                engine = SimpleNamespace(
                    generate=Mock(side_effect=generate), executions={}, caches=SimpleNamespace(bytes=128),
                    counters=lambda: {"compiles": 0, "captures": 0, "replays": 0, "calls": 4})
                torch = fake_torch()
                torch.no_grad = contextlib.nullcontext
                torch.distributed.is_initialized = lambda: False
                torch.npu.max_memory_allocated = lambda device: 1024
                torch.npu.get_device_name = lambda device: "NPU"
                cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=128, mamba_block_size=128),
                                      compilation_config=SimpleNamespace(mode=0, cudagraph_mode="NONE"))
                context = SimpleNamespace(__exit__=Mock())
                extra = SimpleNamespace(operator_opt_config={}, parall_config={})
                path = Path(tmp) / "plan.json"
                path.write_text(json.dumps({"options": {"output": tmp, "mode": "eager", "depth": 0,
                                                       "warmup": 2, "repeats": 2, "max_new_tokens": 32,
                                                       "profile": False, "model_timing": True, "tp": 2}}))
                def reduce(rows, device, tp):
                    operations.append("reduce")
                    self.assertEqual(tp, 2)
                    for entry in rows:
                        entry["wall_ms_max_tp"] = entry.pop("wall_ms")
                with patch.dict(sys.modules, {"torch": torch}), \
                        patch.object(runtime, "setup", return_value=(cfg, "npu:0", rank, context, extra)), \
                        patch.object(runtime, "load_models", return_value=(object(), None, {})), \
                        patch.object(runtime, "DirectEngine", return_value=engine), \
                        patch("vllm_bench.pangu_infer.model_timing.reduce_model_timings", side_effect=reduce) as reduction, \
                        patch("vllm_bench.pangu_infer.model_timing.log_model_timings",
                              side_effect=lambda *args: operations.append("log")) as log, \
                        patch("vllm_bench.pangu_infer.model_timing.log_model_timing_summary") as summary_log, \
                        patch("vllm_bench.pangu_infer.results.versions", return_value={}), \
                        patch("vllm_bench.pangu_infer.results.save_result") as save, \
                        contextlib.redirect_stdout(io.StringIO()):
                    runtime.main(["--plan", str(path)])
                expected = ["warmup", "warmup"]
                expected += ["measured", "reduce", "log"] * 2 if rank == 0 else ["measured", "reduce"] * 2
                self.assertEqual(operations, expected)
                self.assertEqual(reduction.call_count, 2)
                self.assertEqual(log.call_count, 2 if rank == 0 else 0)
                self.assertEqual(summary_log.call_count, 1 if rank == 0 else 0)
                self.assertEqual(save.call_count, 1 if rank == 0 else 0)
                if rank == 0:
                    self.assertEqual(summary_log.call_args.args[0][0]["count"], 2)

    def test_timing_results_are_diagnostic_and_keep_samples_and_summary(self):
        from vllm_bench.pangu_infer.results import save_result
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                plan = {"options": {"engine": "direct", "profile": False, "depth": 0, "model_timing": enabled},
                        "comparison_sha256": "same", "prompt_sha256": "prompt", "model_scope": "full",
                        "checkpoint_identity": {}}
                record = {"token_ids": [1, 2], "decode_tokens": 1, "decode_seconds": .1}
                if enabled:
                    record["model_timings"] = [{"cycle": 0, "model": "target", "input_tokens": 1,
                                                "wall_ms_max_tp": 10.0}]
                result = save_result(Path(tmp), plan, [record], {}, {})
                saved = json.loads((Path(tmp) / "result.json").read_text())
                self.assertEqual(saved, result)
                self.assertEqual(result["measurement"], "diagnostic" if enabled else "benchmark")
                if enabled:
                    self.assertEqual(result["model_timing_summary"][0]["median_ms"], 10.0)
                    self.assertEqual(result["records"][0]["model_timings"], record["model_timings"])
                else:
                    self.assertNotIn("model_timing_summary", result)


if __name__ == "__main__":
    unittest.main()
