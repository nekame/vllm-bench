"""Target trace lifecycle and replay protocol tests without NPU dependencies."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer import run, target_trace as trace
from vllm_bench.pangu_infer.trace_compare import metadata_differences


class Tensor:
    """CPU container for hook/protocol tests; does not emulate model arithmetic."""
    def __init__(self, values):
        self.values = deepcopy(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def clone(self):
        return Tensor(self.values)

    def tolist(self):
        return deepcopy(self.values)

    @property
    def shape(self):
        value, shape = self.values, []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else None
        return tuple(shape)

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        total = 1
        for size in self.shape:
            total *= size
        return total

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def item(self):
        return self.values

    def to(self, *args, **kwargs):
        return self

    def reshape(self, *shape):
        return self

    def index_select(self, dim, indices):
        return Tensor([self.values[index] for index in indices.values])


class Module:
    def __init__(self, forward):
        self.forward, self.pre, self.post = forward, [], []

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        self.pre.append(hook)
        return SimpleNamespace(remove=lambda: self.pre.remove(hook))

    def register_forward_hook(self, hook):
        self.post.append(hook)
        return SimpleNamespace(remove=lambda: self.post.remove(hook))

    def __call__(self, *args, **kwargs):
        for hook in self.pre:
            hook(self, args, kwargs)
        value = self.forward(*args, **kwargs)
        for hook in self.post:
            hook(self, args, value)
        return value


def record(start=4, accepted=1):
    return {"input_ids": Tensor([3, 4]), "positions": Tensor([start, start + 1]),
            "metadata": {"model.layers.0.attn.mome": {"common": {
                "query_start_loc": [0, 2], "num_computed_tokens": [start],
                "num_accepted_tokens": [accepted]}}}}


class TraceProtocolTests(unittest.TestCase):
    def test_layer_selection_rejects_duplicates_and_bad_indices(self):
        self.assertEqual(trace.parse_layers("all", 3), [0, 1, 2])
        self.assertEqual(trace.parse_layers("", 3), [])
        self.assertEqual(trace.parse_layers("0,2", 3), [0, 2])
        for value in ("0,0", "-1", "3", "x", "0,"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                trace.parse_layers(value, 3)

    def test_replay_uses_source_acceptance_and_rejects_unsupported_positions(self):
        self.assertEqual(trace.replay_parameters(record()), (4, 1))
        self.assertEqual(trace.replay_parameters(record(accepted=8)), (4, 8))
        value = record()
        value["positions"] = Tensor([4, 6])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            trace.replay_parameters(value)
        value = record()
        value["metadata"]["model.layers.0.attn.mome"]["common"]["query_start_loc"] = [0, 1, 2]
        with self.assertRaisesRegex(ValueError, "query boundaries"):
            trace.replay_parameters(value)
        value = record()
        value["metadata"]["model.layers.0.attn.mome"]["common"]["num_computed_tokens"] = [3]
        with self.assertRaisesRegex(ValueError, "computed-token"):
            trace.replay_parameters(value)

    def test_proposal_history_uses_source_cursor_including_partial_and_full_acceptance(self):
        self.assertEqual(trace.proposal_parameters(1, 8192, 0, 8192), (8191, 1))
        self.assertEqual(trace.proposal_parameters(2, 8193, 8192, 8), (0, 1))
        self.assertEqual(trace.proposal_parameters(3, 8196, 8193, 8), (2, 3))
        self.assertEqual(trace.proposal_parameters(4, 8204, 8196, 8), (7, 8))
        for start in (8192, 8201):
            with self.assertRaisesRegex(ValueError, "committed-token"):
                trace.proposal_parameters(2, start, 8192, 8)

    def test_metadata_reports_paths_and_missing_fields_with_a_bound(self):
        differences = metadata_differences({"layer": {"accepted": [1], "qsl": [0, 8]}},
                                           {"layer": {"accepted": [8]}})
        self.assertEqual([item["path"] for item in differences], ["layer.accepted", "layer.qsl"])
        self.assertTrue(differences[1]["missing"])
        self.assertEqual(len(metadata_differences({str(i): 0 for i in range(50)}, {}, limit=3)), 3)

    def test_cli_trace_validation_and_manifest_compatibility_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "architectures": ["PanguUltraMoEForCausalLM"], "num_hidden_layers": 2,
                "num_nextn_predict_layers": 3, "vocab_size": 32, "use_mome": True}))
            operator = root / "operator.json"
            operator.write_text(json.dumps({"model_parallel_config": {},
                                           "operator_optimization_config": {"use_noncontiguous_kv": True}}))
            args = ["--model", str(model), "--operator-config", str(operator), "--output", str(root / "out"),
                    "--engine", "omni", "--trace-target", "--repeats", "1", "--temperature", "0",
                    "--context-length", "4", "--max-new-tokens", "8", "--depth", "7", "--trace-layers", "all"]
            source = run.make_plan(run.parser().parse_args(args))
            self.assertEqual(source["trace_layers"], [0, 1])
            for flags, message in ((["--mode", "acl"], "requires"), (["--repeats", "2"], "requires"),
                                   (["--temperature", "1"], "requires"), (["--engine", "direct"], "Use"),
                                   (["--model-timing"], "requires"), (["--profile"], "cannot"),
                                   (["--replay-with-mtp"], "requires")):
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                    run.make_plan(run.parser().parse_args([*args, *flags]))
            directory = root / "reference" / "target-trace"
            rank = directory / "rank-0"
            rank.mkdir(parents=True)
            (rank / "index.json").write_text('{"calls": 2}')
            for call in range(2):
                (rank / f"call-{call:05d}.pt").touch()
            trace.write_manifest(directory, source, {}, {"hardware": "test", "versions": {}})
            candidate = deepcopy(source)
            candidate["options"].update(engine="direct", trace_target=False, replay_target=str(directory.parent))
            self.assertEqual(trace.validate_replay(candidate)[2], 2)
            candidate["options"]["depth"] = 3
            with self.assertRaisesRegex(ValueError, "depth"):
                trace.validate_replay(candidate)
            candidate["options"]["depth"] = 7
            candidate["hf_overrides"] = {"architectures": ["PanguUltraMoEForCausalLM"]}
            with self.assertRaisesRegex(ValueError, "hf_overrides"):
                trace.validate_replay(candidate)
            candidate["hf_overrides"] = {}
            (rank / "call-00001.pt").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                trace.validate_replay(candidate)


class TraceHookTests(unittest.TestCase):
    def fixture(self):
        metadata = {"model.layers.0.attn.mome": SimpleNamespace(
            num_actual_tokens=4, query_start_loc=Tensor([0, 4]),
            num_computed_tokens=Tensor([0]), num_accepted_tokens=Tensor([8]),
            cache_indices=Tensor([1]), prefill=None, decode=None)}
        cache = Tensor([[[0]], [[7]]])
        state_layer = SimpleNamespace(
            kv_cache=(cache,), get_attn_backend=lambda: SimpleNamespace(get_name=lambda: "NPUPanguMome"))
        layer = Module(lambda ids: Tensor([[token, token + 1] for token in ids.values]))
        logits = Module(lambda hidden: Tensor([[0.1, 0.9]]))
        def forward(input_ids, positions):
            result = layer(input_ids)
            cache.values[1][0][0] += 1
            return result
        model = Module(forward)
        model.model = SimpleNamespace(layers=[layer])
        model.logits_processor = logits
        cfg = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={
            "model.layers.0.attn.mome": state_layer}))
        saved = []
        def save(value, path):
            saved.append(deepcopy(value))
            Path(path).touch()
        modules = {"torch": SimpleNamespace(Tensor=Tensor, int64="int64", save=save),
                   "vllm.forward_context": SimpleNamespace(get_forward_context=lambda: SimpleNamespace(attn_metadata=metadata))}
        return model, cfg, saved, modules

    def test_hooks_snapshot_cpu_values_states_and_trim_prefill_layer_rows(self):
        model, cfg, saved, modules = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            recorder = trace.TargetRecorder(model, cfg, tmp, 0, [0])
            ids = Tensor([1, 2, 3, 4])
            hidden = model(input_ids=ids, positions=Tensor([0, 1, 2, 3]))
            model.logits_processor(hidden)
            ids.values[0] = 99
            self.assertEqual(saved[0]["input_ids"].tolist(), [1, 2, 3, 4])
            self.assertEqual(saved[0]["layers"]["0"]["0"].tolist(), [[4, 5]])
            self.assertEqual(saved[0]["states_before"]["model.layers.0.attn.mome"][0].tolist(), [[[7]]])
            self.assertEqual(saved[0]["states_after"]["model.layers.0.attn.mome"][0].tolist(), [[[8]]])
            hidden = model(Tensor([5, 6]), Tensor([4, 5]))
            model.logits_processor(hidden)
            self.assertEqual(saved[1]["layers"]["0"]["0"].shape, (2, 2))
            self.assertEqual(recorder.close(), 2)
            self.assertEqual(model.pre, [])
            self.assertEqual(model.post, [])
            self.assertEqual(model.logits_processor.post, [])
            self.assertEqual(model.model.layers[0].post, [])
            self.assertEqual(json.loads((Path(tmp) / "rank-0" / "index.json").read_text()), {"calls": 2})

    def test_missing_logits_fails_and_close_removes_hooks(self):
        model, cfg, saved, modules = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            recorder = trace.TargetRecorder(model, cfg, tmp, 0, [])
            model(Tensor([1]), Tensor([0]))
            with self.assertRaisesRegex(RuntimeError, "did not reach"):
                model(Tensor([2]), Tensor([1]))
            with self.assertRaisesRegex(RuntimeError, "no logits"):
                recorder.close()
            self.assertEqual(model.pre, [])
            self.assertEqual(model.post, [])
            self.assertEqual(model.logits_processor.post, [])


class TraceReplayTests(unittest.TestCase):
    def test_replay_resets_once_uses_reference_windows_and_never_proposes(self):
        self.check_replay(False)

    def test_optional_mtp_executes_between_forced_windows_and_preserves_target_acceptance(self):
        self.check_replay(True)

    def check_replay(self, with_mtp):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            output.mkdir()
            (output / "target-trace" / "rank-0").mkdir(parents=True)
            plan = {"options": {"engine": "direct", "context_length": 4, "depth": 7,
                                "replay_with_mtp": with_mtp},
                    "trace_layers": [], "prompt_ids": [0, 1, 2, 3]}
            records = [record(0, 8), record(4, 1), record(5, 1)]
            records[0]["input_ids"] = Tensor([0, 1, 2, 3])
            records[0]["positions"] = Tensor([0, 1, 2, 3])
            records[0]["metadata"]["model.layers.0.attn.mome"]["common"]["query_start_loc"] = [0, 4]
            cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=128, mamba_block_size=100))
            hidden_outputs = [Tensor([[index]]) for index in range(3)]
            engine = SimpleNamespace(cfg=cfg, target=object(), device="npu:0", caches=SimpleNamespace(reset=Mock()),
                                     forward=Mock(side_effect=[(hidden, None) for hidden in hidden_outputs]),
                                     propose=Mock(side_effect=[(Tensor([9, 8, 7, 6, 5, 4, 3]), None),
                                                               (Tensor([4, 3, 2, 1, 0, 9, 8]), None)]))
            extra = SimpleNamespace(operator_opt_config={}, parall_config={})
            manifest = {"versions": {}, "hardware": "NPU", "effective": {
                "operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 100}}
            recorder = Mock()
            torch = SimpleNamespace(
                load=lambda path, **kwargs: records[int(path.stem.split("-")[-1])],
                tensor=lambda values, **kwargs: Tensor(values), int64="int64",
                distributed=SimpleNamespace(get_rank=lambda: 0, barrier=Mock()),
                npu=SimpleNamespace(synchronize=Mock(), get_device_name=lambda device: "NPU"))
            cpu_group = object()
            modules = {"torch": torch,
                       "vllm.distributed": SimpleNamespace(get_tp_group=lambda: SimpleNamespace(cpu_group=cpu_group)),
                       "omni_npu.model_config.config_loader.loader": SimpleNamespace(model_extra_config=extra)}
            with patch.dict(sys.modules, modules), \
                    patch.object(trace, "validate_replay", return_value=(Path(tmp), manifest, 3)), \
                    patch.object(trace, "TargetRecorder", return_value=recorder), \
                    patch.object(trace, "write_manifest") as write, \
                    patch("vllm_bench.pangu_infer.results.versions", return_value={}), \
                    patch("vllm_bench.pangu_infer.trace_compare.compare_traces", return_value={"summary": {}}), \
                    patch("sys.stdout", new_callable=io.StringIO):
                trace.replay_target(engine, plan, output)
            engine.caches.reset.assert_called_once()
            torch.distributed.barrier.assert_called_once_with(group=cpu_group)
            if with_mtp:
                self.assertEqual(engine.propose.call_count, 2)
                proposal_calls = engine.propose.call_args_list
                self.assertIs(proposal_calls[0].args[1], hidden_outputs[0])
                self.assertIs(proposal_calls[1].args[1], hidden_outputs[1])
                self.assertEqual([(call.args[2], call.args[4], call.args[6]) for call in proposal_calls],
                                 [(0, 3, 1), (4, 0, 1)])
                self.assertEqual([call.args[5].tolist() for call in proposal_calls], [[3], [3]])
                proposals = json.loads((output / "target-trace" / "rank-0" / "draft-proposals.json").read_text())
                self.assertEqual(proposals[0]["reference"], [4])
                self.assertEqual(proposals[0]["first_difference"], 0)
                self.assertIsNone(proposals[1]["first_difference"])
            else:
                engine.propose.assert_not_called()
            self.assertEqual([(call.args[0], call.args[2], call.args[3])
                              for call in engine.forward.call_args_list],
                             [("target", 0, 8), ("target", 4, 1), ("target", 5, 1)])
            self.assertEqual([call.args[1].tolist() for call in engine.forward.call_args_list],
                             [[0, 1, 2, 3], [3, 4], [3, 4]])
            recorder.close.assert_called_once()
            write.assert_called_once()
            self.assertTrue((output / "trace_comparison.json").is_file())


class TraceBaselineTests(unittest.TestCase):
    def test_hooks_cover_only_measured_request_and_are_removed_on_failure(self):
        from vllm_bench.pangu_infer import baseline

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                events = []
                plan = {"options": {"output": tmp, "mode": "eager", "warmup": 1,
                                    "repeats": 1, "max_new_tokens": 32, "profile": False,
                                    "trace_target": True, "depth": 7}, "trace_layers": [0]}
                path = Path(tmp) / "plan.json"
                path.write_text(json.dumps(plan))
                state = {"operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 100,
                         "compilation": 0, "cudagraph": "NONE", "compile_counter": {}, "hardware": "NPU"}
                def rpc(function, args=()):
                    if function.__name__ == "worker_state":
                        events.append("state")
                        return [state]
                    if function.__name__ == "install_target_trace":
                        events.append("install")
                        self.assertEqual(args, (str(Path(tmp) / "target-trace"), [0]))
                        return [None]
                    if function.__name__ == "finish_target_trace":
                        events.append("finish")
                        return [2]
                    raise AssertionError(function.__name__)
                engine = SimpleNamespace(collective_rpc=rpc)
                def request(engine, plan, request_id, limit):
                    events.append("warmup" if request_id.startswith("warmup") else "measure")
                    if fail and request_id.startswith("measure"):
                        raise RuntimeError("request failed")
                    return {"token_ids": [1, 2]}
                modules = {
                    "vllm.plugins": SimpleNamespace(load_general_plugins=lambda: None),
                    "vllm.engine.arg_utils": SimpleNamespace(EngineArgs=lambda **kwargs: object()),
                    "vllm.v1.engine.llm_engine": SimpleNamespace(
                        LLMEngine=SimpleNamespace(from_engine_args=lambda args: engine)),
                }
                with patch.dict(sys.modules, modules), \
                        patch.object(baseline, "require_npu_platform"), \
                        patch.object(baseline, "engine_options", return_value={}), \
                        patch.object(baseline, "request", side_effect=request), \
                        patch.object(baseline, "versions", return_value={}), \
                        patch.object(baseline, "save_result") as save, \
                        patch.object(trace, "write_manifest") as manifest, \
                        patch("sys.stdout", new_callable=io.StringIO):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "request failed"):
                            baseline.main(["--plan", str(path)])
                    else:
                        baseline.main(["--plan", str(path)])
                expected = ["warmup", "state", "install", "measure", "finish"]
                self.assertEqual(events, expected if fail else [*expected, "state"])
                self.assertEqual(save.call_count, 0 if fail else 1)
                self.assertEqual(manifest.call_count, 0 if fail else 1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is not installed")
class TraceTensorTests(unittest.TestCase):
    def test_full_trace_comparison_reports_first_argmax_and_layer_difference(self):
        import torch
        from vllm_bench.pangu_infer.trace_compare import compare_traces

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"config": {}, "operator_config": {}, "additional_config": {},
                    "checkpoint_identity": {}, "prompt_sha256": "same", "max_model_len": 100,
                    "trace_layers": [0], "options": {"engine": "omni", "tp": 1, "depth": 7,
                                                     "mode": "eager", "temperature": 0}}
            for engine in ("omni", "direct"):
                directory = root / engine
                rank = directory / "rank-0"
                rank.mkdir(parents=True)
                current = deepcopy(plan)
                current["options"]["engine"] = engine
                trace.write_manifest(directory, current, {}, {"versions": {}, "hardware": "test"})
                (rank / "index.json").write_text('{"calls": 2}')
                for call in range(2):
                    output = {"input_ids": torch.tensor([1]), "positions": torch.tensor([call]),
                              "metadata": {"layer": {"accepted": [8]}},
                              "logits": torch.tensor([[1., 2., 0.]]),
                              "layers": {"0": {"0": torch.tensor([[1., 2.]])}},
                              "states_before": {}, "states_after": {}}
                    if engine == "direct" and call == 1:
                        output["logits"] = torch.tensor([[3., 2., 0.]])
                        output["layers"]["0"]["0"][0, 0] = 2
                    if engine == "direct" and call == 0:
                        output["metadata"]["layer"]["accepted"] = [1]
                    torch.save(output, rank / f"call-{call:05d}.pt")
            report = compare_traces(root / "omni", root / "direct")
            summary = report["summary"]
            self.assertTrue(summary["inputs_equal"])
            self.assertFalse(summary["metadata_equal"])
            self.assertEqual(summary["first_metadata_detail"][0], {
                "path": "layer.accepted", "reference": [8], "candidate": [1]})
            self.assertFalse(summary["argmax_equal"])
            self.assertEqual(summary["first_differences"]["argmax"], {"target_call": 1, "rank": 0, "row": 0})
            self.assertEqual(summary["first_argmax_detail"]["reference_top1"], 1)
            self.assertEqual(summary["first_argmax_detail"]["candidate_top1"], 0)
            self.assertEqual(report["calls"][1]["logits"]["max_abs"], 2.)
            self.assertFalse(report["calls"][1]["layer_outputs"]["0.0"]["exact"])

    def test_nonfinite_and_shape_differences_are_explicit(self):
        import torch
        from vllm_bench.pangu_infer.trace_compare import tensor_metrics, logits_metrics

        metrics = tensor_metrics(torch.tensor([float("inf"), 1.]), torch.tensor([float("inf"), 2.]), 0, 0)
        self.assertEqual(metrics["nonfinite_mismatches"], 0)
        self.assertEqual(metrics["max_abs"], 1.)
        metrics = tensor_metrics(torch.tensor([float("nan")]), torch.tensor([1.]), 0, 0)
        self.assertEqual(metrics["nonfinite_mismatches"], 1)
        metrics = logits_metrics(torch.ones(1, 2), torch.ones(2, 2), 0, 0)
        self.assertFalse(metrics["argmax_equal"])


if __name__ == "__main__":
    unittest.main()
