"""Direct-runtime planning, comparison and rejection math without ML packages."""

import contextlib
from copy import deepcopy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

from vllm_bench.common.paths import SRC_ROOT
from vllm_bench.pangu_infer import run, matrix
from vllm_bench.pangu_infer.compare import compare, first_difference
from vllm_bench.pangu_infer.runtime import engine_options, weight_iterator, require_loaded_parameters
from vllm_bench.pangu_infer.sampling import reference_rejection


class DirectPlanningTests(unittest.TestCase):
    def fixture(self, root, *, reduced=False, quantized=True):
        model = root / ("reduced" if reduced else "full")
        model.mkdir()
        config = {"architectures": ["PanguUltraMoEForCausalLM"],
                  "num_hidden_layers": 12 if reduced else 50,
                  "num_nextn_predict_layers": 3, "vocab_size": 32,
                  "use_mome": True, "max_position_embeddings": 16384,
                  "eos_token_id": [0, 1]}
        if quantized:
            config["quantization_config"] = {"quant_method": "mxfp8"}
        (model / "config.json").write_text(json.dumps(config))
        if reduced:
            (model / "extraction_manifest.json").write_text('{"keep_mtp": true}')
        operator = root / "operator.json"
        operator.write_text(json.dumps({"model_parallel_config": {},
                                       "operator_optimization_config": {"use_noncontiguous_kv": True}}))
        return ["--model", str(model), "--operator-config", str(operator),
                "--output", str(root / "output")]

    def test_all_axes_and_cpu_planning(self):
        for reduced in (False, True):
            for quantized in (False, True):
                with tempfile.TemporaryDirectory() as tmp:
                    args = self.fixture(Path(tmp), reduced=reduced, quantized=quantized)
                    for tp in (1, 2, 4, 8):
                        for depth in (0, 3, 7):
                            for mode in run.MODES:
                                a = run.parser().parse_args([*args, "--tp", str(tp), "--depth", str(depth), "--mode", mode])
                                plan = run.make_plan(a)
                                self.assertEqual(len(plan["prompt_ids"]), 8192)
                                self.assertTrue(set(plan["prompt_ids"]).isdisjoint({0, 1}))
                                self.assertEqual(plan["model_scope"], "reduced" if reduced else "full")
                                start = 12 if reduced else 50
                                self.assertEqual(plan["predictor_layers"],
                                                 [] if depth == 0 else [start, start + 1, start + 2] if depth == 3 else [start] * 7)
                                options = engine_options(plan, direct=True)
                                self.assertFalse(options["enforce_eager"])
                                self.assertFalse(options["enable_expert_parallel"])
                                if depth == 0:
                                    self.assertNotIn("speculative_config", options)
                                    self.assertEqual(options["compilation_config"]["cudagraph_capture_sizes"], [1])
                                self.assertEqual(options["compilation_config"]["mode"], 0)
                                self.assertEqual(options["max_model_len"], 8192 + 256 + 8)
                                self.assertGreaterEqual(options["max_num_batched_tokens"], options["max_model_len"])
                                self.assertFalse(options["async_scheduling"])
                    self.assertFalse((Path(tmp) / "output").exists())

    def test_no_mtp_alias_accepts_checkpoints_with_or_without_predictors(self):
        for engine in ("direct", "omni"):
            for predictors in (3, 0, None):
                for temperature in (0, 1):
                    with self.subTest(engine=engine, predictors=predictors, temperature=temperature), tempfile.TemporaryDirectory() as tmp:
                        args = self.fixture(Path(tmp))
                        parsed = run.parser().parse_args(args)
                        config_path = parsed.model / "config.json"
                        config = json.loads(config_path.read_text())
                        if predictors is None:
                            del config["num_nextn_predict_layers"]
                        else:
                            config["num_nextn_predict_layers"] = predictors
                        config_path.write_text(json.dumps(config))
                        operator = json.loads(parsed.operator_config.read_text())
                        operator["operator_optimization_config"]["fix_multi_mtp_kvcache"] = True
                        parsed.operator_config.write_text(json.dumps(operator))
                        common = [*args, "--engine", engine, "--temperature", str(temperature)]
                        plan = run.make_plan(run.parser().parse_args([*common, "--no-mtp"]))
                        explicit = run.make_plan(run.parser().parse_args([*common, "--depth", "0"]))
                        self.assertEqual(plan["options"]["depth"], 0)
                        self.assertEqual(plan["predictor_layers"], [])
                        self.assertEqual(plan["comparison_sha256"], explicit["comparison_sha256"])
                        self.assertNotIn("speculative_config", engine_options(plan, direct=engine == "direct"))

    def test_no_mtp_and_explicit_depth_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.assertEqual(run.parser().parse_args(args).depth, 3)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run.parser().parse_args([*args, "--depth", "7", "--no-mtp"])
            self.assertEqual(error.exception.code, 2)

    def test_matrix_no_mtp_overrides_json_depths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            config_path = root / "full/config.json"
            config = json.loads(config_path.read_text())
            config["num_nextn_predict_layers"] = 0
            config_path.write_text(json.dumps(config))
            spec = root / "matrix.json"
            spec.write_text(json.dumps({"models": ["full"], "operator_configs": {"1": "operator.json"},
                                        "output": "matrix", "tp": [1], "depths": [3, 7], "modes": ["eager"]}))
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                self.assertEqual(matrix.main([str(spec), "--no-mtp"]), 0)
            cases = json.loads(console.getvalue())["cases"]
            self.assertEqual({case["engine"] for case in cases}, {"direct", "omni"})
            self.assertTrue(all(case["depth"] == 0 for case in cases))
            self.assertFalse((root / "matrix").exists())

    def test_cli_without_ml_from_another_directory(self):
        code = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'omni_npu', 'numpy', 'safetensors'}:
        raise RuntimeError('ML imported in CPU CLI: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from vllm_bench.pangu_infer.run import main
sys.exit(main())
'''
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            env = {k: v for k, v in os.environ.items() if not k.startswith("PANGU")}
            env["PYTHONPATH"] = str(SRC_ROOT)
            result = subprocess.run([sys.executable, "-c", code, *args], cwd="/tmp", env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["options"]["context_length"], 8192)
            self.assertFalse((Path(tmp) / "output").exists())

    def test_sampling_arguments_and_prompt_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            for flag, value in (("--temperature", "nan"), ("--temperature", "-1"),
                                ("--top-p", "0"), ("--top-p", "1.01"),
                                ("--top-k", "33"), ("--context-length", "0"),
                                ("--max-new-tokens", "1"), ("--gpu-memory-utilization", "0")):
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    run.make_plan(run.parser().parse_args([*args, flag, value]))
            prompt = Path(tmp) / "prompt.json"
            for ids in ([True] * 4, [32] * 4, [1, 2], {"ids": [1, 2, 3, 4]}):
                prompt.write_text(json.dumps(ids))
                with self.assertRaises(ValueError):
                    run.make_plan(run.parser().parse_args([*args, "--context-length", "4", "--prompt-ids", str(prompt)]))

    def test_operator_config_and_architecture_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            a = run.parser().parse_args(args)
            for operator in (
                {"model_parallel_config": {"ena_seq_parallel": True}, "operator_optimization_config": {"use_noncontiguous_kv": True}},
                {"model_parallel_config": {}, "operator_optimization_config": {}},
                {"model_parallel_config": {}, "operator_optimization_config": {"use_noncontiguous_kv": True, "fix_multi_mtp_kvcache": True}},
            ):
                a.operator_config.write_text(json.dumps(operator))
                with self.assertRaises(ValueError):
                    run.make_plan(a)

    def test_compilation_cannot_silently_bypass_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            default = run.make_plan(run.parser().parse_args([*args, "--mode", "compile"]))
            self.assertIs(default["additional_config"]["npugraph_ex_config"]["enable"], True)
            additional = Path(tmp) / "additional.json"
            additional.write_text('{"npugraph_ex_config": {"enable": false}}')
            with self.assertRaisesRegex(ValueError, "enable=true"):
                run.make_plan(run.parser().parse_args([*args, "--mode", "compile", "--additional-config", str(additional)]))
            with self.assertRaisesRegex(ValueError, "compile-only"):
                run.make_plan(run.parser().parse_args([*args, "--mode", "compile", "--engine", "omni"]))

    def test_checkpoint_changes_comparison_key_and_engines_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            first = run.make_plan(run.parser().parse_args(args))
            second = run.make_plan(run.parser().parse_args([*args, "--engine", "omni"]))
            self.assertEqual(first["comparison_sha256"], second["comparison_sha256"])
            other = run.make_plan(run.parser().parse_args([*args, "--seed", "1"]))
            self.assertNotEqual(first["prompt_sha256"], other["prompt_sha256"])

    def test_matrix_is_sequential_plan_and_omits_false_compile_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            spec = {"models": ["full"], "operator_configs": {str(t): "operator.json" for t in (1, 2, 4, 8)},
                    "output": "matrix"}
            output, items = matrix.plans(spec, root)
            self.assertEqual(len(items), 4 * 2 * 7)
            self.assertTrue(all(not (item["options"]["engine"] == "omni" and item["options"]["mode"] == "compile") for item in items))
            self.assertFalse(output.exists())
            spec["options"] = ["--tp=2"]
            with self.assertRaises(ValueError):
                matrix.plans(spec, root)

    def test_launch_selects_torchrun_or_offline_engine(self):
        for engine in ("direct", "omni"):
            with tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                plan = run.make_plan(run.parser().parse_args([*args, "--engine", engine, "--tp", "2"]))
                with patch.object(run.subprocess, "Popen") as child:
                    process = child.return_value.__enter__.return_value
                    process.stdout = iter(())
                    process.wait.return_value = 0
                    self.assertEqual(run.launch(plan), 0)
                command = child.call_args.args[0]
                self.assertEqual("torch.distributed.run" in command, engine == "direct")
                self.assertIn("vllm_bench.pangu_infer.runtime" if engine == "direct" else "vllm_bench.pangu_infer.baseline", command)
                env = child.call_args.kwargs["env"]
                self.assertEqual(env["OMNI_DRAFT_TOP_K"], "-1")
                self.assertEqual(env["VLLM_USE_V2_MODEL_RUNNER"], "0")
                self.assertTrue({"omni-npu", "omni_npu_patches", "omni_custom_models"}
                                <= set(env["VLLM_PLUGINS"].split(",")))

    def test_launch_streams_stdout_and_stderr_to_console_and_log(self):
        original_popen = subprocess.Popen
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(Path(tmp))
                plan = run.make_plan(run.parser().parse_args(args))
                script = ("import sys\nprint('stdout', flush=True)\n"
                          "sys.stderr.write('stderr\\n'); sys.stderr.flush()\n"
                          f"sys.stdout.write('tail'); sys.stdout.flush()\nraise SystemExit({exit_code})\n")
                def launch_child(command, **kwargs):
                    self.assertEqual(kwargs["env"]["PYTHONUNBUFFERED"], "1")
                    return original_popen([sys.executable, "-c", script], **kwargs)
                console = io.StringIO()
                with patch.object(run.subprocess, "Popen", side_effect=launch_child), contextlib.redirect_stdout(console):
                    self.assertEqual(run.launch(plan), exit_code)
                expected = "stdout\nstderr\ntail"
                self.assertEqual(console.getvalue(), expected)
                output = Path(plan["options"]["output"])
                self.assertEqual((output / "runtime.log").read_text(), expected)
                self.assertEqual((output / "failure.json").exists(), exit_code != 0)
                if exit_code:
                    self.assertEqual(json.loads((output / "failure.json").read_text())["returncode"], exit_code)

    def test_plugin_allowlist_keeps_platform_plugin_and_user_plugins(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            plan = run.make_plan(run.parser().parse_args(args))
            for inherited in ("", "omni_npu_patches,omni_custom_models", "custom,omni-npu,custom"):
                with self.subTest(inherited=inherited), patch.dict(os.environ, {"VLLM_PLUGINS": inherited}):
                    plugins = run.child_environment(plan)["VLLM_PLUGINS"].split(",")
                # This is the entry-point filter used by vLLM for every group.
                available_platforms = [SimpleNamespace(name="omni-npu", device_type="npu")]
                selected = [entry for entry in available_platforms if entry.name in plugins]
                self.assertEqual([entry.device_type for entry in selected], ["npu"])
                self.assertEqual(plugins.count("omni-npu"), 1)
                if "custom" in inherited.split(","):
                    self.assertEqual(plugins.count("custom"), 1)


class RejectionMathTests(unittest.TestCase):
    def test_exact_marginal_distribution(self):
        # Enumerate proposal mass and rejection mass: the resulting first
        # emitted token must have precisely the target distribution.
        for p, q in (([0.2, 0.8], [0.7, 0.3]),
                     ([0, 0.25, 0.75], [0.5, 0.5, 0]),
                     ([0.5, 0.5], [0.5, 0.5])):
            out = [min(a, b) for a, b in zip(p, q)]
            rejection = 1 - sum(out)
            token = next(i for i, prob in enumerate(q) if prob > 0)
            _, residual = reference_rejection(p, q, token, 0)
            self.assertAlmostEqual(sum(residual), 1)
            for i in range(len(out)):
                self.assertAlmostEqual(out[i] + rejection * residual[i], p[i], places=12)

    def test_acceptance_boundary_and_zero_mass(self):
        self.assertTrue(reference_rejection([0.2, 0.8], [0.8, 0.2], 0, 0.249)[0])
        self.assertFalse(reference_rejection([0.2, 0.8], [0.8, 0.2], 0, 0.25)[0])
        self.assertFalse(reference_rejection([0, 1], [1, 0], 0, 0)[0])
        with self.assertRaises(ValueError):
            reference_rejection([0, 1], [1, 0], 1, 0)
        with self.assertRaises(ValueError):
            reference_rejection([math.nan, 1], [1, 0], 0, 0)


class LoadingAndBaselineTests(unittest.TestCase):
    def test_runtime_initialization_and_execution_use_no_grad(self):
        from unittest.mock import Mock
        from vllm_bench.pangu_infer import runtime

        state = {"grad_enabled": True}
        checked = []

        @contextlib.contextmanager
        def no_grad():
            state["grad_enabled"] = False
            try:
                yield
            finally:
                state["grad_enabled"] = True

        def check(label):
            self.assertFalse(state["grad_enabled"], label)
            checked.append(label)

        def load(*args):
            check("weight construction/loading/post-processing")
            return object(), None, {}

        def generate(*args, **kwargs):
            check("warmup/decode")
            return {"decode_seconds": .25}

        engine = SimpleNamespace(
            generate=Mock(side_effect=generate), executions={},
            counters=lambda: {"compiles": 0, "captures": 0, "replays": 0, "calls": 1})

        def create_engine(*args):
            check("cache/buffer construction")
            return engine

        fake_torch = SimpleNamespace(
            no_grad=no_grad, inference_mode=Mock(side_effect=AssertionError("Inference tensors lose version counters")),
            tensor=lambda values, **kwargs: SimpleNamespace(tolist=lambda: values), float64="float64",
            npu=SimpleNamespace(synchronize=lambda: check("synchronize")),
            distributed=SimpleNamespace(all_reduce=lambda *args, **kwargs: None,
                                        ReduceOp=SimpleNamespace(MAX="max"), is_initialized=lambda: False))
        cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=128, mamba_block_size=128),
                              compilation_config=SimpleNamespace(mode=0, cudagraph_mode="NONE"))
        context = SimpleNamespace(__exit__=Mock())
        extra = SimpleNamespace(operator_opt_config={}, parall_config={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            path.write_text(json.dumps({"options": {"output": tmp, "mode": "eager", "depth": 3,
                                                   "warmup": 1, "repeats": 1, "max_new_tokens": 32,
                                                   "profile": False}}))
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(runtime, "setup", return_value=(cfg, "npu:1", 1, context, extra)), \
                    patch.object(runtime, "load_models", side_effect=load), \
                    patch.object(runtime, "DirectEngine", side_effect=create_engine):
                runtime.main(["--plan", str(path)])
        fake_torch.inference_mode.assert_not_called()
        self.assertTrue(state["grad_enabled"])
        self.assertIn("weight construction/loading/post-processing", checked)
        self.assertIn("cache/buffer construction", checked)
        self.assertEqual(engine.generate.call_count, 2)
        context.__exit__.assert_called_once_with(None, None, None)

    def test_platform_check_rejects_empty_device_before_engine_config(self):
        from vllm_bench.pangu_infer.runtime import require_npu_platform
        for device_type in ("", "cpu", "cuda", "npu"):
            platform = SimpleNamespace(current_platform=SimpleNamespace(device_type=device_type))
            with self.subTest(device_type=device_type), patch.dict(sys.modules, {"vllm.platforms": platform}):
                if device_type == "npu":
                    require_npu_platform()
                else:
                    with self.assertRaisesRegex(RuntimeError, "VLLM_PLUGINS must include omni-npu"):
                        require_npu_platform()

    def test_metadata_grouping_preserves_backend_and_full_spec(self):
        from vllm_bench.pangu_infer.cache import compatible_groups
        mla, dsa = object(), object()
        layers = [SimpleNamespace(name="a", backend=mla, spec=(128, 512)),
                  SimpleNamespace(name="b", backend=mla, spec=(128, 512)),
                  SimpleNamespace(name="c", backend=mla, spec=(128, 2048)),
                  SimpleNamespace(name="d", backend=dsa, spec=(128, 512))]
        self.assertEqual([[item.name for item in bucket] for bucket in compatible_groups(layers)],
                         [["a", "b"], ["c"], ["d"]])

    def test_stream_target_and_mtp_separately_with_nested_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "shards").mkdir()
            (root / "shards/weights.safetensors").write_bytes(b"unchanged")
            names = ("model.embed_tokens.weight", "lm_head.weight", "model.layers.0.weight",
                     "model.layers.12.eh_proj.weight", "model.layers.12.eh_proj.weight_scale")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                name: "shards/weights.safetensors" for name in names}}))
            payload = {name: object() for name in names}
            class Shard:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def keys(self):
                    return names
                def get_tensor(self, name):
                    return payload[name]
            with patch.dict(sys.modules, {"safetensors": SimpleNamespace(safe_open=lambda *args, **kwargs: Shard())}):
                target = dict(weight_iterator(root, draft=False, start=12))
                draft = dict(weight_iterator(root, draft=True, start=12))
            self.assertEqual(set(target), set(names[:3]))
            self.assertEqual(set(draft), set(names[3:]))
            for name, value in {**target, **draft}.items():
                self.assertIs(value, payload[name])
            self.assertEqual((root / "shards/weights.safetensors").read_bytes(), b"unchanged")

    def test_missing_parameters_fail_but_shared_target_parameters_are_allowed(self):
        shared, own = object(), object()
        model = SimpleNamespace(named_parameters=lambda: iter((("head", shared), ("own", own))))
        with self.assertRaisesRegex(ValueError, "Uninitialized"):
            require_loaded_parameters(model, {"own"})
        require_loaded_parameters(model, {"own"}, shared={id(shared)})

    def test_offline_decode_boundary_and_output_acceptance(self):
        from vllm_bench.pangu_infer.baseline import request, synchronize_worker
        from unittest.mock import Mock
        engine = Mock()
        engine.has_unfinished_requests.side_effect = [True, True, False]
        engine.step.side_effect = [[SimpleNamespace(request_id="r", outputs=[SimpleNamespace(token_ids=[4])])],
                                   [SimpleNamespace(request_id="r", outputs=[SimpleNamespace(token_ids=[4, 5, 6, 7, 8])])]]
        a = {"temperature": 0, "top_k": 0, "top_p": 1, "seed": 0, "respect_eos": False,
             "depth": 3, "context_length": 8192}
        with patch.dict(sys.modules, {"vllm": SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))}):
            record = request(engine, {"options": a, "prompt_ids": [2, 3]}, "r", 5)
        self.assertEqual(record["decode_tokens"], 4)
        self.assertEqual(record["accepted_drafts"], [3])
        self.assertEqual(record["token_ids"], [4, 5, 6, 7, 8])
        self.assertEqual(engine.collective_rpc.call_count, 3)
        engine.collective_rpc.assert_called_with(synchronize_worker)


class ResultComparisonTests(unittest.TestCase):
    def fixture(self):
        return {"format": "pangu-direct-result-v1", "measurement": "benchmark", "engine": "direct",
                "options": {"model": "/model", "tp": 2, "temperature": 0, "depth": 3,
                            "max_new_tokens": 3, "respect_eos": False},
                "comparison_sha256": "same", "prompt_sha256": "prompt",
                "effective": {"operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 128},
                "details": {"hardware": "Ascend", "versions": {}},
                "records": [{"token_ids": [3, 4, 5]}], "median_tpot_ms": 10}

    def test_speedup_and_mismatched_greedy_tokens(self):
        reference = self.fixture()
        candidate = deepcopy(reference)
        candidate["median_tpot_ms"] = 5
        self.assertEqual(compare(reference, candidate)["speedup"], 2)
        candidate["records"][0]["token_ids"][1] = 7
        result = compare(reference, candidate)
        self.assertIsNone(result["speedup"])
        self.assertFalse(result["greedy_tokens_match"])
        self.assertEqual(result["first_differences"][0]["position"], 1)

    def test_correctness_allows_different_depth_and_mode(self):
        reference, candidate = self.fixture(), self.fixture()
        reference["options"]["depth"] = 0
        candidate["comparison_sha256"] = "different-mode"
        result = compare(reference, candidate, correctness=True)
        self.assertTrue(result["greedy_tokens_match"])
        self.assertIsNone(result["speedup"])

    def test_refuse_different_workload_operators_versions_or_diagnostics(self):
        reference = self.fixture()
        for key in ("workload", "operator", "versions", "diagnostic"):
            candidate = deepcopy(reference)
            if key == "workload":
                candidate["comparison_sha256"] = "other"
            elif key == "operator":
                candidate["effective"]["operator"]["fusion"] = False
            elif key == "versions":
                candidate["details"]["versions"]["torch"] = "different"
            else:
                candidate["measurement"] = "diagnostic"
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(reference, candidate)

    def test_length_difference(self):
        self.assertEqual(first_difference([1], [1, 2])["position"], 1)

    def test_initial_eos_preserves_result_without_a_speedup(self):
        from vllm_bench.pangu_infer.results import save_result
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"options": {"engine": "direct", "profile": False, "depth": 3},
                    "comparison_sha256": "same", "prompt_sha256": "prompt", "model_scope": "full",
                    "checkpoint_identity": {}}
            result = save_result(Path(tmp), plan,
                [{"token_ids": [5], "decode_tokens": 0, "decode_seconds": 0}], {}, {})
            self.assertIsNone(result["median_tpot_ms"])
            self.assertEqual(result["empty_decode_requests"], 1)
        reference = self.fixture()
        reference["records"] = [{"token_ids": [5]}]
        reference["median_tpot_ms"] = None
        with self.assertRaisesRegex(ValueError, "No decode tokens"):
            compare(reference, deepcopy(reference))
        self.assertTrue(compare(reference, deepcopy(reference), correctness=True)["greedy_tokens_match"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional real CPU torch tests; torch is not installed")
class TorchSamplingTests(unittest.TestCase):
    def test_greedy_rejection_positions(self):
        import torch
        from vllm_bench.pangu_infer.sampling import Sampler
        sampler = Sampler({"seed": 0, "temperature": 0, "top_k": 0, "top_p": 1, "tp": 1}, "cpu", 0)
        drafts = torch.tensor([0, 1, 2])
        q = torch.nn.functional.one_hot(drafts, 4).float()
        for expected, targets in ((0, [3, 1, 2, 3]), (1, [0, 3, 2, 3]),
                                  (2, [0, 1, 3, 3]), (3, [0, 1, 2, 3])):
            p = torch.nn.functional.one_hot(torch.tensor(targets), 4).float()
            emitted, accepted = sampler.verify(p, q, drafts)
            self.assertEqual(accepted.item(), expected)
            self.assertEqual(emitted[:expected + 1].tolist(), drafts[:expected].tolist() + [targets[expected]])

    def test_top_k_top_p_and_temperature(self):
        import torch
        from vllm_bench.pangu_infer.sampling import probabilities
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
        self.assertTrue(torch.equal(probabilities(logits, 0, 0, 1), torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.equal(probabilities(logits, 1, 1, 1), torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.equal(probabilities(logits, 1, 0, .5), torch.tensor([[1., 0., 0.]])))
        self.assertTrue(torch.allclose(probabilities(logits, 1, 0, .8), torch.tensor([[2/3, 1/3, 0.]])))

    def test_sampling_emits_target_distribution(self):
        import torch
        from vllm_bench.pangu_infer.sampling import Sampler
        sampler = Sampler({"seed": 8, "temperature": 1, "top_k": 0, "top_p": 1, "tp": 1}, "cpu", 0)
        q = torch.tensor([[.7, .3]])
        p = torch.tensor([[.2, .8], [.2, .8]])
        ones = 0
        for _ in range(4096):
            proposal = sampler.sample(q)
            emitted, _ = sampler.verify(p, q, proposal)
            ones += int(emitted[0])
        self.assertAlmostEqual(ones / 4096, .8, delta=.04)

    def test_real_decode_loop_preserves_committed_prefix_after_rejection(self):
        import torch
        from vllm_bench.pangu_infer.runtime import DirectEngine
        from vllm_bench.pangu_infer.sampling import Sampler
        from unittest.mock import Mock

        for depth in (0, 3, 7):
            for reject_at in (None, 0, 1, 2):
                with self.subTest(depth=depth, reject_at=reject_at):
                    engine = DirectEngine.__new__(DirectEngine)
                    engine.a = {"depth": depth, "context_length": 4, "profile": False,
                                "respect_eos": False, "seed": 0, "temperature": 0,
                                "top_k": 0, "top_p": 1, "tp": 1}
                    engine.plan = {"options": engine.a, "prompt_ids": [1, 2, 3, 4],
                                   "config": {"num_nextn_predict_layers": 3, "eos_token_id": 63}}
                    engine.device = torch.device("cpu")
                    engine.prompt = torch.tensor([1, 2, 3, 4])
                    engine.caches = SimpleNamespace(reset=Mock())
                    engine.sampler = Sampler(engine.a, "cpu", 0)
                    starts = []
                    def forward(self, kind, ids, start, accepted, *, hidden=None, index=None, head=0, draft_index=None):
                        next_ids = (ids + 1) % 64
                        if kind == "draft" and draft_index == reject_at:
                            next_ids = (next_ids + 7) % 64
                        logits = torch.nn.functional.one_hot(next_ids, 64).float() * 1000
                        if kind == "draft":
                            logits = logits.index_select(0, index)
                        else:
                            starts.append(start)
                        return ids.reshape(-1, 1).float(), logits
                    engine.forward = MethodType(forward, engine)
                    with patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True):
                        record = engine.generate(29)
                    self.assertEqual(record["token_ids"], list(range(5, 34)))
                    expected_accepted = min(reject_at, depth) if reject_at is not None and depth else depth
                    self.assertTrue(all(n == expected_accepted for n in record["accepted_drafts"]))
                    self.assertEqual(starts[0], 0)
                    self.assertEqual(starts[1:], [4 + i * (expected_accepted + 1) for i in range(record["cycles"])])
                    engine.a["respect_eos"] = True
                    engine.plan["config"]["eos_token_id"] = 5
                    with patch.object(torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True):
                        stopped = engine.generate(29)
                    self.assertEqual(stopped["token_ids"], [5])
                    self.assertEqual(stopped["decode_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
