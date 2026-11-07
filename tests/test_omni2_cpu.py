"""Native-server adapter routing and streaming lifecycle without ML packages."""

import asyncio
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from vllm_bench.pangu_infer import omni2, run


class PlanningFixture:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({"architectures": [run.ULTRA_ARCHITECTURE],
            "num_hidden_layers": 2, "num_nextn_predict_layers": 1, "vocab_size": 64,
            "max_position_embeddings": 16384, "use_mome": True}))
        self.operator = self.root / "operator.json"
        self.operator.write_text(json.dumps({"model_parallel_config": {},
            "operator_optimization_config": {"use_noncontiguous_kv": True}}))
        self.args = ["--engine", "omni2", "--model", str(self.model), "--operator-config", str(self.operator),
                     "--output", str(self.root / "output"), "--context-length", "8", "--depth", "7",
                     "--temperature", "0", "--max-new-tokens", "16", "--repeats", "2"]

    def plan(self, *flags):
        return run.make_plan(run.parser().parse_args([*self.args, *flags]))


class Omni2PlanningTests(PlanningFixture, unittest.TestCase):
    def test_native_defaults_and_explicit_old_engine_paths(self):
        plan = self.plan()
        self.assertEqual(plan["options"]["warmup"], 0)
        argv = plan["omni2_server_argv"]
        self.assertNotIn("--skip-tokenizer-init", argv)
        spec = json.loads(argv[argv.index("--speculative-config") + 1])
        self.assertEqual(spec, {"method": "mtp", "num_speculative_tokens": 7})
        self.assertEqual(self.plan("--engine", "omni")["options"]["warmup"], 2)
        self.assertEqual(self.plan("--engine", "direct")["options"]["warmup"], 2)
        self.assertNotEqual(plan["comparison_sha256"], self.plan("--engine", "omni")["comparison_sha256"])
        for flags in (("--fix-mome-prefill-state",), ("--trace-mtp-boundary",),
                      ("--trace-target",), ("--mode", "compile")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.plan(*flags)
        self.assertNotIn("--speculative-config", self.plan("--depth", "0")["omni2_server_argv"])
        tokenizer = self.root / "tokenizer"
        tokenizer.mkdir()
        argv = self.plan("--tokenizer", str(tokenizer))["omni2_server_argv"]
        self.assertEqual(argv[argv.index("--tokenizer") + 1], str(tokenizer))

    def test_native_environment_preserves_server_settings(self):
        inherited = {"VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
                     "OMNI_DRAFT_TOP_K": "42", "OMNI_VLLM_PATCHES_DIR": "low_latency",
                     "OMNI_PANGU_V2_HIGH_THROUGHOUT": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "1"}
        with patch.dict(os.environ, inherited):
            env = run.child_environment(self.plan())
            for key, value in inherited.items():
                self.assertEqual(env[key], value)
            old = run.child_environment(self.plan("--engine", "omni"))
            self.assertEqual(old["VLLM_USE_AOT_COMPILE"], "0")
            self.assertEqual(old["OMNI_DRAFT_TOP_K"], "-1")

    def test_server_extras_are_frozen_and_cannot_replace_model_worker_or_axes(self):
        path = self.root / "server.json"
        extra = ["--generation-config", "vllm"]
        path.write_text(json.dumps(extra))
        plan = self.plan("--omni2-server-args", str(path))
        self.assertEqual(plan["omni2_server_argv"][-2:], extra)
        self.assertNotEqual(plan["comparison_sha256"], self.plan()["comparison_sha256"])
        path.write_text('["--generation-config", "auto"]')
        self.assertEqual(omni2.server_argv(plan)[-2:], extra)
        for option in ("--model", "--worker-cls", "--worker-extension-cls", "--speculative-config",
                       "--compilation-config.mode", "--dtype", "--no-enable-prefix-caching", "--config", "-tp"):
            path.write_text(json.dumps([option, "other"]))
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.plan("--omni2-server-args", str(path))
        path.write_text('{}')
        with self.assertRaises(ValueError):
            self.plan("--omni2-server-args", str(path))

    def test_profiler_and_http_request_fields(self):
        plan = self.plan("--profile", "--async-scheduling", "--enable-prefix-caching")
        argv = plan["omni2_server_argv"]
        config = json.loads(argv[argv.index("--profiler-config") + 1])
        self.assertEqual(config["torch_profiler_dir"], str(self.root / "output/traces"))
        fields = omni2.completion_fields(plan, [1, 2], 16)
        self.assertTrue(fields["stream"])
        self.assertTrue(fields["return_token_ids"])
        self.assertFalse(fields["add_special_tokens"])
        self.assertNotIn("min_tokens", fields)
        self.assertNotIn("request_id", fields)
        self.assertNotIn("detokenize", fields)
        self.assertNotIn("top_k", fields)

    def test_launcher_selects_omni2_module_and_keeps_stdout(self):
        plan = self.plan()
        process = SimpleNamespace(stdout=iter(["native launch\n"]), wait=lambda: 0)
        with patch.object(run.subprocess, "Popen") as launch, patch("sys.stdout", new_callable=io.StringIO) as stdout:
            launch.return_value.__enter__.return_value = process
            self.assertEqual(run.launch(plan), 0)
        command = launch.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "vllm_bench.pangu_infer.omni2"])
        self.assertEqual(stdout.getvalue(), "native launch\n")
        self.assertEqual((self.root / "output/runtime.log").read_text(), "native launch\n")

    def test_worker_audit_rejects_silent_execution_changes(self):
        plan = self.plan("--enable-prefix-caching", "--block-size", "128", "--mamba-block-size", "128")
        state = {"async_scheduling": False, "enable_prefix_caching": True, "cache_dtype": "auto",
                 "block_size": 128, "mamba_block_size": 128, "compilation": 0, "compile_counter": {},
                 "execution_settings": {"max_model_len": plan["max_model_len"], "max_num_seqs": 1,
                    "max_num_batched_tokens": plan["max_model_len"], "enable_chunked_prefill": False,
                    "dtype": "torch.bfloat16"}}
        omni2.check_workers(plan, [state])
        for key, value in (("cache_dtype", "bfloat16"), ("mome_prefill_state_fix", True),
                           ("block_size", 256), ("compilation", 3), ("async_scheduling", True)):
            bad = {**state, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                omni2.check_workers(plan, [bad])
        for key, value in (("max_num_seqs", 2), ("max_model_len", 999),
                           ("enable_chunked_prefill", True), ("dtype", "torch.float16")):
            bad = {**state, "execution_settings": {**state["execution_settings"], key: value}}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                omni2.check_workers(plan, [bad])

    def test_main_uses_native_parser_validation_cli_setup_and_uvloop(self):
        plan = self.plan()
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        (self.root / "output").mkdir()
        args = SimpleNamespace(skip_tokenizer_init=False, headless=False, api_server_count=None, model_tag=None)
        native_parser = SimpleNamespace(parse_args=Mock(return_value=args))
        setup, validate = Mock(), Mock()
        called = []
        def uvloop_run(coro):
            called.append("uvloop")
            return asyncio.run(coro)
        modules = {"vllm.entrypoints.openai.cli_args": SimpleNamespace(
                       make_arg_parser=Mock(return_value=native_parser), validate_parsed_serve_args=validate),
                   "vllm.entrypoints.serve.utils.api_utils": SimpleNamespace(cli_env_setup=setup),
                   "vllm.utils.argparse_utils": SimpleNamespace(FlexibleArgumentParser=Mock()),
                   "uvloop": SimpleNamespace(run=uvloop_run)}
        with patch.dict(sys.modules, modules), patch.object(omni2, "run", new_callable=AsyncMock) as execute:
            omni2.main(["--plan", str(path)])
        setup.assert_called_once()
        validate.assert_called_once_with(args)
        native_parser.parse_args.assert_called_once_with(plan["omni2_server_argv"])
        execute.assert_awaited_once_with(plan, args)
        self.assertEqual(called, ["uvloop"])


class Omni2AsyncTests(PlanningFixture, unittest.IsolatedAsyncioTestCase):
    def protocol(self):
        constructor = Mock(side_effect=lambda **fields: SimpleNamespace(**fields))
        return constructor, {"vllm.entrypoints.openai.completion.protocol": SimpleNamespace(CompletionRequest=constructor)}

    def frames(self, plan, *, bad_usage=False, error=False, missing_done=False):
        ids = plan["prompt_ids"]
        first = {"choices": [{"index": 0, "token_ids": [10, 11], "prompt_token_ids": ids,
                               "text": "native ", "finish_reason": None}]}
        second = {"choices": [{"index": 0, "token_ids": [12, 13, 14], "text": "text", "finish_reason": None}]}
        finish = {"choices": [{"index": 0, "token_ids": [], "text": "!", "finish_reason": "length"}]}
        usage = {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": 99 if bad_usage else 5,
                                           "prompt_tokens_details": {"cached_tokens": 4}}}
        data = [first, {"error": {"message": "failed"}}] if error else [first, second, finish, usage]
        return [*("data: " + json.dumps(item) + "\n\n" for item in data),
                *([] if missing_done else ["data: [DONE]\n\n"])]

    async def test_delta_ids_and_timing_exclude_usage_tail_and_allow_coalesced_first_frame(self):
        plan = self.plan()
        closed = []
        async def stream():
            try:
                for frame in self.frames(plan):
                    yield frame
            finally:
                closed.append(True)
        serving = SimpleNamespace(create_completion=AsyncMock(return_value=stream()))
        constructor, modules = self.protocol()
        with patch.dict(sys.modules, modules), patch.object(omni2.time, "perf_counter", side_effect=[0, 1, 3, 100]):
            record = await omni2.request(serving, plan, "model", plan["prompt_ids"], 16)
        self.assertEqual(record["token_ids"], [10, 11, 12, 13, 14])
        self.assertEqual(record["text"], "native text!")
        self.assertEqual((record["decode_seconds"], record["decode_tokens"]), (2, 3))
        self.assertEqual(record["first_output_tokens"], 2)
        self.assertEqual(record["stream_tail_seconds"], 97)
        self.assertEqual(record["cached_prompt_tokens"], 4)
        self.assertIsNone(record["cycles"])
        self.assertEqual(record["accepted_drafts"], [])
        self.assertEqual(closed, [True])
        self.assertNotIn("request_id", constructor.call_args.kwargs)
        self.assertEqual(serving.create_completion.call_args.kwargs, {"raw_request": None})

    async def test_stream_errors_bad_usage_and_missing_done_are_not_success(self):
        plan = self.plan()
        for flags in ({"error": True}, {"bad_usage": True}, {"missing_done": True}):
            closed = []
            async def stream():
                try:
                    for frame in self.frames(plan, **flags):
                        yield frame
                finally:
                    closed.append(True)
            _, modules = self.protocol()
            serving = SimpleNamespace(create_completion=AsyncMock(return_value=stream()))
            with self.subTest(flags=flags), patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
                await omni2.request(serving, plan, "model", plan["prompt_ids"], 16)
            self.assertEqual(closed, [True])
        _, modules = self.protocol()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "did not return a stream"):
            await omni2.request(SimpleNamespace(create_completion=AsyncMock(return_value={"error": "bad"})),
                                plan, "model", plan["prompt_ids"], 16)

    async def execute(self, *, fail=False):
        plan = self.plan()
        (self.root / "output").mkdir(exist_ok=True)
        settings = {"max_model_len": plan["max_model_len"], "max_num_seqs": 1,
                    "max_num_batched_tokens": plan["max_model_len"], "enable_chunked_prefill": False,
                    "dtype": "torch.bfloat16"}
        state = {"operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 128,
                 "cache_dtype": "auto", "mamba_cache_mode": "all", "async_scheduling": False,
                 "enable_prefix_caching": False, "execution_settings": settings, "compile_counter": {},
                 "compilation": 0, "cudagraph": "NONE", "hardware": "NPU"}
        cfg = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1,
                              pipeline_parallel_size=1, enable_expert_parallel=False, worker_cls="native.NPUWorker"),
                              model_config=SimpleNamespace(model=str(self.model), hf_config=SimpleNamespace(
                                  architectures=[run.ULTRA_ARCHITECTURE])),
                              speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=7),
                              compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False)))
        engine = SimpleNamespace(vllm_config=cfg, engine_core=object(),
                                 get_supported_tasks=AsyncMock(return_value=("generate",)),
                                 collective_rpc=AsyncMock(return_value=[state]), do_log_stats=AsyncMock())
        closed, calls = [], []
        @asynccontextmanager
        async def build(args):
            calls.append("native-build")
            try:
                yield engine
            finally:
                closed.append(True)
        serving = SimpleNamespace(default_sampling_params={})
        async def complete(params, raw_request=None):
            calls.append("native-completion")
            async def stream():
                for frame in self.frames(plan, error=fail):
                    yield frame
            return stream()
        serving.create_completion = complete
        async def init(client, state, args, supported_tasks):
            self.assertIs(client, engine)
            self.assertEqual(supported_tasks, ("generate",))
            calls.append("native-init-app-state")
            state.openai_serving_completion = serving
        constructor, modules = self.protocol()
        modules.update({"starlette.datastructures": SimpleNamespace(State=SimpleNamespace),
                        "vllm.entrypoints.openai.api_server": SimpleNamespace(
                            build_async_engine_client=build, init_app_state=init)})
        args = SimpleNamespace(model=str(self.model), served_model_name=["pangu-omni2"], generation_config="auto")
        with patch.dict(sys.modules, modules), patch.object(omni2, "save_result") as save, \
                patch.object(omni2, "versions", return_value={}), patch("sys.stdout", new_callable=io.StringIO):
            if fail:
                with self.assertRaises(RuntimeError):
                    await omni2.run(plan, args)
                save.assert_not_called()
            else:
                await omni2.run(plan, args)
                self.assertEqual(len(save.call_args.args[2]), 2)
                self.assertFalse(save.call_args.args[3]["mome_prefill_state_fix"])
                self.assertEqual(save.call_args.args[4]["frontend"], "SimpleNamespace")
        self.assertEqual(closed, [True])
        self.assertEqual(calls[:2], ["native-build", "native-init-app-state"])
        self.assertEqual(calls.count("native-completion"), 1 if fail else 2)
        # Only audit RPCs at run boundaries. No synchronize RPC or manual step/drain.
        self.assertTrue(all(call.args == ("pangu_bench_worker_state",) for call in engine.collective_rpc.call_args_list))

    async def test_native_server_context_initializes_serving_and_cleans_up(self):
        await self.execute()

    async def test_native_server_context_cleans_up_after_stream_failure(self):
        await self.execute(fail=True)


if __name__ == "__main__":
    unittest.main()
