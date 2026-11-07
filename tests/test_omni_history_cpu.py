"""Request/cache lifecycle experiments without model weights or ML imports."""

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer import baseline, run
from vllm_bench.pangu_infer.results import prefix_cache_groups, save_result
from vllm_bench.pangu_infer.runtime import engine_options


class OmniHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({
            "architectures": [run.ULTRA_ARCHITECTURE], "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1, "vocab_size": 64, "max_position_embeddings": 16384}))
        operator = self.root / "operator.json"
        operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
        self.args = ["--engine", "omni", "--model", str(model), "--operator-config", str(operator),
                     "--output", str(self.root / "output"), "--enable-prefix-caching",
                     "--context-length", "8", "--depth", "7", "--temperature", "0"]

    def plan(self, *flags):
        return run.make_plan(run.parser().parse_args([*self.args, *flags]))

    def test_guards_and_comparison_identity(self):
        plain = self.plan()
        self.assertEqual(plain["comparison_sha256"], self.plan("--omni-cache-history", "reuse",
                         "--omni-warmup-tokens", "17")["comparison_sha256"])
        for flags in (("--omni-cache-history", "cold-first"), ("--omni-cache-history", "cold-each"),
                      ("--warmup", "0"), ("--omni-warmup-tokens", "64")):
            item = self.plan(*flags)
            self.assertNotEqual(plain["comparison_sha256"], item["comparison_sha256"])
        for flags in (("--engine", "direct", "--omni-cache-history", "cold-first"),
                      ("--engine", "direct", "--warmup", "0"),
                      ("--warmup", "-1"), ("--omni-warmup-tokens", "1"),
                      ("--warmup", "0", "--omni-warmup-tokens", "32")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.plan(*flags)
        args = [arg for arg in self.args if arg != "--enable-prefix-caching"]
        with self.assertRaisesRegex(ValueError, "enable-prefix-caching"):
            run.make_plan(run.parser().parse_args([*args, "--omni-cache-history", "cold-each"]))
        # Cache lifecycle changes do not silently alter model EngineArgs.
        self.assertEqual(engine_options(plain, direct=False),
                         engine_options(self.plan("--omni-cache-history", "cold-first"), direct=False))

    def test_custom_warmup_reserves_context_capacity(self):
        plan = self.plan("--max-new-tokens", "32", "--omni-warmup-tokens", "512")
        self.assertEqual(plan["max_model_len"], 8 + 512 + 8)
        with self.assertRaisesRegex(ValueError, "at least"):
            self.plan("--max-new-tokens", "32", "--omni-warmup-tokens", "512", "--max-model-len", "64")

    def execute(self, plan, *, changed_counters=False):
        events, cache = [], {"hit": False}
        state = {"operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 128,
                 "compilation": 0, "cudagraph": "NONE", "compile_counter": {}, "hardware": "NPU",
                 "enable_prefix_caching": True, "async_scheduling": plan["options"].get("async_scheduling", False)}
        after = deepcopy(state)
        if changed_counters:
            after["compile_counter"]["num_graphs_seen"] = 1
        def reset():
            events.append("reset")
            cache["hit"] = False
            return True
        engine = SimpleNamespace(collective_rpc=Mock(side_effect=[[state], [after]]),
                                 has_unfinished_requests=lambda: False, reset_prefix_cache=reset)
        def request(engine, plan, request_id, limit, **kwargs):
            events.append((request_id, limit, cache["hit"]))
            cached = 4 if cache["hit"] else 0
            cache["hit"] = True
            return {"cached_prompt_tokens": cached, "token_ids": [13, 330 if cached else 25],
                    "decode_seconds": .01, "decode_tokens": 1, "accepted_drafts": [2]}
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        with patch.dict(sys.modules, {"vllm.plugins": SimpleNamespace(load_general_plugins=lambda: None)}), \
                patch.object(baseline, "require_npu_platform"), patch.object(baseline, "create_engine", return_value=engine), \
                patch.object(baseline, "request", side_effect=request), patch.object(baseline, "versions", return_value={}), \
                patch.object(baseline, "save_result") as save, patch("sys.stdout", new_callable=io.StringIO):
            baseline.main(["--plan", str(path)])
        return events, save.call_args.args

    def test_prefix_reset_order_and_actual_hits_in_sync_async_and_process_modes(self):
        for flags in ((), ("--async-scheduling",), ("--omni-multiprocessing",),
                      ("--async-scheduling", "--omni-multiprocessing")):
            for history, expected in (("reuse", [4, 4, 4]), ("cold-first", [0, 4, 4]),
                                      ("cold-each", [0, 0, 0])):
                with self.subTest(flags=flags, history=history):
                    events, saved = self.execute(self.plan(*flags, "--warmup", "1", "--repeats", "3",
                                                          "--omni-cache-history", history))
                    records = saved[2]
                    self.assertEqual([r["cached_prompt_tokens"] for r in records], expected)
                    self.assertEqual([r["prefix_cache_reset"] for r in records],
                                     [history == "cold-each" or (history == "cold-first" and i == 0) for i in range(3)])
                    self.assertEqual(events[0], ("warmup-0", 17, False))
                    self.assertEqual(events.count("reset"), {"reuse": 0, "cold-first": 1, "cold-each": 3}[history])
                    self.assertTrue(all(r["decode_seconds"] == .01 for r in records))
                    self.assertEqual(saved[3]["omni_cache_history"], history)

    def test_zero_warmup_does_not_issue_hidden_requests_and_is_diagnostic(self):
        plan = self.plan("--warmup", "0", "--repeats", "3")
        events, saved = self.execute(plan, changed_counters=True)
        self.assertFalse(any(isinstance(e, tuple) and e[0].startswith("warmup") for e in events))
        self.assertEqual([r["cached_prompt_tokens"] for r in saved[2]], [0, 4, 4])
        self.assertTrue(saved[4]["compiled_during_measurement"])
        output = self.root / "result"
        output.mkdir()
        with patch("sys.stdout", new_callable=io.StringIO):
            result = save_result(output, plan, saved[2], saved[3], saved[4])
        self.assertEqual(result["measurement"], "diagnostic")
        self.assertFalse(result["repeat_consistency"]["matches"])
        self.assertTrue(result["prefix_cache_groups"]["hit"]["repeat_consistency"]["matches"])

    def test_warmup_length_is_forwarded_and_normal_benchmark_still_rejects_compilation(self):
        events, saved = self.execute(self.plan("--omni-warmup-tokens", "128"))
        self.assertEqual(events[:2], [("warmup-0", 128, False), ("warmup-1", 128, True)])
        self.assertEqual(saved[4]["warmup_max_tokens"], 128)
        with self.assertRaisesRegex(RuntimeError, "compiled/captured during measurement"):
            self.execute(self.plan(), changed_counters=True)

    def test_reset_refusal_and_inflight_requests_fail_before_measurement(self):
        for active in (False, True):
            engine = SimpleNamespace(has_unfinished_requests=lambda: active, reset_prefix_cache=Mock(return_value=False))
            with self.assertRaises(RuntimeError):
                baseline.reset_prefix_cache(engine, "measure-0")
            self.assertEqual(engine.reset_prefix_cache.call_count, 0 if active else 1)

    def test_grouped_statistics_preserve_global_indices_and_unknown_cache_counts(self):
        records = [{"cached_prompt_tokens": cached, "token_ids": ids, "decode_seconds": duration,
                    "decode_tokens": 1, "accepted_drafts": [accepted]}
                   for cached, ids, duration, accepted in ((0, [1], .01, 0), (4, [2], .002, 7),
                                                          (4, [3], .004, 0), (None, [4], .1, 1))]
        groups = prefix_cache_groups(records, 7, greedy=True)
        self.assertEqual(groups["miss"]["record_indices"], [0])
        self.assertEqual(groups["hit"]["record_indices"], [1, 2])
        self.assertEqual(groups["hit"]["median_tpot_ms"], 3)
        self.assertEqual(groups["hit"]["acceptance_rate"], .5)
        difference = groups["hit"]["repeat_consistency"]["first_differences"][0]
        self.assertEqual((difference["reference_record"], difference["candidate_record"]), (1, 2))
        self.assertEqual(groups["unknown"]["requests"], 1)


if __name__ == "__main__":
    unittest.main()
