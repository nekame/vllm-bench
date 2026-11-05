"""Chunk boundaries, initial MTP inputs and decode timing without NPU."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer import baseline, run
from vllm_bench.pangu_infer.runtime import DirectEngine, engine_options


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    @property
    def shape(self):
        return (len(self.values),)

    def __getitem__(self, index):
        return Tensor(self.values[index]) if isinstance(index, slice) else self.values[index]

    def clone(self):
        return Tensor(self.values)


def engine_fixture(chunk=16):
    engine = DirectEngine.__new__(DirectEngine)
    engine.a = {"context_length": 64, "prefill_chunk_size": chunk, "depth": 7,
                "mode": "compile-acl", "profile": False}
    engine.cfg = object()
    engine.caches, engine.target, engine.draft = object(), object(), object()
    engine.device, engine.model_timings, engine.executions = "npu:0", None, {}
    engine.names = {"target": ["model.layers.0.attn"], "draft": ["draft_model.model.layers.50.attn"]}
    engine.plan = {"config": {"num_hidden_layers": 50, "num_nextn_predict_layers": 3}}
    calls, states = [], {}

    class Execution:
        def __init__(self, model, cfg, caches, names, size, device, **kwargs):
            self.kind = "target" if model is engine.target else "draft"
            self.mode, self.head = kwargs["mode"], kwargs["head"]
            self.query_size = kwargs["query_size"]
            self.output = Tensor([])

        def __call__(self, ids, start, accepted, prompt, **kwargs):
            key = self.kind, self.head
            position, cumulative = states.get(key, (0, 0))
            if start != position:
                raise AssertionError("Chunk does not continue the same cache history")
            inputs = kwargs["hidden"]
            outputs = []
            for index, token in enumerate(ids.values):
                cumulative += token + (inputs.values[index] if inputs is not None else 0)
                outputs.append(cumulative)
            states[key] = start + len(ids.values), cumulative
            calls.append({"kind": self.kind, "start": start, "ids": ids.values,
                          "hidden": inputs.values if inputs is not None else None,
                          "index": kwargs["sample_index"], "accepted": accepted,
                          "head": self.head, "draft_index": kwargs["draft_index"], "mode": self.mode})
            self.output.values = outputs
            return self.output, Tensor(outputs[-1:])

    torch = SimpleNamespace(tensor=lambda values, **kwargs: Tensor(values), int64="int64",
                            cat=lambda tensors, dim=0: Tensor(value for tensor in tensors for value in tensor.values))
    return engine, calls, torch, Execution


class ChunkedPrefillTests(unittest.TestCase):
    def test_target_preserves_order_and_cache_history_then_runs_compiled_decode(self):
        engine, calls, torch, execution = engine_fixture()
        with patch.dict("sys.modules", {"torch": torch}), patch("vllm_bench.pangu_infer.execution.Execution", execution):
            hidden, logits = engine.forward("target", Tensor(range(1, 65)), 0, None)
            self.assertEqual(hidden.values, [index * (index + 1) // 2 for index in range(1, 65)])
            self.assertEqual(logits.values, [2080])
            self.assertEqual([call["start"] for call in calls], [0, 16, 32, 48])
            self.assertTrue(all(call["mode"] == "eager" for call in calls))
            engine.forward("target", Tensor([1] * 8), 64, 1)
        self.assertEqual(calls[-1]["mode"], "compile-acl")
        self.assertEqual(calls[-1]["start"], 64)

    def test_mtp_chunks_keep_shifted_ids_hidden_rows_acceptance_and_head(self):
        engine, calls, torch, execution = engine_fixture()
        shifted, previous = Tensor(range(64)), Tensor(range(100, 164))
        with patch.dict("sys.modules", {"torch": torch}), patch("vllm_bench.pangu_infer.execution.Execution", execution):
            hidden, logits = engine.forward("draft", shifted, 0, 1, hidden=previous,
                                             index=Tensor([63]), head=0, draft_index=0)
        self.assertEqual([value for call in calls for value in call["ids"]], shifted.values)
        self.assertEqual([value for call in calls for value in call["hidden"]], previous.values)
        self.assertTrue(all(call["index"].values == [15] for call in calls))
        self.assertTrue(all(call["accepted"] == 1 and call["head"] == 0 and call["draft_index"] == 0 for call in calls))
        self.assertEqual(hidden.shape, (64,))
        self.assertEqual(logits.values, [sum(shifted.values) + sum(previous.values)])

    def test_default_prefill_stays_one_eager_call(self):
        engine, calls, torch, execution = engine_fixture(chunk=0)
        with patch.dict("sys.modules", {"torch": torch}), patch("vllm_bench.pangu_infer.execution.Execution", execution):
            engine.forward("target", Tensor(range(64)), 0, None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]["ids"]), 64)
        self.assertEqual(calls[0]["mode"], "eager")

    def test_engine_args_and_workload_identity_include_chunking_for_both_engines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"architectures": [run.ULTRA_ARCHITECTURE],
                "num_hidden_layers": 2, "num_nextn_predict_layers": 3, "vocab_size": 32}))
            operator = root / "operator.json"
            operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
            args = ["--model", str(root), "--operator-config", str(operator), "--output", str(root / "out"),
                    "--context-length", "64", "--depth", "7"]
            def plan(*flags):
                return run.make_plan(run.parser().parse_args([*args, *flags]))
            plain = plan()
            direct = plan("--prefill-chunk-size", "16")
            omni = plan("--prefill-chunk-size", "16", "--engine", "omni")
            self.assertEqual(direct["comparison_sha256"], omni["comparison_sha256"])
            self.assertNotEqual(direct["comparison_sha256"], plain["comparison_sha256"])
            self.assertEqual(direct["max_model_len"], plain["max_model_len"])
            for item, is_direct in ((direct, True), (omni, False)):
                options = engine_options(item, direct=is_direct)
                self.assertTrue(options["enable_chunked_prefill"])
                self.assertEqual(options["max_num_batched_tokens"], 16)
            for size in ("-1", "8", "20"):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    plan("--prefill-chunk-size", size)

    def test_omni_decode_cycles_exclude_every_prefill_step(self):
        state = {"step": 0}
        def step():
            state["step"] += 1
            if state["step"] < 4:
                return []
            tokens = [10] if state["step"] == 4 else [10, 11]
            return [SimpleNamespace(request_id="request", outputs=[SimpleNamespace(token_ids=tokens)])]
        engine = SimpleNamespace(collective_rpc=Mock(), add_request=Mock(), step=step,
                                 has_unfinished_requests=lambda: state["step"] < 5)
        plan = {"prompt_ids": [1] * 64, "options": {"temperature": 0, "top_k": 0, "top_p": 1,
            "seed": 0, "depth": 7, "respect_eos": False, "context_length": 64}}
        with patch.dict("sys.modules", {"vllm": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs)}):
            result = baseline.request(engine, plan, "request", 2)
        self.assertEqual(result["cycles"], 1)
        self.assertEqual(result["decode_tokens"], 1)
        self.assertEqual(result["accepted_drafts"], [0])


if __name__ == "__main__":
    unittest.main()
