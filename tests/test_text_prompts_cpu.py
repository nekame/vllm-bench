"""Text workload planning, engine inputs and CSV export without ML packages."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer import baseline, matrix, prompts, run
from vllm_bench.pangu_infer.results import save_result
from vllm_bench.pangu_infer.runtime import DirectEngine


class TextPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(json.dumps({
            "architectures": [run.ULTRA_ARCHITECTURE], "num_hidden_layers": 2,
            "num_nextn_predict_layers": 3, "vocab_size": 64}))
        self.operator = self.root / "operator.json"
        self.operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
        self.source = self.root / "prompts.json"
        self.source.write_text(json.dumps([{"id": 0, "input": "abcde"}, {"id": "two", "input": "z"}]))
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
            encode=Mock(side_effect=lambda text, **kwargs: [1] + [ord(char) % 60 + 3 for char in text]),
            decode=Mock(return_value='Ответ, "текст"\nновая строка'))
        patcher = patch.object(prompts, "load_tokenizer", return_value=self.tokenizer)
        self.loader = patcher.start()
        self.addCleanup(patcher.stop)

    def plan(self, *flags):
        return run.make_plan(run.parser().parse_args([
            "--model", str(self.model), "--operator-config", str(self.operator),
            "--output", str(self.root / "output"), "--prompts", str(self.source),
            "--context-length", "4", "--mode", "eager", *flags]))

    def test_truncation_and_left_padding_use_token_counts(self):
        plan = self.plan()
        first, second = plan["prompts"]
        self.assertEqual(first["prompt_ids"], [1, 40, 41, 42])
        self.assertEqual((first["original_tokens"], first["truncated_tokens"], first["padding_tokens"]), (6, 2, 0))
        self.assertEqual(second["prompt_ids"], [0, 0, 1, 5])
        self.assertEqual(second["padding_tokens"], 2)
        self.assertEqual(plan["prompt_ids"], first["prompt_ids"])
        self.tokenizer.encode.assert_any_call("z", add_special_tokens=True)

    def test_eos_fallback_and_empty_input(self):
        self.tokenizer.pad_token_id = None
        self.tokenizer.encode.return_value = []
        self.tokenizer.encode.side_effect = None
        self.assertEqual(self.plan()["prompt_ids"], [2, 2, 2, 2])
        self.tokenizer.eos_token_id = None
        with self.assertRaisesRegex(ValueError, "PAD or EOS"):
            self.plan()

    def test_invalid_rows_and_token_ids_fail_before_launch(self):
        for data in ([], {}, [None], [{"id": 0}], [{"id": True, "input": "a"}],
                     [{"id": 0, "input": "a"}, {"id": "0", "input": "b"}]):
            with self.subTest(data=data):
                self.source.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    self.plan()
        self.source.write_text('[{"id": 0, "input": "a"}]')
        self.tokenizer.encode.side_effect = lambda *args, **kwargs: [64]
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.plan()

    def test_hash_covers_every_prompt_and_matches_both_engines(self):
        direct = self.plan()
        omni = self.plan("--engine", "omni")
        self.assertEqual(direct["comparison_sha256"], omni["comparison_sha256"])
        rows = json.loads(self.source.read_text())
        rows[1]["input"] = "different"
        self.source.write_text(json.dumps(rows))
        changed = self.plan()
        self.assertEqual(direct["prompt_ids"], changed["prompt_ids"])
        self.assertNotEqual(direct["prompt_sha256"], changed["prompt_sha256"])

    def test_matrix_resolves_prompt_and_tokenizer_paths(self):
        _, plans = matrix.plans({"output": "out", "models": ["model"],
            "operator_configs": {"1": "operator.json"}, "tp": [1], "depths": [0], "modes": ["eager"],
            "options": ["--prompts", "prompts.json", "--tokenizer", "model", "--context-length", "4"]}, self.root)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0]["options"]["prompts"], str(self.source))
        self.assertEqual(plans[0]["comparison_sha256"], plans[1]["comparison_sha256"])

    def test_text_batches_reject_trace_mode(self):
        with self.assertRaisesRegex(ValueError, "trace/replay"):
            self.plan("--engine", "omni", "--trace-target", "--temperature", "0", "--repeats", "1")

    def test_csv_contains_generated_text_and_decode_tpot_in_milliseconds(self):
        plan = self.plan()
        output = self.root / "output"
        output.mkdir()
        records = [{"id": 0, "token_ids": [10, 11], "decode_seconds": .125, "decode_tokens": 1},
                   {"id": "two", "token_ids": [2], "decode_seconds": 0, "decode_tokens": 0}]
        save_result(output, plan, records, {}, {})
        with (output / "generations.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, ["id", "text", "tpot"])
            rows = list(reader)
        self.assertEqual(rows[0], {"id": "0", "text": 'Ответ, "текст"\nновая строка', "tpot": "125.0"})
        self.assertEqual(rows[1]["tpot"], "")
        self.tokenizer.decode.assert_any_call([10, 11], skip_special_tokens=True)

    def test_direct_reuses_prompt_storage_and_validates_length(self):
        engine = DirectEngine.__new__(DirectEngine)
        engine.a, engine.device = {"context_length": 4}, "npu:0"
        engine.prompt = Mock()
        tensor = object()
        torch = SimpleNamespace(tensor=Mock(return_value=tensor), int64=object())
        ids = [0, 0, 1, 5]
        with patch.dict("sys.modules", {"torch": torch}):
            engine.set_prompt(ids)
            with self.assertRaises(ValueError):
                engine.set_prompt([1])
        engine.prompt.copy_.assert_called_once_with(tensor)
        torch.tensor.assert_called_once_with(ids, dtype=torch.int64, device="npu:0")

    def test_omni_runs_each_prompt_and_repeat_without_reloading_engine(self):
        plan = self.plan("--engine", "omni", "--repeats", "2", "--warmup", "1")
        path = self.root / "plan.json"
        path.write_text(json.dumps(plan))
        state = {"operator": {}, "parallel": {}, "block_size": 128, "mamba_block_size": 128,
                 "compilation": 0, "cudagraph": "NONE", "compile_counter": {}, "hardware": "NPU"}
        engine = SimpleNamespace(collective_rpc=Mock(return_value=[state]))
        factory = Mock(return_value=engine)
        modules = {"vllm.plugins": SimpleNamespace(load_general_plugins=Mock()),
                   "vllm.engine.arg_utils": SimpleNamespace(EngineArgs=lambda **kwargs: kwargs),
                   "vllm.v1.engine.llm_engine": SimpleNamespace(LLMEngine=SimpleNamespace(from_engine_args=factory))}
        with patch.dict("sys.modules", modules), patch.object(baseline, "require_npu_platform"), \
                patch.object(baseline, "request", side_effect=lambda *args, **kwargs: {"decode_seconds": .1}) as request, \
                patch.object(baseline, "save_result") as save, patch.object(baseline, "versions", return_value={}):
            baseline.main(["--plan", str(path)])
        factory.assert_called_once()
        self.assertEqual(request.call_count, 5)
        self.assertEqual([call.kwargs["prompt_ids"] for call in request.call_args_list[1:]],
                         [row["prompt_ids"] for row in plan["prompts"] for _ in range(2)])
        self.assertEqual([(row["id"], row["repeat"]) for row in save.call_args.args[2]],
                         [(0, 1), (0, 2), ("two", 1), ("two", 2)])


if __name__ == "__main__":
    unittest.main()
