"""Tests of rejection association and target/MoME hooks, without torch/vLLM."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import spec_rejection as debug
from summarize_rejections import summary


def event(cycle, accepted=1, request="r1", label="test"):
    return {"cycle": cycle, "rank": 0, "label": label,
            "target_forward": {"request_ids": [request],
                               "runner_accepted_gpu": {"values": [2], "truncated": False}},
            "verify": {"accepted": [accepted], "cu_num_draft_tokens": [1]},
            "sampler_output_with_padding": [[10, 20] if accepted else [10, -1]]}


class WindowTests(unittest.TestCase):
    def test_rejection_at_arbitrary_cycle_and_next_count(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = debug.Recorder(directory)
            recorder.record(event(38))
            recorder.record(event(39, accepted=0))
            path = list(Path(directory).rglob("cycle-*.json"))[0]
            self.assertEqual(json.loads(path.read_text())["status"], "awaiting_next_verification")
            following = event(40)
            following["target_forward"]["runner_accepted_gpu"]["values"] = [1]
            recorder.record(following)
            window = json.loads(path.read_text())
            self.assertEqual(window["previous"]["cycle"], 38)
            self.assertEqual(window["next"]["cycle"], 40)
            self.assertTrue(window["next_runner_count_matches_output"])
            self.assertEqual(summary(window)["expected_next_runner_count"], [1])

    def test_bad_count_and_consecutive_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = debug.Recorder(directory)
            recorder.record(event(0, accepted=0))
            recorder.record(event(1, accepted=0))
            recorder.record(event(2))
            windows = [json.loads(p.read_text()) for p in sorted(Path(directory).rglob("cycle-*.json"))]
            self.assertEqual(len(windows), 2)
            self.assertTrue(all(w["status"] == "complete" for w in windows))
            self.assertTrue(all(w["next_runner_count_matches_output"] is False for w in windows))

    def test_no_association_across_request_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = debug.Recorder(directory)
            recorder.record(event(0, accepted=0))
            recorder.record(event(1, request="r2"))
            window = json.loads(list(Path(directory).rglob("cycle-*.json"))[0].read_text())
            self.assertEqual(window["status"], "request_changed_before_next_verification")
            self.assertIsNone(window["next"])

    def test_new_request_gets_no_previous_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = debug.Recorder(directory)
            recorder.record(event(0))
            recorder.record(event(1, accepted=0, request="r2"))
            window = json.loads(list(Path(directory).rglob("cycle-*.json"))[0].read_text())
            self.assertIsNone(window["previous"])

    def test_attach_consumes_target_and_fails_if_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = debug.Recorder(directory)
            recorder.latest_target = {"input_ids": [1, 2]}
            output = {}
            recorder.attach(output)
            self.assertEqual(output["target_forward"]["input_ids"], [1, 2])
            with self.assertRaisesRegex(RuntimeError, "no target forward"):
                recorder.attach({})

    def test_runtime_target_and_mome_hooks(self):
        class ColumnParallelMOMERL:
            prefix = "model.layers.0.self_attn.qa_conv"

            def forward(self, x, conv_states, mome_metadata, inplace=True):
                return x + 1

        conv = ColumnParallelMOMERL()
        meta = types.SimpleNamespace(num_accepted_tokens=1, num_computed_tokens=100)
        context = types.SimpleNamespace(attn_metadata={"layer0": meta})
        forward_context = types.ModuleType("vllm.forward_context")
        forward_context.get_forward_context = lambda: context

        class Runner:
            input_batch = types.SimpleNamespace(num_reqs=1, req_ids=["r1"],
                                                num_accepted_tokens_cpu=[1], num_computed_tokens_cpu=[100])
            model_config = types.SimpleNamespace(is_hybrid=True, enforce_eager=True)
            cache_config = types.SimpleNamespace(enable_prefix_caching=False)
            num_accepted_tokens = types.SimpleNamespace(gpu=[1])

            def _model_forward(self, input_ids=None, positions=None):
                return conv.forward(5, None, meta)

            def get_model(self):
                return types.SimpleNamespace(named_modules=lambda: [("conv", conv)])

        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
                "PANGU_SPEC_DEBUG_DIR": directory, "PANGU_SPEC_DEBUG_MOME_LAYERS": "0",
                "PANGU_SPEC_DEBUG_STATE_FINGERPRINTS": "0"}), \
                patch.dict("sys.modules", {"vllm.forward_context": forward_context}), \
                patch.object(debug, "selected_rank", return_value=True), \
                patch.object(debug, "_RECORDER", None):
            runner = Runner()
            debug.install_runner(runner)
            self.assertEqual(runner._model_forward([10, 11], [100, 101]), 6)
            self.assertIsNone(debug.recorder().latest_target)
            (Path(directory) / "ARM").touch()
            self.assertEqual(runner._model_forward([10, 11], [100, 101]), 6)
            frame = debug.recorder().latest_target
            self.assertEqual(frame["input_ids"]["values"], [10, 11])
            self.assertEqual(frame["request_ids"], ["r1"])
            self.assertEqual(frame["mome_calls"][0]["metadata"]["num_accepted_tokens"], 1)
            self.assertIsNone(debug.TARGET.get())
            runner.model_config.enforce_eager = False
            runner._pangu_rejection_installed = False
            with self.assertRaisesRegex(RuntimeError, "enforce-eager"):
                debug.install_runner(runner)


if __name__ == "__main__":
    unittest.main()
