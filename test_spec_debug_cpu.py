"""CPU-only checks; no torch/vLLM installation needed."""
import types
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import spec_debug


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class DebugTests(unittest.TestCase):
    def test_prefix_acceptance_and_bonus_multiple_requests(self):
        output, accepted = spec_debug.expected_greedy(
            [10, 11, 12, 20, 21], [10, 99, 12, 20, 21], [100, 200], [3, 5])
        self.assertEqual(output, [[10, 99], [20, 21, 200]])
        self.assertEqual(accepted, [1, 2])

    def test_empty_draft(self):
        self.assertEqual(spec_debug.expected_greedy([], [], [7], [0]), ([[7]], [0]))

    def test_actual_verifier_hook_and_corrupt_output_detection(self):
        def compute_probs_and_sample(*args, **kwargs):
            return None

        def rejection_greedy_sample_native(output_token_ids, cu_num_draft_tokens,
                                           draft_token_ids, target_argmax,
                                           bonus_token_ids, is_greedy, max_spec_len):
            output_token_ids.value[:] = [[10, 99, -1]]
            return Tensor([1])

        class Sampler:
            def forward(self, *args, **kwargs):
                return None

        module = types.SimpleNamespace(compute_probs_and_sample=compute_probs_and_sample,
                                       rejection_greedy_sample_native=rejection_greedy_sample_native,
                                       NPURejectionSampler=Sampler)
        with patch.dict("os.environ", {"PANGU_SPEC_DEBUG_DIR": "/tmp/unused-spec-debug"}):
            spec_debug.patch_sampler(module)
        for target, expected in (([10, 99], True), ([10, 98], False)):
            event = {}
            token = spec_debug.ACTIVE.set(event)
            try:
                module.rejection_greedy_sample_native(
                    Tensor([]), Tensor([2]), Tensor([10, 11]),
                    Tensor(target), Tensor([[100]]), None, 2)
            finally:
                spec_debug.ACTIVE.reset(token)
            self.assertEqual(event["verify"]["matches_reference"], expected)

    def test_forward_arming_and_limit(self):
        def compute_probs_and_sample(logits, *args, **kwargs):
            return Tensor([10]), logits

        def rejection_greedy_sample_native(output_token_ids, cu_num_draft_tokens,
                                           draft_token_ids, target_argmax,
                                           bonus_token_ids, is_greedy, max_spec_len):
            output_token_ids.value[:] = [[10, 20]]
            return Tensor([1])

        class IndexedLogits:
            def __getitem__(self, index):
                return self

        class Sampler:
            def forward(self, metadata, draft_probs, logits, sampling_metadata):
                module.compute_probs_and_sample(logits)
                result = Tensor([])
                module.rejection_greedy_sample_native(
                    result, Tensor([1]), Tensor([10]), Tensor([10]), Tensor([[20]]), None, 1)
                return types.SimpleNamespace(sampled_token_ids=result)

        module = types.SimpleNamespace(compute_probs_and_sample=compute_probs_and_sample,
                                       rejection_greedy_sample_native=rejection_greedy_sample_native,
                                       NPURejectionSampler=Sampler)
        dist = types.ModuleType("torch.distributed")
        dist.is_initialized = lambda: True
        dist.get_rank = lambda: 0
        torch = types.ModuleType("torch")
        torch.__path__ = []
        torch.distributed = dist
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {
                "PANGU_SPEC_DEBUG_DIR": directory, "PANGU_SPEC_DEBUG_LIMIT": "1"}), \
                patch.dict("sys.modules", {"torch": torch, "torch.distributed": dist}), \
                patch.object(spec_debug, "top2", return_value={"rows": []}):
            spec_debug.patch_sampler(module)
            sampler = Sampler()
            metadata = types.SimpleNamespace(num_draft_tokens=[1],
                                             target_logits_indices=Tensor([0]),
                                             bonus_logits_indices=Tensor([1]))
            sampling = types.SimpleNamespace(all_greedy=True, temperature=Tensor([0]))
            sampler.forward(metadata, None, IndexedLogits(), sampling)
            self.assertFalse(list(Path(directory).glob("*.jsonl")))
            (Path(directory) / "ARM").touch()
            (Path(directory) / "LABEL").write_text("request-1")
            for _ in range(2):
                sampler.forward(metadata, None, IndexedLogits(), sampling)
            lines = list(Path(directory).glob("*.jsonl"))[0].read_text().splitlines()
            self.assertEqual(len(lines), 1)
            event = json.loads(lines[0])
            self.assertEqual(event["label"], "request-1")
            self.assertTrue(event["verify"]["matches_reference"])
            self.assertEqual(event["sampler_output_with_padding"], [[10, 20]])


if __name__ == "__main__":
    unittest.main()
