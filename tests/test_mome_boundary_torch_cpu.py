"""Real Omni builder + reference convolution, tiny CPU tensors, no model weights.

Passing reproduction tests mean that the current-source counterexample was
reproduced, not that the current builder is correct. The local adapter is
checked against an uninterrupted oracle and a temporary patched-source control.
"""

from contextlib import ExitStack
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

from mome_cpu_support import Fixture, MOME_FILE, REFERENCE_FILE, load_local_methods, load_original_methods, patched_copy


ROOT = Path(__file__).resolve().parents[1]


def metadata_values(metadata):
    fields = ("num_computed_tokens", "num_accepted_tokens", "cache_indices",
              "block_idx_last_computed_token", "block_idx_first_scheduled_token",
              "block_idx_last_scheduled_token", "num_actual_tokens", "max_query_len",
              "num_decodes", "num_prefills")
    return {name: value.tolist() if hasattr(value, "tolist") else value
            for name in fields for value in (getattr(metadata, name),)}


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU PyTorch is required for numerical MoME tests")
class MomeBoundaryTests(unittest.TestCase):
    omni_source = ROOT.parent / "omniinfer"
    fix_patch = ROOT / "patches/omniinfer/mome-target-prefill-state.patch"
    observations = []

    @classmethod
    def setUpClass(cls):
        import torch

        for relative in (MOME_FILE, REFERENCE_FILE):
            if not (cls.omni_source / relative).is_file():
                raise unittest.SkipTest("Omni source checkout required; use scripts/check_mome_cpu.py --omni-source")
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.torch, cls.observations = torch, []
        cls.original = cls.stack.enter_context(load_original_methods(cls.omni_source, torch, name="original"))
        directory = cls.stack.enter_context(tempfile.TemporaryDirectory(prefix="pangu-mome-cpu-"))
        cls.fixed_source = patched_copy(cls.omni_source, directory, cls.fix_patch)
        cls.source_hashes = {"original_builder": hashlib.sha256((cls.omni_source / MOME_FILE).read_bytes()).hexdigest(),
                             "patched_builder": hashlib.sha256((cls.fixed_source / MOME_FILE).read_bytes()).hexdigest(),
                             "local_builder": hashlib.sha256((ROOT / "src/vllm_bench/pangu_infer/mome_prefill.py").read_bytes()).hexdigest(),
                             "reference_convolution": hashlib.sha256((cls.omni_source / REFERENCE_FILE).read_bytes()).hexdigest(),
                             "candidate_patch": hashlib.sha256(cls.fix_patch.read_bytes()).hexdigest()}
        cls.patched = cls.stack.enter_context(load_original_methods(cls.fixed_source, torch, name="patched"))
        cls.corrected = cls.stack.enter_context(load_local_methods(cls.original))

    def fixture(self, methods=None, **kwargs):
        return Fixture(self.torch, self.original if methods is None else methods, **kwargs)

    def oracle(self, fixture, prompt, query, weight):
        output, _ = self.original.convolution(self.torch.cat([prompt, query]), weight, None, None,
                                               activation=None)
        return output[fixture.context:]

    def assert_exact(self, actual, expected):
        self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_aligned_prefill_transition_selects_different_history_in_current_and_fixed_sources(self):
        for depth in (3, 7):
            with self.subTest(depth=depth):
                original = self.fixture(depth=depth)
                corrected = self.fixture(self.corrected, depth=depth)
                prefill = original.metadata(7680, 512)
                self.assertEqual(prefill.block_idx_last_scheduled_token.tolist(), [63])
                a = original.metadata(8192, depth + 1)
                b = corrected.metadata(8192, depth + 1)
                self.assertEqual(a.block_idx_last_computed_token.tolist(), [64])
                self.assertEqual(a.num_accepted_tokens.tolist(), [1])
                self.assertEqual(b.block_idx_last_computed_token.tolist(), [63])
                self.assertEqual(b.num_accepted_tokens.tolist(), [depth + 1])
                self.observations.append({"case": "aligned_prefill", "depth": depth,
                    "current": {"initial_block": 64, "accepted": 1},
                    "corrected": {"initial_block": 63, "accepted": depth + 1}})

    def test_recycled_future_page_changes_original_output_but_not_corrected_output(self):
        torch = self.torch
        for depth in (3, 7):
            for persistent in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(depth=depth, persistent=persistent, dtype=dtype):
                        fixture = self.fixture(depth=depth, persistent=persistent)
                        prompt, query, weight = fixture.tensors(dtype)
                        expected = self.oracle(fixture, prompt, query, weight)
                        outputs, fixed_outputs = [], []
                        for poison in (123, -456):
                            cache = fixture.cache_after_prefill(prompt, weight, poison=poison)
                            metadata = fixture.metadata(fixture.context, depth + 1)
                            outputs.append(fixture.forward(query, weight, cache, metadata))
                            corrected = self.fixture(self.corrected, depth=depth, persistent=persistent)
                            fixed_cache = corrected.cache_after_prefill(prompt, weight, poison=poison)
                            fixed_metadata = corrected.metadata(corrected.context, depth + 1)
                            fixed_outputs.append(corrected.forward(query, weight, fixed_cache, fixed_metadata))
                        self.assertFalse(torch.equal(outputs[0], outputs[1]))
                        self.assertFalse(torch.equal(outputs[0], expected))
                        # Width-three convolution can depend on old history only in its first two rows.
                        self.assert_exact(outputs[0][2:], expected[2:])
                        for output in fixed_outputs:
                            self.assert_exact(output, expected)
                        self.assert_exact(fixed_outputs[0], fixed_outputs[1])
                        self.observations.append({"case": "future_page_reuse", "depth": depth,
                            "persistent_cpu_buffers": persistent, "dtype": str(dtype),
                            "current_outputs_equal": False, "corrected_matches_reference": True,
                            "current_output_max_abs_diff": float((outputs[0].float() - outputs[1].float()).abs().max())})

    def test_apc_off_offset_can_be_wrong_even_when_the_page_is_valid(self):
        torch = self.torch
        for depth in (3, 7):
            with self.subTest(depth=depth):
                fixture = self.fixture(depth=depth, apc=False)
                prompt, query, weight = fixture.tensors(torch.float32)
                cache = fixture.cache_after_prefill(prompt, weight, poison=0)
                # A completely valid rolling window still needs its final two rows after prefill.
                cache[1].copy_(prompt[-(depth + 2):])
                expected = self.oracle(fixture, prompt, query, weight)
                current = fixture.forward(query, weight, cache, fixture.metadata(fixture.context, depth + 1))
                corrected = self.fixture(self.corrected, depth=depth, apc=False)
                fixed = corrected.forward(query, weight, cache, corrected.metadata(corrected.context, depth + 1))
                self.assertFalse(torch.equal(current, expected))
                self.assert_exact(fixed, expected)
                self.observations.append({"case": "apc_off_rolling_window", "depth": depth,
                    "current_matches_reference": False, "corrected_matches_reference": True})

    def test_target_only_is_a_correct_control_with_and_without_apc(self):
        torch = self.torch
        for apc in (False, True):
            with self.subTest(apc=apc):
                fixture = self.fixture(depth=0, apc=apc)
                prompt, query, weight = fixture.tensors(torch.float32)
                cache = fixture.cache_after_prefill(prompt, weight, poison=123)
                output = fixture.forward(query, weight, cache, fixture.metadata(fixture.context, 1))
                self.assert_exact(output, self.oracle(fixture, prompt, query, weight))

    def test_chunked_prefill_history_matches_uninterrupted_reference(self):
        torch = self.torch
        for methods in (self.original, self.corrected):
            with self.subTest(methods=methods.builder.__module__):
                fixture = self.fixture(methods)
                prompt, query, weight = fixture.tensors(torch.float32)
                cache = torch.zeros((fixture.columns + 1, 9, fixture.channels), device="cpu")
                pieces, start = [], 0
                for count in (2048, 2048, 2048, 1536, 512):
                    metadata = fixture.metadata(start, count).prefill
                    initial = int(metadata.block_idx_last_computed_token[0])
                    last = int(metadata.block_idx_last_scheduled_token[0])
                    history = None if start == 0 else cache[int(fixture.table[0, initial])].unsqueeze(0)
                    output, _ = methods.convolution(prompt[start:start + count], weight, None, history,
                        return_final_states=True,
                        final_states_out=cache[int(fixture.table[0, last])].unsqueeze(0), activation=None)
                    pieces.append(output)
                    start += count
                expected, _ = methods.convolution(prompt, weight, None, None, activation=None)
                self.assert_exact(torch.cat(pieces), expected)
                tail = cache[int(fixture.table[0, 63]), -2:]
                self.assert_exact(tail, prompt[-2:])

    def test_draft_metadata_and_later_target_steps_are_unchanged_by_candidate_fix(self):
        for depth in (3, 7):
            for apc in (False, True):
                for persistent in (False, True):
                    for start in (8191, 8192, 8193):
                        a = self.fixture(depth=depth, apc=apc, persistent=persistent)
                        b = self.fixture(self.corrected, depth=depth, apc=apc, persistent=persistent)
                        for draft_index, accepted in ((0, None), (1, 1)):
                            with self.subTest(depth=depth, apc=apc, start=start, draft_index=draft_index):
                                before = a.metadata(start, depth + 1, accepted=accepted, draft_index=draft_index)
                                after = b.metadata(start, depth + 1, accepted=accepted, draft_index=draft_index)
                                self.assertEqual(metadata_values(before), metadata_values(after))
                    for accepted in (1, depth + 1):
                        before = a.metadata(8321, depth + 1, accepted=accepted)
                        after = b.metadata(8321, depth + 1, accepted=accepted)
                        self.assertEqual(metadata_values(before), metadata_values(after))

    def test_pd_boundary_metadata_is_unchanged_by_candidate_fix(self):
        for depth in (3, 7):
            a = self.fixture(depth=depth, pd=True)
            b = self.fixture(self.corrected, depth=depth, pd=True)
            self.assertEqual(metadata_values(a.metadata(8192, depth + 1)),
                             metadata_values(b.metadata(8192, depth + 1)))

    def test_local_adapter_matches_reviewed_patch_for_target_and_draft_metadata(self):
        for depth in (3, 7):
            for apc in (False, True):
                for persistent in (False, True):
                    for pd in (False, True):
                        a = self.fixture(self.patched, depth=depth, apc=apc, persistent=persistent, pd=pd)
                        b = self.fixture(self.corrected, depth=depth, apc=apc, persistent=persistent, pd=pd)
                        for start in (0, 7680, 8192, 8193, 8321):
                            for draft_index, accepted in ((None, 1), (0, None), (1, 1)):
                                with self.subTest(depth=depth, apc=apc, persistent=persistent, pd=pd,
                                                  start=start, draft_index=draft_index):
                                    before = a.metadata(start, depth + 1, accepted=accepted, draft_index=draft_index)
                                    after = b.metadata(start, depth + 1, accepted=accepted, draft_index=draft_index)
                                    self.assertEqual(metadata_values(before), metadata_values(after))

    def test_cpu_persistent_buffers_are_used_and_metadata_views_alias_them(self):
        fixture = self.fixture(persistent=True)
        first = fixture.metadata(8192, 8)
        self.assertEqual(first.num_accepted_tokens.device.type, "cpu")
        self.assertEqual(first.cache_indices.data_ptr(), fixture.builder.cache_indices_tensor.data_ptr())
        self.assertEqual(first.num_accepted_tokens.data_ptr(), fixture.builder.num_accepted_tokens.data_ptr())
        self.assertEqual(first.block_idx_last_computed_token.tolist(), [64])
        fixture.metadata(8321, 8)
        self.assertEqual(first.block_idx_last_computed_token.tolist(), [65])

    def test_contexts_on_both_sides_of_a_block_boundary_match_reference_after_fix(self):
        torch = self.torch
        for context in (8184, 8191, 8192, 8193, 8320):
            for depth in (3, 7):
                with self.subTest(context=context, depth=depth):
                    fixture = self.fixture(self.corrected, context=context, depth=depth)
                    prompt, query, weight = fixture.tensors(torch.float32)
                    cache = fixture.cache_after_prefill(prompt, weight, poison=456)
                    output = fixture.forward(query, weight, cache, fixture.metadata(context, depth + 1))
                    self.assert_exact(output, self.oracle(fixture, prompt, query, weight))


if __name__ == "__main__":
    unittest.main()
