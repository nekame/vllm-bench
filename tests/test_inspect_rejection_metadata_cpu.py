"""Read real capture tree structure using mocked CPU tensor payloads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from vllm_bench.mtp_profile.inspect_rejection_metadata import inspect_snapshot


class InspectRejectionMetadataTests(unittest.TestCase):
    def snapshot(self, root, fields):
        before = root / "rank-000/before"
        before.mkdir(parents=True)
        (before / "tree.json").write_text(json.dumps({"$dict": [
            ["inputs", {"$dict": list(fields.items())}],
            ["cache_config", {"$object": "unavailable.module:NotImported"}]]}))
        return before

    def test_selects_tensor_payloads_without_decoding_runtime_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = self.snapshot(Path(tmp), {
                "num_rejected_tokens_gpu": {"$tensor": "3", "device": "npu:0"},
                "token_indices_to_sample": {"$tensor": "4", "device": "npu:0"},
                "sampling_metadata": {"$object": "unavailable.module:NotImported"}})
            loader = Mock(return_value={"3": Mock(tolist=lambda: [4]), "4": Mock(tolist=lambda: [3])})
            self.assertEqual(inspect_snapshot(before, tensor_loader=loader), {
                "num_rejected_tokens_gpu": [4], "token_indices_to_sample": [3]})
            loader.assert_called_once_with(before / "tensors.pt")

    def test_null_inputs_need_no_torch_or_tensor_file_and_accept_rank_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = self.snapshot(Path(tmp), {"num_rejected_tokens_gpu": None, "token_indices_to_sample": None})
            loader = Mock(side_effect=AssertionError("Unexpected tensor load"))
            self.assertEqual(inspect_snapshot(before.parent, tensor_loader=loader), {
                "num_rejected_tokens_gpu": None, "token_indices_to_sample": None})
            loader.assert_not_called()

    def test_missing_field_and_missing_tensor_are_errors_instead_of_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            before = self.snapshot(Path(tmp), {"token_indices_to_sample": None})
            with self.assertRaisesRegex(ValueError, "Missing recorded input"):
                inspect_snapshot(before)
            path = before / "tree.json"
            tree = json.loads(path.read_text())
            tree["$dict"][0][1]["$dict"].insert(0, ["num_rejected_tokens_gpu", {"$tensor": "absent"}])
            path.write_text(json.dumps(tree))
            with self.assertRaisesRegex(ValueError, "Missing tensor"):
                inspect_snapshot(before, tensor_loader=lambda _: {})
