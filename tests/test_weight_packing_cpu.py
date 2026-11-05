"""Packing failures preserve the operator exception and identify bad shards."""

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.weights import pack_weights, packing_context


class WeightPackingTests(unittest.TestCase):
    def model(self):
        bad = SimpleNamespace(prefix="model.layers.0.mlp.down_proj", tp_size=8,
                              weight=SimpleNamespace(shape=(5120, 2016)),
                              weight_scale=SimpleNamespace(shape=(5120, 63)))
        good = SimpleNamespace(weight=SimpleNamespace(shape=(5120, 4032)),
                               weight_scale=SimpleNamespace(shape=(5120, 126)))
        packed = SimpleNamespace(weight=SimpleNamespace(shape=(2016, 5120), is_mxfp8_packed=True),
                                 weight_scale=SimpleNamespace(shape=(32, 5120, 2)))
        return SimpleNamespace(named_modules=lambda: iter([
            ("target.model.layers.0.mlp.down_proj", bad), ("aligned", good), ("packed", packed), ("norm", object())]))

    def test_context_contains_actual_tp_prefix_and_shapes(self):
        result = packing_context(self.model(), "target")
        self.assertEqual(result["scope"], "target")
        self.assertEqual(len(result["unaligned_linears"]), 1)
        row = result["unaligned_linears"][0]
        self.assertEqual(row["prefix"], "model.layers.0.mlp.down_proj")
        self.assertEqual(row["tp_size"], 8)
        self.assertEqual(row["weight_shape"], [5120, 2016])
        self.assertEqual(row["scale_shape"], [5120, 63])

    def test_failure_keeps_original_exception_and_prints_context(self):
        error = AssertionError("num_blocks (63) must be even")
        process = Mock(side_effect=error)
        module = SimpleNamespace(process_weights_after_loading=process)
        output = io.StringIO()
        with patch.dict("sys.modules", {"vllm.model_executor.model_loader.utils": module}), redirect_stdout(output):
            with self.assertRaises(AssertionError) as caught:
                pack_weights(self.model(), object(), "npu:0", scope="target")
        self.assertIs(caught.exception, error)
        row = json.loads(output.getvalue().removeprefix("[weight-pack] "))
        self.assertEqual(row["scope"], "target")
        self.assertEqual(row["unaligned_linears"][0]["tp_size"], 8)
        process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
