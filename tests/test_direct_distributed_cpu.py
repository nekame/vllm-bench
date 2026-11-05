"""Statistics and sampling must use existing TP communication groups."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.distributed import reduce_cpu_max
from vllm_bench.pangu_infer.sampling import Sampler


class DirectDistributedTests(unittest.TestCase):
    def test_statistics_use_cpu_tensor_and_existing_cpu_group(self):
        values = [.12, .25, .31]
        tensor = SimpleNamespace(tolist=lambda: [.14, .26, .32])
        cpu_group, device_group = object(), object()
        torch = SimpleNamespace(tensor=Mock(return_value=tensor), float64=object(),
                                distributed=SimpleNamespace(all_reduce=Mock(), ReduceOp=SimpleNamespace(MAX=object())))
        modules = {"torch": torch, "vllm.distributed": SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(cpu_group=cpu_group, device_group=device_group))}
        with patch.dict("sys.modules", modules):
            result = reduce_cpu_max(values, 8, dtype=torch.float64)
        self.assertEqual(result, [.14, .26, .32])
        torch.tensor.assert_called_once_with(values, dtype=torch.float64, device="cpu")
        torch.distributed.all_reduce.assert_called_once_with(tensor, op=torch.distributed.ReduceOp.MAX, group=cpu_group)

    def test_tp_one_does_not_allocate_or_reduce_statistics(self):
        torch = SimpleNamespace(tensor=Mock(side_effect=AssertionError("No tensor required")))
        values = [.12, .25]
        with patch.dict("sys.modules", {"torch": torch}):
            self.assertEqual(reduce_cpu_max(values, 1, dtype=object()), values)
        torch.tensor.assert_not_called()

    def test_sample_broadcast_uses_existing_device_group(self):
        device_group = object()
        torch = SimpleNamespace(distributed=SimpleNamespace(broadcast=Mock()))
        modules = {"torch": torch, "vllm.distributed": SimpleNamespace(
            get_tp_group=lambda: SimpleNamespace(device_group=device_group))}
        sampler = Sampler.__new__(Sampler)
        sampler.options = {"tp": 8}
        token = object()
        with patch.dict("sys.modules", modules):
            self.assertIs(sampler.broadcast(token), token)
        torch.distributed.broadcast.assert_called_once_with(token, src=0, group=device_group)


if __name__ == "__main__":
    unittest.main()
