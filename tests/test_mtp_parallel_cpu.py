"""Separate MTP groups, proposal handoff and timing alignment without NPU."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer import run, runtime
from vllm_bench.pangu_infer.mtp_parallel import MtpParallel, record_idle_mtp
from vllm_bench.pangu_infer.runtime import DirectEngine


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 8


class Tensor:
    def __init__(self, values):
        self.values = deepcopy(values)

    @property
    def shape(self):
        return (len(self.values),)

    def copy_(self, other):
        self.values = deepcopy(other.values)
        return self


def fixture(rank=0, tp=2, temperature=0):
    cfg = SimpleNamespace(parallel_config=ParallelConfig(), compilation_config=object())
    target = SimpleNamespace(local_rank=rank, world_size=8, device_group=object())
    state = SimpleNamespace(_TP=target, _EP=object(), _MOE_DISPATCH_EP=object(), cfg=cfg)
    created = []

    def init_group(ranks, local_rank, backend, group_name):
        members = next(group for group in ranks if rank in group)
        group = SimpleNamespace(world_size=len(members), rank_in_group=members.index(rank),
                                device_group=object(), destroy=Mock())
        created.append((deepcopy(ranks), group_name, group))
        return group

    @contextmanager
    def tp_context(group):
        old, state._TP = state._TP, group
        try:
            yield
        finally:
            state._TP = old

    @contextmanager
    def config_context(config):
        old, state.cfg = state.cfg, config
        try:
            yield
        finally:
            state.cfg = old

    def empty(*shape, **kwargs):
        return Tensor([0] * shape[0] if len(shape) == 1 else [[0] * shape[1] for _ in range(shape[0])])

    state.get_tp_group = lambda: state._TP
    state.init_model_parallel_group = init_group
    state.patch_tensor_parallel_group = tp_context
    torch = SimpleNamespace(empty=empty, int64="int64", float32="float32",
                            distributed=SimpleNamespace(broadcast=Mock()))
    modules = {"torch": torch, "vllm.config": SimpleNamespace(set_current_vllm_config=config_context),
               "vllm.distributed": SimpleNamespace(parallel_state=state),
               "vllm.distributed.parallel_state": state}
    plan = {"options": {"tp": 8, "mtp_tp": tp, "depth": 7, "temperature": temperature},
            "config": {"vocab_size": 4}}
    return cfg, target, state, created, torch, modules, plan


class MtpGroupTests(unittest.TestCase):
    def test_rank_partition_and_target_restored_on_exception(self):
        for tp in (1, 2, 4):
            for rank in range(8):
                with self.subTest(tp=tp, rank=rank):
                    cfg, target, state, created, torch, modules, plan = fixture(rank, tp)
                    old_ep, old_dispatch = state._EP, state._MOE_DISPATCH_EP
                    with patch.dict("sys.modules", modules):
                        mtp = MtpParallel(plan, cfg, "npu:0", rank)
                        self.assertEqual(mtp.active, rank < tp)
                        self.assertEqual([row[1] for row in created], ["tp", "ep", "moe_dispatch_ep"])
                        self.assertEqual(created[0][0], [list(range(tp))] + [[r] for r in range(tp, 8)])
                        if mtp.active:
                            with self.assertRaisesRegex(ValueError, "forward failed"):
                                with mtp.context():
                                    self.assertIs(state._TP, mtp.tp_group)
                                    self.assertIs(state._EP, mtp.ep_group)
                                    self.assertIs(state._MOE_DISPATCH_EP, mtp.dispatch_group)
                                    self.assertEqual(state.cfg.parallel_config.tensor_parallel_size, tp)
                                    self.assertEqual(cfg.parallel_config.tensor_parallel_size, 8)
                                    raise ValueError("forward failed")
                        else:
                            with self.assertRaisesRegex(RuntimeError, "does not own MTP"):
                                with mtp.context():
                                    pass
                        self.assertIs(state._TP, target)
                        self.assertIs(state._EP, old_ep)
                        self.assertIs(state._MOE_DISPATCH_EP, old_dispatch)
                        self.assertIs(state.cfg, cfg)
                        self.assertIs(mtp.cfg.compilation_config, cfg.compilation_config)
                        mtp.close()
                        for _, _, group in created:
                            group.destroy.assert_called_once()

    def test_proposals_and_sampling_distributions_use_target_group(self):
        for temperature in (0, 1):
            with self.subTest(temperature=temperature):
                cfg, target, state, created, torch, modules, plan = fixture(temperature=temperature)
                with patch.dict("sys.modules", modules):
                    mtp = MtpParallel(plan, cfg, "npu:0", 0)
                    ids, probs = Tensor(list(range(7))), Tensor([[.25] * 4 for _ in range(7)])
                    output = mtp.broadcast((ids, probs if temperature else None))
                self.assertEqual(output[0].values, ids.values)
                self.assertEqual(torch.distributed.broadcast.call_count, 2 if temperature else 1)
                for call in torch.distributed.broadcast.call_args_list:
                    self.assertIs(call.kwargs["group"], target.device_group)
                    self.assertEqual(call.kwargs["src"], 0)
                if temperature:
                    self.assertEqual(output[1].values, probs.values)
                else:
                    self.assertIsNone(output[1])

    def test_inactive_rank_receives_without_executing_or_sampling(self):
        cfg, target, state, created, torch, modules, plan = fixture(rank=7)
        with patch.dict("sys.modules", modules):
            mtp = MtpParallel(plan, cfg, "npu:7", 7)
            engine = DirectEngine.__new__(DirectEngine)
            engine.mtp, engine.a = mtp, plan["options"]
            engine.plan = {"config": {"num_nextn_predict_layers": 3}}
            engine.model_timings = SimpleNamespace(cycle=1, records=[])
            engine._propose = Mock(side_effect=AssertionError("Inactive rank executed MTP"))
            expected = list(range(7))
            torch.distributed.broadcast.side_effect = lambda tensor, **kwargs: tensor.copy_(Tensor(expected))
            ids, probs = engine.propose(Tensor(list(range(8))), object(), 8192, object(), 0, object(), 1)
        self.assertEqual(ids.values, expected)
        self.assertEqual(len(engine.model_timings.records), 7)
        self.assertEqual([row["input_tokens"] for row in engine.model_timings.records], [8, 1, 1, 1, 1, 1, 1])
        engine._propose.assert_not_called()

    def test_active_proposal_runs_in_mtp_scope_and_handoff_in_target_scope(self):
        cfg, target, state, created, torch, modules, plan = fixture()
        with patch.dict("sys.modules", modules):
            mtp = MtpParallel(plan, cfg, "npu:0", 0)
            engine = DirectEngine.__new__(DirectEngine)
            engine.mtp = mtp
            proposal = Tensor(list(range(7))), None
            def propose(*args):
                self.assertIs(state._TP, mtp.tp_group)
                return proposal
            def broadcast(tensor, **kwargs):
                self.assertIs(state._TP, target)
            engine._propose = Mock(side_effect=propose)
            torch.distributed.broadcast.side_effect = broadcast
            result = engine.propose(object(), object(), 8192, object(), 0, object(), 1)
        engine._propose.assert_called_once()
        self.assertEqual(result[0].values, list(range(7)))

    def test_idle_timing_rows_cover_multi_head_padding_and_prefill_exclusion(self):
        for depth, count, pad, expected in ((3, 3, False, [4, 4, 4]), (3, 1, False, [4, 1, 1]),
                                            (7, 3, True, [8] * 7), (7, 3, False, [8, 1, 1, 1, 1, 1, 1])):
            timing = SimpleNamespace(cycle=None, records=[])
            a = {"depth": depth, "mtp_pad_to_8": pad}
            record_idle_mtp(timing, a, count, depth + 1)
            self.assertEqual(timing.records, [])
            timing.cycle = 2
            record_idle_mtp(timing, a, count, depth + 1)
            self.assertEqual([row["input_tokens"] for row in timing.records], expected)
            self.assertTrue(all(row["wall_ms"] == 0 for row in timing.records))


class MtpPlanningTests(unittest.TestCase):
    def test_modes_depths_and_tp_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"architectures": [run.ULTRA_ARCHITECTURE],
                "num_hidden_layers": 2, "num_nextn_predict_layers": 3, "vocab_size": 32}))
            operator = root / "operator.json"
            operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
            args = ["--model", str(root), "--operator-config", str(operator), "--output", str(root / "out"), "--tp", "8"]
            def plan(*flags):
                return run.make_plan(run.parser().parse_args([*args, *flags]))
            normal = plan()
            self.assertEqual(normal["comparison_sha256"], plan("--mtp-tp", "8")["comparison_sha256"])
            for tp in (1, 2, 4):
                for depth in (3, 7):
                    for mode in run.MODES:
                        self.assertEqual(plan("--mtp-tp", str(tp), "--depth", str(depth), "--mode", mode)["options"]["mtp_tp"], tp)
            self.assertNotEqual(normal["comparison_sha256"], plan("--mtp-tp", "2")["comparison_sha256"])
            for flags in (("--mtp-tp", "2", "--engine", "omni"), ("--mtp-tp", "2", "--no-mtp"),
                          ("--mtp-tp", "4", "--tp", "2")):
                with self.assertRaises(ValueError):
                    plan(*flags)
            operator.write_text('{"model_parallel_config": {"layer_parallel_config": {"q_proj": {}}}, "operator_optimization_config": {}}')
            with self.assertRaisesRegex(ValueError, "layer_parallel_config"):
                plan("--mtp-tp", "2")


class MtpLoadingTests(unittest.TestCase):
    def test_only_active_ranks_construct_load_and_pack_mtp_under_its_groups(self):
        for rank in (0, 7):
            with self.subTest(rank=rank):
                cfg, target_group, state, created, torch, modules, plan = fixture(rank)
                cfg.model_config = SimpleNamespace(dtype="bf16")
                plan["options"]["model"] = "/tmp/model"
                plan["config"]["num_hidden_layers"] = 12
                events = []
                class Model:
                    def __init__(self, label):
                        self.label, self.parameter = label, object()
                        events.append(("construct", label, state._TP.world_size))
                    def parameters(self):
                        return iter([self.parameter])
                    def named_parameters(self):
                        return iter([("weight", self.parameter)])
                    def load_weights(self, weights):
                        list(weights)
                        events.append(("load", self.label, state._TP.world_size))
                        return {"weight"}
                    def post_weight_load(self):
                        events.append(("post_load", self.label, state._TP.world_size))
                    def eval(self):
                        return self
                target_factory = Mock(side_effect=lambda **kwargs: Model("target"))
                def draft_factory(**kwargs):
                    self.assertIsNone(kwargs["shared_target"])
                    self.assertEqual(kwargs["vllm_config"].parallel_config.tensor_parallel_size, 2)
                    return Model("draft")
                draft_factory_mock = Mock(side_effect=draft_factory)
                torch.get_default_dtype = lambda: "float32"
                torch.set_default_dtype = Mock()
                torch.device = lambda value: nullcontext()
                torch.nn = SimpleNamespace(ModuleDict=lambda models: models)
                torch.npu = SimpleNamespace(synchronize=Mock())
                state.prepare_communication_buffer_for_model = Mock()
                modules.update({
                    "vllm.model_executor.model_loader.utils": SimpleNamespace(
                        process_weights_after_loading=lambda model, *args: events.append(("pack", "target" if isinstance(model, dict) else "draft", state._TP.world_size))),
                    "omni_npu.v1.layers.attention.weight_utils": SimpleNamespace(refresh_mla_wrappers=Mock()),
                    "vllm_bench.pangu_infer.model.pangu_ultra_moe": SimpleNamespace(PanguUltraMoEForCausalLM=target_factory),
                    "vllm_bench.pangu_infer.model.pangu_ultra_moe_mtp": SimpleNamespace(OpenPanguMTP=draft_factory_mock),
                })
                with patch.dict("sys.modules", modules), patch.object(runtime, "weight_iterator", return_value=iter(())) as weights:
                    mtp = MtpParallel(plan, cfg, "npu:0", rank)
                    target, draft, details = runtime.load_models(plan, cfg, "npu:0", mtp=mtp)
                self.assertTrue(all(size == (2 if label == "draft" else 8) for _, label, size in events))
                target_factory.assert_called_once()
                if rank == 0:
                    draft_factory_mock.assert_called_once()
                    self.assertIsNotNone(draft)
                    self.assertTrue(weights.call_args.kwargs["include_shared"])
                    self.assertEqual(details["draft_loaded_parameters"], 1)
                else:
                    draft_factory_mock.assert_not_called()
                    self.assertIsNone(draft)
                    self.assertEqual(weights.call_count, 1)
                    self.assertEqual(details["draft_loaded_parameters"], 0)
                self.assertIs(state._TP, target_group)


if __name__ == "__main__":
    unittest.main()
