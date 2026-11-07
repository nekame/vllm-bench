"""Local adapter wiring and API contracts without model/framework imports.

Numerical behavior is covered separately with real CPU torch.
"""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.baseline import create_engine


ROOT = Path(__file__).resolve().parents[1] / "src/vllm_bench/pangu_infer"


def load(name):
    spec = importlib.util.spec_from_file_location("_test_local_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MomeAdapterTests(unittest.TestCase):
    def test_default_engine_uses_original_factory(self):
        for multiprocessing in (False, True):
            args = Mock()
            factory = Mock()
            modules = {"vllm.engine.arg_utils": SimpleNamespace(EngineArgs=args),
                       "vllm.v1.engine.llm_engine": SimpleNamespace(LLMEngine=factory)}
            with patch.dict(sys.modules, modules):
                result = create_engine({"options": {"omni_multiprocessing": multiprocessing}}, {"model": "model"})
            args.assert_called_once_with(model="model")
            factory.from_engine_args.assert_called_once_with(
                args.return_value, **({"enable_multiprocessing": True} if multiprocessing else {}))
            args.return_value.create_engine_config.assert_not_called()
            factory.assert_not_called()
            self.assertIs(result, factory.from_engine_args.return_value)

    def test_corrected_engine_selects_worker_after_native_config_creation(self):
        for multiprocessing in (False, True):
            cfg = SimpleNamespace(parallel_config=SimpleNamespace(worker_cls="native.Worker"))
            args = Mock(return_value=SimpleNamespace(create_engine_config=lambda: cfg, disable_log_stats=True))
            factory, executor = Mock(), Mock()
            modules = {"vllm.engine.arg_utils": SimpleNamespace(EngineArgs=args),
                       "vllm.v1.engine.llm_engine": SimpleNamespace(LLMEngine=factory),
                       "vllm.v1.executor": SimpleNamespace(Executor=executor)}
            options = {"model": "model", "worker_extension_cls": "named.RpcExtension"}
            with patch.dict(sys.modules, modules):
                create_engine({"options": {"fix_mome_prefill_state": True,
                                           "omni_multiprocessing": multiprocessing}}, options)
            args.assert_called_once_with(**options)
            self.assertEqual(cfg.parallel_config.worker_cls,
                             "vllm_bench.pangu_infer.omni_worker.MomePrefillWorker")
            executor.get_class.assert_called_once_with(cfg)
            factory.assert_called_once_with(vllm_config=cfg, executor_class=executor.get_class.return_value,
                                            log_stats=False, multiprocess_mode=multiprocessing)
            factory.from_engine_args.assert_not_called()

    def builder_module(self):
        class Original:
            def build(self, *args, **kwargs):
                return kwargs
        modules = {"omni_npu.attention.backends.mome":
                   SimpleNamespace(NPUMomeAttentionMetadataBuilder=Original)}
        with patch.dict(sys.modules, modules):
            return load("mome_prefill")

    def test_target_normalization_preserves_caller_counts_and_resolves_bound_prompt(self):
        cls = self.builder_module().MomePrefillMetadataBuilder
        builder = cls()
        builder.runner_num_prompt_tokens = [8192]
        builder.fake_num_spec, builder.is_pd_disagg = 7, False
        computed, accepted = Mock(), Mock()
        common = SimpleNamespace(num_reqs=1, compute_num_computed_tokens=lambda: computed)
        with patch.object(type(computed), "__eq__", create=True, return_value="boundary-mask"):
            result = builder.build(0, common, num_accepted_tokens=accepted)
        accepted.masked_fill.assert_called_once_with("boundary-mask", 8)
        self.assertIs(result["num_accepted_tokens"], accepted.masked_fill.return_value)
        self.assertEqual(result["num_prompt_tokens"], [8192])

    def test_draft_pd_and_missing_prompt_do_not_normalize(self):
        cls = self.builder_module().MomePrefillMetadataBuilder
        for mode in ("draft", "draft-step0", "pd", "unbound"):
            builder = cls()
            builder.runner_num_prompt_tokens = [8192]
            builder.fake_num_spec, builder.is_pd_disagg = 7, mode == "pd"
            accepted = Mock()
            common = SimpleNamespace(num_reqs=1, compute_num_computed_tokens=Mock())
            if mode == "draft":
                result = builder.build_for_drafting(common, 1, accepted, [8192])
            else:
                result = builder.build(0, common, num_accepted_tokens=accepted,
                                       draft_step0=mode == "draft-step0",
                                       allow_runner_num_prompt_tokens=mode != "unbound")
            accepted.masked_fill.assert_not_called()
            common.compute_num_computed_tokens.assert_not_called()
            self.assertIs(result["num_accepted_tokens"], accepted)

    def worker_module(self):
        class Original:
            pass
        class Local(Original):
            def __init__(self, *args):
                self.constructor_args = args
        class Worker:
            def initialize_from_config(self, config):
                self.initialized = config
        modules = {"omni_npu.attention.backends.mome": SimpleNamespace(NPUMomeAttentionMetadataBuilder=Original),
                   "omni_npu.worker.npu_worker": SimpleNamespace(NPUWorker=Worker),
                   "vllm_bench.pangu_infer.mome_prefill": SimpleNamespace(MomePrefillMetadataBuilder=Local)}
        with patch.dict(sys.modules, modules):
            return load("omni_worker"), Original, Local

    def test_worker_replaces_only_mome_and_refreshes_native_drafter_references(self):
        module, original, local = self.worker_module()
        old = original()
        old.kv_cache_spec, old.runner_num_prompt_tokens = object(), object()
        unrelated = object()
        group = SimpleNamespace(metadata_builders=[old, unrelated], layer_names=["mome"], spec=object())
        runner = SimpleNamespace(attn_groups=[[group]], vllm_config=object(), device="npu:0",
                                 kv_cache_config=object(), drafter=SimpleNamespace(initialize_attn_backend=Mock()))
        worker = module.MomePrefillWorker()
        worker.use_v2_model_runner, worker.model_runner = False, runner
        worker.initialize_from_config(runner.kv_cache_config)
        replacement = group.metadata_builders[0]
        self.assertIsInstance(replacement, local)
        self.assertEqual(replacement.constructor_args,
                         (old.kv_cache_spec, group.layer_names, runner.vllm_config, runner.device))
        self.assertIs(replacement.runner_num_prompt_tokens, old.runner_num_prompt_tokens)
        self.assertIs(group.metadata_builders[1], unrelated)
        runner.drafter.initialize_attn_backend.assert_called_once_with(runner.kv_cache_config)
        self.assertIs(worker.initialized, runner.kv_cache_config)
        self.assertTrue(worker.mome_prefill_state_fix)
        self.assertEqual(worker.mome_prefill_builder_count, 1)
        self.assertEqual(module.install_mome_builders(runner), 1)
        self.assertIs(group.metadata_builders[0], replacement)

    def test_worker_rejects_absent_mome_builders(self):
        module, _, _ = self.worker_module()
        with self.assertRaisesRegex(RuntimeError, "no MoME"):
            module.install_mome_builders(SimpleNamespace(attn_groups=[]))


if __name__ == "__main__":
    unittest.main()
