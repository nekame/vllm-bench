"""Per-model compilation configuration without torch or NPU execution."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.compilation import MtpNpuGraphExAdaptor, without_static_kernels


class MtpCompilationTests(unittest.TestCase):
    def test_config_copy_disables_both_static_switches_and_preserves_shared_registry(self):
        options = {"enable": True, "static_kernel_compile": True,
                   "super_kernel_optimize": True, "frozen_parameter": True, "clone_input": False}
        registry = object()
        original = SimpleNamespace(additional_config={"npugraph_ex_config": options, "other": 7},
                                   compilation_config=registry)
        draft = without_static_kernels(original)
        self.assertIs(draft.compilation_config, registry)
        self.assertIsNot(draft.additional_config, original.additional_config)
        self.assertTrue(options["static_kernel_compile"])
        self.assertTrue(options["super_kernel_optimize"])
        self.assertEqual(draft.additional_config["npugraph_ex_config"], {
            **options, "static_kernel_compile": False, "super_kernel_optimize": False})
        self.assertEqual(draft.additional_config["other"], 7)

    def test_backend_keeps_optimizations_and_passes_without_static_kernel_coordination(self):
        for limit in (64, 16):
            with self.subTest(capture_limit=limit):
                config = SimpleNamespace(debug=SimpleNamespace(aclgraph=SimpleNamespace()))
                compiled, graph, inputs, post_pass = object(), object(), [object()], object()
                backend = Mock(return_value=compiled)
                package = SimpleNamespace(CompilerConfig=lambda: config, get_npu_backend=Mock(return_value=backend))
                options = {"enable": True, "static_kernel_compile": True, "super_kernel_optimize": True,
                           "frozen_parameter": True, "clone_input": False, "clone_output": True,
                           "remove_noop_ops": False, "inplace_pass": False, "input_inplace_pass": False,
                           "pattern_fusion_pass": False, "capture_limit": limit}
                modules = {"npugraph_ex": package,
                    "torch._inductor.compile_fx": SimpleNamespace(graph_returns_tuple=lambda graph: True),
                    "omni_npu.compilation.npugraph_ex": SimpleNamespace(
                        _tracing_context_matching_inputs=lambda inputs: nullcontext(),
                        graph_output_is_tuple=Mock(side_effect=AssertionError("already tuple")))}
                with patch.dict("sys.modules", modules):
                    result = MtpNpuGraphExAdaptor(options).compile(
                        graph, inputs, {"post_grad_custom_post_pass": post_pass})
                self.assertEqual(result, (compiled, None))
                backend.assert_called_once_with(graph, inputs)
                package.get_npu_backend.assert_called_once_with(compiler_config=config)
                self.assertFalse(config.static_kernel_compile)
                self.assertFalse(config.super_kernel_optimize)
                self.assertTrue(config.force_eager)
                self.assertEqual(config.mode, "npugraph_ex")
                for name in ("frozen_parameter", "clone_input", "clone_output", "remove_noop_ops",
                             "inplace_pass", "input_inplace_pass", "pattern_fusion_pass"):
                    self.assertEqual(getattr(config, name), options[name])
                self.assertIs(config.post_grad_custom_post_pass, post_pass)
                if limit != 64:
                    self.assertEqual(config.debug.aclgraph.static_capture_size_limit, limit)
                self.assertTrue(options["static_kernel_compile"])


if __name__ == "__main__":
    unittest.main()
