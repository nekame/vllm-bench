# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Per-execution compilation settings without changing Omni's global config.

The non-static backend path is adapted from omni/compilation/npugraph_ex.py.
See doc/pangu_infer/UPSTREAM.md.
"""

import copy


def without_static_kernels(cfg):
    cfg = copy.copy(cfg)
    cfg.additional_config = dict(cfg.additional_config or {})
    options = dict(cfg.additional_config.get("npugraph_ex_config", {}))
    options.update(static_kernel_compile=False, super_kernel_optimize=False)
    cfg.additional_config["npugraph_ex_config"] = options
    return cfg


class MtpNpuGraphExAdaptor:
    """Omni's non-static compilation path with explicit MTP options."""

    def __init__(self, options):
        self.options = dict(options)

    def compile(self, graph, example_inputs, compiler_config):
        import npugraph_ex
        from torch._inductor.compile_fx import graph_returns_tuple
        from omni_npu.compilation.npugraph_ex import (
            _tracing_context_matching_inputs, graph_output_is_tuple,
        )

        if not self.options.get("enable", False):
            raise ValueError("MTP compilation requires npugraph_ex_config.enable=true")
        if not graph_returns_tuple(graph) and not graph_output_is_tuple(graph):
            output_node = graph.graph.output_node()
            with graph.graph.inserting_before(output_node):
                tuple_node = graph.graph.create_node("call_function", tuple, args=([output_node.args[0]],))
            output_node.args = (tuple_node,)
            graph.recompile()

        config = npugraph_ex.CompilerConfig()
        config.mode = "npugraph_ex"
        config.force_eager = True
        config.static_kernel_compile = False
        config.super_kernel_optimize = False
        if self.options.get("capture_limit", 64) != 64:
            config.debug.aclgraph.static_capture_size_limit = self.options["capture_limit"]
        for name, default in (
            ("clone_input", True), ("clone_output", False), ("remove_noop_ops", True),
            ("inplace_pass", True), ("input_inplace_pass", True),
            ("pattern_fusion_pass", True), ("frozen_parameter", False),
        ):
            setattr(config, name, self.options.get(name, default))
        for name in ("post_grad_custom_post_pass", "post_grad_custom_pre_pass"):
            if name in compiler_config:
                setattr(config, name, compiler_config[name])
        backend = npugraph_ex.get_npu_backend(compiler_config=config)
        with _tracing_context_matching_inputs(example_inputs):
            compiled = backend(graph, example_inputs)
        return compiled, None
