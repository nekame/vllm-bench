# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Own graph-task handles and workspaces for distinct direct executions.

Adapted from omni/compilation/acl_graph.py; see doc/pangu_infer/UPSTREAM.md.
Omni caches workspaces by physical size and operation. Separate TP layouts
and packed/padded graphs must capture and retain their own workspace sizes.
"""

import torch
from vllm.forward_context import get_forward_context
from omni_npu.compilation.acl_graph import ACLGraphWrapper, get_graph_params


class DirectACLGraphWrapper(ACLGraphWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.owned_tasks = {}
        self.owned_workspaces = {}

    def __call__(self, *args, **kwargs):
        context = get_forward_context()
        descriptor = context.batch_descriptor
        entry = self.concrete_aclgraph_entries.get(descriptor)
        capturing = (context.cudagraph_runtime_mode == self.runtime_mode
                     and (entry is None or entry.aclgraph is None))
        if not capturing:
            return super().__call__(*args, **kwargs)

        params = get_graph_params()
        size = descriptor.num_tokens
        previous_tasks = params.task_entries[size]
        previous_workspaces = params.workspaces[size]
        # A workspace selected for target TP8 is not necessarily large enough
        # for MTP TP2 at the same token count. Recompute it within this graph.
        params.task_entries[size] = {}
        params.workspaces[size] = {}
        try:
            result = super().__call__(*args, **kwargs)
            self.owned_tasks[descriptor] = {
                name: task for name, task in params.task_entries[descriptor.num_tokens].items()
                if name in self.attn_layer_names}
            self.owned_workspaces[descriptor] = params.workspaces[descriptor.num_tokens].copy()
            return result
        finally:
            params.task_entries[size] = previous_tasks
            params.workspaces[size] = previous_workspaces

    def reset_captured_graphs(self):
        super().reset_captured_graphs()
        self.owned_tasks.clear()
        self.owned_workspaces.clear()

    def _update_graph_tasks(self, update_stream, forward_context):
        if forward_context.attn_metadata is None:
            raise RuntimeError("Attention metadata is required to update direct graph tasks")
        descriptor = forward_context.batch_descriptor
        tasks = self.owned_tasks[descriptor]
        workspaces = self.owned_workspaces[descriptor]
        # Same update/event protocol as Omni, using this graph's captured handles.
        with torch.npu.stream(update_stream):
            for layer_name, task in tasks.items():
                update_fn = task.op_desc.compute_dynamic_kwargs
                if update_fn is None:
                    task.event.record(update_stream)
                    continue
                dynamic_kwargs = update_fn(forward_context, layer_name, self.vllm_config)
                if dynamic_kwargs is None:
                    task.event.record(update_stream)
                    continue
                updated_kwargs = {**task.captured_kwargs, **dynamic_kwargs}
                if (updated_kwargs.get("input_layout", "").startswith("BSND")
                        and "actual_seq_lengths" in updated_kwargs):
                    updated_kwargs["actual_seq_lengths"] = None
                torch.npu.graph_task_update_begin(update_stream, task.handle)
                task.op_desc.op_out_fn(
                    **updated_kwargs, workspace=workspaces[task.op_desc.workspace_fn], out=task.out_tensors)
                torch.npu.graph_task_update_end(update_stream)
                task.event.record(update_stream)
