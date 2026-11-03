"""Direct torch.compile and ACL Graph calls on persistent input buffers."""

from contextlib import nullcontext
import time


class Execution:
    def __init__(self, model, cfg, caches, names, size, device, *, head=None, mode="eager", profile=False):
        import torch

        self.cfg, self.size, self.head = cfg, size, head
        self.mode = mode
        self.profile = profile
        self.lane = caches.lane(names, size)
        self.ids = torch.empty(size, dtype=torch.int64, device=device)
        self.hidden = (torch.empty(size, cfg.model_config.hf_config.hidden_size,
                                   dtype=cfg.model_config.dtype, device=device)
                       if head is not None else None)
        self.sample_index = torch.empty(1, dtype=torch.int64, device=device)
        self.compiles = 0
        self.calls = 0
        self.captures = 0
        self.replays = 0
        self.backend_compile_seconds = 0
        self.capture_seconds = 0

        class Step(torch.nn.Module):
            def __init__(step):
                super().__init__()
                step.model = model

            def forward(step, ids, positions, previous, sample_index):
                if head is None:
                    hidden = step.model(ids, positions)
                    # Prefill needs only the final distribution; verification
                    # and target-only decode need every input position.
                    depth = getattr(cfg.speculative_config, "num_speculative_tokens", 0)
                    sample = hidden[-1:] if size > depth + 1 else hidden
                    return hidden, step.model.compute_logits(sample)
                hidden = step.model(ids, positions, previous, spec_step_idx=head)
                return hidden, step.model.compute_logits(hidden.index_select(0, sample_index), spec_step_idx=head)

        self.step = Step().eval()
        self.runnable = self.step
        if "compile" in mode:
            from omni_npu.compilation.npugraph_ex import NpuGraphExAdaptor
            backend = NpuGraphExAdaptor()

            def compile_backend(graph, inputs):
                self.compiles += 1
                started = time.perf_counter()
                compiled, _ = backend.compile(graph, inputs, compiler_config={})
                self.backend_compile_seconds += time.perf_counter() - started
                if compiled is None:
                    raise RuntimeError("NPU backend returned no compiled callable")
                return compiled

            self.runnable = torch.compile(self.step, backend=compile_backend,
                                          fullgraph=True, dynamic=False)
        self.wrapper = None
        if "acl" in mode:
            from vllm.config import CUDAGraphMode
            from vllm.compilation.cuda_graph import CUDAGraphOptions
            from omni_npu.compilation.acl_graph import ACLGraphWrapper

            self.wrapper = ACLGraphWrapper(
                self.runnable, cfg, CUDAGraphMode.FULL,
                cudagraph_options=CUDAGraphOptions(gc_disable=False, weak_ref_output=False),
                update_stream=torch.npu.Stream(), attn_layer_names=names)
            self.runnable = self.wrapper

    def __call__(self, ids, start, accepted, prompt, *, hidden=None, sample_index=None, draft_index=None,
                 model_timing=None):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context, get_forward_context, BatchDescriptor

        with torch.profiler.record_function("direct/prepare_buffers_metadata") if self.profile else nullcontext():
            self.ids.copy_(ids)
            if self.hidden is not None:
                self.hidden.copy_(hidden)
                self.sample_index.copy_(sample_index)
            metadata, slots = self.lane.prepare(start, accepted, prompt, draft_index)
        mode = CUDAGraphMode.FULL if self.wrapper else CUDAGraphMode.NONE
        descriptor = BatchDescriptor(num_tokens=self.size, num_reqs=1, uniform=True)
        with set_forward_context(
            attn_metadata=metadata, vllm_config=self.cfg, num_tokens=self.size,
            cudagraph_runtime_mode=mode, batch_descriptor=descriptor, slot_mapping=slots):
            get_forward_context().capturing = False
            capturing = False
            if self.wrapper:
                entry = self.wrapper.concrete_aclgraph_entries.get(descriptor)
                if entry is not None and entry.aclgraph is not None:
                    self.replays += 1
                else:
                    # Compile and allocate workspaces before entering capture.
                    self.wrapper.runnable(self.ids, self.lane.positions, self.hidden, self.sample_index)
                    torch.npu.synchronize()
                    self.captures += 1
                    capturing = True
            started = time.perf_counter() if capturing else None
            label = "direct/graph_replay_update" if self.wrapper else "direct/model_launch"
            with torch.profiler.record_function(label) if self.profile else nullcontext():
                if model_timing is not None and model_timing.cycle is not None:
                    result = model_timing.measure(
                        self.runnable, (self.ids, self.lane.positions, self.hidden, self.sample_index),
                        model="target" if self.head is None else "mtp", input_tokens=self.size,
                        step=draft_index, head=self.head)
                else:
                    result = self.runnable(self.ids, self.lane.positions, self.hidden, self.sample_index)
            if capturing:
                torch.npu.synchronize()
                self.capture_seconds += time.perf_counter() - started
        self.calls += 1
        return result
