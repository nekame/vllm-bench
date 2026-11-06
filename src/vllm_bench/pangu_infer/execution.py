"""Direct torch.compile and ACL Graph calls on persistent input buffers."""

from contextlib import nullcontext
import time


class Execution:
    def __init__(self, model, cfg, caches, names, size, device, *, head=None, mode="eager", profile=False,
                 query_size=None, isolate_graph_tasks=False, memory_diagnostics=False,
                 disable_static_kernels=False):
        import torch

        if disable_static_kernels:
            from vllm_bench.pangu_infer.compilation import without_static_kernels
            cfg = without_static_kernels(cfg)
        self.cfg, self.size, self.head = cfg, size, head
        self.query_size = size if query_size is None else query_size
        if not 1 <= self.query_size <= size or (self.query_size < size and head is None):
            raise ValueError("Padding requires an MTP execution with a valid query size")
        self.mode = mode
        self.profile = profile
        self.device = device
        self.memory_diagnostics = memory_diagnostics
        self.memory_phase = "model"
        self.lane = caches.lane(names, size, query_size=self.query_size, cfg=cfg)
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
            if disable_static_kernels:
                from vllm_bench.pangu_infer.compilation import MtpNpuGraphExAdaptor
                backend = MtpNpuGraphExAdaptor(cfg.additional_config["npugraph_ex_config"])
            else:
                from omni_npu.compilation.npugraph_ex import NpuGraphExAdaptor
                backend = NpuGraphExAdaptor()

            def compile_backend(graph, inputs):
                self.memory_phase = "backend_compile"
                self.log_memory("compile_begin")
                self.compiles += 1
                started = time.perf_counter()
                compiled, _ = backend.compile(graph, inputs, compiler_config={})
                self.backend_compile_seconds += time.perf_counter() - started
                if compiled is None:
                    raise RuntimeError("NPU backend returned no compiled callable")
                self.memory_phase = "compiled_forward"
                self.log_memory("compile_end")
                return compiled

            self.runnable = torch.compile(self.step, backend=compile_backend,
                                          fullgraph=True, dynamic=False)
        self.wrapper = None
        if "acl" in mode:
            from vllm.config import CUDAGraphMode
            from vllm.compilation.cuda_graph import CUDAGraphOptions
            from omni_npu.compilation.acl_graph import ACLGraphWrapper

            if isolate_graph_tasks:
                from vllm_bench.pangu_infer.model.acl_graph import DirectACLGraphWrapper
                ACLGraphWrapper = DirectACLGraphWrapper

            self.wrapper = ACLGraphWrapper(
                self.runnable, cfg, CUDAGraphMode.FULL,
                cudagraph_options=CUDAGraphOptions(gc_disable=False, weak_ref_output=False),
                update_stream=torch.npu.Stream(), attn_layer_names=names)
            self.runnable = self.wrapper

    def diagnostic_call(self, ids, start, accepted, prompt, *, hidden=None, sample_index=None, draft_index=None,
                        model_timing=None):
        first_call = self.calls == 0
        self.memory_phase = "model"
        if first_call:
            self.log_memory("model_begin", start=start, draft_step=draft_index)
        try:
            result = self(ids, start, accepted, prompt, hidden=hidden, sample_index=sample_index,
                          draft_index=draft_index, model_timing=model_timing)
        except Exception:
            self.log_memory("model_failed", phase=self.memory_phase, start=start, draft_step=draft_index)
            raise
        if first_call:
            self.log_memory("model_end", start=start, draft_step=draft_index)
        return result

    def log_memory(self, stage, **context):
        if self.memory_diagnostics:
            from vllm_bench.pangu_infer.memory import log_memory
            log_memory(stage, self.device, model="target" if self.head is None else "mtp", head=self.head,
                       mode=self.mode, tokens=self.size, query_tokens=self.query_size, **context)

    def log_progress(self, stage, **context):
        if self.memory_diagnostics:
            from vllm.distributed import parallel_state
            from vllm_bench.pangu_infer.memory import log_progress

            group = parallel_state.get_tp_group()
            log_progress(stage, model="target" if self.head is None else "mtp", head=self.head,
                         mode=self.mode, tokens=self.size, query_tokens=self.query_size,
                         tp_group=group.unique_name, tp_ranks=group.ranks, **context)

    def __call__(self, ids, start, accepted, prompt, *, hidden=None, sample_index=None, draft_index=None,
                 model_timing=None):
        import torch
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context, get_forward_context, BatchDescriptor

        with torch.profiler.record_function("direct/prepare_buffers_metadata") if self.profile else nullcontext():
            if self.query_size < self.size:
                self.ids[:self.query_size].copy_(ids)
                self.ids[self.query_size:].zero_()
            else:
                self.ids.copy_(ids)
            if self.hidden is not None:
                if self.query_size < self.size:
                    self.hidden[:self.query_size].copy_(hidden)
                    self.hidden[self.query_size:].zero_()
                else:
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
                    self.memory_phase = "graph_warmup"
                    self.log_memory("graph_warmup_begin")
                    self.log_progress("graph_warmup_forward_begin")
                    warmup_started = time.perf_counter()
                    self.wrapper.runnable(self.ids, self.lane.positions, self.hidden, self.sample_index)
                    self.log_progress("graph_warmup_forward_end",
                                      elapsed_seconds=round(time.perf_counter() - warmup_started, 6))
                    self.memory_phase = "graph_warmup_sync"
                    self.log_progress("graph_warmup_sync_begin")
                    sync_started = time.perf_counter()
                    torch.npu.synchronize()
                    self.log_progress("graph_warmup_sync_end",
                                      elapsed_seconds=round(time.perf_counter() - sync_started, 6))
                    self.log_memory("graph_warmup_end")
                    self.captures += 1
                    capturing = True
            started = time.perf_counter() if capturing else None
            label = "direct/graph_replay_update" if self.wrapper else "direct/model_launch"
            if capturing:
                self.memory_phase = "acl_wrapper"
                self.log_memory("acl_wrapper_begin")
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
                self.log_memory("acl_wrapper_end")
        self.calls += 1
        if self.query_size < self.size:
            return result[0][:self.query_size], result[1]
        return result
