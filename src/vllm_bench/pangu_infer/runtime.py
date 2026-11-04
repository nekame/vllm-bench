"""NPU worker: explicit model, caches and a batch-one speculative decode loop."""

from contextlib import nullcontext
import argparse
import dataclasses
from enum import Enum
import json
import os
from pathlib import Path
import sys
import time

from vllm_bench.pangu_infer.run import write_json


def config_json(value):
    """Serialize effective configuration, rejecting unexpected opaque objects."""
    if isinstance(value, Enum):
        return config_json(value.value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if dataclasses.is_dataclass(value):
        return {f.name: config_json(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): config_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_json(v) for v in value]
    raise TypeError(f"Cannot serialize effective config {type(value).__name__}")


def engine_options(plan, *, direct):
    a = plan["options"]
    compile_enabled = "compile" in a["mode"]
    graph_enabled = "acl" in a["mode"]
    # Keep enforce_eager=False in every mode: it changes operator optimizations.
    # The direct model is compiled explicitly, outside the vLLM decorators.
    options = dict(
        model=a["model"], trust_remote_code=True, skip_tokenizer_init=True,
        dtype="bfloat16", tensor_parallel_size=a["tp"], data_parallel_size=1,
        enable_expert_parallel=False, enforce_eager=False,
        max_model_len=plan["max_model_len"], max_num_seqs=1,
        max_num_batched_tokens=plan["max_model_len"],
        enable_chunked_prefill=False, enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=a["gpu_memory_utilization"], seed=a["seed"],
        compilation_config={"mode": 0 if direct or not compile_enabled else 3,
                            "cudagraph_mode": "FULL" if graph_enabled else "NONE",
                            "cudagraph_capture_sizes": sorted({1, a["depth"] + 1})},
        additional_config=plan["additional_config"])
    if a["depth"]:
        options["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": a["depth"],
            "draft_sample_method": "greedy" if a["temperature"] == 0 else "probabilistic",
            "rejection_sample_method": "standard"}
    if direct:
        options["distributed_executor_backend"] = "external_launcher"
    return options


def require_npu_platform():
    from vllm.platforms import current_platform

    if current_platform.device_type != "npu":
        raise RuntimeError(
            f"Expected vLLM NPU platform, got device_type={current_platform.device_type!r}. "
            "VLLM_PLUGINS must include omni-npu, omni_npu_patches and omni_custom_models; "
            "the installed OmniInfer must register the omni-npu platform entry point. "
            "Restart through vllm_bench.pangu_infer.run after fixing the environment."
        )


def setup(plan):
    import torch
    import torch_npu  # noqa: F401
    from vllm.plugins import load_general_plugins

    require_npu_platform()
    load_general_plugins()
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager
    from omni_npu.model_config.config_loader.loader import load_model_extra_config, model_extra_config
    from omni_npu.worker.npu_worker import (
        init_worker_distributed_environment, _patch_npu_triton_capabilities,
        _enable_batch_invariant_determinism,
    )
    from omni_npu.v1.distributed.parallel_state_ext import ensure_layer_parallel_initialized
    from omni_npu.compilation.npugraph_ex_config import init_aclgraph_config

    a = plan["options"]
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != a["tp"]:
        raise ValueError("torchrun world size must equal TP")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    _patch_npu_triton_capabilities()
    torch.distributed.init_process_group(backend="hccl", init_method="env://")
    cfg = EngineArgs(**engine_options(plan, direct=True)).create_engine_config()
    if plan["config"].get("use_mome") and cfg.cache_config.mamba_block_size is None:
        # With APC disabled one request owns one running MoME state. The stock
        # state spans the request's maximum length, without prefix-cache
        # block transitions. Resolve it before constructing the state specs.
        cfg.cache_config.mamba_block_size = cfg.model_config.max_model_len
    context = set_current_vllm_config(cfg)
    context.__enter__()
    init_worker_distributed_environment(cfg, rank, "env://", local_rank, current_platform.dist_backend)
    load_model_extra_config(cfg.model_config, cfg, cfg.scheduler_config)
    _enable_batch_invariant_determinism(model_extra_config)
    from omni_npu.configs.validators import ValidationContext, validate_all
    validate_all(ValidationContext(model_extra_config=model_extra_config, vllm_config=cfg))
    # Effective settings, rather than just the source JSON, control the layers.
    if model_extra_config.parall_config.ena_seq_parallel or model_extra_config.parall_config.ena_context_parallel:
        raise ValueError("Effective operator config enabled sequence/context parallelism")
    if cfg.parallel_config.pipeline_parallel_size != 1 or cfg.parallel_config.enable_expert_parallel:
        raise ValueError("Direct runtime supports pure TP only")
    if a["depth"]:
        draft_hf = cfg.speculative_config.draft_model_config.hf_config
        if getattr(draft_hf, "n_predict", 1) != plan["config"]["num_nextn_predict_layers"]:
            raise ValueError("Effective OmniInfer predictor count disagrees with checkpoint")
    if cfg.model_config.hf_config.use_mome and not model_extra_config.operator_opt_config.use_noncontiguous_kv:
        raise ValueError("Effective operator config disabled noncontiguous MoME")
    ensure_layer_parallel_initialized(backend=current_platform.dist_backend)
    init_workspace_manager(device, 1)
    init_aclgraph_config(cfg)
    torch.manual_seed(a["seed"])
    torch.npu.manual_seed_all(a["seed"])
    return cfg, device, rank, context, model_extra_config


def weight_iterator(model_dir, *, draft, start):
    """Stream each selected safetensors tensor once; never dequantize weights."""
    from safetensors import safe_open
    import re

    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        paths = sorted({model_dir / name for name in mapping.values()})
    else:
        mapping = None
        paths = sorted(model_dir.glob("*.safetensors"))
    if not paths:
        raise ValueError("No safetensors weight shards")
    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    for path in paths:
        if not path.resolve().is_relative_to(model_dir.resolve()):
            raise ValueError("External checkpoint shard")
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if mapping is not None and mapping.get(name) != path.relative_to(model_dir).as_posix():
                    continue
                match = layer_pattern.match(name)
                is_draft = match is not None and int(match[1]) >= start
                if is_draft == draft:
                    yield name, shard.get_tensor(name)


def load_models(plan, cfg, device):
    import torch
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
    from omni_npu.v1.layers.attention.weight_utils import refresh_mla_wrappers
    from vllm_bench.pangu_infer.model.pangu_ultra_moe import PanguUltraMoEForCausalLM

    a = plan["options"]
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(cfg.model_config.dtype)
        with torch.device(device):
            target = PanguUltraMoEForCausalLM(vllm_config=cfg)
            draft = None
            if a["depth"]:
                from vllm_bench.pangu_infer.model.pangu_ultra_moe_mtp import OpenPanguMTP
                draft = OpenPanguMTP(vllm_config=cfg, prefix="draft_model", shared_target=target)
    finally:
        torch.set_default_dtype(original_dtype)
    start = plan["config"]["num_hidden_layers"]
    loaded_target = target.load_weights(weight_iterator(a["model"], draft=False, start=start))
    loaded_draft = draft.load_weights(weight_iterator(a["model"], draft=True, start=start)) if draft else set()
    if not loaded_target or draft is not None and not loaded_draft:
        raise ValueError("Checkpoint did not load target/MTP parameters")
    require_loaded_parameters(target, loaded_target)
    if draft is not None:
        require_loaded_parameters(draft, loaded_draft, shared={id(p) for p in target.parameters()})
    # One root ensures the shared embedding and LM head are packed only once.
    root = torch.nn.ModuleDict({"target": target, **({"draft": draft} if draft else {})})
    process_weights_after_loading(root, cfg.model_config, device)
    for model in (target, draft):
        if model is not None:
            refresh_mla_wrappers(model, cfg.model_config.dtype)
            model.post_weight_load()
            prepare_communication_buffer_for_model(model)
            model.eval()
    torch.npu.synchronize()
    return target, draft, {"target_loaded_parameters": len(loaded_target),
                           "draft_loaded_parameters": len(loaded_draft)}


def require_loaded_parameters(model, loaded, shared=()):
    missing = [name for name, parameter in model.named_parameters()
               if id(parameter) not in shared and name not in loaded]
    if missing:
        raise ValueError(f"Uninitialized {type(model).__name__} parameters: {missing[:20]}")


class DirectEngine:
    def __init__(self, plan, cfg, device, rank, target, draft):
        import torch
        from vllm.platforms import current_platform
        from vllm_bench.pangu_infer.cache import Caches
        from vllm_bench.pangu_infer.sampling import Sampler

        self.plan, self.cfg, self.device = plan, cfg, device
        self.a = plan["options"]
        self.target, self.draft = target, draft
        self.model_timings = None
        if plan["config"].get("use_mome"):
            current_platform._align_hybrid_block_size(cfg, None)
        self.caches = Caches(cfg, device)
        self.executions = {}
        self.sampler = Sampler(self.a, device, rank)
        self.prompt = torch.tensor(plan["prompt_ids"], dtype=torch.int64, device=device)
        self.names = {"target": [g.name for g in self.caches.groups if not g.name.startswith("draft_model.")],
                      "draft": [g.name for g in self.caches.groups if g.name.startswith("draft_model.")]}
        if "acl" in self.a["mode"]:
            from omni_npu.compilation.acl_graph import set_graph_params
            set_graph_params({1, self.a["depth"] + 1})

    def forward(self, kind, ids, start, accepted, *, hidden=None, index=None, head=0, draft_index=None):
        from vllm_bench.pangu_infer.execution import Execution

        size = ids.shape[0]
        prefill = size == self.a["context_length"] and start == 0
        key = (kind, size, head if kind == "draft" else None, prefill)
        if key not in self.executions:
            names = self.names[kind]
            if kind == "draft":
                prefix = f"draft_model.model.layers.{self.plan['config']['num_hidden_layers'] + head}."
                names = [name for name in names if name.startswith(prefix)]
            self.executions[key] = Execution(
                self.target if kind == "target" else self.draft, self.cfg, self.caches,
                names, size, self.device, head=head if kind == "draft" else None,
                mode="eager" if prefill else self.a["mode"], profile=self.a["profile"])
        return self.executions[key](ids, start, accepted, self.a["context_length"],
                                    hidden=hidden, sample_index=index, draft_index=draft_index,
                                    model_timing=self.model_timings)

    def propose(self, ids, hidden, start, index, index_value, anchor, accepted):
        import torch

        depth = self.a["depth"]
        count = self.plan["config"]["num_nextn_predict_layers"]
        multi = count > 1 and depth <= count
        shifted = torch.roll(ids, -1)
        shifted.scatter_(0, index, anchor.reshape(1))
        predictions, distributions = [], []
        positions_start = start
        for step in range(depth):
            if step and not multi:
                # First single-token continuation reads the accepted tail in
                # the packed first pass; later continuations read its latest tail.
                positions_start = start + index_value + step
                shifted = predictions[-1].reshape(1)
                hidden = hidden.index_select(0, index) if step == 1 else hidden
                index_for_call = torch.zeros_like(index)
                accepted_for_call = accepted if step == 1 else None
            else:
                index_for_call = index
                accepted_for_call = accepted
            head = step if multi else 0
            hidden, logits = self.forward(
                "draft", shifted, positions_start, accepted_for_call, hidden=hidden,
                index=index_for_call, head=head, draft_index=step)
            probs = self.sampler.distribution(logits)
            token = self.sampler.sample(probs).reshape(1)
            predictions.append(token)
            if self.a["temperature"] != 0:
                distributions.append(probs)
            if multi and step + 1 < depth:
                shifted = torch.roll(shifted, -1)
                shifted.scatter_(0, index, token)
        return torch.cat(predictions), torch.cat(distributions) if distributions else None

    def counters(self):
        return {"compiles": sum(e.compiles for e in self.executions.values()),
                "captures": sum(e.captures for e in self.executions.values()),
                "replays": sum(e.replays for e in self.executions.values()),
                "calls": sum(e.calls for e in self.executions.values())}

    def generate(self, limit, *, profile=False, model_timing=False):
        import torch

        if model_timing:
            from vllm_bench.pangu_infer.model_timing import ModelTimings
            self.model_timings = ModelTimings()
        else:
            self.model_timings = None
        self.caches.reset()
        self.sampler.reset()
        torch.npu.synchronize()
        prefill_start = time.perf_counter()
        with torch.profiler.record_function("direct/prefill") if profile else nullcontext():
            hidden, logits = self.forward("target", self.prompt, 0, None)
            anchor = self.sampler.sample(self.sampler.distribution(logits[-1:])).reshape(1)
            index = torch.tensor([len(self.plan["prompt_ids"]) - 1], dtype=torch.int64, device=self.device)
            initial_proposal = (self.propose(self.prompt, hidden, 0, index,
                                           len(self.plan["prompt_ids"]) - 1, anchor, self.a["depth"] + 1)
                                if self.a["depth"] else None)
        torch.npu.synchronize()
        prefill_seconds = time.perf_counter() - prefill_start
        # The first sampled token belongs to prefill, just as in offline baseline.
        output = [anchor]
        produced, cursor, accepted = 1, self.a["context_length"], self.a["depth"] + 1
        ids, previous_start = self.prompt, 0
        index_value = len(self.plan["prompt_ids"]) - 1
        cycle_counts, cycles = [], 0
        eos = self.plan["config"].get("eos_token_id", [])
        eos = eos if isinstance(eos, list) else [eos]
        stopped = self.a["respect_eos"] and int(anchor.item()) in eos
        decode_start = time.perf_counter()
        while produced < limit and not stopped:
            if self.model_timings is not None:
                self.model_timings.cycle = cycles
            if self.a["depth"]:
                with torch.profiler.record_function("direct/draft") if profile else nullcontext():
                    drafts, q = (initial_proposal if cycles == 0 else
                                 self.propose(ids, hidden, previous_start, index, index_value, anchor, accepted))
                verify_ids = torch.cat((anchor, drafts))
            else:
                verify_ids = anchor
            with torch.profiler.record_function("direct/target") if profile else nullcontext():
                hidden, logits = self.forward("target", verify_ids, cursor, accepted)
            with torch.profiler.record_function("direct/sample_commit") if profile else nullcontext():
                p = self.sampler.distribution(logits)
                if self.a["depth"]:
                    emitted, naccepted = self.sampler.verify(p, q, drafts)
                    # One unavoidable host decision per speculative cycle in
                    # this intentionally simple batch-one engine.
                    accepted_drafts = int(naccepted.item())
                    emitted = emitted[:accepted_drafts + 1]
                else:
                    accepted_drafts = 0
                    emitted = self.sampler.sample(p[-1:]).reshape(1)
                output.append(emitted)
                accepted = accepted_drafts + 1
                produced += accepted
                previous_start, ids = cursor, verify_ids
                cursor += accepted
                index.fill_(accepted_drafts)
                index_value = accepted_drafts
                anchor = emitted[-1:]
                cycle_counts.append(accepted_drafts)
                cycles += 1
            if self.a["respect_eos"]:
                # Explicit optional synchronization; fixed-length benchmarks
                # do not execute this branch.
                if any(t in eos for t in emitted.tolist()):
                    break
        torch.npu.synchronize()
        decode_seconds = time.perf_counter() - decode_start
        if self.model_timings is not None:
            self.model_timings.cycle = None
        tokens = torch.cat(output).tolist()[:limit]
        if self.a["respect_eos"]:
            first_eos = next((i for i, token in enumerate(tokens) if token in eos), None)
            if first_eos is not None:
                tokens = tokens[:first_eos + 1]
        record = {"token_ids": tokens, "prefill_seconds": prefill_seconds,
                "decode_seconds": decode_seconds, "decode_tokens": len(tokens) - 1,
                "cycles": cycles, "accepted_drafts": cycle_counts,
                "decode_target_calls": cycles,
                "decode_draft_calls": max(0, cycles - 1) * self.a["depth"],
                "acceptance_measurement": "exact verification counts before terminal length truncation",
                "initial_context": self.a["context_length"], "final_computed_context": cursor}
        if self.model_timings is not None:
            record["model_timings"] = self.model_timings.records
        return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    import torch
    a, output = plan["options"], Path(plan["options"]["output"])
    started = time.perf_counter()
    cfg, device, rank, context, extra = setup(plan)
    setup_seconds = time.perf_counter() - started
    try:
        # NPU weight packing and compilation may inspect tensor version
        # counters. Disable autograd without creating inference tensors.
        with torch.no_grad():
            started = time.perf_counter()
            target, draft, loading = load_models(plan, cfg, device)
            load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            engine = DirectEngine(plan, cfg, device, rank, target, draft)
            torch.npu.synchronize()
            cache_init_seconds = time.perf_counter() - started
            if a.get("replay_target"):
                from vllm_bench.pangu_infer.target_trace import replay_target
                replay_target(engine, plan, output)
                return
            warmup_start = time.perf_counter()
            for _ in range(a["warmup"]):
                engine.generate(max(2 * (a["depth"] + 1) + 1, 17))
            warmup_seconds = time.perf_counter() - warmup_start
            before = engine.counters()
            if "compile" in a["mode"] and before["compiles"] == 0:
                raise RuntimeError("Requested torch.compile did not compile any decode graph")
            if "acl" in a["mode"] and before["replays"] == 0:
                raise RuntimeError("Warmup did not replay any ACL graph")
            for key, execution in engine.executions.items():
                if "compile" in execution.mode and not execution.compiles:
                    raise RuntimeError(f"Decode lane {key} did not compile")
                if "acl" in execution.mode and not execution.replays:
                    raise RuntimeError(f"Decode lane {key} did not replay an ACL graph")
            if rank == 0 and a.get("model_timing", False):
                print("[model-time] diagnostic=sync_wall boundary=forward+lm_head"
                      " aggregation=max_tp output=after_decode", flush=True)
            if a["profile"]:
                import torch_npu
                activities = [torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU]
                profiler = torch_npu.profiler.profile(
                    activities=activities,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / f"trace-rank-{rank}")),
                    record_shapes=True, with_stack=False)
            else:
                profiler = nullcontext()
            with profiler:
                records = []
                for repeat in range(1, a["repeats"] + 1):
                    record = engine.generate(a["max_new_tokens"], profile=a["profile"],
                                             model_timing=a.get("model_timing", False))
                    if a.get("model_timing", False):
                        from vllm_bench.pangu_infer.model_timing import reduce_model_timings, log_model_timings
                        reduce_model_timings(record["model_timings"], device, a["tp"])
                        if rank == 0:
                            log_model_timings(record["model_timings"], repeat)
                    records.append(record)
            after = engine.counters()
            if any(after[k] != before[k] for k in ("compiles", "captures")):
                raise RuntimeError("Compilation/capture occurred during measured requests; increase warmup or fix specialization")
            # Compare the slowest rank, including its host overhead; never add
            # a distributed barrier to each measured cycle.
            times = torch.tensor([r["decode_seconds"] for r in records], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
            for record, seconds in zip(records, times.tolist()):
                record["decode_seconds"] = seconds
            effective = {"operator": config_json(extra.operator_opt_config),
                         "parallel": config_json(extra.parall_config),
                         "compilation": {"mode": a["mode"], "vllm": config_json(cfg.compilation_config.mode),
                                         "cudagraph": config_json(cfg.compilation_config.cudagraph_mode)},
                         "block_size": cfg.cache_config.block_size,
                         "mamba_block_size": cfg.cache_config.mamba_block_size}
            from vllm_bench.pangu_infer.results import save_result, versions
            if rank == 0:
                if a.get("model_timing", False):
                    from vllm_bench.pangu_infer.model_timing import summarize_model_timings, log_model_timing_summary
                    log_model_timing_summary(summarize_model_timings(records))
                save_result(output, plan, records, effective, {
                    **loading, "warmup_seconds": warmup_seconds,
                    "setup_seconds": setup_seconds, "load_seconds": load_seconds,
                    "cache_init_seconds": cache_init_seconds,
                    "counters_before": before, "counters_after": after,
                    "execution_lanes": [{"kind": key[0], "tokens": key[1], "head": key[2],
                                         "prefill": key[3], "mode": e.mode, "compiles": e.compiles,
                                         "captures": e.captures, "replays": e.replays,
                                         "backend_compile_seconds": e.backend_compile_seconds,
                                         "capture_seconds": e.capture_seconds}
                                        for key, e in engine.executions.items()],
                    "cache_bytes_per_rank": engine.caches.bytes,
                    "peak_allocated_bytes_rank0": torch.npu.max_memory_allocated(device),
                    "versions": versions(), "hardware": torch.npu.get_device_name(device)})
    finally:
        context.__exit__(None, None, None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
