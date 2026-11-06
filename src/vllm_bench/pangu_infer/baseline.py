"""Offline stock OmniInfer baseline, with a streaming engine-step boundary."""

import argparse
import json
from pathlib import Path
import sys
import time

from vllm_bench.pangu_infer.runtime import engine_options, config_json, require_npu_platform
from vllm_bench.pangu_infer.results import save_result, versions
from vllm_bench.pangu_infer.run import write_json


def synchronize_worker(worker):
    import torch
    torch.npu.synchronize()


def drain_async_batches(engine):
    """Finish the in-process core's work after the final visible output.

    Match EngineCoreProc.has_work for DP=1: a finished request may leave
    queued batches or scheduler cleanup. Device synchronization alone does
    not consume these results. Use LLMEngine.step for its normal post_step
    and output-processing protocol; do not modify scheduler/worker state.
    """
    core = engine.engine_core.engine_core
    queued_batches = len(core.batch_queue or ())
    steps = 0
    while core.batch_queue or core.scheduler.has_requests():
        engine.step()
        steps += 1
    return queued_batches, steps


def worker_state(worker):
    """Read-only RPC, outside timings, on every stock OmniInfer worker."""
    import torch
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from vllm.compilation.counter import compilation_counter

    cfg = worker.model_runner.vllm_config
    return {"operator": config_json(model_extra_config.operator_opt_config),
            "parallel": config_json(model_extra_config.parall_config),
            "async_scheduling": bool(cfg.scheduler_config.async_scheduling),
            "enable_prefix_caching": bool(cfg.cache_config.enable_prefix_caching),
            "mamba_cache_mode": config_json(cfg.cache_config.mamba_cache_mode),
            "cache_dtype": cfg.cache_config.cache_dtype,
            "block_size": cfg.cache_config.block_size,
            "mamba_block_size": cfg.cache_config.mamba_block_size,
            "compilation": config_json(cfg.compilation_config.mode),
            "cudagraph": config_json(cfg.compilation_config.cudagraph_mode),
            "compile_counter": {k: v for k, v in vars(compilation_counter).items() if type(v) is int},
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "hardware": torch.npu.get_device_name()}


def request(engine, plan, request_id, limit, *, prompt_ids=None, boundary=None):
    from vllm import SamplingParams

    a = plan["options"]
    async_scheduling = a.get("async_scheduling", False)
    multiprocessing = a.get("omni_multiprocessing", False)
    host_timing = async_scheduling or multiprocessing
    params = SamplingParams(
        temperature=a["temperature"], top_k=a["top_k"] or -1, top_p=a["top_p"],
        seed=a["seed"], max_tokens=limit, ignore_eos=not a["respect_eos"], detokenize=False)
    engine.collective_rpc(synchronize_worker)
    start = time.perf_counter()
    internal_request_id = engine.add_request(
        request_id, {"prompt_token_ids": plan["prompt_ids"] if prompt_ids is None else prompt_ids}, params)
    if boundary is not None:
        from vllm_bench.pangu_infer import boundary_trace
        if not isinstance(internal_request_id, str) or not internal_request_id:
            raise RuntimeError("Boundary trace requires LLMEngine.add_request to return the internal request ID")
        engine.collective_rpc(boundary_trace.arm, args=(internal_request_id, *boundary))
    first_time, first_count, first_cycle, tokens, cycles = None, 0, 0, [], 0
    cached_prompt_tokens = None
    accepted_drafts = []
    queued_batches, cleanup_steps, cleanup_seconds = 0, 0, 0.0
    while engine.has_unfinished_requests():
        outputs = engine.step()
        cycles += 1
        for output in outputs:
            if output.request_id != request_id or not output.outputs:
                continue
            previous_count = len(tokens)
            tokens = list(output.outputs[0].token_ids)
            cached_prompt_tokens = getattr(output, "num_cached_tokens", cached_prompt_tokens)
            if first_time is not None and a["depth"] and len(tokens) > previous_count:
                accepted_drafts.append(max(0, len(tokens) - previous_count - 1))
            if first_time is None and tokens:
                # Async scheduling or EngineCoreProc may already be advancing
                # decode. Synchronizing here would move the first-token
                # boundary past the first host-visible output and distort TPOT.
                if not host_timing:
                    engine.collective_rpc(synchronize_worker)
                first_time, first_count = time.perf_counter(), len(tokens)
                first_cycle = cycles
    if host_timing:
        end = time.perf_counter()
        cleanup_started = end
        if multiprocessing:
            # EngineCoreProc advances its own queue and scheduler, like HTTP.
            # The frontend has no in-process core to step or inspect here.
            queued_batches, cleanup_steps = None, None
        else:
            queued_batches, cleanup_steps = drain_async_batches(engine)
        engine.collective_rpc(synchronize_worker)
        cleanup_seconds = time.perf_counter() - cleanup_started
    else:
        engine.collective_rpc(synchronize_worker)
        end = time.perf_counter()
    if first_time is None:
        raise RuntimeError("Offline engine produced no tokens")
    if first_count != 1:
        raise RuntimeError("Expected one prefill output token; decode timing boundaries are not comparable")
    if host_timing:
        print("[request-end] " + json.dumps({"request_id": request_id,
            "cached_prompt_tokens": cached_prompt_tokens, "queued_batches": queued_batches,
            "cleanup_steps": cleanup_steps, "cleanup_seconds": cleanup_seconds,
            "cleanup_owner": "engine_core_process" if multiprocessing else "frontend"}), flush=True)
    return {"token_ids": tokens, "prefill_seconds": first_time - start,
            "decode_seconds": end - first_time, "decode_tokens": len(tokens) - first_count,
            "cycles": cycles - first_cycle, "accepted_drafts": accepted_drafts,
            "acceptance_measurement": "from output increments; final truncated cycle is a lower bound",
            "initial_context": a["context_length"], "cached_prompt_tokens": cached_prompt_tokens,
            "post_request_queued_batches": queued_batches, "post_request_cleanup_steps": cleanup_steps,
            "post_request_cleanup_seconds": cleanup_seconds}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    args = p.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    from vllm.plugins import load_general_plugins
    require_npu_platform()
    load_general_plugins()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    a, output = plan["options"], Path(plan["options"]["output"])
    if a["mode"] == "compile":
        # The stock Omni decorator _bypass_prefill bypasses compilation when
        # runtime mode != FULL. Calling that "compile-only" would be false.
        raise ValueError("Stock OmniInfer bypasses torch.compile without FULL ACL Graph; use eager, acl or compile-acl baseline")
    options = engine_options(plan, direct=False)
    if a["profile"]:
        options["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(output / "traces"),
                                      "ignore_frontend": True, "torch_profiler_with_stack": False}
    started = time.perf_counter()
    factory_options = {"enable_multiprocessing": True} if a.get("omni_multiprocessing", False) else {}
    engine = LLMEngine.from_engine_args(EngineArgs(**options), **factory_options)
    startup_seconds = time.perf_counter() - started
    warmup_start = time.perf_counter()
    for i in range(a["warmup"]):
        request(engine, plan, f"warmup-{i}", max(2 * (a["depth"] + 1) + 1, 17))
    warmup_seconds = time.perf_counter() - warmup_start
    states_before = engine.collective_rpc(worker_state)
    requested_async = a.get("async_scheduling", False)
    multiprocessing = a.get("omni_multiprocessing", False)
    host_timing = requested_async or multiprocessing
    core_mode = "process" if multiprocessing else "inproc"
    if any(s.get("async_scheduling", False) != requested_async for s in states_before):
        raise RuntimeError("Effective Omni async_scheduling differs from the requested setting")
    requested_prefix = a.get("enable_prefix_caching", False)
    if any(s.get("enable_prefix_caching", False) != requested_prefix for s in states_before):
        raise RuntimeError("Effective Omni enable_prefix_caching differs from the requested setting")
    requested_cache_dtype = a.get("kv_cache_dtype", "auto")
    if any(s.get("cache_dtype", "auto") != requested_cache_dtype for s in states_before):
        raise RuntimeError("Effective Omni cache_dtype differs from the requested --kv-cache-dtype")
    print(f"[execution] async_scheduling={str(requested_async).lower()} "
          f"engine_core={core_mode} "
          f"timing_boundary={'host_outputs' if host_timing else 'device_complete'}", flush=True)
    cache_state = states_before[0]
    print(f"[cache] cache_dtype={cache_state.get('cache_dtype', 'auto')} "
          f"enable_prefix_caching={str(requested_prefix).lower()} "
          f"block_size={cache_state['block_size']} mamba_block_size={cache_state['mamba_block_size']} "
          f"mamba_cache_mode={cache_state.get('mamba_cache_mode')}", flush=True)
    if "acl" in a["mode"] and not all(s["compile_counter"].get("num_cudagraph_captured", 0) for s in states_before):
        raise RuntimeError("Offline engine did not capture ACL graphs")
    if "compile" in a["mode"] and not all(
            s["compile_counter"].get("num_graphs_seen", 0) or s["compile_counter"].get("stock_torch_compile_count", 0)
            for s in states_before):
        raise RuntimeError("Offline engine did not trace a compiled model")
    if a["profile"]:
        engine.start_profile()
    trace_directory = output / "target-trace"
    if a.get("trace_target", False):
        from vllm_bench.pangu_infer.target_trace import install_target_trace, finish_target_trace
        engine.collective_rpc(install_target_trace, args=(str(trace_directory), plan["trace_layers"]))
    boundary_reports = []
    if a.get("trace_mtp_boundary", False):
        from vllm_bench.pangu_infer import boundary_trace
        engine.collective_rpc(boundary_trace.install,
                              args=(str(output / "mtp-boundary"), a["context_length"], a["depth"]))
    try:
        records = []
        for prompt_index, prompt in enumerate(plan.get("prompts", [None])):
            prompt_ids = None if prompt is None else prompt["prompt_ids"]
            for repeat in range(1, a["repeats"] + 1):
                request_id = f"measure-{len(records)}"
                diagnostic = {}
                if a.get("trace_mtp_boundary", False):
                    diagnostic["boundary"] = (str(prompt_index), len(records))
                record = request(engine, plan, request_id, a["max_new_tokens"],
                                 prompt_ids=prompt_ids, **diagnostic)
                if a.get("trace_mtp_boundary", False):
                    reports = engine.collective_rpc(boundary_trace.finish)
                    boundary_reports.append(reports)
                    print("[mtp-boundary] " + json.dumps(reports), flush=True)
                if prompt is not None:
                    record.update(id=prompt["id"], repeat=repeat)
                records.append(record)
    finally:
        if a.get("trace_mtp_boundary", False):
            engine.collective_rpc(boundary_trace.close)
        if a.get("trace_target", False):
            counts = engine.collective_rpc(finish_target_trace)
            if len(set(counts)) != 1 or min(counts) < 2:
                raise RuntimeError("Target trace requires matching TP call counts including decode")
    if a["profile"]:
        engine.stop_profile()
    states_after = engine.collective_rpc(worker_state)
    if any(before["compile_counter"] != after["compile_counter"]
           for before, after in zip(states_before, states_after)):
        raise RuntimeError("Stock engine compiled/captured during measurement; increase warmup")
    first = states_after[0]
    if any(any(state[key] != first[key] for key in ("operator", "parallel", "block_size", "mamba_block_size"))
           for state in states_after):
        raise ValueError("Offline ranks disagree on effective configuration")
    effective = {key: first[key] for key in ("operator", "parallel", "block_size", "mamba_block_size")}
    effective["async_scheduling"] = first.get("async_scheduling", False)
    effective["enable_prefix_caching"] = first.get("enable_prefix_caching", False)
    effective["mamba_cache_mode"] = first.get("mamba_cache_mode")
    effective["cache_dtype"] = first.get("cache_dtype", "auto")
    effective["engine_core_mode"] = core_mode
    effective["compilation"] = {"mode": a["mode"], "vllm": first["compilation"], "cudagraph": first["cudagraph"]}
    details = {
        "startup_seconds": startup_seconds, "warmup_seconds": warmup_seconds, "workers_before": states_before,
        "workers_after": states_after, "versions": versions(), "hardware": first["hardware"],
        "timing_boundary": ("offline first-to-final host-visible outputs; final device drain excluded; no HTTP"
                            if host_timing else
                            "offline device-complete first-to-final outputs, including final synchronization; no HTTP")}
    if a.get("trace_mtp_boundary", False):
        details["mtp_boundary_reports"] = boundary_reports
        write_json(output / "mtp-boundary" / "report.json", boundary_reports)
    save_result(output, plan, records, effective, details)
    if a.get("trace_target", False):
        from vllm_bench.pangu_infer.target_trace import write_manifest
        write_manifest(trace_directory, plan, effective, details)
        print(f"[target-trace] calls_per_rank={counts[0]} directory={trace_directory}", flush=True)


if __name__ == "__main__":
    main()
