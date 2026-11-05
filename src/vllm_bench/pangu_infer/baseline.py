"""Offline stock OmniInfer baseline, with a streaming engine-step boundary."""

import argparse
import json
from pathlib import Path
import sys
import time

from vllm_bench.pangu_infer.runtime import engine_options, config_json, require_npu_platform
from vllm_bench.pangu_infer.results import save_result, versions


def synchronize_worker(worker):
    import torch
    torch.npu.synchronize()


def worker_state(worker):
    """Read-only RPC, outside timings, on every stock OmniInfer worker."""
    import torch
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from vllm.compilation.counter import compilation_counter

    cfg = worker.model_runner.vllm_config
    return {"operator": config_json(model_extra_config.operator_opt_config),
            "parallel": config_json(model_extra_config.parall_config),
            "block_size": cfg.cache_config.block_size,
            "mamba_block_size": cfg.cache_config.mamba_block_size,
            "compilation": config_json(cfg.compilation_config.mode),
            "cudagraph": config_json(cfg.compilation_config.cudagraph_mode),
            "compile_counter": {k: v for k, v in vars(compilation_counter).items() if type(v) is int},
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "hardware": torch.npu.get_device_name()}


def request(engine, plan, request_id, limit, *, prompt_ids=None):
    from vllm import SamplingParams

    a = plan["options"]
    params = SamplingParams(
        temperature=a["temperature"], top_k=a["top_k"] or -1, top_p=a["top_p"],
        seed=a["seed"], max_tokens=limit, ignore_eos=not a["respect_eos"], detokenize=False)
    engine.collective_rpc(synchronize_worker)
    start = time.perf_counter()
    engine.add_request(request_id, {"prompt_token_ids": plan["prompt_ids"] if prompt_ids is None else prompt_ids}, params)
    first_time, first_count, tokens, cycles = None, 0, [], 0
    accepted_drafts = []
    while engine.has_unfinished_requests():
        outputs = engine.step()
        cycles += 1
        for output in outputs:
            if output.request_id != request_id or not output.outputs:
                continue
            previous_count = len(tokens)
            tokens = list(output.outputs[0].token_ids)
            if first_time is not None and a["depth"] and len(tokens) > previous_count:
                accepted_drafts.append(max(0, len(tokens) - previous_count - 1))
            if first_time is None and tokens:
                # Synchronous scheduling prevents the engine from starting
                # the next step while we establish this device-complete bound.
                engine.collective_rpc(synchronize_worker)
                first_time, first_count = time.perf_counter(), len(tokens)
    engine.collective_rpc(synchronize_worker)
    end = time.perf_counter()
    if first_time is None:
        raise RuntimeError("Offline engine produced no tokens")
    if first_count != 1:
        raise RuntimeError("Expected one prefill output token; decode timing boundaries are not comparable")
    return {"token_ids": tokens, "prefill_seconds": first_time - start,
            "decode_seconds": end - first_time, "decode_tokens": len(tokens) - first_count,
            "cycles": cycles - 1, "accepted_drafts": accepted_drafts,
            "acceptance_measurement": "from output increments; final truncated cycle is a lower bound",
            "initial_context": a["context_length"]}


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
    engine = LLMEngine.from_engine_args(EngineArgs(**options))
    startup_seconds = time.perf_counter() - started
    warmup_start = time.perf_counter()
    for i in range(a["warmup"]):
        request(engine, plan, f"warmup-{i}", max(2 * (a["depth"] + 1) + 1, 17))
    warmup_seconds = time.perf_counter() - warmup_start
    states_before = engine.collective_rpc(worker_state)
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
    try:
        records = []
        for prompt in plan.get("prompts", [None]):
            prompt_ids = None if prompt is None else prompt["prompt_ids"]
            for repeat in range(1, a["repeats"] + 1):
                record = request(engine, plan, f"measure-{len(records)}", a["max_new_tokens"], prompt_ids=prompt_ids)
                if prompt is not None:
                    record.update(id=prompt["id"], repeat=repeat)
                records.append(record)
    finally:
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
    effective["compilation"] = {"mode": a["mode"], "vllm": first["compilation"], "cudagraph": first["cudagraph"]}
    details = {
        "startup_seconds": startup_seconds, "warmup_seconds": warmup_seconds, "workers_before": states_before,
        "workers_after": states_after, "versions": versions(), "hardware": first["hardware"],
        "timing_boundary": "offline step outputs, including engine scheduling and IPC; no HTTP"}
    save_result(output, plan, records, effective, details)
    if a.get("trace_target", False):
        from vllm_bench.pangu_infer.target_trace import write_manifest
        write_manifest(trace_directory, plan, effective, details)
        print(f"[target-trace] calls_per_rank={counts[0]} directory={trace_directory}", flush=True)


if __name__ == "__main__":
    main()
