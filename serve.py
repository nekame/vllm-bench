#!/usr/bin/env python3
"""Print or launch a persistent vLLM/OmniInfer profiling server."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESET_ENV = ["OMNI_HYBRID_ALIGNED_DECODE", "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
             "OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD", "OMNI_DP_ROUND_ROBIN",
             "USE_MOE_FORCE_LOAD_BALANCE", "OMNI_PROFILE_TOKEN_THRESHOLD",
             "PROFILER_TOKEN_THRESHOLD", "OMNI_ENABLE_PREFILL_PROFILER",
             "ENABLE_PREFILL_PROFILER"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--operator-config", type=Path, required=True,
                   help="Your target OmniInfer operator/parallel config, with the same TP settings")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="pangu12")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=33024)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--kv-cache-memory-bytes", type=int)
    p.add_argument("--num-gpu-blocks-override", type=int)
    p.add_argument("--patches", default=os.getenv("OMNI_VLLM_PATCHES_DIR", "low_latency"))
    p.add_argument("--attention-backend", choices=["pangu", "deepseek"], default="pangu")
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--annotate-layers", action="store_true",
                   help="Add CPU record_function layer ranges; benchmark timings include instrumentation overhead")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--profile-memory", action="store_true")
    p.add_argument("--engine-cli-json", type=Path, help="JSON array of additional vllm serve arguments")
    p.add_argument("--execute", action="store_true", help="Without this flag only print the launch plan")
    a = p.parse_args()
    model, operator, output = a.model.resolve(), a.operator_config.resolve(), a.output.resolve()
    cfg = json.loads((model / "config.json").read_text())
    if cfg.get("architectures") != ["OpenPanguV2ForCausalLM"] or cfg.get("num_nextn_predict_layers") != 0:
        p.error("Expected an extracted V2 checkpoint with MTP disabled")
    if not operator.is_file():
        p.error("Operator config does not exist")
    # Check shape now, before asking vLLM to initialize distributed workers.
    opt = json.loads(operator.read_text())
    if not {"model_parallel_config", "operator_optimization_config"} <= opt.keys():
        p.error("Operator config needs model_parallel_config and operator_optimization_config")
    if a.tp <= 0 or min(a.max_model_len, a.max_num_seqs, a.max_num_batched_tokens) <= 0:
        p.error("TP and workload limits must be positive")
    if not a.chunked_prefill and a.max_num_batched_tokens < a.max_model_len:
        p.error("Without chunked prefill, max-num-batched-tokens must cover max-model-len")
    forbidden = {"pangu_v2_benchmark", "decode_profile_sync", "pd_bench_aligned_decode"}
    if forbidden & {s.strip() for s in a.patches.split(",")}:
        p.error("Remove benchmark gates / decode sync patches from this profiling server")
    profiler = {"profiler": "torch", "torch_profiler_dir": str(output / "traces"),
                "ignore_frontend": True, "torch_profiler_with_stack": False,
                "torch_profiler_with_flops": False, "torch_profiler_record_shapes": a.record_shapes,
                "torch_profiler_with_memory": a.profile_memory}
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--served-model-name", a.name, "--host", a.host, "--port", str(a.port),
               "--tensor-parallel-size", str(a.tp), "--data-parallel-size", "1",
               "--dtype", "bfloat16", "--trust-remote-code", "--generation-config", "vllm",
               "--max-model-len", str(a.max_model_len), "--max-num-seqs", str(a.max_num_seqs),
               "--max-num-batched-tokens", str(a.max_num_batched_tokens),
               "--gpu-memory-utilization", str(a.gpu_memory_utilization),
               "--no-enable-prefix-caching", "--api-server-count", "1",
               "--enable-chunked-prefill" if a.chunked_prefill else "--no-enable-chunked-prefill",
               "--profiler-config", json.dumps(profiler)]
    for key, value in [("--kv-cache-memory-bytes", a.kv_cache_memory_bytes),
                       ("--num-gpu-blocks-override", a.num_gpu_blocks_override)]:
        if value is not None:
            command += [key, str(value)]
    if a.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    if a.enforce_eager:
        command.append("--enforce-eager")
    if a.engine_cli_json:
        extra = json.loads(a.engine_cli_json.read_text())
        if not isinstance(extra, list) or not all(isinstance(s, str) for s in extra):
            p.error("engine-cli-json must be an array of CLI argument strings")
        blocked = ("--model", "--hf-overrides", "--speculative-config", "--data-parallel",
                   "--profiler-config", "--enable-prefix-caching", "--tensor-parallel-size")
        if any(s.startswith(blocked) for s in extra):
            p.error("Extra CLI arguments cannot replace architecture, parallelism, cache or profiler settings")
        command += extra
    env = os.environ.copy()
    for key in RESET_ENV:
        env.pop(key, None)
    settings = {"OMNI_VLLM_PATCHES_DIR": a.patches,
                "OMNI_CUSTOM_MODEL_CONFIG_PATH": str(operator),
                "OMNI_PANGU_V2_HIGH_THROUGHOUT": "1" if a.attention_backend == "deepseek" else "0"}
    env.update(settings)
    if a.annotate_layers:
        env["PANGU12_ANNOTATE_LAYERS"] = "1"
        env["PYTHONPATH"] = str(HERE / "instrumentation") + os.pathsep + env.get("PYTHONPATH", "")
    else:
        env.pop("PANGU12_ANNOTATE_LAYERS", None)
    plan = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": command,
            "settings": settings, "cleared_environment": RESET_ENV,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "model_config": cfg, "operator_config": opt,
            "annotation_overhead": a.annotate_layers}
    print("Environment:", json.dumps(settings, indent=2), flush=True)
    print("Command:", shlex.join(command), flush=True)
    if not a.execute:
        print("Plan only. Add --execute on the NPU server to start.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "server_manifest.json"
    if metadata.exists():
        p.error("Output already contains a server manifest; choose a new directory")
    (output / "operator_config.json").write_text(json.dumps(opt, indent=2) + "\n")
    metadata.write_text(json.dumps(plan, indent=2) + "\n")
    # Keep stdout/stderr attached: tee can record logs without hiding startup errors.
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
