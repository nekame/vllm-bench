#!/usr/bin/env python3
"""Plan or launch standalone MTP profiling. ML imports happen only with --execute."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from vllm_bench.mtp_profile.mtp_common import TOPOLOGIES, checkpoint_config, layer_sequence, topology_world_size, validate_record, write_json
from vllm_bench.common.runtime_env import RESET_ENV, instrumentation_pythonpath


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--operator-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--topology", choices=TOPOLOGIES, required=True)
    p.add_argument("--input", choices=("synthetic", "replay"), default="synthetic")
    p.add_argument("--capture", type=Path, help="Complete capture directory containing every rank of the selected topology")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--batch-per-dp-rank", type=int, default=1)
    p.add_argument("--depths", type=int, nargs="+", help="Defaults to 3 7; replay uses captured depth")
    p.add_argument("--execution", choices=("eager", "compiled", "graph"), default="eager")
    p.add_argument("--compilation-config", type=Path, help="JSON object with target compilation settings")
    p.add_argument("--additional-config", type=Path, help="Original vLLM additional-config JSON")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--prefill-chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--diagnostic-stages", action="store_true",
                   help="Separate synchronized stage run, excluded from baseline latency")
    p.add_argument("--atol", type=float, default=0.05)
    p.add_argument("--rtol", type=float, default=0.05)
    p.add_argument("--patches")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    world_size = topology_world_size(TOPOLOGIES[a.topology])
    a.depths = a.depths or ([3, 7] if a.input == "synthetic" else [])
    if min(a.context_length, a.batch_per_dp_rank, a.repeats, a.prefill_chunk, *a.depths) < 1 or a.warmup < 1:
        p.error("Lengths, depths, repeats and warmup must be positive")
    if min(a.atol, a.rtol) < 0:
        p.error("Tolerances must be nonnegative")
    if a.input == "replay" and a.capture is None:
        p.error("Replay requires --capture")
    if a.input == "synthetic" and a.capture is not None:
        p.error("--capture is only used with --input replay")
    config, manifest = checkpoint_config(a.model.resolve())
    operator = json.loads(a.operator_config.read_text())
    if not {"model_parallel_config", "operator_optimization_config"} <= operator.keys():
        p.error("Invalid operator config sections")
    compilation = json.loads(a.compilation_config.read_text()) if a.compilation_config else {}
    additional = json.loads(a.additional_config.read_text()) if a.additional_config else {}
    replay_meta = None
    if a.input == "replay":
        for rank in range(world_size):
            meta = json.loads((a.capture / f"rank-{rank:03d}" / "capture.json").read_text())
            validate_record(meta, topology=a.topology, config=config)
            if not meta.get("complete") or meta["rank"] != rank or meta["world_size"] != world_size:
                p.error(f"Capture must contain {world_size} complete rank records for {a.topology}")
            if meta["operator_config"] != operator:
                p.error("Operator config differs from capture")
            if replay_meta and any(meta[k] != replay_meta[k] for k in ("depth", "versions", "runtime_limits", "additional_config", "input_tokens", "batch_per_dp_rank", "padded_tokens")):
                p.error("Rank recordings disagree on runtime configuration")
            replay_meta = meta
        a.depths = a.depths or [replay_meta["depth"]]
        a.batch_per_dp_rank = replay_meta["batch_per_dp_rank"]
        if a.depths != [replay_meta["depth"]]:
            p.error("Replay accepts only the captured depth, once")
        if not a.additional_config:
            additional = replay_meta["additional_config"] or {}
        elif additional != (replay_meta["additional_config"] or {}):
            p.error("Additional config differs from capture")
        if not a.compilation_config:
            compilation = replay_meta["compilation_options"].copy()
    if len(set(a.depths)) != len(a.depths):
        p.error("Depths must be unique")
    if not isinstance(compilation, dict) or not isinstance(additional, dict):
        p.error("Compilation and additional config must be JSON objects")
    if a.execution == "eager":
        compilation.update(mode=0, cudagraph_mode="NONE")
    elif a.execution == "compiled":
        compilation.update(cudagraph_mode="NONE")
        compilation.setdefault("mode", 3)
    else:
        compilation.setdefault("mode", 3)
        compilation.setdefault("cudagraph_mode", "FULL")
        if compilation["cudagraph_mode"] != "FULL":
            p.error("Standalone graph mode currently requires cudagraph_mode=FULL")
    if a.execution != "eager" and compilation.get("mode") in (0, "NONE", "none"):
        p.error("Compiled/graph execution requires compilation enabled")
    options = vars(a).copy()
    for key, value in list(options.items()):
        if isinstance(value, Path):
            options[key] = str(value.resolve())
    patches = a.patches or (replay_meta.get("patches") if replay_meta else None) or os.getenv("OMNI_VLLM_PATCHES_DIR") or (
        "high_throughout" if config["architectures"] == ["PanguUltraMoEForCausalLM"] else "low_latency")
    forbidden = {"pangu_v2_benchmark", "decode_profile_sync", "pd_bench_aligned_decode"}
    if forbidden & set(patches.split(",")):
        p.error("Remove benchmark gates and profiler-sync patches")
    plan = {"format": "pangu-mtp-run-v1", "options": options, "topology": TOPOLOGIES[a.topology],
            "world_size": world_size,
            "config": config, "checkpoint_manifest": manifest, "operator_config": operator,
            "compilation_config": compilation, "additional_config": additional, "patches": patches,
            "expected_layers": {str(d): layer_sequence(config, d) for d in a.depths},
            "capture_metadata": replay_meta,
            "total_sequences": a.batch_per_dp_rank * TOPOLOGIES[a.topology]["dp"],
            "server_started": False, "target_weights_loaded": False,
            "validation": "CPU planning only; NPU execution requires server validation"}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    try:
        child_pythonpath = instrumentation_pythonpath(os.environ, None)
    except ValueError as error:
        p.error(str(error))
    if not a.execute:
        return 0
    output = a.output.resolve()
    if output.exists():
        p.error("Output must not exist")
    output.mkdir(parents=True)
    write_json(output / "launch_plan.json", plan)
    env = os.environ.copy()
    env["PYTHONPATH"] = child_pythonpath
    env.update(OMNI_CUSTOM_MODEL_CONFIG_PATH=str(a.operator_config.resolve()),
               OMNI_VLLM_PATCHES_DIR=patches, PANGU_MTP_CAPTURE_DIR="",
               OMNI_PANGU_V2_HIGH_THROUGHOUT="1" if config["architectures"] == ["PanguUltraMoEForCausalLM"] else "0")
    if replay_meta:
        for key, value in replay_meta["draft_environment"].items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
    plugins = [x for x in env.get("VLLM_PLUGINS", "").split(",") if x]
    for plugin in ("omni-npu", "omni_npu_patches", "omni_custom_models"):
        if plugin not in plugins:
            plugins.append(plugin)
    env["VLLM_PLUGINS"] = ",".join(plugins)
    for key in RESET_ENV:
        env.pop(key, None)
    env.pop("PANGU12_ANNOTATE_LAYERS", None)
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    # Use Omni's HCCL + separate gloo groups. vLLM's split-PG validation assumes
    # an externally initialized CUDA device-bound default group.
    env["VLLM_DISTRIBUTED_USE_SPLIT_GROUP"] = "0"
    commands, summaries = [], []
    # Speculative depth affects buffer/cache sizes, not just the proposer loop.
    # Separate jobs preserve the configured-depth semantics of real deployments.
    for depth in a.depths:
        folder = output / f"depth-{depth}"
        folder.mkdir()
        child = deepcopy(plan)
        child["options"].update(depths=[depth], output=str(folder))
        child["expected_layers"] = {str(depth): plan["expected_layers"][str(depth)]}
        path = folder / "launch_plan.json"
        write_json(path, child)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                   f"--nproc-per-node={world_size}", "--module", "vllm_bench.mtp_profile.mtp_runtime", "--plan", str(path)]
        commands.append(command)
        write_json(output / "command.json", commands)
        status = subprocess.run(command, env=env, check=False).returncode
        if status:
            return status
        summaries.extend(json.loads((folder / "summary.json").read_text()))
    write_json(output / "summary.json", summaries)
    with (output / "summary.csv").open("w", newline="") as stream:
        fields = ["depth", "critical_path_median_s", "total_sequences", "proposed_tokens_per_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in summaries)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
