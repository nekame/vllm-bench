#!/usr/bin/env python3
"""Print or launch a persistent vLLM/OmniInfer profiling server."""
import argparse
import json
import os
import shlex
import subprocess
import sys

from vllm_bench.common.server_launch import add_server_arguments, build_launch
from vllm_bench.common.runtime_env import instrumentation_pythonpath


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(p)
    p.add_argument("--annotate-layers", action="store_true", help="Add CPU layer ranges")
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--profile-memory", action="store_true")
    a = p.parse_args()
    output = a.output.resolve()
    profiler = {"profiler": "torch", "torch_profiler_dir": str(output / "traces"),
                "ignore_frontend": True, "torch_profiler_with_stack": False,
                "torch_profiler_with_flops": False, "torch_profiler_record_shapes": a.record_shapes,
                "torch_profiler_with_memory": a.profile_memory}
    try:
        base_env = os.environ.copy()
        if a.annotate_layers:
            base_env["PYTHONPATH"] = instrumentation_pythonpath(base_env, "layer_profile")
        plan, env = build_launch(a, profiler_config=profiler, base_env=base_env)
    except ValueError as error:
        p.error(str(error))
    if a.annotate_layers:
        env["PYTHONPATH"] = base_env["PYTHONPATH"]
        env["PANGU12_ANNOTATE_LAYERS"] = "1"
    else:
        env.pop("PANGU12_ANNOTATE_LAYERS", None)
    plan["annotation_overhead"] = a.annotate_layers
    command, settings, opt = plan["command"], plan["settings"], plan["operator_config"]
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
