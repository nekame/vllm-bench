"""Launch an eager Ultra target with Ki capture and no profiler."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from vllm_bench.common.runtime_env import instrumentation_pythonpath
from vllm_bench.common.server_launch import add_server_arguments, build_launch


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_server_arguments(p)
    p.set_defaults(tp=2, name="pangu-ki", enable_mtp=True, num_speculative_tokens=1,
                   enforce_eager=True, max_num_seqs=1, max_model_len=8320, max_num_batched_tokens=8192)
    p.add_argument("--kv-cache-dtype", choices=("hif8_ds_mla",), default="hif8_ds_mla")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--capture-output", type=Path, help="Default OUTPUT/capture; must be new")
    p.add_argument("--mome-layers", default="0")
    p.add_argument("--attention-steps-per-request", type=int, default=32)
    p.add_argument("--indexer-ki-steps-per-request", type=int, default=32)
    p.add_argument("--indexer-ki-prolog", action="store_true")
    p.add_argument("--indexer-ki-prolog-positions", default="1920:2048")
    p.add_argument("--tensor-max-mib", type=int, default=16)
    p.add_argument("--tensor-budget-mib", type=int, default=512)
    p.add_argument("--limit", type=int, default=1000)
    return p


def build_debug_plan(a):
    from vllm_bench.mtp_debug.indexer_prolog_trace import parse_positions
    parse_positions(a.indexer_ki_prolog_positions)
    if a.tp not in (2, 8) or a.max_num_seqs != 1 or not a.enforce_eager:
        raise ValueError("Ki debugging requires TP2/TP8, DP1/PP1, eager and max-num-seqs=1")
    if not 1 <= a.port <= 65535 or a.block_size < 1:
        raise ValueError("Invalid port or block size")
    if min(a.attention_steps_per_request, a.indexer_ki_steps_per_request,
           a.tensor_max_mib, a.tensor_budget_mib, a.limit) < 1:
        raise ValueError("Capture limits must be positive")
    if a.mome_layers != "all" and (not a.mome_layers or any(int(x) < 0 for x in a.mome_layers.split(","))):
        raise ValueError("--mome-layers must be all or nonnegative layer indices")
    capture = (a.capture_output or a.output / "capture").resolve()
    if capture.exists():
        raise ValueError("Use a new capture directory")
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = instrumentation_pythonpath(base_env, None)
    base_env.pop("VLLM_TORCH_PROFILER_DIR", None)
    plan, env = build_launch(a, profiler_config={"profiler": None}, base_env=base_env,
        extra_blocked=("--kv-cache-dtype", "--block-size", "--max-num-seqs", "--no-enforce-eager",
                       "--pipeline-parallel", "--async-scheduling", "--host", "--port", "--served-model-name",
                       "--max-model-len", "--max-num-batched-tokens", "--gpu-memory-utilization",
                       "--kv-cache-memory-bytes", "--num-gpu-blocks-override", "--api-server-count", "--generation-config"))
    if plan["model_config"]["architectures"] != ["PanguUltraMoEForCausalLM"]:
        raise ValueError("Ki debugging requires PanguUltraMoEForCausalLM")
    opt = plan["operator_config"]["operator_optimization_config"]
    if opt.get("use_quant_lightning_indexer_v2") is not True or opt.get("use_noncontiguous_kv") is not True:
        raise ValueError("Operator config must enable use_quant_lightning_indexer_v2 and use_noncontiguous_kv")
    engine = plan["command"]
    command = [sys.executable, "-m", "vllm_bench.mtp_debug.debug_spec", "--output", str(capture),
               "--rejection-windows", "--indexer-ki", "--attention-ranks", "all", "--mome-layers", a.mome_layers]
    for option in ("attention_steps_per_request", "indexer_ki_steps_per_request", "tensor_max_mib", "tensor_budget_mib", "limit"):
        command += ["--" + option.replace("_", "-"), str(getattr(a, option))]
    if a.indexer_ki_prolog:
        command += ["--indexer-ki-prolog", "--indexer-ki-prolog-positions", a.indexer_ki_prolog_positions]
    command += ["--execute", "--", *engine]
    plan.update(command=command, engine_command=engine, profiler_enabled=False,
                diagnostic_overhead=True, capture_output=str(capture))
    plan["cleared_environment"] = [*plan["cleared_environment"], "VLLM_TORCH_PROFILER_DIR"]
    return plan, env


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        plan, env = build_debug_plan(a)
        output = a.output.resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new or empty server output directory")
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.error(str(error))
    print("Command:", shlex.join(plan["command"]), flush=True)
    if not a.execute:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "server_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "operator_config.json").write_text(json.dumps(plan["operator_config"], indent=2) + "\n")
    return subprocess.call(plan["command"], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
