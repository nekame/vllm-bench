"""Plan or launch direct Pangu Ultra inference / an offline OmniInfer baseline."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from vllm_bench.common.checkpoint import quantization_method
from vllm_bench.common.runtime_env import instrumentation_pythonpath, RESET_ENV


MODES = ("eager", "compile", "acl", "compile-acl")
ULTRA_ARCHITECTURE = "PanguUltraMoEForCausalLM"
UPSTREAM = "4cd855b29b5eb0a78bca645fad39e3181c35d099"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def predictor_sequence(config, depth):
    start, count = config["num_hidden_layers"], config["num_nextn_predict_layers"]
    return [start + i for i in range(depth)] if 1 < count and depth <= count else [start] * depth


def checkpoint_identity(model):
    """Cheap change detection; do not hash terabytes of tensor payloads."""
    index = model / "model.safetensors.index.json"
    if index.exists():
        raw = index.read_bytes()
        names = sorted(set(json.loads(raw)["weight_map"].values()))
    else:
        raw = b""
        names = sorted(path.name for path in model.glob("*.safetensors"))
    shards = {}
    for name in names:
        path = (model / name).resolve()
        if not path.is_relative_to(model.resolve()):
            raise ValueError("Checkpoint index references an external shard")
        stat = path.stat()
        shards[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    manifest = model / "extraction_manifest.json"
    return {"index_sha256": hashlib.sha256(raw).hexdigest() if raw else None,
            "extraction_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
            "shard_metadata": shards, "payload_hashes_verified": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ignore-architecture", action="store_true",
                   help="Force PanguUltraMoEForCausalLM regardless of config.architectures; retain all other config fields")
    p.add_argument("--operator-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--engine", choices=("direct", "omni"), default="direct")
    p.add_argument("--async-scheduling", action="store_true",
                   help="Omni only: enable async scheduling and time decode between host-visible outputs")
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--mtp-tp", type=int, choices=(1, 2, 4, 8),
                   help="Direct only: MTP TP size on the first N ranks, for prefill and decode; default: target TP")
    p.add_argument("--compact-kv-cache", action="store_true",
                   help="Direct only: allocate each layer's cache with its own page size, without hybrid page padding")
    mtp = p.add_mutually_exclusive_group()
    mtp.add_argument("--depth", type=int, choices=(0, 3, 7), default=3,
                     help="MTP proposal depth; 0 disables MTP for greedy or sampled generation")
    mtp.add_argument("--no-mtp", dest="depth", action="store_const", const=0,
                     help="Run only the target model; equivalent to --depth 0")
    p.add_argument("--mtp-pad-to-8", action="store_true",
                   help="Direct only: pad recurrent single-token MTP calls to 8 input rows")
    p.add_argument("--mtp-no-static-kernels", action="store_true",
                   help="Direct compile modes only: disable MTP static kernels and super-kernel optimization; retain target settings")
    p.add_argument("--mode", choices=MODES, default="eager")
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument("--prompt-ids", type=Path, help="JSON array of exactly context-length token IDs")
    prompt.add_argument("--prompts", type=Path, help="JSON array of {id, input} text prompts")
    p.add_argument("--tokenizer", type=Path, help="Tokenizer directory for --prompts; default: model directory")
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--prefill-chunk-size", type=int, default=0,
                   help="Prefill token budget for both engines; 0 disables chunking, e.g. 2048 for context 8192")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2, help="Complete warmup requests")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--respect-eos", action="store_true", help="Default: ignore EOS for fixed-length benchmarks")
    p.add_argument("--profile", action="store_true", help="Diagnostic CPU/NPU traces; not benchmark timings")
    p.add_argument("--memory-diagnostics", action="store_true",
                   help="Direct only: log NPU memory at startup, first model calls, compilation and graph capture")
    p.add_argument("--model-timing", action="store_true",
                   help="Direct-only diagnostic: synchronized decode forward + LM head times, logged after each request")
    trace = p.add_mutually_exclusive_group()
    trace.add_argument("--trace-target", action="store_true",
                       help="Record Omni target inputs, metadata and logits for one eager request")
    trace.add_argument("--replay-target", type=Path,
                       help="Replay Omni target windows in direct; draft generation is disabled by default")
    p.add_argument("--replay-with-mtp", action="store_true",
                   help="Execute direct MTP between forced target windows; requires --replay-target and positive depth")
    p.add_argument("--trace-layers", default="",
                   help="Also record layer outputs and MoME states: all or comma-separated target layer indices")
    p.add_argument("--additional-config", type=Path)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="Offline engine memory budget")
    p.add_argument("--execute", action="store_true")
    return p


def make_plan(a):
    options = {key: str(value.resolve()) if isinstance(value, Path) else value
               for key, value in vars(a).items()}
    if a.compact_kv_cache and a.engine != "direct":
        raise ValueError("--compact-kv-cache requires --engine direct")
    if a.async_scheduling and a.engine != "omni":
        raise ValueError("--async-scheduling requires --engine omni")
    if a.mtp_tp is not None:
        if a.depth == 0 or a.mtp_tp > a.tp or a.tp % a.mtp_tp:
            raise ValueError("--mtp-tp requires positive MTP depth and must divide target TP")
        if a.engine != "direct" and a.mtp_tp != a.tp:
            raise ValueError("Different --mtp-tp is supported only by --engine direct")
    split_mtp = a.mtp_tp is not None and a.mtp_tp != a.tp
    if a.memory_diagnostics and a.engine != "direct":
        raise ValueError("--memory-diagnostics requires --engine direct")
    if a.model_timing and a.engine != "direct":
        raise ValueError("--model-timing requires --engine direct; stock OmniInfer instrumentation is not supported")
    if a.mtp_pad_to_8 and (a.engine != "direct" or a.depth == 0):
        raise ValueError("--mtp-pad-to-8 requires --engine direct and positive depth")
    if a.mtp_no_static_kernels and (a.engine != "direct" or not a.depth or "compile" not in a.mode):
        raise ValueError("--mtp-no-static-kernels requires direct, positive depth and a compile mode")
    diagnostic_trace = a.trace_target or a.replay_target is not None
    if a.async_scheduling and diagnostic_trace:
        raise ValueError("Target trace/replay requires synchronous scheduling")
    if a.tokenizer and not a.prompts:
        raise ValueError("--tokenizer requires --prompts")
    if a.prompts and diagnostic_trace:
        raise ValueError("Text prompt batches cannot be combined with target trace/replay; use --prompt-ids")
    if a.replay_with_mtp and (a.replay_target is None or a.depth == 0):
        raise ValueError("--replay-with-mtp requires --replay-target and positive depth")
    if diagnostic_trace:
        if a.mode != "eager" or a.temperature != 0 or a.repeats != 1:
            raise ValueError("Target trace/replay requires --mode eager --temperature 0 --repeats 1")
        if a.profile or a.model_timing:
            raise ValueError("Target trace/replay cannot be combined with profiling or model timing")
        if (a.trace_target and a.engine != "omni") or (a.replay_target and a.engine != "direct"):
            raise ValueError("Use --trace-target with omni and --replay-target with direct")
    elif a.trace_layers:
        raise ValueError("--trace-layers requires target trace or replay")
    if a.engine == "omni" and a.mode == "compile":
        raise ValueError("Stock OmniInfer bypasses compile-only without FULL graphs; use a direct run or another baseline mode")
    if min(a.context_length, a.repeats, a.warmup) < 1 or a.max_new_tokens < 2:
        raise ValueError("Context, warmup and repeats must be positive; max-new-tokens must be >= 2")
    if a.prefill_chunk_size:
        if a.prefill_chunk_size <= a.depth + 1:
            raise ValueError("prefill-chunk-size must exceed the decode verification window (depth + 1)")
        if a.prefill_chunk_size < a.context_length and a.context_length % a.prefill_chunk_size:
            raise ValueError("For chunked prefill, context-length must be a multiple of prefill-chunk-size")
        if diagnostic_trace:
            raise ValueError("Target trace/replay currently requires unchunked prefill")
    if (not math.isfinite(a.temperature) or a.temperature < 0 or
            not math.isfinite(a.top_p) or not 0 < a.top_p <= 1 or a.top_k < 0):
        raise ValueError("Require finite temperature >= 0, 0 < top-p <= 1 and top-k >= 0")
    if not 0 < a.gpu_memory_utilization <= 1:
        raise ValueError("gpu-memory-utilization must be in (0, 1]")
    config = json.loads((a.model / "config.json").read_text())
    if not a.ignore_architecture and config.get("architectures") != [ULTRA_ARCHITECTURE]:
        raise ValueError("This tool requires PanguUltraMoEForCausalLM; use --ignore-architecture to force Ultra")
    hf_overrides = {"architectures": [ULTRA_ARCHITECTURE]} if a.ignore_architecture else {}
    quantization = quantization_method(config)
    if type(config.get("num_hidden_layers")) is not int or config["num_hidden_layers"] <= 0:
        raise ValueError("Invalid target layer count")
    if a.depth and (type(config.get("num_nextn_predict_layers")) is not int or config["num_nextn_predict_layers"] < 1):
        raise ValueError("MTP layers missing; prepare the reduced checkpoint with --layers 12 --keep-mtp")
    vocab = config["vocab_size"]
    if type(vocab) is not int or vocab <= 1 or a.top_k > vocab:
        raise ValueError("Invalid vocabulary or top-k exceeds vocabulary")
    prompts = None
    if a.prompts:
        from vllm_bench.pangu_infer.prompts import load_prompts
        prompts = load_prompts(a.prompts, a.tokenizer or a.model, a.context_length, vocab)
        prompt = prompts[0]["prompt_ids"]
    elif a.prompt_ids:
        prompt = json.loads(a.prompt_ids.read_text())
    else:
        # Deterministic synthetic prompt, shared byte-for-byte by both engines.
        import random
        rng = random.Random(a.seed)
        excluded = config.get("eos_token_id", [])
        excluded = set(excluded if isinstance(excluded, list) else [excluded])
        prompt = []
        while len(prompt) < a.context_length:
            token = rng.randrange(vocab)
            if token not in excluded:
                prompt.append(token)
    if (not isinstance(prompt, list) or len(prompt) != a.context_length or
            any(type(t) is not int or not 0 <= t < vocab for t in prompt)):
        raise ValueError("prompt-ids must contain context-length valid integer token IDs")
    operator = json.loads(a.operator_config.read_text())
    if (not isinstance(operator, dict) or
            not {"model_parallel_config", "operator_optimization_config"} <= operator.keys()):
        raise ValueError("Operator config needs model_parallel_config and operator_optimization_config")
    parallel = operator["model_parallel_config"]
    opt = operator["operator_optimization_config"]
    if not isinstance(parallel, dict) or not isinstance(opt, dict):
        raise ValueError("Operator config sections must be objects")
    if split_mtp:
        from vllm_bench.pangu_infer.mtp_parallel import validate_layer_parallel_config
        validate_layer_parallel_config(parallel.get("layer_parallel_config", {}))
    # These paths require additional batch/state protocols and are deliberately
    # rejected rather than silently changing the user's operator configuration.
    forbidden = ("ena_seq_parallel", "ena_context_parallel", "ena_swa_attn_seq_parallel",
                 "enable_flashcomm2", "ena_dp_lmhead_parallel", "ena_local_lmhead_parallel")
    if any(parallel.get(key) for key in forbidden):
        raise ValueError("Use a pure TP operator config without SP/CP/FlashComm2/DP LM head")
    if config.get("use_mome") and not opt.get("use_noncontiguous_kv", False):
        raise ValueError("Direct MoME adapter requires use_noncontiguous_kv=true in BOTH compared runs")
    if a.depth and opt.get("fix_multi_mtp_kvcache", False):
        raise ValueError("This adapter implements the default fix_multi_mtp_kvcache=false proposal policy")
    if config.get("cla_explicit_mapping"):
        raise ValueError("CLA cache aliases are not supported by this PoC")
    additional = json.loads(a.additional_config.read_text()) if a.additional_config else {}
    if not isinstance(additional, dict):
        raise ValueError("additional-config must be an object")
    extension = additional.setdefault("npugraph_ex_config", {})
    if not isinstance(extension, dict):
        raise ValueError("npugraph_ex_config must be an object")
    extension.setdefault("enable", True)
    if "compile" in a.mode and extension["enable"] is not True:
        raise ValueError("Compile modes require npugraph_ex_config.enable=true")
    if extension.get("merge_dynamic_quant", False):
        raise ValueError("merge_dynamic_quant requires vLLM piecewise pass context; not supported by direct fullgraph compilation")
    # Reserve the largest supported verification window in every run. Besides
    # allowing warmup of short benchmarks, this keeps APC-off MoME block sizes
    # identical for target-only and MTP correctness comparisons.
    max_len = a.context_length + max(a.max_new_tokens, 17) + 8
    if max_len > config.get("max_position_embeddings", max_len):
        raise ValueError("Prompt + generation + speculative window exceeds model position limit")
    scope = "reduced" if (a.model / "extraction_manifest.json").is_file() else "full"
    identity = checkpoint_identity(a.model)
    prompt_hash = fingerprint(prompts if prompts is not None else prompt)
    match = {"model": options["model"], "config": config, "operator": operator,
             "checkpoint_identity": identity,
             "additional": additional, "prompt_sha256": prompt_hash,
             "tp": a.tp, "depth": a.depth, "mode": a.mode,
             "temperature": a.temperature, "top_k": a.top_k, "top_p": a.top_p,
             "seed": a.seed, "max_new_tokens": a.max_new_tokens, "respect_eos": a.respect_eos}
    if hf_overrides:
        match["hf_overrides"] = hf_overrides
    if split_mtp:
        match["mtp_tp"] = a.mtp_tp
    if a.mtp_no_static_kernels:
        match["mtp_no_static_kernels"] = True
    if a.async_scheduling:
        match["async_scheduling"] = True
    if a.prefill_chunk_size:
        match["prefill_chunk_size"] = a.prefill_chunk_size
    plan = {"format": "pangu-direct-plan-v1", "options": options, "config": config,
            "hf_overrides": hf_overrides,
            "operator_config": operator, "additional_config": additional,
            "prompt_ids": prompt, "prompt_sha256": prompt_hash,
            "comparison_sha256": fingerprint(match), "max_model_len": max_len,
            "checkpoint_identity": identity,
            "quantization": quantization, "model_scope": scope,
            "predictor_layers": predictor_sequence(config, a.depth) if a.depth else [],
            "upstream_commit": UPSTREAM, "validation": "NPU execution not yet performed"}
    if prompts is not None:
        plan["prompts"] = prompts
    if diagnostic_trace:
        from vllm_bench.pangu_infer.target_trace import parse_layers, validate_replay
        plan["trace_layers"] = parse_layers(a.trace_layers, config["num_hidden_layers"])
        if a.replay_target:
            validate_replay(plan)
    return plan


def child_environment(plan):
    env = os.environ.copy()
    env["PYTHONPATH"] = instrumentation_pythonpath(env, None)
    for name in RESET_ENV:
        env.pop(name, None)
    env.update(OMNI_CUSTOM_MODEL_CONFIG_PATH=plan["options"]["operator_config"],
               OMNI_VLLM_PATCHES_DIR="high_throughout", OMNI_PANGU_V2_HIGH_THROUGHOUT="1",
               VLLM_USE_V2_MODEL_RUNNER="0", VLLM_USE_AOT_COMPILE="0", OMNI_DRAFT_TOP_K="-1",
               PYTHONUNBUFFERED="1")
    if plan["options"]["engine"] == "omni":
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    plugins = [name for name in env.get("VLLM_PLUGINS", "").split(",") if name]
    # vLLM applies this allow-list to platform plugins as well as general
    # plugins. Omitting omni-npu leaves current_platform.device_type empty.
    env["VLLM_PLUGINS"] = ",".join(dict.fromkeys([
        *plugins, "omni-npu", "omni_npu_patches", "omni_custom_models",
    ]))
    return env


def launch(plan):
    output = Path(plan["options"]["output"])
    if output.exists():
        raise ValueError("Output already exists; use a fresh directory")
    model = Path(plan["options"]["model"])
    if checkpoint_identity(model) != plan["checkpoint_identity"] or json.loads((model / "config.json").read_text()) != plan["config"]:
        raise ValueError("Checkpoint changed after planning")
    if json.loads(Path(plan["options"]["operator_config"]).read_text()) != plan["operator_config"]:
        raise ValueError("Operator config changed after planning")
    env = child_environment(plan)
    output.mkdir(parents=True)
    plan_path = output / "launch_plan.json"
    write_json(plan_path, plan)
    module = "vllm_bench.pangu_infer.runtime" if plan["options"]["engine"] == "direct" else "vllm_bench.pangu_infer.baseline"
    if plan["options"]["engine"] == "direct":
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={plan['options']['tp']}", "--module", module, "--plan", str(plan_path)]
    else:
        command = [sys.executable, "-m", module, "--plan", str(plan_path)]
    with (output / "runtime.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
    if code:
        write_json(output / "failure.json", {"returncode": code, "log": str(output / "runtime.log")})
    return code


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        plan = make_plan(a)
        # Validate inherited hooks even in dry-run, without importing torch.
        child_environment(plan)
        print(json.dumps({k: v for k, v in plan.items() if k not in ("prompt_ids", "prompts")}, indent=2, ensure_ascii=False))
        return launch(plan) if a.execute else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
