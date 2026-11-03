"""Shared, ML-free target-server validation, command and environment construction."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys

from vllm_bench.common.checkpoint import SUPPORTED_ARCHITECTURES, quantization_method
from vllm_bench.common.runtime_env import RESET_ENV, source_pythonpath


def add_server_arguments(p):
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-scope", choices=["reduced", "full"], default="reduced",
                   help="reduced: extracted target checkpoint; full: original target checkpoint")
    p.add_argument("--enable-mtp", action="store_true",
                   help="Enable MTP speculative decoding; requires a checkpoint retaining MTP weights")
    p.add_argument("--num-speculative-tokens", type=int,
                   help="MTP depth; requires --enable-mtp")
    p.add_argument("--operator-config", type=Path, required=True,
                   help="Your target OmniInfer operator/parallel config, with the same TP settings")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="pangu12")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp", type=int, default=8,
                   help="Tensor parallel size; DP stays 1. "
                        "Supply an operator config matching the selected TP")
    p.add_argument("--max-model-len", type=int, default=33024)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--kv-cache-memory-bytes", type=int)
    p.add_argument("--num-gpu-blocks-override", type=int)
    p.add_argument("--patches", help="Defaults to existing environment, then low_latency for V2 / high_throughout for Ultra")
    p.add_argument("--attention-backend", choices=["auto", "pangu", "deepseek"], default="auto")
    p.add_argument("--enable-expert-parallel", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--engine-cli-json", type=Path, help="JSON array of additional vllm serve arguments")
    p.add_argument("--execute", action="store_true", help="Without this flag only print the launch plan")


def build_launch(a, *, profiler_config=None, base_env=None, extra_blocked=()):
    model, operator, output = a.model.resolve(), a.operator_config.resolve(), a.output.resolve()
    cfg = json.loads((model / "config.json").read_text())
    if cfg.get("architectures") not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Expected a supported V2 or Ultra checkpoint")
    if a.num_speculative_tokens is not None and not a.enable_mtp:
        raise ValueError("--num-speculative-tokens requires --enable-mtp")
    if a.model_scope == "reduced":
        if not a.enable_mtp and cfg.get("num_nextn_predict_layers") != 0:
            raise ValueError("Expected an extracted V2 or Ultra checkpoint with MTP disabled; "
                    "use --enable-mtp for a checkpoint prepared with --keep-mtp")
    else:
        layers = cfg.get("num_hidden_layers")
        if type(layers) is not int or layers <= 0:
            raise ValueError("Full checkpoint needs a positive integer num_hidden_layers")
        if any((model / name).exists() for name in
               ("extraction_manifest.json", "mtp_checkpoint_manifest.json")):
            raise ValueError("Full mode requires the original target checkpoint, not an extracted checkpoint")
    speculative = None
    if a.enable_mtp:
        predictors = cfg.get("num_nextn_predict_layers")
        if type(predictors) is not int or predictors <= 0:
            raise ValueError("MTP requires a positive integer num_nextn_predict_layers in the checkpoint")
        if a.num_speculative_tokens is None:
            a.num_speculative_tokens = 3
        if a.num_speculative_tokens <= 0:
            raise ValueError("--num-speculative-tokens must be positive")
        speculative = {"method": "mtp", "num_speculative_tokens": a.num_speculative_tokens}
    quantization = quantization_method(cfg)
    ultra = cfg["architectures"] == ["PanguUltraMoEForCausalLM"]
    if a.attention_backend == "auto":
        a.attention_backend = "deepseek" if ultra else "pangu"
    if ultra and a.attention_backend == "pangu":
        raise ValueError("Ultra uses DSA/MLA attention; choose --attention-backend auto or deepseek")
    if a.patches is None:
        a.patches = os.getenv("OMNI_VLLM_PATCHES_DIR") or ("high_throughout" if ultra else "low_latency")
    if not operator.is_file():
        raise ValueError("Operator config does not exist")
    # Check shape now, before asking vLLM to initialize distributed workers.
    opt = json.loads(operator.read_text())
    if not {"model_parallel_config", "operator_optimization_config"} <= opt.keys():
        raise ValueError("Operator config needs model_parallel_config and operator_optimization_config")
    if a.tp <= 0 or min(a.max_model_len, a.max_num_seqs, a.max_num_batched_tokens) <= 0:
        raise ValueError("TP and workload limits must be positive")
    if not a.chunked_prefill and a.max_num_batched_tokens < a.max_model_len:
        raise ValueError("Without chunked prefill, max-num-batched-tokens must cover max-model-len")
    forbidden = {"pangu_v2_benchmark", "decode_profile_sync", "pd_bench_aligned_decode"}
    if forbidden & {s.strip() for s in a.patches.split(",")}:
        raise ValueError("Remove benchmark gates / decode sync patches from this profiling server")
    command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(model),
               "--served-model-name", a.name, "--host", a.host, "--port", str(a.port),
               "--tensor-parallel-size", str(a.tp), "--data-parallel-size", "1",
               "--dtype", "bfloat16", "--trust-remote-code", "--generation-config", "vllm",
               "--max-model-len", str(a.max_model_len), "--max-num-seqs", str(a.max_num_seqs),
               "--max-num-batched-tokens", str(a.max_num_batched_tokens),
               "--gpu-memory-utilization", str(a.gpu_memory_utilization),
               "--no-enable-prefix-caching", "--api-server-count", "1",
               "--enable-chunked-prefill" if a.chunked_prefill else "--no-enable-chunked-prefill"]
    if profiler_config is not None:
        command += ["--profiler-config", json.dumps(profiler_config)]
    if getattr(a, "kv_cache_dtype", None) is not None:
        command += ["--kv-cache-dtype", a.kv_cache_dtype]
    if getattr(a, "block_size", None) is not None:
        command += ["--block-size", str(a.block_size)]
    if speculative is not None:
        command += ["--speculative-config", json.dumps(speculative)]
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
            raise ValueError("engine-cli-json must be an array of CLI argument strings")
        blocked = ("--model", "--hf-overrides", "--speculative-config", "--data-parallel",
                   "--profiler-config", "--enable-prefix-caching", "--tensor-parallel-size", *extra_blocked)
        if any(s.replace("_", "-").startswith(blocked) for s in extra):
            raise ValueError("Extra CLI arguments cannot replace architecture, parallelism, cache or profiler settings")
        command += extra
    env = os.environ.copy() if base_env is None else base_env.copy()
    env["PYTHONPATH"] = source_pythonpath(env)
    for key in RESET_ENV:
        env.pop(key, None)
    settings = {"OMNI_VLLM_PATCHES_DIR": a.patches,
                "OMNI_CUSTOM_MODEL_CONFIG_PATH": str(operator),
                "OMNI_PANGU_V2_HIGH_THROUGHOUT": "1" if a.attention_backend == "deepseek" else "0"}
    if quantization == "mxfp8":
        # Mxfp8Config.get_quant_method explicitly requires this custom-model plugin.
        plugins = [s.strip() for s in env.get("VLLM_PLUGINS", "").split(",") if s.strip()]
        for required in ("omni-npu", "omni_npu_patches", "omni_custom_models"):
            if required not in plugins:
                plugins.append(required)
        settings["VLLM_PLUGINS"] = ",".join(plugins)
    env.update(settings)
    plan = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": command,
            "settings": settings, "cleared_environment": RESET_ENV,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
            "model_config": cfg, "operator_config": opt, "quantization": quantization,
            "model_scope": a.model_scope, "speculative_config": speculative,
            "annotation_overhead": False}
    return plan, env
