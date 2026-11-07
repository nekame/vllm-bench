"""Native vLLM OpenAI completion path, consumed locally without an HTTP socket."""

import argparse
from contextlib import aclosing
import json
from pathlib import Path
import time

from vllm_bench.pangu_infer.runtime import config_json, engine_options
from vllm_bench.pangu_infer.audit import execution_settings, runtime_manifest
from vllm_bench.pangu_infer.results import save_result, versions
from vllm_bench.pangu_infer.run import write_json


def server_argv(plan):
    """Use the server's parser/defaults, including native speculative defaults."""
    options = engine_options(plan, direct=False)
    options.pop("skip_tokenizer_init")
    if plan["options"].get("async_scheduling") is None:
        options.pop("async_scheduling")
    if plan["options"]["mode"] == "compile-acl":
        # HTTP leaves compilation mode to native vLLM; only graph policy is explicit.
        options["compilation_config"].pop("mode")
    if "speculative_config" in options:
        options["speculative_config"] = {"method": "mtp", "num_speculative_tokens": plan["options"]["depth"]}
    if plan["options"].get("tokenizer"):
        options["tokenizer"] = plan["options"]["tokenizer"]
    if plan["options"].get("omni2_audit", False):
        options["worker_extension_cls"] = "vllm_bench.pangu_infer.baseline.OmniBaselineWorkerExtension"
    options["served_model_name"] = "pangu-omni2"
    options["enable_prompt_tokens_details"] = True
    if plan["options"]["profile"]:
        options["profiler_config"] = {"profiler": "torch", "ignore_frontend": True,
                                      "torch_profiler_dir": str(Path(plan["options"]["output"]) / "traces")}
    extra = plan.get("omni2_server_args", [])
    protected = set(options) | {"async_scheduling", "model_tag", "worker_cls", "worker_extension_cls", "tokenizer", "skip_tokenizer_init", "config",
                                "headless", "api_server_count", "grpc", "pipeline_parallel_size",
                                "revision", "code_revision", "tokenizer_revision", "quantization",
                                "load_format", "model_impl", "enable_lora", "lora_modules",
                                "tool_parser_plugin", "reasoning_parser_plugin", "profiler_config"}
    protected = {name.replace("_", "-") for name in protected}
    for token in extra:
        if token.startswith("-") and not token.startswith("--"):
            try:
                float(token)
            except ValueError:
                raise ValueError("omni2-server-args requires full --option names instead of short aliases") from None
        if token.startswith("--"):
            name = token[2:].split("=", 1)[0].split(".", 1)[0].replace("_", "-")
            name = name.removeprefix("no-")
            if name in protected or name.startswith("data-parallel"):
                raise ValueError(f"omni2-server-args cannot override controlled option {token}; use run.py arguments")
    argv = []
    for name, value in options.items():
        flag = name.replace("_", "-")
        if isinstance(value, bool):
            argv.append("--" + ("" if value else "no-") + flag)
        else:
            argv.extend(("--" + flag, json.dumps(value) if isinstance(value, dict) else str(value)))
    return [*argv, *extra]


def completion_fields(plan, prompt_ids, limit):
    a = plan["options"]
    fields = {"prompt": prompt_ids, "max_tokens": limit, "stream": True,
              "temperature": a["temperature"], "top_p": a["top_p"], "seed": a["seed"],
              "ignore_eos": not a["respect_eos"], "add_special_tokens": False,
              "return_token_ids": True, "stream_options": {"include_usage": True}}
    if a["top_k"]:
        fields["top_k"] = a["top_k"]
    return fields


async def request(serving, plan, model, prompt_ids, limit):
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest

    # CompletionRequest supplies its native random request ID and defaults.
    params = CompletionRequest(model=model, **completion_fields(plan, prompt_ids, limit))
    started = time.perf_counter()
    stream = await serving.create_completion(params, raw_request=None)
    if not hasattr(stream, "__aiter__"):
        raise RuntimeError(f"Native completion did not return a stream: {stream}")
    tokens, text, counts = [], [], []
    first_time, last_time, first_count = None, None, 0
    cached_prompt_tokens, usage, done, finish_reason = None, None, False, None
    async with aclosing(stream):
        async for frame in stream:
            if isinstance(frame, bytes):
                frame = frame.decode("utf-8")
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    continue
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"Native completion stream error: {event['error']}")
                if event.get("usage") is not None:
                    usage = event["usage"]
                    details = usage.get("prompt_tokens_details") or {}
                    cached_prompt_tokens = details.get("cached_tokens", cached_prompt_tokens)
                for choice in event.get("choices", []):
                    if choice["index"] != 0:
                        raise RuntimeError("omni2 expects exactly one completion choice")
                    returned_prompt = choice.get("prompt_token_ids")
                    if returned_prompt is not None and returned_prompt != prompt_ids:
                        raise RuntimeError("Native completion changed the input token IDs")
                    delta = choice.get("token_ids")
                    if delta:
                        if any(type(token) is not int or token < 0 for token in delta):
                            raise RuntimeError("Native completion returned invalid token IDs")
                        now = time.perf_counter()
                        tokens.extend(delta)
                        counts.append(len(delta))
                        last_time = now
                        if first_time is None:
                            first_time, first_count = now, len(delta)
                    elif delta is None and choice.get("text"):
                        raise RuntimeError("Native completion omitted token IDs; return_token_ids support is required")
                    text.append(choice.get("text", ""))
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
    ended = time.perf_counter()
    if not done or finish_reason is None or first_time is None or usage is None:
        raise RuntimeError("Incomplete native completion stream (tokens, finish reason, usage and DONE are required)")
    if usage["completion_tokens"] != len(tokens) or usage["prompt_tokens"] != len(prompt_ids):
        raise RuntimeError("Native completion usage does not match returned token IDs")
    return {"token_ids": tokens, "text": "".join(text), "prefill_seconds": first_time - started,
            "decode_seconds": last_time - first_time, "decode_tokens": len(tokens) - first_count,
            "first_output_tokens": first_count, "cycles": None, "stream_token_counts": counts,
            "accepted_drafts": [], "acceptance_measurement": "unavailable from SSE; use native Omni speculative stats",
            "initial_context": len(prompt_ids), "cached_prompt_tokens": cached_prompt_tokens,
            "requested_tokens": limit, "finish_reason": finish_reason,
            "stream_tail_seconds": ended - last_time}


def check_settings(plan, state):
    a = plan["options"]
    expected = {"enable_prefix_caching": a.get("enable_prefix_caching", False),
                "cache_dtype": a.get("kv_cache_dtype", "auto")}
    if a.get("async_scheduling") is not None:
        expected["async_scheduling"] = a["async_scheduling"]
    if state.get("mome_prefill_state_fix", False):
        raise RuntimeError("omni2 requires the unmodified native MoME builder")
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError(f"Native server {key} differs from the requested setting")
    for key in ("block_size", "mamba_block_size"):
        if a.get(key) is not None and state[key] != a[key]:
            raise RuntimeError(f"Native server {key} differs from the requested setting")
    settings = state["execution_settings"]
    if a["mode"] != "compile-acl" and state["compilation"] != 0:
        raise RuntimeError("Native server compilation mode differs from the plan")
    if settings["dtype"] != "torch.bfloat16":
        raise RuntimeError("Native server dtype differs from the planned bfloat16")
    for key, value in (("max_model_len", plan["max_model_len"]), ("max_num_seqs", 1),
                       ("max_num_batched_tokens", a.get("prefill_chunk_size", 0) or plan["max_model_len"])):
        if settings[key] != value:
            raise RuntimeError(f"Native server {key} differs; align run.py and omni2-server-args")
    if settings["enable_chunked_prefill"] != bool(a.get("prefill_chunk_size", 0)):
        raise RuntimeError("Native server chunked prefill differs from the requested setting")
    if a.get("cudagraph_capture_sizes") is not None and sorted(settings["cudagraph_capture_sizes"]) != a["cudagraph_capture_sizes"]:
        raise RuntimeError("Native server capture sizes differ from the requested setting")


def check_workers(plan, states):
    if not states:
        raise RuntimeError("Native server returned no worker audit state")
    a = plan["options"]
    for state in states:
        check_settings(plan, state)
        if "acl" in a["mode"] and not state["compile_counter"].get("num_cudagraph_captured", 0):
            raise RuntimeError("Native server did not capture ACL graphs")
        if "compile" in a["mode"] and not (state["compile_counter"].get("num_graphs_seen", 0)
                                           or state["compile_counter"].get("stock_torch_compile_count", 0)):
            raise RuntimeError("Native server did not trace a compiled model")


def frontend_state(cfg):
    """Only frontend-known settings; no claims about worker-local state."""
    cache = cfg.cache_config
    return {"operator": None, "parallel": None, "hardware": None,
            "block_size": cache.block_size, "mamba_block_size": cache.mamba_block_size,
            "cache_dtype": cache.cache_dtype, "mamba_cache_mode": config_json(cache.mamba_cache_mode),
            "async_scheduling": bool(cfg.scheduler_config.async_scheduling),
            "enable_prefix_caching": bool(cache.enable_prefix_caching),
            "compilation": config_json(cfg.compilation_config.mode),
            "cudagraph": config_json(cfg.compilation_config.cudagraph_mode),
            "execution_settings": config_json(execution_settings(cfg))}


async def run_session(plan, args, engine, app, startup):
    a, output = plan["options"], Path(plan["options"]["output"])
    serving = app.state.openai_serving_completion
    if serving is None:
        raise RuntimeError("Native server did not initialize completion serving")
    cfg = engine.vllm_config
    if (cfg.parallel_config.tensor_parallel_size != a["tp"] or cfg.parallel_config.data_parallel_size != 1
            or cfg.parallel_config.pipeline_parallel_size != 1 or cfg.parallel_config.enable_expert_parallel):
        raise RuntimeError("Native server parallel configuration differs from the planned pure TP")
    if cfg.model_config.model != a["model"] or cfg.model_config.hf_config.architectures != ["PanguUltraMoEForCausalLM"]:
        raise RuntimeError("Native server model differs from the planned Ultra checkpoint")
    spec = cfg.speculative_config
    if (0 if spec is None else spec.num_speculative_tokens) != a["depth"]:
        raise RuntimeError("Native server MTP depth differs from the plan")
    expected_graph = a.get("cudagraph_mode") or ("FULL_DECODE_ONLY" if "acl" in a["mode"] else "NONE")
    if cfg.compilation_config.cudagraph_mode.name != expected_graph:
        raise RuntimeError("Native server graph mode differs from the plan")
    frontend = frontend_state(cfg)
    check_settings(plan, frontend)
    model = args.served_model_name[0] if args.served_model_name else args.model
    warmup_tokens = a.get("omni_warmup_tokens") or max(2 * (a["depth"] + 1) + 1, 17)
    started = time.perf_counter()
    for _ in range(a["warmup"]):
        await request(serving, plan, model, plan["prompt_ids"], warmup_tokens)
    warmup = time.perf_counter() - started
    audit = a.get("omni2_audit", False)
    before = await engine.collective_rpc("pangu_bench_worker_state") if audit else None
    if audit:
        check_workers(plan, before)
    history = a.get("omni_cache_history", "reuse")
    print(f"[execution] engine=omni2 client={type(engine).__name__} core={type(engine.engine_core).__name__} "
          f"native_lifespan=true worker_audit={str(audit).lower()} "
          f"mome_prefill_state_fix=false async_scheduling={frontend['async_scheduling']} "
          f"cudagraph_mode={cfg.compilation_config.cudagraph_mode.name}", flush=True)
    records = []
    if a["profile"]:
        await engine.start_profile()
    try:
        for prompt in plan.get("prompts", [None]):
            ids = plan["prompt_ids"] if prompt is None else prompt["prompt_ids"]
            for repeat in range(1, a["repeats"] + 1):
                reset = history == "cold-each" or (history == "cold-first" and not records)
                if reset:
                    if not await engine.reset_prefix_cache():
                        raise RuntimeError("Native server refused prefix-cache reset")
                    print(f"[prefix-cache-reset] before_record={len(records)}", flush=True)
                record = await request(serving, plan, model, ids, a["max_new_tokens"])
                record.update(repeat=repeat, prefix_cache_reset=reset)
                if prompt is not None:
                    record["id"] = prompt["id"]
                records.append(record)
                print("[request-result] " + json.dumps({"record": len(records) - 1,
                      "cached_prompt_tokens": record["cached_prompt_tokens"],
                      "first_output_tokens": record["first_output_tokens"]}), flush=True)
    finally:
        if a["profile"]:
            await engine.stop_profile()
    after = await engine.collective_rpc("pangu_bench_worker_state") if audit else None
    changed = None
    if audit:
        check_workers(plan, after)
        if len(before) != len(after):
            raise RuntimeError("Native server worker audit rank count changed")
        changed = any(left["compile_counter"] != right["compile_counter"] for left, right in zip(before, after))
        if changed and a["warmup"]:
            raise RuntimeError("Native server compiled/captured during measurement; increase warmup")
    first = after[0] if audit else frontend
    keys = ("operator", "parallel", "block_size", "mamba_block_size", "cache_dtype",
            "mamba_cache_mode", "async_scheduling", "enable_prefix_caching", "execution_settings")
    if audit and any(any(item[key] != first[key] for key in keys) for item in after):
        raise RuntimeError("Native server workers disagree on effective settings")
    effective = {key: first[key] for key in keys}
    effective.update(engine_core_mode="async-process", mome_prefill_state_fix=False,
                     settings_source="worker_audit" if audit else "frontend_vllm_config",
                     omni_cache_history=history, compilation={"mode": a["mode"],
                     "vllm": first["compilation"], "cudagraph": first["cudagraph"]})
    details = {"startup_seconds": startup, "warmup_seconds": warmup, "warmup_max_tokens": warmup_tokens,
               "workers_before": before, "workers_after": after, "worker_audit": audit,
               "native_lifespan": True, "versions": versions(),
               "hardware": first["hardware"], "compiled_during_measurement": changed,
               "server_argv": server_argv(plan), "frontend": type(engine).__name__,
               "frontend_runtime_manifest": runtime_manifest(),
               "engine_client": type(engine.engine_core).__name__,
               "timing_boundary": "native completion first-to-last token-bearing SSE frame; no network or per-request NPU sync"}
    spec = cfg.speculative_config
    write_json(output / "native_server_config.json", {
        "worker_cls": cfg.parallel_config.worker_cls, "effective": effective,
        "speculative": {"method": spec.method, "num_speculative_tokens": spec.num_speculative_tokens}
        if spec is not None else None, "generation_config": args.generation_config,
        "default_sampling_params": repr(serving.default_sampling_params)})
    save_result(output, plan, records, effective, details)


async def run(plan, args):
    from vllm.entrypoints.openai.api_server import build_app, build_async_engine_client, init_app_state

    started = time.perf_counter()
    async with build_async_engine_client(args) as engine:
        tasks = await engine.get_supported_tasks()
        app = build_app(args, tasks, engine.model_config)
        await init_app_state(engine, app.state, args, tasks)
        async with app.router.lifespan_context(app):
            await run_session(plan, args, engine, app, time.perf_counter() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parsed = parser.parse_args(argv)
    plan = json.loads(parsed.plan.read_text())
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    import uvloop

    cli_env_setup()
    args = make_arg_parser(FlexibleArgumentParser()).parse_args(server_argv(plan))
    validate_parsed_serve_args(args)
    if args.skip_tokenizer_init or args.headless or args.api_server_count not in (None, 1) or args.model_tag is not None:
        raise ValueError("omni2 requires tokenizer initialization and one native API client")
    write_json(Path(plan["options"]["output"]) / "native_server_argv.json", server_argv(plan))
    uvloop.run(run(plan, args))


if __name__ == "__main__":
    main()
