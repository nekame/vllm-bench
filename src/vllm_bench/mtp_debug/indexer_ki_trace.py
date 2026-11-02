"""Exact raw ki at the QLIv2 cache-write boundary, including prefill."""
import contextvars
import functools
import json
import os
import re

from vllm_bench.mtp_debug.attention_trace import paged_by_slots, save, scalar, writer
from vllm_bench.mtp_debug.spec_debug import values

UPDATE = contextvars.ContextVar("pangu_indexer_ki_update", default=None)


def full_values(value):
    if value is None:
        return None
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    return values(value)


def prepare_ki_frame(runner, frame, bound, metadata):
    from vllm_bench.mtp_debug.spec_rejection import recorder
    rec = recorder()
    label = rec.output / "LABEL"
    request = tuple(frame["request_ids"])
    step = runner._pangu_ki_counts.get(request, 0)
    prefills = [getattr(meta, "num_prefills", None) for meta in (metadata or {}).values()]
    phase = "prefill" if any(isinstance(n, int) and n > 0 for n in prefills) else "decode"
    trace = {"format": "pangu-indexer-ki-v1", "rank": frame["rank"], "pid": os.getpid(),
             "request_ids": list(request), "label": label.read_text().strip() if label.exists() else None,
             "forward_index": frame["forward_index"], "phase": phase, "step": step,
             "status": "capturing", "calls": []}
    frame["_indexer_ki"] = trace
    if step >= int(os.environ.get("PANGU_SPEC_DEBUG_INDEXER_KI_STEPS", "32")):
        trace.update(status="skipped", reason="per_request_step_quota")
        return
    runner._pangu_ki_counts[request] = step + 1
    trace["input_ids"] = full_values(bound.get("input_ids"))
    trace["positions"] = full_values(bound.get("positions"))


def begin_write(frame, prefix, layer, arguments):
    trace = frame["_indexer_ki"]
    frame["_ki_tensor_slot"] = frame.get("_ki_tensor_slot", 20000) + 1
    call = {"prefix": prefix, "layer": layer, "slot": frame["_ki_tensor_slot"], "status": "capturing",
            "operator": "indexer_quant_cache", "parameters": {
                name: scalar(arguments.get(name)) for name in ("quant_mode", "round_scale")}}
    trace["calls"].append(call)
    ki, slots = arguments["ki"], full_values(arguments["slots"])
    call["input_shape"] = list(ki.shape)
    if (len(ki.shape) != 2 or ki.shape[1] != 128 or not isinstance(slots, list)
            or len(slots) != ki.shape[0] or arguments.get("quant_mode") != "mxfp8"):
        call.update(status="unsupported", reason="Expected MXFP8 ki[T,128] and slots[T]")
        return call
    positions, ids = trace.get("positions"), trace.get("input_ids")
    if (not isinstance(positions, list) or not isinstance(ids, list)
            or len(positions) != ki.shape[0] or len(ids) != ki.shape[0]):
        call["forward_row_counts"] = {"positions": len(positions) if isinstance(positions, list) else None,
                                      "input_ids": len(ids) if isinstance(ids, list) else None}
        call.update(status="unsupported", reason="Full forward input IDs/positions unavailable for ki row alignment")
        return call
    call["row_positions"] = positions[:ki.shape[0]]
    call["row_input_ids"] = ids[:ki.shape[0]]
    call["slots"] = slots
    call["cache_layout"] = {name: {"shape": list(arguments[name].shape), "dtype": str(arguments[name].dtype),
        "strides": list(arguments[name].stride()) if hasattr(arguments[name], "stride") else None,
        "storage_offset": arguments[name].storage_offset() if hasattr(arguments[name], "storage_offset") else None}
        for name in ("key", "scales")}
    call["active_rows"] = [i for i, slot in enumerate(slots) if slot >= 0]
    call["active_positions"] = [call["row_positions"][i] for i in call["active_rows"]]
    if len(set(call["active_positions"])) != len(call["active_positions"]):
        call.update(status="unsupported", reason="Ambiguous active ki positions")
        return call
    call["ki_before"] = save(frame, call, "ki-before", ki)
    return call


def end_write(frame, call, arguments, result):
    if call["status"] != "capturing":
        return
    call["return_kind"] = type(result).__name__
    call["ki_after"] = save(frame, call, "ki-after", arguments["ki"])
    active_slots = [call["slots"][i] for i in call["active_rows"]]
    for name, tensor in (("key_after", arguments["key"]), ("scales_after", arguments["scales"])):
        call[name] = paged_by_slots(frame, call, name.replace("_", "-"), tensor, active_slots, opaque=True)
        call[name]["logical_positions"] = call["active_positions"]
    fields = ("ki_before", "ki_after", "key_after", "scales_after")
    call["status"] = "complete" if all(call[field].get("status") == "complete" for field in fields) else "partial"
    if call["status"] != "complete":
        call["reason"] = writer().stopped_reason or "tensor_not_recorded"


def patch_cache_operator(owner):
    original = getattr(owner, "indexer_quant_cache", None)
    if original is None:
        raise RuntimeError("QLIv2 indexer_quant_cache operator is not registered")
    if getattr(original, "_pangu_ki_wrapped", False):
        return

    @functools.wraps(original)
    def invoke(*args, **kwargs):
        context = UPDATE.get()
        if context is None:
            return original(*args, **kwargs)
        frame, prefix, layer = context
        arguments = {**dict(zip(("key", "scales", "ki", "slots", "quant_mode", "round_scale"), args)), **kwargs}
        call = begin_write(frame, prefix, layer, arguments)
        result = original(*args, **kwargs)
        end_write(frame, call, arguments, result)
        return result

    invoke._pangu_ki_wrapped = True
    owner.indexer_quant_cache = invoke


def scope_update(indexer, prefix, layer):
    original = indexer._update_cache

    @functools.wraps(original)
    def update(*args, **kwargs):
        from vllm_bench.mtp_debug.spec_rejection import TARGET
        frame = TARGET.get()
        if frame is None or frame.get("_indexer_ki", {}).get("status") != "capturing":
            return original(*args, **kwargs)
        token = UPDATE.set((frame, prefix, layer))
        try:
            return original(*args, **kwargs)
        finally:
            UPDATE.reset(token)

    indexer._update_cache = update


def install_ki(runner, allowed):
    import torch
    runner._pangu_ki_counts = {}
    modules = []
    for name, module in runner.get_model().named_modules():
        if not callable(getattr(module, "_apply_attn_absorb", None)):
            continue
        prefix = getattr(module, "prefix", name)
        match = re.search(r"layers\.(\d+)(?:\.|$)", prefix)
        if not match or allowed is not None and int(match[1]) not in allowed:
            continue
        indexer = getattr(module, "indexer", None)
        if indexer is None:
            continue
        if not getattr(indexer, "use_qli_v2", False):
            raise RuntimeError("--indexer-ki requires QLIv2 MXFP8 indexer")
        scope_update(indexer, prefix, int(match[1]))
        modules.append({"prefix": prefix, "layer": int(match[1])})
    if not modules:
        raise RuntimeError("No selected QLIv2 indexer cache writers found")
    patch_cache_operator(torch.ops.cann_ops_transformer)
    return {"format": "pangu-indexer-ki-hooks-v1", "modules": modules, "prefill": True, "decode": True}


def finalize_ki_frame(frame):
    trace = frame.pop("_indexer_ki")
    trace["reads"] = []
    for layer in frame.get("attention_trace", {}).get("layers", []):
        for stage in layer.get("stages", []):
            if stage["name"] == "indexer_kernel":
                tensors = stage.get("tensors", {})
                trace["reads"].append({"layer": layer["layer"], "status": stage["status"],
                    "row_positions": stage.get("row_positions", []), "kv_length": stage.get("kv_length"),
                    "key_rows": tensors.get("key_rows"), "key_scales": tensors.get("key_scales")})
    if trace["status"] == "capturing":
        trace["status"] = "complete" if trace["calls"] and all(c["status"] == "complete" for c in trace["calls"]) else "partial"
    from vllm_bench.mtp_debug.spec_rejection import recorder
    path = recorder().output / f"ki-rank-{trace['rank']}-pid-{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(trace, ensure_ascii=False) + "\n")
