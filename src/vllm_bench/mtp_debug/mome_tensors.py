"""Capture exact selected decode convolution arguments around an unchanged call."""
import importlib.metadata
import os
import platform
import sys

from vllm_bench.mtp_debug.spec_debug import values
from vllm_bench.mtp_debug.tensor_io import TensorWriter, tensor_identity
from vllm_bench.mtp_debug.ranks import current_rank


def runtime_info():
    versions = {}
    for package in ("torch", "torch-npu", "vllm", "omni-infer"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "machine": platform.machine(), "packages": versions}


def parameters(module, original):
    loader = sys.modules.get("omni_npu.model_config.config_loader.loader")
    config = getattr(getattr(loader, "model_extra_config", None), "operator_opt_config", None)
    weight = getattr(module, "weight", None)
    return {"weight": tensor_identity(weight) if weight is not None else None,
            "runtime": runtime_info(),
            "kernel_width": getattr(module, "kernel_width", None),
            "on_ascend950": getattr(module, "on_ascend950", None),
            "enable_precision_strong_consistency": getattr(config, "enable_precision_strong_consistency", None),
            "callable": original.__module__ + "." + original.__qualname__}


def begin_capture(recorder, frame, event, arguments, params):
    meta = arguments["mome_metadata"]
    trace = {"format": "pangu-mome-tensors-v1", "status": "capturing",
             "parameters": {**params, "inplace": arguments.get("inplace", True)},
             "states_before": [], "states_after": []}
    event["tensor_capture"] = trace
    attention = frame.get("attention_trace", {})
    if attention.get("status") == "skipped":
        trace.update(status="skipped", reason=attention.get("reason"))
        return None
    if getattr(meta, "num_prefills", None) != 0 or getattr(meta, "num_decodes", None) != 1:
        trace.update(status="skipped", reason="not_single_request_decode")
        return None
    cache = arguments["conv_states"]
    indices = values(meta.cache_indices.reshape(-1))
    blocks = sorted({int(i) for i in indices if 0 <= int(i) < cache.shape[0]})
    if len(blocks) != 1:
        trace.update(status="skipped", reason="expected_one_referenced_state_block")
        return None
    if recorder.tensor_writer is None:
        recorder.tensor_writer = TensorWriter(recorder.output, os.getpid(),
            rank=current_rank(),
            max_tensor_bytes=int(os.environ.get("PANGU_SPEC_DEBUG_TENSOR_MAX_MIB", "16")) * 1024**2,
            budget_bytes=int(os.environ.get("PANGU_SPEC_DEBUG_TENSOR_BUDGET_MIB", "512")) * 1024**2)
    writer = recorder.tensor_writer
    call = len(frame["mome_calls"])  # Caller appends the event before invoking this helper.
    forward = frame["forward_index"]

    def save(tensor, name):
        return writer.save(tensor, forward, call, name)

    positions = frame.get("positions") or {}
    count = getattr(meta, "num_actual_tokens", None)
    trace["row_positions"] = (positions.get("values", [])[:count]
        if isinstance(count, int) and not positions.get("truncated") else [])
    trace["valid_token_rows"] = count
    trace["x_before"] = save(arguments["x"], "x-before")
    block = blocks[0]
    trace["states_before"] = [{"block": block, "tensor": save(cache[block], "state-before")}]
    accepted = values(meta.num_accepted_tokens)
    width = params.get("kernel_width")
    candidate = list(range(accepted[0] - 1, accepted[0] - 1 + width - 1)) if (
        isinstance(accepted, list) and len(accepted) == 1 and isinstance(width, int) and width > 1) else None
    trace["candidate_history_rows"] = candidate
    trace["history_mapping"] = "inferred_from_wrapper_contract_not_verified_kernel"
    return trace, save, cache, block, writer


def end_capture(context, arguments, result):
    if context is None:
        return
    trace, save, cache, block, writer = context
    inplace = arguments.get("inplace", True)
    trace["return_kind"] = "none" if result is None else "tensor"
    if result is None and inplace:
        # Some inplace kernels have no return value; their observable result is x.
        trace["y_source"] = "inplace_x"
        trace["y_after"] = save(arguments["x"], "y-after")
    elif result is None:
        trace["y_source"] = "unavailable"
        trace["y_after"] = {"status": "not_recorded", "reason": "none_return_without_inplace"}
    else:
        trace["y_source"] = "returned_tensor"
        trace["y_after"] = save(result, "y-after")
    trace["states_after"] = [{"block": block, "tensor": save(cache[block], "state-after")}]
    if inplace:
        trace["x_after"] = save(arguments["x"], "x-after")
    trace["status"] = "stopped" if writer.stopped_reason else "complete"
    if writer.stopped_reason:
        trace["reason"] = writer.stopped_reason
