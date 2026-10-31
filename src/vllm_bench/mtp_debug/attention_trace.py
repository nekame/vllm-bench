"""Scoped exact HiF8 DSA operator capture. Imported only by enabled workers."""
import contextvars
import functools
import inspect
import math
import os
import re

from vllm_bench.mtp_debug.attention_layout import PACKED_WIDTH, page_coordinates, selected_positions
from vllm_bench.mtp_debug.mome_tensors import runtime_info
from vllm_bench.mtp_debug.ranks import current_rank
from vllm_bench.mtp_debug.spec_debug import values
from vllm_bench.mtp_debug.tensor_io import TensorWriter, WIDTH, tensor_bytes, tensor_identity

OP = contextvars.ContextVar("pangu_attention_operator", default=None)


def writer():
    from vllm_bench.mtp_debug.spec_rejection import recorder
    rec = recorder()
    if rec.tensor_writer is None:
        rec.tensor_writer = TensorWriter(rec.output, os.getpid(), current_rank(),
            int(os.environ.get("PANGU_SPEC_DEBUG_TENSOR_MAX_MIB", "16")) * 1024**2,
            int(os.environ.get("PANGU_SPEC_DEBUG_TENSOR_BUDGET_MIB", "512")) * 1024**2)
    return rec.tensor_writer


def topology(runner):
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    config = runner.vllm_config.parallel_config
    if config.data_parallel_size != 1 or group.world_size != 8 or getattr(config, "pipeline_parallel_size", 1) != 1:
        raise RuntimeError("attention trace v1 requires DP=1, TP=8, PP=1")
    return {"rank": current_rank(), "tp_rank": group.rank_in_group, "tp_size": group.world_size}


def prepare_frame(runner, frame, metadata):
    counter = runner._pangu_attention_counts
    request = tuple(frame["request_ids"])
    decode = any(getattr(meta, "num_prefills", None) == 0 and getattr(meta, "num_decodes", None) == 1
                 for meta in (metadata or {}).values())
    trace = {"format": "pangu-attention-trace-v1", "topology": runner._pangu_attention_topology,
             "status": "capturing", "layers": [], "step": counter.get(request, 0)}
    frame["attention_trace"] = trace
    if not decode:
        trace.update(status="skipped", reason="not_single_request_decode")
    elif counter.get(request, 0) >= int(os.environ.get("PANGU_SPEC_DEBUG_ATTENTION_STEPS", "32")):
        trace.update(status="skipped", reason="per_request_step_quota")
    else:
        counter[request] = counter.get(request, 0) + 1


def layer_trace(frame, module):
    trace = frame.get("attention_trace", {})
    if trace.get("status") != "capturing":
        return None
    prefix = module.prefix
    for layer in trace["layers"]:
        if layer["prefix"] == prefix:
            return layer
    layer = {"prefix": prefix, "layer": module.layer_idx, "status": "capturing", "stages": [],
             "config": module._pangu_attention_config}
    trace["layers"].append(layer)
    return layer


def stage(frame, layer, name, operator):
    frame["_attention_tensor_slot"] = frame.get("_attention_tensor_slot", 10000) + 1
    item = {"name": name, "operator": operator, "status": "capturing", "tensors": {},
            "scalars": {}, "slot": frame["_attention_tensor_slot"]}
    layer["stages"].append(item)
    return item


def save(frame, item, name, tensor, opaque=False):
    if tensor is None:
        return {"status": "not_recorded", "reason": "none_tensor"}
    source_dtype = str(tensor.dtype)
    if opaque or source_dtype.removeprefix("torch.") not in WIDTH:
        import torch
        tensor = tensor.contiguous().view(torch.uint8)
        opaque = True
    # The full payload SHA suffices here; thousands of per-cache-row samples
    # would make each JSONL event larger than its useful tensor evidence.
    descriptor = writer().save(tensor, frame["forward_index"], item["slot"], name, row_summaries=False)
    if opaque:
        descriptor.update(comparison_mode="bytes_only", source_dtype=source_dtype)
    return descriptor


def paged(frame, item, name, tensor, positions, table, opaque=False):
    if tensor is None or tensor.dim() < 2:
        return {"status": "not_recorded", "reason": "unsupported_paged_tensor"}
    pages, slots = page_coordinates(positions, table, tensor.shape[1], tensor.shape[0])
    row_bytes = math.prod(tensor.shape[2:]) * tensor.element_size()
    expected = len(positions) * row_bytes
    if not writer().can_save(expected):
        return {"status": "not_recorded", "reason": writer().stopped_reason}
    source_dtype = str(tensor.dtype)
    dtype = source_dtype.removeprefix("torch.")
    opaque = opaque or dtype not in WIDTH
    # NPU advanced indexing (tensor[pages, slots]) is not supported for all
    # quantized cache layouts. Copy each referenced page once using basic
    # slicing, then gather exact rows on CPU by byte offsets. Never copy the
    # entire cache or run an indexing kernel on it.
    grouped = {}
    for index, (page, slot) in enumerate(zip(pages, slots)):
        grouped.setdefault(page, []).append((index, slot))
    raw = bytearray(expected)
    for page, rows in grouped.items():
        cpu_page = tensor[page:page + 1].detach().cpu().contiguous()
        if opaque:
            import torch
            cpu_page = cpu_page.view(torch.uint8)
        content = tensor_bytes(cpu_page)
        if len(content) != tensor.shape[1] * row_bytes:
            raise ValueError("Copied KV page byte size mismatch")
        for index, slot in rows:
            raw[index * row_bytes:(index + 1) * row_bytes] = content[slot * row_bytes:(slot + 1) * row_bytes]
    shape = [len(positions), *tensor.shape[2:]]
    if opaque:
        shape = [len(positions), *tensor.shape[2:-1], tensor.shape[-1] * tensor.element_size()] if tensor.dim() > 2 else [len(positions), tensor.element_size()]
        dtype = "uint8"
    descriptor = writer().save_bytes(raw, dtype, shape, frame["forward_index"], item["slot"], name)
    if opaque:
        descriptor.update(comparison_mode="bytes_only", source_dtype=source_dtype)
    descriptor["gather_method"] = "basic_page_slice_to_cpu_then_byte_offsets"
    descriptor["logical_positions"] = positions
    # Physical mappings are audit data, never used as equality evidence.
    descriptor["physical_pages"] = pages
    descriptor["page_slots"] = slots
    return descriptor


def scalar(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [scalar(x) for x in value]
    return str(value)


def capture_inputs(ctx, arguments):
    frame, layer, kind = ctx["frame"], ctx["layer"], ctx["kind"]
    item = stage(frame, layer, kind, ctx["operator"])
    tensors = item["tensors"]
    lengths = arguments.get("actual_seq_lengths_kv", arguments.get("actual_seq_lengths_key", arguments.get("seqused_k")))
    query_lengths = arguments.get("actual_seq_lengths_query", arguments.get("cu_seqlens_q"))
    length_values = values(lengths)
    q_values = values(query_lengths)
    table = values(arguments.get("block_table"))
    if not isinstance(length_values, list) or len(length_values) != 1 or not q_values or len(table or []) != 1:
        item.update(status="unsupported", reason="single-request operator lengths/block table required")
        return item
    length = int(length_values[0])
    valid_rows = int(q_values[-1])
    if length < 0 or valid_rows < 1:
        raise ValueError("Invalid operator sequence lengths")
    positions = frame.get("positions") or {}
    item["row_positions"] = positions.get("values", [])[:valid_rows] if not positions.get("truncated") else []
    item["kv_length"] = length
    item["query_lengths"] = q_values
    item["layout_complete"] = len(item["row_positions"]) == valid_rows
    indices = arguments.get("sparse_indices")
    if kind == "attention_kernel":
        key = arguments["key"]
        query = arguments["query"]
        if (indices is None or len(indices.shape) != 3 or list(indices.shape[1:]) != [1, 2048] or len(key.shape) != 4
                or list(key.shape[-2:]) != [1, PACKED_WIDTH]
                or len(query.shape) != 3 or query.shape[-1] != 576
                or valid_rows > query.shape[0] or indices.shape[0] != query.shape[0]):
            item.update(status="unsupported", reason="expected HiF8 sparse count 2048 and packed width 656")
            return item
        logical_positions, validation = selected_positions(values(indices), length)
        item["selection_validation"] = validation
        if validation["invalid_indices"]:
            item["layout_complete"] = False
    else:
        logical_positions = list(range(length))
    for name, tensor in arguments.items():
        if name == "key":
            tensors["key_rows"] = paged(frame, item, "key-rows", tensor, logical_positions, table, opaque=True)
            if kind == "attention_kernel":
                tensors["key_rows"]["semantic"] = "hif8_packed_512_128_16"
        elif name in ("key_dequant_scale", "key_scale"):
            tensors["key_scales"] = paged(frame, item, "key-scales", tensor, logical_positions, table)
        elif hasattr(tensor, "detach"):
            opaque = ((kind == "indexer_kernel" and name == "query" and arguments.get("query_dtype") is not None)
                      or (kind == "attention_kernel" and name == "value"))
            tensors[name] = save(frame, item, name.replace("_", "-"), tensor, opaque)
        else:
            item["scalars"][name] = scalar(tensor)
    return item


def first_tensor(result):
    if hasattr(result, "detach"):
        return result
    if isinstance(result, (tuple, list)) and result:
        return result[0] if hasattr(result[0], "detach") else None
    return None


def patch_operator(owner, attribute, kind, positional=()):
    original = getattr(owner, attribute, None)
    if original is None or getattr(original, "_pangu_attention_wrapped", False):
        return

    @functools.wraps(original)
    def call(*args, **kwargs):
        ctx = OP.get()
        if ctx is None or ctx["kind"] != kind:
            return original(*args, **kwargs)
        bound = {**dict(zip(positional, args)), **kwargs}
        ctx["operator"] = attribute
        previous_stages = len(ctx["layer"]["stages"])
        try:
            item = capture_inputs(ctx, bound)
        except (ValueError, IndexError, TypeError, KeyError) as error:
            item = ctx["layer"]["stages"][-1] if len(ctx["layer"]["stages"]) > previous_stages else stage(ctx["frame"], ctx["layer"], kind, attribute)
            item.update(status="unsupported", reason=str(error))
        result = original(*args, **kwargs)
        if item["status"] == "capturing":
            tensor = first_tensor(result)
            field = "indices" if kind == "indexer_kernel" else "output"
            item["tensors"][field] = save(ctx["frame"], item, field, tensor)
            item["return_kind"] = type(result).__name__
            item["status"] = "stopped" if writer().stopped_reason else "complete"
            if not item["layout_complete"] or any(d.get("status") != "complete" for d in item["tensors"].values()):
                item["status"] = "partial"
        return result

    call._pangu_attention_wrapped = True
    setattr(owner, attribute, call)


def install_operators():
    import torch
    import torch_npu
    patch_operator(torch.ops.custom, "npu_ai_infra_kv_quant_sparse_flash_attention", "attention_kernel")
    patch_operator(torch_npu, "npu_quant_lightning_indexer", "indexer_kernel")
    patch_operator(torch.ops.cann_ops_transformer, "quant_lightning_indexer", "indexer_kernel",
                   ("query", "key", "weights", "query_dequant_scale", "key_dequant_scale", "sparse_count", "quant_mode"))


def scope_method(obj, method_name, module, kind):
    original = getattr(obj, method_name)

    @functools.wraps(original)
    def call(*args, **kwargs):
        from vllm_bench.mtp_debug.spec_rejection import TARGET
        frame = TARGET.get()
        layer = layer_trace(frame, module) if frame is not None else None
        if layer is None:
            return original(*args, **kwargs)
        token = OP.set({"frame": frame, "layer": layer, "kind": kind})
        try:
            return original(*args, **kwargs)
        finally:
            OP.reset(token)

    setattr(obj, method_name, call)


def projection_method(module):
    original = module._post_attn_absorb

    @functools.wraps(original)
    def call(*args, **kwargs):
        from vllm_bench.mtp_debug.spec_rejection import TARGET
        frame = TARGET.get()
        layer = layer_trace(frame, module) if frame is not None else None
        if layer is None:
            return original(*args, **kwargs)
        item = stage(frame, layer, "projection", original.__qualname__)
        bound = inspect.signature(original).bind(*args, **kwargs).arguments
        item["tensors"]["input"] = save(frame, item, "projection-input", bound["out"])
        weight = bound.get("w_uv")
        if weight is not None:
            item["weight"] = tensor_identity(weight)
        else:
            item["weight"] = module._pangu_attention_config["W_UV"]
        result = original(*args, **kwargs)
        item["tensors"]["output"] = save(frame, item, "projection-output", first_tensor(result))
        item["row_positions"] = next((s["row_positions"] for s in reversed(layer["stages"]) if s["name"] == "attention_kernel" and "row_positions" in s), [])
        item["status"] = "stopped" if writer().stopped_reason else "complete"
        if any(d.get("status") != "complete" for d in item["tensors"].values()):
            item["status"] = "partial"
        return result

    module._post_attn_absorb = call


def install_attention(runner, allowed):
    runner._pangu_attention_topology = topology(runner)
    runner._pangu_attention_counts = {}
    installed = []
    for name, module in runner.get_model().named_modules():
        if not callable(getattr(module, "_apply_attn_absorb", None)):
            continue
        match = re.search(r"layers\.(\d+)(?:\.|$)", getattr(module, "prefix", name))
        if not match or (allowed is not None and int(match[1]) not in allowed):
            continue
        if (module.cache_dtype != "hif8_ds_mla" or not module.on_ascend950 or not module.noncontiguous_kv
                or module.kv_lora_rank != 512 or module.qk_rope_head_dim != 64
                or getattr(module, "ena_cp", False) or getattr(module, "ena_kvsp", False)):
            raise RuntimeError("attention trace supports only A5 HiF8 DSA 512/64 noncontiguous KV without CP")
        module._pangu_attention_config = {
            "runtime": runtime_info(),
            "cache_dtype": module.cache_dtype, "packed_width": PACKED_WIDTH, "layer": int(match[1]),
            "num_heads": module.num_heads, "v_head_dim": module.v_head_dim,
            "num_local_heads": module.num_local_heads,
            "skip_topk": module.skip_topk, "sink_count": module.param_sink_number,
            "W_UV": tensor_identity(module.attn.W_UV),
            "o_all_gather": module.kv_b_proj.tp_size > 1,
            "o_gather_size": module.kv_b_proj.tp_size}
        scope_method(module, "_apply_attn_absorb", module, "attention_kernel")
        if module.indexer is not None:
            scope_method(module.indexer, "_apply_lightning_indexer", module, "indexer_kernel")
        projection_method(module)
        installed.append({"prefix": module.prefix, **module._pangu_attention_config})
    if not installed:
        raise RuntimeError("No selected HiF8 DSA attention modules found")
    install_operators()
    return {"format": "pangu-attention-hooks-v1", "topology": runner._pangu_attention_topology,
            "layers": installed, "runtime": runtime_info()}


def finalize_frame(frame):
    trace = frame["attention_trace"]
    if trace["status"] != "capturing":
        return
    complete = bool(trace["layers"])
    for layer in trace["layers"]:
        controls = []
        for call in frame["mome_calls"]:
            match = re.search(r"layers\.(\d+)(?:\.|$)", call["prefix"])
            if match and int(match[1]) == layer["layer"]:
                controls.append(call)
        layer["mome_controls"] = controls
        stages = layer["stages"]
        required = {"attention_kernel", "projection"}
        if not layer["config"]["skip_topk"]:
            required.add("indexer_kernel")
        observed = {s["name"] for s in stages if s["status"] == "complete"}
        layer["status"] = "complete" if required <= observed else "partial"
        layer["missing_stages"] = sorted(required - observed)
        complete &= layer["status"] == "complete"
    trace["status"] = "complete" if complete else "partial"
