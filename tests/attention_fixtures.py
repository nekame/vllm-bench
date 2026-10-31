"""Portable synthetic multi-rank capture fixtures (no ML packages)."""
import hashlib
import json
from pathlib import Path
import struct


def payload(root, name, dtype, shape, data, opaque=False, logical=None):
    path = Path(root) / "tensors" / (name + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else struct.pack("<" + {"float32": "f", "int32": "i", "uint8": "B"}[dtype] * len(data), *data)
    path.write_bytes(raw)
    descriptor = {"status": "complete", "path": path.relative_to(root).as_posix(), "dtype": dtype,
                  "shape": shape, "nbytes": len(raw), "byteorder": "little", "sha256": hashlib.sha256(raw).hexdigest()}
    if opaque:
        descriptor["comparison_mode"] = "bytes_only"
    if logical is not None:
        descriptor["logical_positions"] = logical
        descriptor["physical_pages"] = [99] * len(logical)
    return descriptor


def fixture(root, rank, side, mutation=None, query_position=100):
    affected = side == "right" and rank == 3
    def p(name, dtype, shape, data, **kwargs):
        return payload(root, f"{side}-{rank}-{name}", dtype, shape, data, **kwargs)
    packed = bytes(640) + struct.pack("<4f", 1, 1, 1, 1)
    indices = [0, 2 if affected and mutation in ("indices", "indexer_output") else 1] + [-1] * 2046
    positions = [0, indices[1]]
    packed_right = bytearray(packed)
    if affected and mutation == "kv":
        packed_right[0] = 1
    packed_right = bytes(packed_right)
    layer_config = {"cache_dtype": "hif8_ds_mla", "packed_width": 656, "num_local_heads": 1, "v_head_dim": 2,
                    "o_all_gather": True, "o_gather_size": 8, "skip_topk": False, "sink_count": 0,
                    "runtime": {"test": "same"}, "W_UV": {"sha256": "same"}}
    stage = lambda name, tensors, scalars=None: {"name": name, "operator": name, "status": "complete",
              "row_positions": [query_position], "tensors": tensors, "scalars": scalars or {}}
    indexer = stage("indexer_kernel", {
        "query": p("li-query", "uint8", [1, 1, 2], [2 if affected and mutation == "indexer_query" else 1, 2], opaque=True),
        "weights": p("li-weights", "float32", [1, 1], [1]),
        "query_dequant_scale": p("li-query-scale", "float32", [1, 1], [1]),
        "key_rows": p("li-key", "uint8", [query_position + 1, 1, 2], bytes([2 if affected and mutation == "indexer_history" else 1, 2]) * (query_position + 1), opaque=True, logical=list(range(query_position + 1))),
        "key_scales": p("li-scale", "float32", [query_position + 1, 1], [1] * (query_position + 1), logical=list(range(query_position + 1))),
        "indices": p("li-indices", "int32", [1, 1, 2048], indices)}, {"sparse_count": 2048, "sparse_mode": 3})
    output = [rank + 1. + (0.5 if affected and mutation == "kernel_output" else 0), rank + 2.]
    kernel = stage("attention_kernel", {
        "query": p("query", "float32", [1, 1, 2], [4 if affected and mutation == "query" else 3, 4]),
        "key_rows": p("kv", "uint8", [2, 1, 656], packed_right * 2, opaque=True, logical=positions),
        "value": p("dummy", "uint8", [1, 1, 1, 656], bytes(656), opaque=True),
        "actual_seq_lengths_query": p("q-len", "int32", [1], [1]),
        "actual_seq_lengths_kv": p("kv-len", "int32", [1], [query_position + 1]),
        "sparse_indices": p("indices", "int32", [1, 1, 2048], indices),
        "output": p("attn-out", "float32", [1, 1, 2], output)}, {"sparse_mode": 3, "key_dtype": "hif8"})
    projected = [rank * 10 + 1. + (0.5 if affected and mutation in ("kernel_output", "projection") else 0), rank * 10 + 2.]
    projection = stage("projection", {
        "input": p("proj-in", "float32", [1, 1, 2], output),
        "output": p("proj-out", "float32", [1, 2], projected)})
    projection["weight"] = {"sha256": "same"}
    global_o = [x for r in range(8) for x in (r * 10 + 1. + (0.5 if side == "right" and r == 3 and mutation in ("kernel_output", "projection") else 0), r * 10 + 2.)]
    if affected and mutation == "gather":
        global_o[0] += 0.5
    controls = []
    for conv in ("qa_conv", "compresskv_conv", "o_conv"):
        data = global_o if conv == "o_conv" else [1., 2.]
        controls.append({"prefix": "layers.0.attn.mome." + conv, "tensor_capture": {
            "status": "complete", "row_positions": [query_position],
            "parameters": {"weight": {"sha256": "same"}, "inplace": True},
            "x_before": p(conv + "-x", "float32", [1, len(data)], data),
            "y_after": p(conv + "-y", "float32", [1, len(data)], data),
            "states_before": [], "states_after": []}})
    layer = {"prefix": "layers.0.self_attn", "layer": 0, "status": "complete", "config": layer_config,
             "stages": [indexer, kernel, projection], "mome_controls": controls}
    token = 10 if side == "left" else 20
    event = {"rank": rank, "pid": rank + 1, "label": "request-01" if side == "left" else "request-02",
        "cycle": 0 if side == "left" else 68,
        "target_forward": {"request_ids": [side], "input_ids": {"values": [50, 999]},
             "positions": {"values": [query_position, query_position + 1]}, "mome_calls": controls,
             "attention_trace": {"status": "complete", "topology": {"rank": rank, "tp_rank": rank, "tp_size": 8}, "layers": [layer]}},
        "verify": {"accepted": [0], "cu_num_draft_tokens": [1], "draft_ids": [999], "target_argmax": [token],
                   "bonus_ids": [42], "matches_reference": True}, "sampler_output_with_padding": [[token, -1]]}
    return {"event": event, "sources": [f"{side}:{rank}"]}


def dataset(root, mutation=None):
    return [fixture(root, rank, side, mutation) for rank in range(8) for side in ("left", "right")]
