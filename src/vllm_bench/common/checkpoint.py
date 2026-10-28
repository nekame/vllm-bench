"""Byte-preserving safetensors and checkpoint helpers; Python standard library only."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1,
               "F8_E5M2": 1, "F8_E8M0": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
               "I64": 8, "U64": 8, "F64": 8}
MAX_HEADER = 100 * 1024 * 1024
SUPPORTED_ARCHITECTURES = (["OpenPanguV2ForCausalLM"], ["PanguUltraMoEForCausalLM"])


def quantization_method(config: dict):
    quant = config.get("quantization_config")
    if quant is None or quant == {}:
        return None
    if not isinstance(quant, dict) or quant.get("quant_method") != "mxfp8":
        raise ValueError("Supported checkpoint formats: BF16 and quantization_config.quant_method='mxfp8'")
    return "mxfp8"


def resolve_config(source: Path, config_path: Path):
    """Preserve source quantization even when an older structural override lacks it."""
    raw = config_path.read_bytes()
    config = json.loads(raw)
    quantization_method(config)
    source_path = source / "config.json"
    inherited = False
    if source_path.is_file():
        source_config = json.loads(source_path.read_text())
        source_quant = quantization_method(source_config)
        keys = ("architectures", "num_hidden_layers", "hidden_size", "vocab_size",
                "num_attention_heads", "n_routed_experts", "moe_intermediate_size",
                "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
        different = [key for key in keys if key in source_config and source_config[key] != config.get(key)]
        if different:
            raise ValueError(f"Provided config disagrees with SOURCE/config.json: {different}. "
                             "Use the source config, or an override with the same architecture and dimensions.")
        if source_quant:
            if config.get("quantization_config"):
                if config["quantization_config"] != source_config["quantization_config"]:
                    raise ValueError("Provided quantization_config disagrees with SOURCE/config.json; "
                                     "the extractor does not change quantization")
            else:
                config["quantization_config"] = source_config["quantization_config"]
                inherited = True
    return config, raw, inherited


def read_header(path: Path):
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= MAX_HEADER:
            raise ValueError(f"Invalid header size in {path}: {size}")
        header = json.loads(f.read(size))
    entries = {k: v for k, v in header.items() if k != "__metadata__"}
    intervals = []
    for name, item in entries.items():
        a, b = item["data_offsets"]
        shape = item["shape"]
        if (not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b
                or any(not isinstance(x, int) or x < 0 for x in shape)):
            raise ValueError(f"Invalid tensor descriptor: {path}:{name}")
        dtype = item["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"Unsupported safetensors dtype {dtype!r}: {name}")
        if math.prod(shape) * DTYPE_BYTES[dtype] != b - a:
            raise ValueError(f"Tensor byte size mismatch: {path}:{name}")
        intervals.append((a, b))
    cursor = 0
    for a, b in sorted(intervals):
        if a != cursor:
            raise ValueError(f"Overlapping or non-contiguous payload: {path}")
        cursor = b
    if 8 + size + cursor != path.stat().st_size:
        raise ValueError(f"File size does not match header: {path}")
    return entries, 8 + size


def discover(source: Path):
    index = source / "model.safetensors.index.json"
    expected = None
    if index.is_file():
        expected = json.loads(index.read_text())["weight_map"]
        names = sorted(set(expected.values()))
        files = []
        for name in names:
            path = (source / name).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise ValueError(f"Missing or external shard: {name}")
            files.append(path)
    else:
        files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors found. PyTorch .bin checkpoints are not supported.")
    tensors = {}
    for path in files:
        header, start = read_header(path)
        for name, item in header.items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor name: {name}")
            tensors[name] = {"source": path, "start": start, **item}
    if expected is not None:
        if set(expected) != set(tensors):
            raise ValueError("Index and shard tensor names disagree")
        for name, shard in expected.items():
            if tensors[name]["source"] != (source / shard).resolve():
                raise ValueError(f"Index points to the wrong shard: {name}")
    return tensors


def plan_shards(selected, limit):
    shards, current, size = [], [], 0
    for name in sorted(selected, key=lambda n: (str(selected[n]["source"]),
                                               selected[n]["data_offsets"][0])):
        item = selected[name]
        length = item["data_offsets"][1] - item["data_offsets"][0]
        if current and size + length > limit:
            shards.append(current)
            current, size = [], 0
        current.append(name)
        size += length
    if current:
        shards.append(current)
    return shards


def write_shard(path, names, selected, buffer_bytes, hashes):
    header = {"__metadata__": {"format": "pt"}}
    cursor = 0
    for name in names:
        item = selected[name]
        a, b = item["data_offsets"]
        header[name] = {"dtype": item["dtype"], "shape": item["shape"],
                        "data_offsets": [cursor, cursor + b - a]}
        cursor += b - a
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    source_file, handle = None, None
    try:
        with path.open("xb") as out:
            out.write(struct.pack("<Q", len(raw)))
            out.write(raw)
            for name in names:
                item = selected[name]
                if item["source"] != source_file:
                    if handle:
                        handle.close()
                    source_file = item["source"]
                    handle = source_file.open("rb")
                a, b = item["data_offsets"]
                handle.seek(item["start"] + a)
                remaining = b - a
                digest = hashlib.sha256() if hashes is not None else None
                while remaining:
                    chunk = handle.read(min(buffer_bytes, remaining))
                    if not chunk:
                        raise ValueError(f"Unexpected EOF copying {name}")
                    out.write(chunk)
                    if digest:
                        digest.update(chunk)
                    remaining -= len(chunk)
                if digest:
                    hashes[name] = digest.hexdigest()
        read_header(path)
    finally:
        if handle:
            handle.close()


def copy_assets(source, destination):
    """Copy code/tokenizers/config assets, never weights, traces or old manifests."""
    suffixes = {".py", ".json", ".txt", ".model", ".tiktoken", ".jinja", ".jinja2", ".vocab", ".merges"}
    copied = []
    for path in sorted(source.iterdir()):
        if (not path.is_file() or path.suffix not in suffixes
                or path.name in {"config.json", "generation_config.json"}
                or path.name.endswith(".index.json") or path.name.startswith("extraction_")):
            continue
        if path.stat().st_size > 256 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large asset: {path}; copy required assets explicitly")
        shutil.copyfile(path, destination / path.name)
        copied.append(path.name)
    # Generation defaults must not secretly re-enable speculative generation.
    generation = source / "generation_config.json"
    if generation.is_file():
        cfg = json.loads(generation.read_text())
        for key in ("num_assistant_tokens", "num_assistant_tokens_schedule", "assistant_confidence_threshold"):
            cfg.pop(key, None)
        (destination / generation.name).write_text(json.dumps(cfg, indent=2) + "\n")
        copied.append(generation.name)
    return copied
