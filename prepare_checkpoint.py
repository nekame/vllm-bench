#!/usr/bin/env python3
"""Extract a prefix checkpoint using only the Python standard library.

Tensor payloads are copied byte-for-byte. No torch import, tensor allocation,
model construction, or NPU initialization takes place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid

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


def reduced_config(config: dict, layers: int):
    cfg = json.loads(json.dumps(config))
    original = cfg["num_hidden_layers"]
    if cfg.get("architectures") not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Supported architectures: OpenPanguV2ForCausalLM and PanguUltraMoEForCausalLM")
    quantization_method(cfg)
    if not 3 < layers < original:
        raise ValueError(f"--layers must be between 4 and {original - 1}")
    if cfg.get("cla_explicit_mapping"):
        raise ValueError("CLA mappings need explicit remapping; unsupported by this prefix extractor")
    swa = cfg.get("swa_layers", [])
    windows = cfg.get("sliding_window_list", [])
    if len(swa) != len(windows):
        raise ValueError("swa_layers and sliding_window_list have different lengths")
    pairs = [(i, w) for i, w in zip(swa, windows) if i < layers]
    cfg["swa_layers"] = [i for i, _ in pairs]
    cfg["sliding_window_list"] = [w for _, w in pairs]
    for key in ("dsa_layers", "block_post_layernorm_idx"):
        if key in cfg:
            cfg[key] = [i for i in cfg[key] if i < layers]
    for key in ("indexer_types", "index_topk_pattern", "layer_types"):
        if key in cfg:
            cfg[key] = cfg[key][:layers]
    cfg["num_hidden_layers"] = layers
    cfg["num_nextn_predict_layers"] = 0
    dsa, local = set(cfg["dsa_layers"]), set(cfg["swa_layers"])
    if dsa & local or dsa | local != set(range(layers)):
        raise ValueError("DSA/SWA lists do not partition the retained layers")
    return cfg


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to SOURCE/config.json")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--layer-prefix", default="model.layers.")
    parser.add_argument("--max-shard-gib", type=float, default=4)
    parser.add_argument("--buffer-mib", type=int, default=16)
    parser.add_argument("--hash-tensors", action="store_true", help="Record source payload SHA256 while copying")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_shard_gib <= 0 or args.buffer_mib <= 0:
        parser.error("Shard and buffer sizes must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Source and output must be separate, non-nested directories")
    if output.exists():
        parser.error("Output already exists; choose a new directory (no overwrite)")
    config_path = args.config or source / "config.json"
    original, original_raw, inherited_quantization = resolve_config(source, config_path)
    config = reduced_config(original, args.layers)
    tensors = discover(source)
    layer_re = re.compile("^" + re.escape(args.layer_prefix) + r"(\d+)\.")
    selected, found_layers = {}, set()
    for name, item in tensors.items():
        match = layer_re.match(name)
        if match:
            layer = int(match[1])
            if layer >= args.layers:
                continue
            found_layers.add(layer)
        selected[name] = item
    if found_layers != set(range(args.layers)):
        raise ValueError(f"Missing retained layers or wrong prefix: {sorted(found_layers)}")
    if not any("embed_tokens" in n for n in selected) or not any("lm_head" in n for n in selected):
        raise ValueError("Expected untied embedding and lm_head tensors are missing")
    quantization = quantization_method(config)
    weight_dtypes = {"BF16", "F32"}
    if quantization == "mxfp8":
        weight_dtypes.add("F8_E4M3")
    bad_weights = [n for n, item in selected.items()
                   if n.endswith(".weight") and item["dtype"] not in weight_dtypes]
    if bad_weights:
        raise ValueError(f"Weight dtype does not match quantization={quantization!r}: {bad_weights[:5]}. "
                         "FP8 weights require the original MXFP8 quantization_config.")
    shards = plan_shards(selected, int(args.max_shard_gib * 1024**3))
    total = sum(x["data_offsets"][1] - x["data_offsets"][0] for x in selected.values())
    summary = {"source": str(source), "output": str(output), "layers": args.layers,
               "retained_tensors": len(selected), "discarded_tensors": len(tensors) - len(selected),
               "tensor_bytes": total, "tensor_gib": round(total / 1024**3, 3),
               "output_shards": len(shards), "dsa_layers": config["dsa_layers"],
               "swa_layers": config["swa_layers"], "mtp_layers": 0,
               "quantization": quantization, "inherited_source_quantization": inherited_quantization,
               "tensor_dtypes": sorted({item["dtype"] for item in selected.values()})}
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".partial-" + uuid.uuid4().hex[:10])
    stage.mkdir()
    try:
        free = shutil.disk_usage(stage).free
        if free < total + 64 * 1024**2:
            raise ValueError(f"Not enough free disk space: need at least {total} bytes")
        weight_map, hashes = {}, {} if args.hash_tensors else None
        for i, names in enumerate(shards, 1):
            filename = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            print(f"[{i}/{len(shards)}] {filename}: {len(names)} tensors", flush=True)
            write_shard(stage / filename, names, selected, args.buffer_mib * 1024**2, hashes)
            weight_map.update({name: filename for name in names})
        assets = copy_assets(source, stage)
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
        manifest = {**summary, "original_config": original,
                    "original_config_sha256": hashlib.sha256(original_raw).hexdigest(),
                    "config_file": str(config_path.resolve()),
                    "effective_config_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                    "retained_layer_mapping": {str(i): i for i in range(args.layers)},
                    "copied_assets": assets, "tensor_sha256": hashes,
                    "limitations": ["Synthetic profiling checkpoint; not for model quality evaluation",
                                    "MTP removed", "Original final stack layers 48/49 are not represented"]}
        (stage / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        discover(stage)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException:
        print(f"Incomplete output retained for inspection: {stage}", file=sys.stderr)
        raise
    print(f"Checkpoint ready: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
