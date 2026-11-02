#!/usr/bin/env python3
"""Extract a prefix checkpoint using only the Python standard library.

Tensor payloads are copied byte-for-byte. No torch import, tensor allocation,
model construction, or NPU initialization takes place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

from vllm_bench.common.checkpoint import (
    SUPPORTED_ARCHITECTURES, copy_assets, discover, plan_shards,
    quantization_method, resolve_config, write_shard,
)


def layer_mapping(config: dict, layers: int, keep_mtp: bool = False):
    mapping = {i: i for i in range(layers)}
    if keep_mtp:
        count = config.get("num_nextn_predict_layers")
        if type(count) is not int or count <= 0:
            raise ValueError("--keep-mtp requires a positive integer num_nextn_predict_layers in the source")
        mapping.update({config["num_hidden_layers"] + i: layers + i for i in range(count)})
    return mapping


def remap_quantization(config, mapping):
    """Remap explicit module references; omit entries for discarded layers."""
    omitted = object()
    pattern = re.compile(r"(?<!\w)(model(?:\\?\.)layers(?:\\?\.))(\d+)(?=\\?\.|$)")

    def visit(value):
        if isinstance(value, str):
            if any(int(m[2]) not in mapping for m in pattern.finditer(value)):
                return omitted
            return pattern.sub(lambda m: m[1] + str(mapping[int(m[2])]), value)
        if isinstance(value, list):
            return [result for item in value if (result := visit(item)) is not omitted]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                key, item = visit(key), visit(item)
                if key is not omitted and item is not omitted:
                    result[key] = item
            return result
        return value

    return visit(config)


def reduced_config(config: dict, layers: int, keep_mtp: bool = False):
    cfg = json.loads(json.dumps(config))
    original = cfg["num_hidden_layers"]
    if cfg.get("architectures") not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Supported architectures: OpenPanguV2ForCausalLM and PanguUltraMoEForCausalLM")
    quantization_method(cfg)
    if not 3 < layers < original:
        raise ValueError(f"--layers must be between 4 and {original - 1}")
    if cfg.get("cla_explicit_mapping"):
        raise ValueError("CLA mappings need explicit remapping; unsupported by this prefix extractor")
    mapping = layer_mapping(config, layers, keep_mtp)
    swa = cfg.get("swa_layers", [])
    windows = cfg.get("sliding_window_list", [])
    if len(swa) != len(windows):
        raise ValueError("swa_layers and sliding_window_list have different lengths")
    pairs = [(mapping[i], w) for i, w in zip(swa, windows) if i in mapping]
    cfg["swa_layers"] = [i for i, _ in pairs]
    cfg["sliding_window_list"] = [w for _, w in pairs]
    for key in ("dsa_layers", "block_post_layernorm_idx"):
        if key in cfg:
            cfg[key] = [mapping[i] for i in cfg[key] if i in mapping]
    for key in ("indexer_types", "index_topk_pattern", "layer_types"):
        if key in cfg:
            cfg[key] = [cfg[key][i] for i in mapping if i < len(cfg[key])]
    cfg["num_hidden_layers"] = layers
    cfg["num_nextn_predict_layers"] = config["num_nextn_predict_layers"] if keep_mtp else 0
    if keep_mtp and "quantization_config" in cfg:
        cfg["quantization_config"] = remap_quantization(cfg["quantization_config"], mapping)
    dsa, local = set(cfg["dsa_layers"]), set(cfg["swa_layers"])
    if dsa & local or dsa | local != set(mapping.values()):
        raise ValueError("DSA/SWA lists do not partition the retained layers")
    return cfg


def validate_mtp_tensors(tensors, indices, prefix="model.layers."):
    for index in indices:
        base = f"{prefix}{index}."
        for suffix in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"):
            if base + suffix not in tensors:
                raise ValueError(f"Missing predictor-specific tensor: {base + suffix}")
        for part in ("self_attn.", "mlp."):
            if not any(name.startswith(base + part) and name.endswith(".weight") for name in tensors):
                raise ValueError(f"Missing MTP {part} weights for layer {index}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to SOURCE/config.json")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--keep-mtp", action="store_true",
                        help="Keep all source MTP layers and renumber them after the retained target layers")
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
    if args.keep_mtp and (source / "config.json").is_file():
        source_config = json.loads((source / "config.json").read_text())
        if original.get("num_nextn_predict_layers") != source_config.get("num_nextn_predict_layers"):
            raise ValueError("--keep-mtp requires the original source num_nextn_predict_layers; "
                             "a config override cannot change the number of predictors")
    config = reduced_config(original, args.layers, args.keep_mtp)
    mapping = layer_mapping(original, args.layers, args.keep_mtp)
    tensors = discover(source)
    layer_re = re.compile("^" + re.escape(args.layer_prefix) + r"(\d+)\.")
    selected, found_layers = {}, set()
    for name, item in tensors.items():
        match = layer_re.match(name)
        if match:
            layer = int(match[1])
            if layer not in mapping:
                continue
            found_layers.add(layer)
            name = name[:match.start(1)] + str(mapping[layer]) + name[match.end(1):]
        selected[name] = item
    if found_layers != set(mapping):
        raise ValueError(f"Missing retained layers or wrong prefix: {sorted(found_layers)}")
    if args.keep_mtp:
        validate_mtp_tensors(selected, range(args.layers, args.layers + config["num_nextn_predict_layers"]),
                             args.layer_prefix)
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
               "swa_layers": config["swa_layers"], "mtp_layers": config["num_nextn_predict_layers"],
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
                    "retained_layer_mapping": {str(i): target for i, target in mapping.items()},
                    "keep_mtp": args.keep_mtp, "layer_prefix": args.layer_prefix,
                    "copied_assets": assets, "tensor_sha256": hashes,
                    "limitations": ["Synthetic profiling checkpoint; not for model quality evaluation",
                                    "MTP sees shortened target hidden states; not full-model speculative acceptance"
                                    if args.keep_mtp else "MTP removed",
                                    "Original final stack layers 48/49 are not represented"]}
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
