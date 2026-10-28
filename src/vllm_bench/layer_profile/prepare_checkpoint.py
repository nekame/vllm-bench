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
