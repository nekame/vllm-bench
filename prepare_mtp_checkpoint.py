#!/usr/bin/env python3
"""Copy only MTP and shared tensors, without importing ML libraries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

from prepare_checkpoint import (SUPPORTED_ARCHITECTURES, copy_assets, discover,
                                plan_shards, quantization_method, resolve_config,
                                write_shard)


def select_mtp(config, tensors):
    if config.get("architectures") not in SUPPORTED_ARCHITECTURES:
        raise ValueError("Expected Pangu V2 or Ultra architecture")
    quantization_method(config)
    start, count = config["num_hidden_layers"], config.get("num_nextn_predict_layers", 0)
    if start < 1 or count < 1:
        raise ValueError("Source must contain MTP; a reduced 12-layer checkpoint cannot be used")
    indices = set(range(start, start + count))
    selected, found = {}, set()
    for name, item in tensors.items():
        match = re.match(r"^model\.layers\.(\d+)\.", name)
        if match:
            index = int(match[1])
            if index not in indices:
                continue
            found.add(index)
        elif not name.startswith(("model.embed_tokens.", "lm_head.")):
            continue
        selected[name] = item
    if found != indices:
        raise ValueError(f"Missing MTP layers: {sorted(indices - found)}")
    # Standalone loading follows set_shared_weight(target): target globals win.
    for prefix in ("model.embed_tokens.", "lm_head."):
        if not any(name.startswith(prefix) and name.endswith("weight") for name in selected):
            raise ValueError(f"Missing target shared weights: {prefix}")
    for index in indices:
        for part in ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight"):
            name = f"model.layers.{index}.{part}"
            if name not in selected:
                raise ValueError(f"Missing predictor-specific tensor: {name}")
        if not any(n.startswith(f"model.layers.{index}.self_attn.") for n in selected):
            raise ValueError(f"Missing attention weights for MTP layer {index}")
        if not any(n.startswith(f"model.layers.{index}.mlp.") for n in selected):
            raise ValueError(f"Missing MLP weights for MTP layer {index}")
    allowed = {"BF16", "F32"} | ({"F8_E4M3"} if quantization_method(config) else set())
    bad = [n for n, v in selected.items() if n.endswith(".weight") and v["dtype"] not in allowed]
    if bad:
        raise ValueError(f"Weights disagree with quantization metadata: {bad[:5]}")
    return selected, sorted(indices)


def prepare(source, output, *, config_path=None, shard_bytes=4 * 1024**3,
            buffer_bytes=16 * 1024**2, hash_tensors=False, dry_run=False):
    source, output = source.resolve(), output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output must be separate non-nested directories")
    if output.exists():
        raise ValueError("Output already exists; no overwrite")
    if min(shard_bytes, buffer_bytes) <= 0:
        raise ValueError("Shard and buffer sizes must be positive")
    config, original, inherited = resolve_config(source, config_path or source / "config.json")
    tensors = discover(source)
    selected, indices = select_mtp(config, tensors)
    shards = plan_shards(selected, shard_bytes)
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in selected.values())
    manifest = {"format": "pangu-mtp-checkpoint-v1", "source": str(source),
                "mtp_layers": indices, "original_num_hidden_layers": config["num_hidden_layers"],
                "architectures": config["architectures"], "quantization": quantization_method(config),
                "retained_tensors": len(selected), "tensor_bytes": total,
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "source_config_sha256": hashlib.sha256(original).hexdigest(),
                "inherited_quantization": inherited,
                "shared_weight_policy": "target embedding and lm_head; predictor head norms retained",
                "target_layers_present": False}
    print(json.dumps(manifest, indent=2), flush=True)
    if dry_run:
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".partial-" + uuid.uuid4().hex[:10])
    stage.mkdir()
    try:
        if shutil.disk_usage(stage).free < total + 64 * 1024**2:
            raise ValueError("Not enough free disk space")
        hashes, weight_map = ({} if hash_tensors else None), {}
        for i, names in enumerate(shards, 1):
            name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            write_shard(stage / name, names, selected, buffer_bytes, hashes)
            weight_map.update({n: name for n in names})
        copy_assets(source, stage)
        # Keep the original architecture and full layer-index semantics. The runtime
        # explicitly constructs MTP; this folder is NOT a serve-able target model.
        (stage / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")
        manifest["tensor_sha256"] = hashes
        (stage / "mtp_checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        discover(stage)
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    except BaseException:
        print(f"Incomplete output retained: {stage}", file=sys.stderr)
        raise
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--max-shard-gib", type=float, default=4)
    p.add_argument("--buffer-mib", type=int, default=16)
    p.add_argument("--hash-tensors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    prepare(a.source, a.output, config_path=a.config,
            shard_bytes=int(a.max_shard_gib * 1024**3), buffer_bytes=a.buffer_mib * 1024**2,
            hash_tensors=a.hash_tensors, dry_run=a.dry_run)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
