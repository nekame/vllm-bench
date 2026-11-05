#!/usr/bin/env python3
"""Read-only verification of an extracted checkpoint, without ML dependencies."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import sys

from vllm_bench.common.checkpoint import discover
from vllm_bench.layer_profile.prepare_checkpoint import layer_mapping, reduced_config, validate_mtp_tensors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--hashes", action="store_true", help="Read every output payload; requires --hash-tensors at extraction")
    a = p.parse_args()
    root = a.checkpoint.resolve()
    manifest = json.loads((root / "extraction_manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    keep_mtp = manifest.get("keep_mtp", False)
    if config != reduced_config(manifest["original_config"], manifest["layers"], keep_mtp):
        raise ValueError("Config differs from the recorded prefix extraction")
    tensors = discover(root)
    if keep_mtp:
        mapping = layer_mapping(manifest["original_config"], manifest["layers"], True)
        if manifest["retained_layer_mapping"] != {str(i): target for i, target in mapping.items()}:
            raise ValueError("Layer mapping differs from the recorded prefix extraction")
        prefix = manifest.get("layer_prefix", "model.layers.")
        pattern = re.compile("^" + re.escape(prefix) + r"(\d+)\.")
        indices = {int(m[1]) for name in tensors if (m := pattern.match(name))}
        if indices != set(mapping.values()):
            raise ValueError("Unexpected target/MTP tensor layer indices")
        validate_mtp_tensors(tensors, range(manifest["layers"], manifest["layers"] + config["num_nextn_predict_layers"]), prefix)
    total = sum(t["data_offsets"][1] - t["data_offsets"][0] for t in tensors.values())
    if total != manifest["tensor_bytes"] or len(tensors) != manifest["retained_tensors"]:
        raise ValueError("Tensor counts/bytes do not match extraction manifest")
    if a.hashes:
        expected = manifest.get("tensor_sha256")
        if not expected or set(expected) != set(tensors):
            raise ValueError("No complete tensor hashes in manifest; extract with --hash-tensors")
        for number, (name, item) in enumerate(tensors.items(), 1):
            digest = hashlib.sha256()
            start, end = item["data_offsets"]
            remaining = end - start
            with item["source"].open("rb") as f:
                f.seek(item["start"] + start)
                while remaining:
                    chunk = f.read(min(16 * 1024**2, remaining))
                    if not chunk:
                        raise ValueError(f"Unexpected EOF: {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Hash mismatch: {name}")
            if number % 1000 == 0:
                print(f"Verified {number}/{len(tensors)} tensors", flush=True)
    print(json.dumps({"valid": True, "layers": config["num_hidden_layers"],
                      "mtp_layers": config["num_nextn_predict_layers"],
                      "tensors": len(tensors), "bytes": total, "hashes_checked": a.hashes}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
