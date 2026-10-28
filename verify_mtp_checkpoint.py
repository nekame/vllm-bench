#!/usr/bin/env python3
"""Validate an extracted MTP checkpoint using only the standard library."""
import argparse
import hashlib
from pathlib import Path
import sys

from mtp_common import checkpoint_config
from prepare_checkpoint import discover
from prepare_mtp_checkpoint import select_mtp


def verify(path, hashes=False):
    path = path.resolve()
    config, manifest = checkpoint_config(path)
    tensors = discover(path)
    selected, indices = select_mtp(config, tensors)
    if selected.keys() != tensors.keys() or indices != manifest["mtp_layers"]:
        raise ValueError("Unexpected target tensors or wrong MTP indices")
    total = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values())
    if total != manifest["tensor_bytes"] or len(tensors) != manifest["retained_tensors"]:
        raise ValueError("Tensor inventory differs from manifest")
    if hashes:
        expected = manifest.get("tensor_sha256")
        if not expected or expected.keys() != tensors.keys():
            raise ValueError("Extract with --hash-tensors to verify payload hashes")
        for name, item in tensors.items():
            begin, end = item["data_offsets"]
            digest = hashlib.sha256()
            with item["source"].open("rb") as f:
                f.seek(item["start"] + begin)
                remaining = end - begin
                while remaining:
                    chunk = f.read(min(16 * 1024**2, remaining))
                    if not chunk:
                        raise ValueError(f"Truncated tensor: {name}")
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Tensor hash mismatch: {name}")
    return {"mtp_layers": indices, "tensors": len(tensors), "tensor_bytes": total, "hashes_verified": hashes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--hash-tensors", action="store_true")
    a = p.parse_args()
    import json
    print(json.dumps(verify(a.model, a.hash_tensors), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        sys.exit(f"ERROR: {error}")
