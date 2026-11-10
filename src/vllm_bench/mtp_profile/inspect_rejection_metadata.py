"""Print rejection metadata from an MTP capture on CPU, without vLLM imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


FIELDS = ("num_rejected_tokens_gpu", "token_indices_to_sample")


def load_tensors(path):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Reading tensors.pt requires PyTorch; NPU and vLLM are not required") from error
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def inspect_snapshot(folder, *, tensor_loader=load_tensors):
    folder = Path(folder)
    if not (folder / "tree.json").is_file() and (folder / "before/tree.json").is_file():
        folder = folder / "before"
    tree = json.loads((folder / "tree.json").read_text())
    try:
        inputs = dict(dict(tree["$dict"])["inputs"]["$dict"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Expected a capture before/tree.json containing inputs") from error
    result, tensors = {}, None
    for name in FIELDS:
        if name not in inputs:
            raise ValueError(f"Missing recorded input: {name}")
        node = inputs[name]
        if node is None:
            result[name] = None
            continue
        if not isinstance(node, dict) or "$tensor" not in node:
            raise ValueError(f"Expected a tensor reference or null for {name}")
        if tensors is None:
            tensors = tensor_loader(folder / "tensors.pt")
        key = node["$tensor"]
        if key not in tensors:
            raise ValueError(f"Missing tensor {key!r} for {name}")
        result[name] = tensors[key].tolist()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="rank-NNN/before directory, or rank-NNN itself")
    parser.add_argument("--json", action="store_true", help="Print a JSON object")
    args = parser.parse_args(argv)
    result = inspect_snapshot(args.snapshot)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for name, value in result.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"ERROR: {error}")
