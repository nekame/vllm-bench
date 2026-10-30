"""Explicit JSON tree + weights-only tensor serialization for MTP recordings.

No pickle object loading. Unsupported Python state fails rather than silently
dropping metadata. ML libraries are imported only when encoding/decoding state.
"""
from __future__ import annotations

import enum
import importlib
import importlib.metadata
import json
from pathlib import Path

from vllm_bench.mtp_profile.mtp_common import write_json


def versions():
    result = {}
    for name in ("vllm", "torch", "torch-npu", "omni-npu"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


class Codec:
    def __init__(self, tensors=None, device="cpu", limit_bytes=None, reuse_buffers=None):
        self.tensors = {} if tensors is None else tensors
        self.device, self.limit_bytes = device, limit_bytes
        self.bytes = 0
        self.memo = {}
        self.keepalive = []
        self.reuse_buffers = reuse_buffers

    def encode(self, value):
        import torch
        import numpy as np
        if value is None or type(value) in (bool, str, int, float):
            return value
        if isinstance(value, torch.Tensor):
            if id(value) not in self.memo:
                size = value.numel() * value.element_size()
                if self.limit_bytes is not None and self.bytes + size > self.limit_bytes:
                    raise ValueError("Capture exceeds byte limit; increase PANGU_MTP_CAPTURE_MAX_GIB")
                self.bytes += size
                key = str(len(self.tensors))
                self.tensors[key] = value.detach().cpu().clone()
                self.memo[id(value)] = key
                self.keepalive.append(value)
            return {"$tensor": self.memo[id(value)], "device": str(value.device)}
        if isinstance(value, torch.dtype):
            return {"$dtype": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device):
            return {"$device": str(value)}
        if isinstance(value, torch.Generator):
            return {"$generator": self.encode(value.get_state()), "device": str(value.device)}
        if isinstance(value, np.ndarray):
            return {"$numpy": self.encode(torch.from_numpy(value.copy()))}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, enum.Enum):
            return {"$enum": self.type_name(value), "value": self.encode(value.value)}
        if isinstance(value, dict):
            return {"$dict": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple, set)):
            return {"$" + type(value).__name__: [self.encode(x) for x in value]}
        if hasattr(value, "__dict__"):
            name = self.type_name(value)
            return {"$object": name, "state": self.encode(vars(value))}
        raise TypeError(f"Unsupported recorded state: {type(value)}")

    @staticmethod
    def type_name(value):
        cls = type(value)
        if not cls.__module__.startswith(("vllm.", "omni_npu.")) or "<locals>" in cls.__qualname__:
            raise TypeError(f"Unsupported state class: {cls}")
        return cls.__module__ + ":" + cls.__qualname__

    @staticmethod
    def resolve(name):
        module, qualname = name.split(":", 1)
        if not module.startswith(("vllm.", "omni_npu.")) or any(x.startswith("__") for x in qualname.split(".")):
            raise ValueError(f"Unsupported state type: {name}")
        value = importlib.import_module(module)
        for part in qualname.split("."):
            value = getattr(value, part)
        return value

    def decode(self, tree):
        import torch
        if not isinstance(tree, dict):
            return tree
        if "$tensor" in tree:
            key = (tree["$tensor"], tree["device"])
            if key not in self.memo:
                device = "cpu" if tree["device"].startswith("cpu") else self.device
                source = self.tensors[tree["$tensor"]]
                if self.reuse_buffers is not None and key in self.reuse_buffers:
                    destination = self.reuse_buffers[key]
                    copy_tree(destination, source)
                else:
                    destination = source.to(device).clone()
                    if self.reuse_buffers is not None:
                        self.reuse_buffers[key] = destination
                self.memo[key] = destination
            return self.memo[key]
        if "$dtype" in tree:
            value = getattr(torch, tree["$dtype"])
            if not isinstance(value, torch.dtype):
                raise ValueError("Invalid dtype")
            return value
        if "$device" in tree:
            return torch.device("cpu" if tree["$device"].startswith("cpu") else self.device)
        if "$generator" in tree:
            device = "cpu" if tree["device"].startswith("cpu") else self.device
            gen = torch.Generator(device=device)
            gen.set_state(self.decode(tree["$generator"]).cpu())
            return gen
        if "$numpy" in tree:
            return self.decode(tree["$numpy"]).cpu().numpy().copy()
        if "$enum" in tree:
            return self.resolve(tree["$enum"])(self.decode(tree["value"]))
        for name, cls in (("list", list), ("tuple", tuple), ("set", set)):
            if "$" + name in tree:
                return cls(self.decode(x) for x in tree["$" + name])
        if "$dict" in tree:
            return {self.decode(k): self.decode(v) for k, v in tree["$dict"]}
        if "$object" in tree:
            obj = object.__new__(self.resolve(tree["$object"]))
            for key, value in self.decode(tree["state"]).items():
                object.__setattr__(obj, key, value)
            return obj
        raise ValueError("Unknown state tree node")


def save_state(path, value, limit_bytes=None):
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    codec = Codec(limit_bytes=limit_bytes)
    tree = codec.encode(value)
    write_json(path / "tree.json", tree)
    torch.save(codec.tensors, path / "tensors.pt")
    return codec.bytes


def load_state(path, device, fields=None, reuse_buffers=None):
    import torch
    path = Path(path)
    tensors = torch.load(path / "tensors.pt", map_location="cpu", weights_only=True, mmap=True)
    tree = json.loads((path / "tree.json").read_text())
    if fields is not None:
        tree = {"$dict": [[k, v] for k, v in tree["$dict"] if k in fields]}
    return Codec(tensors, device, reuse_buffers=reuse_buffers).decode(tree)


def cache_views(proposer):
    """Only drafter caches; never snapshot target attention caches."""
    context = proposer.vllm_config.compilation_config.static_forward_context
    return {name: context[name].kv_cache for name in proposer.attn_layer_names}


def snapshot_cache_blocks(proposer, group_ids, batch, limit_bytes):
    """Capture referenced draft pages, never the backing shared with target.

    Keep original page IDs and full destination shapes; no slot remapping.
    Supported NPU cache leaves have the physical block on axis zero.
    """
    import torch
    runner = proposer.runner
    views = cache_views(proposer)
    result, used = {}, 0
    for gid in group_ids:
        group = runner.kv_cache_config.kv_cache_groups[gid]
        table = runner.input_batch.block_table[gid]
        if batch > len(table.num_blocks_per_row):
            raise ValueError("Capture batch exceeds block table rows")
        ids = set()
        for row in range(batch):
            count = int(table.num_blocks_per_row[row])
            if count < 0 or count > table.block_table.cpu.shape[1]:
                raise ValueError("Invalid draft block table row length")
            ids.update(int(x) for x in table.block_table.cpu[row, :count].tolist())
        # Zero is the reserved null block; negative values are not physical IDs.
        ids.discard(0)
        if any(x < 0 or x >= runner.kv_cache_config.num_blocks for x in ids):
            raise ValueError("Invalid draft block ID")
        indices = torch.tensor(sorted(ids), dtype=torch.int64, device="cpu")

        def snapshot(value):
            nonlocal used
            if isinstance(value, torch.Tensor):
                if not value.shape or value.shape[0] != runner.kv_cache_config.num_blocks:
                    raise ValueError(f"Unsupported draft cache block axis: {tuple(value.shape)}")
                block_bytes = value[0].numel() * value.element_size()
                used += len(ids) * block_bytes + indices.numel() * indices.element_size()
                if used > limit_bytes:
                    raise ValueError("Capture exceeds byte limit; increase PANGU_MTP_CAPTURE_MAX_GIB")
                values = torch.empty((len(ids), *value.shape[1:]), dtype=value.dtype, device="cpu")
                # Bounded gather temporary: the full model may fill NPU memory.
                chunk = max(1, (16 * 1024**2) // max(1, block_bytes))
                for start in range(0, len(ids), chunk):
                    selected = indices[start:start + chunk].to(value.device)
                    values[start:start + chunk].copy_(value.detach().index_select(0, selected).cpu())
                return {"$mtp_cache_blocks": 1, "shape": list(value.shape),
                        "indices": indices, "values": values}
            if isinstance(value, (list, tuple)):
                return type(value)(snapshot(x) for x in value)
            raise TypeError(f"Unsupported draft cache container: {type(value)}")

        for name in group.layer_names:
            if name in views:
                if name in result:
                    raise ValueError(f"Draft layer belongs to multiple cache groups: {name}")
                result[name] = snapshot(views[name])
    if result.keys() != views.keys():
        raise ValueError("Draft cache groups do not cover every draft layer")
    return result


def copy_tree(destination, source):
    import torch
    if isinstance(destination, torch.Tensor):
        if isinstance(source, dict) and source.get("$mtp_cache_blocks") == 1:
            values, indices = source["values"], source["indices"]
            if (list(destination.shape) != source["shape"] or destination.dtype != values.dtype
                    or tuple(values.shape) != (indices.numel(), *destination.shape[1:])
                    or len(indices.shape) != 1 or indices.dtype != torch.int64):
                raise ValueError("Sparse cache layout mismatch")
            destination.zero_()
            if indices.numel():
                # Bound transfer temporaries during repeated state restoration.
                block_bytes = max(1, values[0].numel() * values.element_size())
                chunk = max(1, (16 * 1024**2) // block_bytes)
                for start in range(0, indices.numel(), chunk):
                    destination.index_copy_(0, indices[start:start + chunk].to(destination.device),
                                            values[start:start + chunk].to(destination.device))
            return
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError(f"Cache layout mismatch: {destination.shape}/{destination.dtype} vs {source.shape}/{source.dtype}")
        destination.copy_(source)
    elif isinstance(destination, (list, tuple)):
        if len(destination) != len(source):
            raise ValueError("Cache sequence length mismatch")
        for dst, src in zip(destination, source):
            copy_tree(dst, src)
    elif isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise ValueError("Cache key mismatch")
        for key in destination:
            copy_tree(destination[key], source[key])
    elif destination != source:
        raise ValueError("Cache scalar state mismatch")


def ensure_draft_token_buffer(runner, *, dtype=None, size=None):
    """Standalone initialization/replay only; never invent state in a recorder."""
    buf = getattr(runner, "cu_num_draft_tokens", None)
    if buf is None:
        if dtype is None:
            import torch
            dtype = torch.int32
        buf = runner._make_buffer(runner.max_num_reqs if size is None else size, dtype=dtype)
        buf.cpu.zero_()
        buf.copy_to_gpu()
        runner.cu_num_draft_tokens = buf
    return buf


def runner_buffers(runner):
    """Required buffers plus the optional multi-MTP cumulative draft counts."""
    for name in ("num_accepted_tokens", "num_prompt_tokens"):
        yield name, getattr(runner, name)
    buf = getattr(runner, "cu_num_draft_tokens", None)
    if buf is not None:
        yield "cu_num_draft_tokens", buf


def runner_state(runner, group_ids):
    """Mutable runner buffers consumed by patched EagleProposer."""
    result = {"buffers": {}, "block_tables": []}
    if (getattr(runner.drafter, "fix_multi_mtp_kvcache", False)
            and getattr(runner, "cu_num_draft_tokens", None) is None):
        raise ValueError("fix_multi_mtp_kvcache requires cu_num_draft_tokens before capturing runner state")
    for name, buf in runner_buffers(runner):
        result["buffers"][name] = {"gpu": buf.gpu, "cpu": buf.cpu}
    for gid in group_ids:
        table = runner.input_batch.block_table[gid]
        result["block_tables"].append({"gpu": table.block_table.gpu,
                                        "cpu": table.block_table.cpu,
                                        "rows": table.num_blocks_per_row,
                                        "slot_gpu": table.slot_mapping.gpu,
                                        "slot_cpu": table.slot_mapping.cpu})
    if getattr(runner.drafter, "fix_multi_mtp_kvcache", False):
        result["input_cache"] = {name: getattr(runner.input_batch, name)
                                 for name in ("target_token_ids_cache", "target_model_hidden_states_cache")}
    return result


def restore_runner(runner, state):
    import numpy as np
    for name, values in state["buffers"].items():
        if name == "cu_num_draft_tokens":
            buf = ensure_draft_token_buffer(runner, dtype=values["gpu"].dtype, size=values["gpu"].shape[0])
        else:
            buf = getattr(runner, name)
        copy_tree(buf.gpu, values["gpu"])
        copy_tree(buf.cpu, values["cpu"])
    for gid, values in enumerate(state["block_tables"]):
        table = runner.input_batch.block_table[gid]
        for key, dst in (("gpu", table.block_table.gpu), ("cpu", table.block_table.cpu),
                         ("slot_gpu", table.slot_mapping.gpu), ("slot_cpu", table.slot_mapping.cpu)):
            copy_tree(dst, values[key])
        np.copyto(table.num_blocks_per_row, values["rows"])
    for name, value in state.get("input_cache", {}).items():
        copy_tree(getattr(runner.input_batch, name), value)
