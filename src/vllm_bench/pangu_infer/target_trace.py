"""Eager target recording and teacher-forced replay, using ordinary module hooks.

No model methods are replaced. Tensor files contain CPU tensors and plain
containers and are loaded with weights_only=True. ML imports remain local.
"""

import json
from pathlib import Path
import re

from vllm_bench.pangu_infer.run import write_json


FORMAT = "pangu-target-trace-v1"
METADATA_FIELDS = (
    "num_reqs", "num_actual_tokens", "num_prefills", "num_prefill_tokens",
    "num_decodes", "num_decode_tokens", "max_query_len", "max_seq_len",
    "query_start_loc", "seq_lens", "num_computed_tokens", "num_accepted_tokens",
    "query_cumlens", "query_cumlens_list", "seq_lens_list", "num_tokens",
    "causal", "decode_threshold", "B_size", "use_mome_sp",
)


def parse_layers(value, count):
    if value == "all":
        return list(range(count))
    if not value:
        return []
    try:
        layers = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError("trace-layers must be all or comma-separated layer indices") from error
    if len(layers) != len(set(layers)) or any(i < 0 or i >= count for i in layers):
        raise ValueError("trace-layers contains duplicate or out-of-range indices")
    return layers


def trace_directory(path):
    path = Path(path)
    if (path / "target-trace" / "manifest.json").is_file():
        path = path / "target-trace"
    return path


def read_manifest(path):
    path = trace_directory(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != FORMAT:
        raise ValueError("Unexpected target trace format")
    return path, manifest


def validate_replay(plan):
    path, manifest = read_manifest(plan["options"]["replay_target"])
    if manifest.get("engine") != "omni":
        raise ValueError("Replay requires a reference trace recorded by Omni")
    source = manifest["plan"]
    if source.get("hf_overrides", {}) != plan.get("hf_overrides", {}):
        raise ValueError("Target replay differs in hf_overrides")
    for key in ("config", "operator_config", "additional_config", "checkpoint_identity",
                "prompt_sha256", "max_model_len"):
        if source[key] != plan[key]:
            raise ValueError(f"Target replay differs in {key}")
    for key in ("model", "tp", "depth", "mode", "temperature", "top_k", "top_p",
                "seed", "max_new_tokens", "respect_eos"):
        if source["options"][key] != plan["options"][key]:
            raise ValueError(f"Target replay differs in {key}")
    if not set(plan["trace_layers"]) <= set(source["trace_layers"]):
        raise ValueError("Requested replay layers were not recorded in the reference")
    counts = []
    for rank in range(plan["options"]["tp"]):
        header = json.loads((path / f"rank-{rank}" / "index.json").read_text())
        count = header["calls"]
        if type(count) is not int or count < 2:
            raise ValueError("Target trace must contain prefill and decode")
        for call in range(count):
            if not (path / f"rank-{rank}" / f"call-{call:05d}.pt").is_file():
                raise ValueError(f"Incomplete target trace on rank {rank}, call {call}")
        counts.append(count)
    if len(set(counts)) != 1:
        raise ValueError("Target trace call counts differ between TP ranks")
    return path, manifest, counts[0]


def cpu_tensor(value):
    return value.detach().cpu().clone()


def metadata_snapshot(metadata):
    """Logical fields only: physical block IDs/addresses differ between engines."""
    import torch

    result = {}
    for name, value in sorted(metadata.items()):
        if name.startswith("draft_model."):
            continue
        item = {}
        for section, obj in (("common", value), ("prefill", getattr(value, "prefill", None)),
                             ("decode", getattr(value, "decode", None))):
            if obj is None:
                item[section] = None
                continue
            fields = {}
            for field in METADATA_FIELDS:
                if not hasattr(obj, field):
                    continue
                field_value = getattr(obj, field)
                if isinstance(field_value, torch.Tensor):
                    field_value = field_value.detach().cpu().tolist()
                if field_value is None or isinstance(field_value, (bool, int, float, list)):
                    fields[field] = field_value
            item[section] = fields
        result[name] = item
    if not result:
        raise ValueError("Target trace found no target attention metadata")
    return result


def replay_parameters(record):
    """Validate the supported batch-one protocol, retaining source acceptance."""
    positions = record["positions"].tolist()
    ids = record["input_ids"].tolist()
    if not positions or positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("Target replay requires contiguous one-dimensional positions")
    if len(ids) != len(positions):
        raise ValueError("Target replay input IDs and positions have different sizes")
    start, size = positions[0], len(ids)
    accepted = None
    found = False
    for name, value in record["metadata"].items():
        common = value["common"]
        qsl = common.get("query_start_loc")
        if qsl is not None and qsl != [0, size]:
            raise ValueError(f"Unsupported target query boundaries in {name}: {qsl}")
        if "num_computed_tokens" in common:
            if common["num_computed_tokens"] != [start]:
                raise ValueError(f"Source computed-token count differs from positions in {name}")
            counts = common.get("num_accepted_tokens")
            if counts is not None:
                if len(counts) != 1 or type(counts[0]) is not int or counts[0] < 1:
                    raise ValueError("Invalid source MoME acceptance count")
                if found and counts[0] != accepted:
                    raise ValueError("Source MoME layers disagree on acceptance")
                accepted, found = counts[0], True
    return start, accepted


def proposal_parameters(call, start, previous_start, previous_size):
    """Committed positions come from the source cursor, independently of logits."""
    if call == 1:
        # Prefill's sampler produced one valid output before the first proposal.
        return previous_size - 1, 1
    committed = start - previous_start
    if not 1 <= committed <= previous_size:
        raise ValueError("Source target windows have an invalid committed-token advance")
    return committed - 1, committed


class TargetRecorder:
    def __init__(self, model, cfg, directory, rank, layers):
        self.directory = Path(directory) / f"rank-{rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cfg, self.layers = cfg, set(layers)
        self.calls, self.pending, self.handles = 0, None, []
        try:
            self.handles.append(model.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(model.register_forward_hook(self.after))
            self.handles.append(model.logits_processor.register_forward_hook(self.logits))
            for index in layers:
                layer = model.model.layers[index]
                self.handles.append(layer.register_forward_hook(self.layer_hook(index)))
        except BaseException:
            for handle in self.handles:
                handle.remove()
            raise

    def states(self, metadata):
        import torch

        result = {}
        for name, layer in self.cfg.compilation_config.static_forward_context.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if (name.startswith("draft_model.") or match is None
                    or int(match[1]) not in self.layers):
                continue
            value = metadata.get(name)
            if value is None or not hasattr(value, "cache_indices") or not hasattr(layer, "kv_cache"):
                continue
            # MoME states are small. Never snapshot the prompt's full MLA/DSA KV cache.
            backend = layer.get_attn_backend()
            if backend.get_name() != "NPUPanguMome":
                continue
            indices = value.cache_indices.reshape(-1).to(dtype=torch.int64)
            if indices.numel() != 1 or int(indices[0].item()) < 0:
                raise ValueError("Target trace requires one active MoME state per layer")
            result[name] = [cpu_tensor(cache.index_select(0, indices)) for cache in layer.kv_cache]
        return result

    def before(self, module, args, kwargs):
        import torch
        from vllm.forward_context import get_forward_context

        if self.pending is not None:
            raise RuntimeError("Target forward did not reach logits_processor before the next call")
        ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        if not isinstance(ids, torch.Tensor) or not isinstance(positions, torch.Tensor):
            raise ValueError("Target trace requires explicit input_ids and positions tensors")
        if ids.ndim != 1 or positions.ndim != 1:
            raise ValueError("Target trace supports only one-dimensional text inputs")
        metadata = get_forward_context().attn_metadata
        self.pending = {"call": self.calls, "input_ids": cpu_tensor(ids),
                        "positions": cpu_tensor(positions), "metadata": metadata_snapshot(metadata),
                        "layers": {}, "states_before": self.states(metadata)}

    def layer_hook(self, index):
        def capture(module, args, output):
            import torch

            if self.pending is None:
                raise RuntimeError("Layer hook ran outside a target forward")
            tensors = output if isinstance(output, tuple) else (output,)
            values = {}
            for position, value in enumerate(tensors):
                if isinstance(value, torch.Tensor):
                    # Prefill: retain only the final row; decode: retain every query row.
                    if self.calls == 0 and value.ndim:
                        value = value[-1:]
                    values[str(position)] = cpu_tensor(value)
            self.pending["layers"][str(index)] = values
        return capture

    def after(self, module, args, output):
        from vllm.forward_context import get_forward_context

        self.pending["states_after"] = self.states(get_forward_context().attn_metadata)

    def logits(self, module, args, output):
        import torch

        if self.pending is None or not isinstance(output, torch.Tensor):
            raise RuntimeError("Target logits hook did not receive a tensor for the pending call")
        if set(self.pending["layers"]) != {str(index) for index in self.layers}:
            raise RuntimeError("Requested layer hooks did not run; target trace requires uncompiled eager layers")
        self.pending["logits"] = cpu_tensor(output)
        torch.save(self.pending, self.directory / f"call-{self.calls:05d}.pt")
        self.pending = None
        self.calls += 1

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.pending is not None:
            raise RuntimeError("Incomplete target trace: forward has no logits")
        write_json(self.directory / "index.json", {"calls": self.calls})
        return self.calls


def install_target_trace(worker, directory, layers):
    import torch

    runner = worker.model_runner
    worker._pangu_target_recorder = TargetRecorder(
        runner.get_model(), runner.vllm_config, directory, torch.distributed.get_rank(), layers)


def finish_target_trace(worker):
    recorder = worker._pangu_target_recorder
    try:
        return recorder.close()
    finally:
        del worker._pangu_target_recorder


def write_manifest(directory, plan, effective, details):
    write_json(Path(directory) / "manifest.json", {
        "format": FORMAT, "engine": plan["options"]["engine"], "measurement": "diagnostic",
        "plan": plan, "effective": effective,
        "versions": details["versions"], "hardware": details["hardware"],
        "tensor_files": "CPU tensors; torch.load(weights_only=True)",
        "state_capture": "Selected layers: active MoME state before/after; physical block IDs excluded",
    })


def replay_target(engine, plan, output):
    import torch
    from omni_npu.model_config.config_loader.loader import model_extra_config
    from vllm_bench.pangu_infer.results import versions
    from vllm_bench.pangu_infer.runtime import config_json
    from vllm_bench.pangu_infer.trace_compare import compare_traces

    source, manifest, calls = validate_replay(plan)
    rank = torch.distributed.get_rank()
    cfg = engine.cfg
    effective = {"operator": config_json(model_extra_config.operator_opt_config),
                 "parallel": config_json(model_extra_config.parall_config),
                 "block_size": cfg.cache_config.block_size,
                 "mamba_block_size": cfg.cache_config.mamba_block_size}
    details = {"versions": versions(), "hardware": torch.npu.get_device_name(engine.device)}
    for key in ("versions", "hardware"):
        if details[key] != manifest[key]:
            raise ValueError(f"Target replay differs in {key}")
    for key, value in effective.items():
        if manifest["effective"].get(key) != value:
            raise ValueError(f"Target replay differs in effective {key}")
    directory = Path(output) / "target-trace"
    recorder = TargetRecorder(engine.target, cfg, directory, rank, plan["trace_layers"])
    engine.caches.reset()
    previous_ids, previous_hidden, previous_start = None, None, None
    proposals = []
    try:
        for call in range(calls):
            record = torch.load(source / f"rank-{rank}" / f"call-{call:05d}.pt",
                                map_location="cpu", weights_only=True)
            start, accepted = replay_parameters(record)
            if call == 0 and (start != 0 or record["input_ids"].tolist() != plan["prompt_ids"]):
                raise ValueError("First target trace call must be the full prompt prefill")
            if call and (start < plan["options"]["context_length"]
                         or record["input_ids"].numel() > plan["options"]["depth"] + 1):
                raise ValueError("Unsupported decode window in target trace")
            ids = record["input_ids"].to(engine.device, dtype=torch.int64)
            if call and plan["options"].get("replay_with_mtp", False):
                index_value, committed = proposal_parameters(
                    call, start, previous_start, previous_ids.numel())
                index = torch.tensor([index_value], dtype=torch.int64, device=engine.device)
                drafts, _ = engine.propose(
                    previous_ids, previous_hidden, previous_start, index, index_value,
                    ids[:1], committed)
                reference = record["input_ids"][1:].tolist()
                candidate = drafts.detach().cpu().tolist()
                if len(candidate) != plan["options"]["depth"]:
                    raise ValueError("MTP replay returned an unexpected proposal length")
                difference = next((row for row, (a, b) in enumerate(zip(reference, candidate)) if a != b), None)
                proposals.append({"target_call": call, "committed": committed,
                                  "sample_index": index_value, "reference": reference, "candidate": candidate,
                                  "compared_tokens": len(reference), "first_difference": difference})
            result = engine.forward("target", ids, start, accepted)
            if plan["options"].get("replay_with_mtp", False):
                previous_ids, previous_hidden, previous_start = ids, result[0], start
    finally:
        recorder.close()
    torch.npu.synchronize()
    if plan["options"].get("replay_with_mtp", False):
        write_json(directory / f"rank-{rank}" / "draft-proposals.json", proposals)
    from vllm.distributed import get_tp_group
    torch.distributed.barrier(group=get_tp_group().cpu_group)
    if rank == 0:
        write_manifest(directory, plan, effective, details)
        report = compare_traces(source, directory)
        write_json(Path(output) / "trace_comparison.json", report)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
        print(f"[target-trace] report={Path(output) / 'trace_comparison.json'}", flush=True)
