"""Opt-in recorder attached to a real OmniInfer runner, outside ACL replay."""
from __future__ import annotations

from copy import deepcopy
import dataclasses
import functools
import inspect
import json
import os
from pathlib import Path

from vllm_bench.mtp_profile.mtp_common import TOPOLOGIES, compilation_input_options, digest, proposer_globals, topology_name, topology_world_size, write_json
from vllm_bench.mtp_profile.mtp_state import recordable_inputs, runner_state, save_state, snapshot_cache_blocks, versions


def compilation_options(config):
    """Public JSON-valued execution options; never serialize static modules."""
    result = {"mode": int(config.mode), "cudagraph_mode": config.cudagraph_mode.name}
    for key in ("backend", "custom_ops", "splitting_ops", "compile_sizes", "pass_config",
                "cudagraph_capture_sizes", "cudagraph_num_of_warmups", "max_cudagraph_capture_size",
                "inductor_compile_config"):
        value = getattr(config, key)
        result[key] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
    json.dumps(result)  # Reject non-JSON custom compiler callbacks explicitly.
    return compilation_input_options(result)


def draft_cache_config(proposer):
    config = deepcopy(proposer.runner.kv_cache_config)
    names = set(proposer.attn_layer_names)
    groups, group_ids = [], []
    for gid, group in enumerate(config.kv_cache_groups):
        retained = [n for n in group.layer_names if n in names]
        if retained:
            group.layer_names = retained
            groups.append(group)
            group_ids.append(gid)
    config.kv_cache_groups = groups
    tensors = []
    covered = set()
    for tensor in config.kv_cache_tensors:
        retained = [n for n in tensor.shared_by if n in names]
        if retained:
            if getattr(tensor, "block_stride", 0) or getattr(tensor, "offset", 0):
                raise ValueError("Packed cross-layer cache allocations are not supported by MTP capture")
            # Hybrid pools alias layers from different groups. Replay gives each
            # draft layer an independent allocation with identical block numbers.
            for name in retained:
                independent = deepcopy(tensor)
                independent.shared_by = [name]
                tensors.append(independent)
                covered.add(name)
    if covered != names:
        raise ValueError(f"Draft cache allocations missing: {sorted(names - covered)}")
    config.kv_cache_tensors = tensors
    return config, group_ids


def attach(runner):
    import torch
    destination = os.environ.get("PANGU_MTP_CAPTURE_DIR")
    if not destination or getattr(runner, "_pangu_mtp_recorder", False):
        return
    proposer = getattr(runner, "drafter", None)
    if proposer is None or getattr(proposer, "method", None) != "mtp":
        raise ValueError("MTP recorder requires an MTP EagleProposer")
    original = proposer.propose
    count, skip = 0, int(os.getenv("PANGU_MTP_CAPTURE_SKIP", "2"))
    limit = int(os.getenv("PANGU_MTP_CAPTURE_COUNT", "1"))
    max_bytes = int(float(os.getenv("PANGU_MTP_CAPTURE_MAX_GIB", "8")) * 1024**3)
    if min(skip, limit) < 0 or max_bytes <= 0:
        raise ValueError("Invalid capture limits")
    rank = torch.distributed.get_rank()
    parallel = proposer.vllm_config.parallel_config
    topology = topology_name(parallel.tensor_parallel_size, parallel.data_parallel_size, parallel.enable_expert_parallel)
    if torch.distributed.get_world_size() != topology_world_size(TOPOLOGIES[topology]):
        raise ValueError(f"Recorder world size differs from topology {topology}; extra parallel dimensions are unsupported")

    @functools.wraps(original)
    def record(*args, **kwargs):
        nonlocal count
        index = count
        count += 1
        if not skip <= index < skip + limit:
            return original(*args, **kwargs)
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        sampling = inputs["sampling_metadata"]
        # Arbitrary stateful processors cannot safely be replayed without target.
        if not sampling.no_penalties or sampling.allowed_token_ids_mask is not None or sampling.bad_words_token_ids:
            raise ValueError("Recorder v1 supports draft sampling without penalties/token constraints")
        config = json.loads((Path(proposer.vllm_config.model_config.model) / "config.json").read_text())
        cache_config, gids = draft_cache_config(proposer)
        torch.npu.synchronize()
        state = {"inputs": recordable_inputs(inputs), "cache_config": cache_config,
                 "caches": snapshot_cache_blocks(proposer, gids, inputs["next_token_ids"].shape[0], max_bytes),
                 "runner": runner_state(runner, gids),
                 "batch_execution": runner.batch_execution_and_padding_state,
                 "rng_cpu": torch.get_rng_state(), "rng_npu": torch.npu.get_rng_state()}
        torch.npu.synchronize()
        folder = Path(destination) / f"cycle-{index - skip:04d}" / f"rank-{rank:03d}"
        size = save_state(folder / "before", state, max_bytes)
        references = {"hidden_states": [], "logits": []}
        # This function runs outside per-layer graph wrappers, also for graph replay.
        globals_ = proposer_globals(original)
        call_forward = globals_["_forward_draft_model"]
        call_logits = proposer.model.compute_logits

        def forward(p, **kw):
            result = call_forward(p, **kw)
            if p is proposer:
                hidden = result[0] if isinstance(result, tuple) else result
                references["hidden_states"].append(hidden.detach().cpu().clone())
            return result

        def logits(*a, **kw):
            result = call_logits(*a, **kw)
            references["logits"].append(result.detach().cpu().clone())
            return result

        model = proposer.model
        had_logits = "compute_logits" in vars(model)
        old_logits = vars(model).get("compute_logits")
        globals_["_forward_draft_model"] = forward
        model.compute_logits = logits
        try:
            result = original(*args, **kwargs)
        finally:
            globals_["_forward_draft_model"] = call_forward
            if had_logits:
                model.compute_logits = old_logits
            else:
                delattr(model, "compute_logits")
        torch.npu.synchronize()
        references["tokens"] = result
        save_state(folder / "reference", references, max_bytes)
        operator_path = os.environ.get("OMNI_CUSTOM_MODEL_CONFIG_PATH")
        if not operator_path:
            raise ValueError("Recorder requires explicit OMNI_CUSTOM_MODEL_CONFIG_PATH")
        cfg = proposer.vllm_config
        meta = {"format": "pangu-mtp-capture-v1", "rank": rank, "world_size": torch.distributed.get_world_size(),
                "topology": topology, "config_sha256": digest(config), "versions": versions(),
                "depth": inputs["num_speculative_tokens"], "snapshot_bytes": size,
                "cache_snapshot": "draft-blocks-v1",
                "input_tokens": inputs["target_token_ids"].numel(),
                "batch_per_dp_rank": inputs["next_token_ids"].shape[0],
                "padded_tokens": state["batch_execution"][1].num_tokens if state["batch_execution"] else None,
                "operator_config": json.loads(Path(operator_path).read_text()),
                "runtime_limits": {"max_model_len": cfg.model_config.max_model_len,
                                   "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                                   "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens},
                "additional_config": cfg.additional_config,
                "cache_settings": {key: getattr(cfg.cache_config, key) for key in
                                   ("enable_prefix_caching", "cache_dtype", "block_size", "mamba_cache_mode", "mamba_block_size")},
                "speculative_tokens_configured": cfg.speculative_config.num_speculative_tokens,
                "speculative_options": {key: getattr(cfg.speculative_config, key) for key in
                                        ("draft_sample_method", "rejection_sample_method", "use_local_argmax_reduction")},
                "draft_environment": {key: os.environ.get(key) for key in
                                      ("OMNI_DRAFT_TOP_K", "OMNI_PANGU_V2_HIGH_THROUGHOUT")},
                "patches": os.environ.get("OMNI_VLLM_PATCHES_DIR"),
                "compilation_config_repr": str(cfg.compilation_config),
                "compilation_options": compilation_options(cfg.compilation_config),
                "complete": True}
        write_json(folder / "capture.json", meta)
        print(f"[pangu-mtp] captured {folder}", flush=True)
        return result

    proposer.propose = record
    runner._pangu_mtp_recorder = True
    print(f"[pangu-mtp] recorder attached on rank {rank}; capture adds synchronization and CPU copies", flush=True)


def patch_runner(cls):
    if getattr(cls.load_model, "_pangu_mtp_hook", False):
        return
    original = cls.load_model

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        attach(self)
        return result

    load._pangu_mtp_hook = True
    cls.load_model = load
