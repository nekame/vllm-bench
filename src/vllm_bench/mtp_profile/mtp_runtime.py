#!/usr/bin/env python3
"""NPU-only standalone runtime; invoked by mtp_profile.profile_mtp, never by CPU checks.

Uses private vLLM 0.25.1 / OmniInfer APIs deliberately. All adapter failures are
fatal; there is no fake attention, random-weight fallback, or eager fallback.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
import dataclasses
from enum import Enum
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from vllm_bench.mtp_profile.mtp_common import SUMMARY_FIELDS, compilation_input_options, layer_sequence, proposer_globals, topology_world_size, validate_record, write_json
from vllm_bench.mtp_profile.mtp_state import (bind_sampling_runtime, cache_views, copy_tree, ensure_draft_token_buffer,
                       load_state, restore_runner, runner_buffers, runner_state, save_state, versions)


def diagnostic_config(value):
    """JSON values for effective-config reporting, without importing ML packages.

    This is intentionally separate from checkpoint/capture serialization and
    hashing. Unknown objects (including tensors) must not be silently stringified.
    """
    if isinstance(value, Enum):
        return diagnostic_config(value.value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: diagnostic_config(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: diagnostic_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    torch = sys.modules.get("torch")
    if torch is not None:
        scalar_types = tuple(cls for name in ("dtype", "device")
                             if isinstance(cls := getattr(torch, name, None), type))
        if scalar_types and isinstance(value, scalar_types):
            return str(value)
    raise TypeError(f"Unsupported effective-config value: {type(value).__module__}.{type(value).__qualname__}")


def install_standalone_loader(config):
    """Supply target-shared weights through the existing MTP weight loader.

    Scoped to this new process. No target class is instantiated. Share the head
    before loading to avoid allocating three standalone copies of lm_head.
    """
    from vllm.model_executor.models import ModelRegistry
    ultra = config["architectures"] == ["PanguUltraMoEForCausalLM"]
    if ultra:
        from omni_npu.v1.models.pangu.pangu_ultra_moe_mtp import OpenPanguMTP as Base
        architecture = "OpenPanguMTPModel"
    else:
        from omni_npu.v1.models.pangu.pangu_v2_moe_mtp import OpenPanguV2MTP as Base
        architecture = "OpenPanguV2MTPModel"
    start = config["num_hidden_layers"]

    class StandaloneMTP(Base):
        # vLLM initialize_model inspects these exact parameter names to select
        # the new-style constructor. **kwargs alone triggers its legacy path.
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if ultra:
                # Full Ultra inference replaces draft's plain embedding through
                # set_shared_weight(target). Reproduce that module type before
                # loading the target-shared payload, including its SP interface.
                from omni_npu.v1.layers.vocab_parallel_embedding import NPUVocabParallelEmbedding
                hf_config = vllm_config.model_config.hf_config
                embedding_prefix = f"{prefix}.model.embed_tokens" if prefix else "model.embed_tokens"
                self.model.embed_tokens = NPUVocabParallelEmbedding(
                    hf_config.vocab_size, hf_config.hidden_size,
                    params_dtype=self.model.embed_tokens.weight.dtype,
                    quant_config=vllm_config.quant_config, prefix=embedding_prefix)
            layers = list(self.model.layers.values())
            head = layers[0].shared_head.head
            for layer in layers[1:]:
                layer.shared_head.head = head

        def load_weights(self, weights):
            seen_shared = set()

            def remap():
                for name, tensor in weights:
                    if name.startswith("model.embed_tokens."):
                        seen_shared.add("embedding")
                        yield name.replace("model.embed_tokens.", f"model.layers.{start}.embed_tokens.", 1), tensor
                    elif name.startswith("lm_head."):
                        seen_shared.add("head")
                        yield name.replace("lm_head.", f"model.layers.{start}.shared_head.head.", 1), tensor
                    elif ".embed_tokens." in name or ".shared_head.head." in name:
                        # The full engine overwrites these with target-shared weights.
                        continue
                    else:
                        yield name, tensor

            loaded = super().load_weights(remap())
            if seen_shared != {"embedding", "head"}:
                raise ValueError("Target-shared embedding/head were not loaded")
            return loaded

    ModelRegistry.register_model(architecture, StandaloneMTP)


def require_full_decode_graph(cfg, stage):
    from vllm.config import CUDAGraphMode
    mode = cfg.compilation_config.cudagraph_mode
    if mode.decode_mode() != CUDAGraphMode.FULL:
        raise ValueError(f"{stage}: FULL decode graphs required; effective cudagraph_mode={mode}")


def create_runtime(plan, record=None):
    import torch
    import torch_npu  # noqa: F401
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    from vllm.config import CUDAGraphMode, CompilationMode, set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm.distributed.parallel_state import prepare_communication_buffer_for_model
    from vllm.v1.worker.workspace import init_workspace_manager
    from omni_npu.worker.npu_worker import (init_worker_distributed_environment,
                                           _patch_npu_triton_capabilities,
                                           _enable_batch_invariant_determinism)
    from omni_npu.model_config.config_loader.loader import load_model_extra_config, model_extra_config
    from omni_npu.v1.distributed.parallel_state_ext import ensure_layer_parallel_initialized
    from omni_npu.worker.npu_model_runner import NPUModelRunner

    a, topology = plan["options"], plan["topology"]
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world_size = topology_world_size(topology)
    if int(os.environ["WORLD_SIZE"]) != world_size:
        raise ValueError(f"Topology {a['topology']} requires {world_size} torchrun ranks")
    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    _patch_npu_triton_capabilities()
    # external_launcher must see the complete torchrun world, including DP ranks.
    torch.distributed.init_process_group(backend="hccl", init_method="env://")
    limits = record["runtime_limits"] if record else {
        "max_model_len": a["context_length"] + max(a["depths"]) + 32 + (a.get("synthetic_query_length") or 1) - 1,
        "max_num_seqs": a["batch_per_dp_rank"],
        "max_num_batched_tokens": max(a["prefill_chunk"] * a["batch_per_dp_rank"],
                                      a["batch_per_dp_rank"] * max(max(a["depths"]) + 1,
                                                                  a.get("synthetic_query_length") or 1))}
    cache_settings = record["cache_settings"] if record else {"enable_prefix_caching": False}
    extra_cache = {key: value for key, value in cache_settings.items() if key != "cache_dtype"}
    if "cache_dtype" in cache_settings:
        extra_cache["kv_cache_dtype"] = cache_settings["cache_dtype"]
    speculative = {"method": "mtp", "num_speculative_tokens": record["speculative_tokens_configured"] if record else max(a["depths"])}
    if record:
        speculative.update(record["speculative_options"])
    args = EngineArgs(model=a["model"], trust_remote_code=True, dtype="bfloat16",
                      skip_tokenizer_init=True, tensor_parallel_size=topology["tp"],
                      data_parallel_size=topology["dp"],
                      data_parallel_rank=rank if topology["dp"] > 1 else None,
                      enable_expert_parallel=topology["ep"],
                      distributed_executor_backend="external_launcher",
                      enforce_eager=a["execution"] == "eager",
                      enable_chunked_prefill=True,
                      speculative_config=speculative,
                      compilation_config=compilation_input_options(plan["compilation_config"]),
                      additional_config=plan["additional_config"], **limits, **extra_cache)
    cfg = args.create_engine_config()
    if cfg.use_v2_model_runner:
        raise ValueError("Adapter requires V1 runner: set VLLM_USE_V2_MODEL_RUNNER=0")
    cfg.parallel_config.data_parallel_rank = rank if topology["dp"] > 1 else 0
    if cfg.speculative_config is None or cfg.speculative_config.method != "mtp":
        raise ValueError("Expected OmniInfer MTP speculative config")
    expected_arch = ["OpenPanguMTPModel"] if plan["config"]["architectures"] == ["PanguUltraMoEForCausalLM"] else ["OpenPanguV2MTPModel"]
    if cfg.speculative_config.draft_model_config.hf_config.architectures != expected_arch:
        raise ValueError("Draft architecture is not the expected MTP class; refusing model construction")
    if a["execution"] != "eager" and cfg.compilation_config.mode == CompilationMode.NONE:
        raise ValueError("Compilation was disabled by effective platform config")
    expected_backend = plan.get("compilation_platform_backend")
    if expected_backend and a["execution"] != "eager" and cfg.compilation_config.backend != expected_backend:
        raise ValueError(f"Platform selected compile backend {cfg.compilation_config.backend!r}; expected {expected_backend!r}")
    if a["execution"] == "graph":
        require_full_decode_graph(cfg, "Effective runtime config")
    context = set_current_vllm_config(cfg)
    context.__enter__()
    # torch's world is initialized above, so this only creates vLLM groups and
    # performs Omni's normal distributed setup; it does not launch any workers.
    init_worker_distributed_environment(cfg, rank, "env://", local_rank, current_platform.dist_backend)
    load_model_extra_config(cfg.model_config, cfg, cfg.scheduler_config)
    _enable_batch_invariant_determinism(model_extra_config)
    from omni_npu.configs.validators import ValidationContext, validate_all
    validate_all(ValidationContext(model_extra_config=model_extra_config, vllm_config=cfg))
    ensure_layer_parallel_initialized(backend=current_platform.dist_backend)
    init_workspace_manager(device, 1)
    torch.manual_seed(a["seed"])
    torch.npu.manual_seed_all(a["seed"])
    install_standalone_loader(plan["config"])
    runner = NPUModelRunner(cfg, device)
    proposer = runner.drafter
    if proposer is None or not hasattr(proposer, "n_predict"):
        raise ValueError("OmniInfer patched EagleProposer was not installed")
    proposer_globals(proposer.propose)
    if proposer.n_predict != plan["config"]["num_nextn_predict_layers"]:
        raise ValueError("Configured predictor count differs from extracted checkpoint")
    proposer.model = proposer._get_model()
    runner.model = proposer.model
    runner._model_refresh_mla_weights(proposer.model)
    runner._model_post_weight_load(proposer.model)
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    layers = get_layers_from_vllm_config(cfg, AttentionLayerBase)
    proposer.attn_layer_names = sorted(n for n, layer in layers.items() if layer.get_kv_cache_spec(cfg) is not None)
    proposer._draft_attn_layer_names = set(proposer.attn_layer_names)
    if not proposer.attn_layer_names:
        raise ValueError("MTP did not register attention/cache layers")
    # No target module may have been created through the loading path.
    start = plan["config"]["num_hidden_layers"]
    if set(proposer.model.model.layers) != {str(start + i) for i in range(plan["config"]["num_nextn_predict_layers"])}:
        raise ValueError("Runtime constructed unexpected model layers")
    prepare_communication_buffer_for_model(proposer.model)
    # Mirrors vLLM's platform cache alignment, after real draft layers exist.
    if plan["config"].get("use_mome"):
        current_platform._align_hybrid_block_size(cfg, None)
    if record:
        kv_config = record["state"]["cache_config"]
        for group in kv_config.kv_cache_groups:
            for name in group.layer_names:
                layers[name].page_size_padded = group.kv_cache_spec.page_size_bytes
        cfg.cache_config.mamba_page_size_padded = max(g.kv_cache_spec.page_size_bytes for g in kv_config.kv_cache_groups)
    else:
        kv_config = synthetic_cache_config(runner, limits, a["batch_per_dp_rank"], max(a["depths"]))
    runner.initialize_kv_cache(kv_config)
    proposer.initialize_attn_backend(kv_config, runner._kernel_block_sizes)
    if a["execution"] == "graph":
        require_full_decode_graph(cfg, "Attention backend graph configuration")
    if not proposer.draft_attn_groups:
        raise ValueError("No draft attention groups initialized")
    return runner, proposer, cfg, context


def synthetic_cache_config(runner, limits, batch, depth):
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
    specs = runner.get_kv_cache_spec()
    if not specs:
        raise ValueError("No MTP KV cache specs")
    blocks = 1 + batch * max(max(math.ceil(limits["max_model_len"] / spec.block_size) for spec in specs.values()), depth + 1)
    # No shared raw allocations: easier exact restore and per-layer cache auditing.
    groups = [KVCacheGroupSpec(layer_names=[name], kv_cache_spec=spec, is_eagle_group=True)
              for name, spec in sorted(specs.items())]
    tensors = [KVCacheTensor(size=blocks * spec.page_size_bytes, shared_by=[name])
               for name, spec in sorted(specs.items())]
    return KVCacheConfig(num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def greedy_metadata(batch, device):
    import torch
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.logits_processor import LogitsProcessors
    zeros = torch.zeros(batch, dtype=torch.float32, device=device)
    return SamplingMetadata(temperature=zeros, all_greedy=True, all_random=False,
                            top_p=None, top_k=None, generators={}, max_num_logprobs=None,
                            no_penalties=True, prompt_token_ids=None,
                            frequency_penalties=zeros, presence_penalties=zeros,
                            repetition_penalties=torch.ones_like(zeros), output_token_ids=[[] for _ in range(batch)],
                            allowed_token_ids_mask=None, bad_words_token_ids={}, logitsprocs=LogitsProcessors())


def synthetic_query_length(options, depth, *, n_predict, fix_multi_mtp_kvcache):
    minimum = depth + 1 if fix_multi_mtp_kvcache and depth <= n_predict else 1
    length = options.get("synthetic_query_length")
    if length is None:
        return minimum
    if length < minimum:
        raise ValueError(f"Synthetic query length must be >= {minimum} for depth={depth} and this cache configuration")
    return length


def synthetic_inputs(runner, proposer, start, length, depth, seed):
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata
    a, device = runner._mtp_options, runner.device
    batch = a["batch_per_dp_rank"]
    q_cpu = torch.arange(0, (batch + 1) * length, length, dtype=torch.int32)
    q = q_cpu.to(device)
    positions = torch.arange(start, start + length, dtype=torch.int64, device=device).repeat(batch)
    runner.input_batch.block_table.compute_slot_mapping(batch, q, positions)
    base = runner.input_batch.block_table[proposer.kv_cache_gid]
    seq_cpu = torch.full((batch,), start + length, dtype=torch.int32)
    computed_cpu = torch.full((batch,), start, dtype=torch.int32)
    common = CommonAttentionMetadata(query_start_loc=q, query_start_loc_cpu=q_cpu,
                                     seq_lens=seq_cpu.to(device), _seq_lens_cpu=seq_cpu,
                                     _num_computed_tokens_cpu=computed_cpu, seq_lens_cpu_upper_bound=seq_cpu.clone(),
                                     num_reqs=batch, num_actual_tokens=batch * length,
                                     max_query_len=length, max_seq_len=start + length,
                                     block_table_tensor=base.get_device_tensor(batch),
                                     slot_mapping=base.slot_mapping.gpu[:batch * length], causal=True)
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    gen = torch.Generator(device="cpu").manual_seed(seed + dp_rank)
    vocab = proposer.model.config.vocab_size
    return {"num_speculative_tokens": depth,
            "target_token_ids": torch.randint(0, vocab, (batch * length,), generator=gen, dtype=torch.int32).to(device),
            "target_positions": positions,
            "target_hidden_states": torch.randn(batch * length, proposer.hidden_size, generator=gen, dtype=torch.bfloat16).to(device),
            "next_token_ids": torch.randint(0, vocab, (batch,), generator=gen, dtype=torch.int32).to(device),
            "token_indices_to_sample": None, "common_attn_metadata": common,
            "sampling_metadata": greedy_metadata(batch, device)}


def initialize_synthetic(runner, proposer, a):
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec
    runner._mtp_options = a
    batch, max_len = a["batch_per_dp_rank"], runner.max_model_len
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        block_size = group.kv_cache_spec.block_size
        count = (1 + group.kv_cache_spec.num_speculative_blocks if isinstance(group.kv_cache_spec, MambaSpec)
                 else math.ceil(max_len / block_size))
        for row in range(batch):
            runner.input_batch.block_table[gid].add_row(list(range(1 + row * count, 1 + (row + 1) * count)), row)
        runner.input_batch.block_table[gid].commit_block_table(batch)
    if proposer.fix_multi_mtp_kvcache:
        ensure_draft_token_buffer(runner)
    for name, buf in runner_buffers(runner):
        buf.cpu.zero_()
        buf.cpu[:batch].fill_(a["context_length"] if name == "num_prompt_tokens" else 1 if name == "num_accepted_tokens" else 0)
        buf.copy_to_gpu()
    if proposer.fix_multi_mtp_kvcache:
        runner.input_batch.target_token_ids_cache.zero_()
        runner.input_batch.target_model_hidden_states_cache.zero_()
        if min(a["prefill_chunk"], a["context_length"]) < proposer.n_predict + 1 or (
            a["context_length"] % a["prefill_chunk"] and a["context_length"] % a["prefill_chunk"] < proposer.n_predict + 1
        ):
            raise ValueError("fix_multi_mtp_kvcache requires every synthetic prefill chunk to have n_predict+1 tokens or more")
    for value in cache_views(proposer).values():
        def zero(tree):
            if isinstance(tree, torch.Tensor):
                tree.zero_()
            elif isinstance(tree, (list, tuple)):
                for child in tree:
                    zero(child)
        zero(value)
    # Populate every predictor's history. Calling the real proposer with depth
    # n_predict uses all heads on a packed prefill chunk, matching Omni's path.
    for start in range(0, a["context_length"], a["prefill_chunk"]):
        length = min(a["prefill_chunk"], a["context_length"] - start)
        inputs = synthetic_inputs(runner, proposer, start, length, proposer.n_predict, a["seed"] + start)
        execute(runner, proposer, inputs, graph=False)
    torch.npu.synchronize()


def execute(runner, proposer, inputs, graph, template=None):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    tokens = inputs["target_token_ids"].shape[0]
    dp = runner.vllm_config.parallel_config.data_parallel_size
    import torch
    across = torch.full((dp,), tokens, dtype=torch.int32) if dp > 1 else None
    mode = CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE
    # The patched proposer consumes this state instead of re-running target's
    # batch dispatcher. Synthetic/replay workloads have coordinated rank shapes.
    if template:
        _, descriptor, across = template
        runner.batch_execution_and_padding_state = (mode, descriptor, across)
    else:
        runner.batch_execution_and_padding_state = (mode, BatchDescriptor(tokens), across)
    return proposer.propose(**inputs)


def prepare_graph_state(cfg, token_count):
    from omni_npu.compilation.acl_graph import set_graph_params
    sizes = set(cfg.compilation_config.cudagraph_capture_sizes or []) | {token_count}
    set_graph_params(sizes)


def setup_graphs(proposer, cfg):
    import torch
    from vllm.config import CUDAGraphMode
    from omni_npu.compilation.acl_graph import ACLGraphWrapper
    stream = torch.npu.Stream()
    model = proposer.model

    class CheckedACLGraphWrapper(ACLGraphWrapper):
        """Never replay a graph with stale pointers, including kwargs inputs."""
        def __call__(self, *args, **kwargs):
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            if context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                addresses = [(str(i), x.data_ptr(), tuple(x.shape), tuple(x.stride()), str(x.dtype))
                             for i, x in list(enumerate(args)) + sorted(kwargs.items())
                             if isinstance(x, torch.Tensor) and x.device.type != "cpu"]
                book = self.__dict__.setdefault("_mtp_input_addresses", {})
                key = context.batch_descriptor
                if key in book and book[key] != addresses:
                    raise ValueError("ACL replay inputs changed device pointers/layout; capture is unsafe in this runtime path")
                book[key] = addresses
            return super().__call__(*args, **kwargs)

    if proposer.n_predict == 1:
        wrapper = CheckedACLGraphWrapper(model, cfg, runtime_mode=CUDAGraphMode.FULL,
                                         update_stream=stream, attn_layer_names=proposer.attn_layer_names)
        proposer.model = wrapper
        return [wrapper]
    model.model.wrapped_layers = {
        str(index): CheckedACLGraphWrapper(layer, cfg, runtime_mode=CUDAGraphMode.FULL,
                                   update_stream=stream,
                                   attn_layer_names=[n for n in proposer.attn_layer_names if model.get_spec_layer(n) == int(index)])
        for index, layer in model.model.layers.items()}
    return list(model.model.wrapped_layers.values())


def warmup_execution(runner, proposer, cfg, restore, *, graph, token_count, warmup, template=None):
    """Compile on FULL metadata before attaching any lazy-capture wrappers."""
    import torch
    wrappers = None
    if graph:
        prepare_graph_state(cfg, token_count)
        # NONE would bypass compilation in omni_npu.compilation.decorators.
        # FULL without ACLGraphWrapper activates compilation without capture.
        for _ in range(warmup):
            inputs = restore()
            execute(runner, proposer, inputs, True, template)
            torch.npu.synchronize()
        wrappers = setup_graphs(proposer, cfg)
    for _ in range(warmup + (1 if graph else 0)):
        inputs = restore()
        execute(runner, proposer, inputs, graph, template)
        torch.npu.synchronize()
    return wrappers


@contextmanager
def observe(proposer, *, references=False, synchronize=False):
    """Separate diagnostic path; baseline has neither per-stage sync nor copies."""
    import torch
    timings, outputs = {}, {"hidden_states": [], "logits": []}
    globals_ = proposer_globals(proposer.propose)
    originals = {"forward": globals_["_forward_draft_model"], "logits": proposer.model.compute_logits,
                 "sampling": proposer._sample_from_logits}
    existed = {"logits": "compute_logits" in vars(proposer.model), "sampling": "_sample_from_logits" in vars(proposer)}

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            if synchronize:
                torch.npu.synchronize()
            start = time.perf_counter()
            with torch.autograd.profiler.record_function("mtp/" + name):
                result = fn(*args, **kwargs)
            if synchronize:
                torch.npu.synchronize()
                timings.setdefault(name, []).append(time.perf_counter() - start)
            if references and name in ("forward", "logits"):
                tensor = result[0] if isinstance(result, tuple) else result
                outputs["hidden_states" if name == "forward" else "logits"].append(tensor.detach().cpu().clone())
            return result
        return wrapped

    globals_["_forward_draft_model"] = wrap("forward", originals["forward"])
    proposer.model.compute_logits = wrap("logits", originals["logits"])
    proposer._sample_from_logits = wrap("sampling", originals["sampling"])
    try:
        yield timings, outputs
    finally:
        globals_["_forward_draft_model"] = originals["forward"]
        for obj, attr, key in ((proposer.model, "compute_logits", "logits"), (proposer, "_sample_from_logits", "sampling")):
            if existed[key]:
                setattr(obj, attr, originals[key])
            else:
                delattr(obj, attr)


def compare(reference, actual, atol, rtol):
    import torch
    result = {"tokens_equal": torch.equal(reference["tokens"].cpu(), actual["tokens"].cpu()), "stages": {}}
    passed = result["tokens_equal"]
    for name in ("hidden_states", "logits"):
        if len(reference[name]) != len(actual[name]):
            raise ValueError(f"Replay {name} count differs")
        metrics = []
        for expected, observed in zip(reference[name], actual[name]):
            x, y = expected.cpu().float(), observed.cpu().float()
            if x.shape != y.shape:
                raise ValueError(f"Replay {name} shape differs")
            ok = torch.allclose(x, y, atol=atol, rtol=rtol)
            metrics.append({"max_abs_error": (x - y).abs().max().item(), "allclose": ok})
            passed = passed and ok
        result["stages"][name] = metrics
    result["passed"] = passed
    return result


def trace_propose(runner, proposer, inputs, graph, template=None):
    """Whole profiled cycle, including all inter-step work and NPU drain."""
    import torch
    with torch.autograd.profiler.record_function("mtp/propose"):
        start = time.perf_counter()
        result = execute(runner, proposer, inputs, graph, template)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
    return result, elapsed


def run(plan):
    a = plan["options"]
    if a["execution"] == "graph":
        os.environ["VLLM_USE_AOT_COMPILE"] = "0"
    import torch
    import torch_npu  # noqa: F401
    from vllm.compilation.counter import compilation_counter
    if not versions()["vllm"].startswith("0.25.1"):
        raise RuntimeError("Standalone private-API adapter targets vLLM 0.25.1; use its OmniInfer environment")
    rank = int(os.environ["RANK"])
    output = Path(a["output"]) / f"rank-{rank:03d}"
    output.mkdir(exist_ok=False)
    record = None
    if a["input"] == "replay":
        folder = Path(a["capture"]) / f"rank-{rank:03d}"
        meta = json.loads((folder / "capture.json").read_text())
        validate_record(meta, topology=a["topology"], config=plan["config"], versions=versions())
        if meta["operator_config"] != plan["operator_config"]:
            raise ValueError("Replay operator config differs; use capture configuration")
        record = {**meta, "state": load_state(folder / "before", "cpu")}
    with torch.inference_mode():
        runner, proposer, cfg, context = create_runtime(plan, record)
    try:
        with torch.inference_mode():
            if record is None:
                initialize_synthetic(runner, proposer, a)
            else:
                copy_tree(cache_views(proposer), record["state"]["caches"])
                restore_runner(runner, record["state"]["runner"])
            gids = list(range(len(runner.kv_cache_config.kv_cache_groups)))
            if record:
                baseline = {key: record["state"][key] for key in ("caches", "runner")}
            else:
                save_state(output / "baseline", {"caches": cache_views(proposer), "runner": runner_state(runner, gids)})
                # Keep baseline on CPU; only copy into existing device allocations.
                baseline = load_state(output / "baseline", "cpu")
            graph = a["execution"] == "graph"
            results = []
            effective = {"versions": versions(), "compilation": str(cfg.compilation_config),
                         "parallel": str(cfg.parallel_config), "model_class": type(proposer.model).__name__,
                         "attention_layers": proposer.attn_layer_names,
                         "target_weights_loaded": False, "npu_validation": "executing"}
            effective["runtime_environment"] = {"VLLM_USE_AOT_COMPILE": os.environ.get("VLLM_USE_AOT_COMPILE")}
            from omni_npu.model_config.config_loader.loader import model_extra_config
            effective["operator_config"] = diagnostic_config(model_extra_config)
            write_json(output / "effective_config.json", effective)
            for depth in a["depths"]:
                if record:
                    inputs = load_state(Path(a["capture"]) / f"rank-{rank:03d}" / "before", runner.device, fields={"inputs"})["inputs"]
                    if depth != record["depth"]:
                        raise ValueError("Replay depth must equal the captured depth")
                    reference = load_state(Path(a["capture"]) / f"rank-{rank:03d}" / "reference", "cpu")
                else:
                    query_length = synthetic_query_length(a, depth, n_predict=proposer.n_predict,
                                                          fix_multi_mtp_kvcache=proposer.fix_multi_mtp_kvcache)
                    inputs = synthetic_inputs(runner, proposer, a["context_length"], query_length, depth, a["seed"])
                    reference = None
                    effective["synthetic_input"] = {
                        "query_length": query_length, "batch_per_dp_rank": a["batch_per_dp_rank"],
                        "input_tokens": inputs["target_token_ids"].numel(),
                        "descriptor_tokens": inputs["target_token_ids"].numel(), "depth": depth}
                    write_json(output / "effective_config.json", effective)
                # Keep CPU encoded inputs: propose mutates metadata and input views.
                save_state(output / f"inputs-{depth}", inputs)
                input_buffers = {}

                def restore():
                    copy_tree(cache_views(proposer), baseline["caches"])
                    restore_runner(runner, baseline["runner"])
                    if record:
                        torch.set_rng_state(record["state"]["rng_cpu"].cpu())
                        torch.npu.set_rng_state(record["state"]["rng_npu"].cpu())
                    restored = load_state(output / f"inputs-{depth}", runner.device, reuse_buffers=input_buffers)
                    if record:
                        bind_sampling_runtime(restored, runner)
                    torch.npu.synchronize()
                    return restored

                template = None
                if record:
                    template = load_state(Path(a["capture"]) / f"rank-{rank:03d}" / "before", runner.device, fields={"batch_execution"})["batch_execution"]
                graph_wrappers = warmup_execution(
                    runner, proposer, cfg, restore, graph=graph,
                    token_count=template[1].num_tokens if template else inputs["target_token_ids"].numel(),
                    warmup=a["warmup"], template=template)
                if graph and not any(w.concrete_aclgraph_entries for w in graph_wrappers):
                    raise ValueError("Requested graph mode produced no captured ACL graphs")
                graph_counts = [len(w.concrete_aclgraph_entries) for w in graph_wrappers] if graph else None
                backend_compilations = compilation_counter.num_backend_compilations
                if reference:
                    call_inputs = restore()
                    with observe(proposer, references=True) as (_, actual):
                        actual["tokens"] = execute(runner, proposer, call_inputs, graph, template)
                    torch.npu.synchronize()
                    comparison = compare(reference, actual, a["atol"], a["rtol"])
                    write_json(output / f"comparison-{depth}.json", comparison)
                    if not comparison["passed"]:
                        raise ValueError("Replay validation failed; inspect comparison JSON before interpreting timings")
                latencies = []
                for repeat in range(a["repeats"]):
                    call_inputs = restore()
                    torch.distributed.barrier()
                    torch.npu.synchronize()
                    start = time.perf_counter()
                    tokens = execute(runner, proposer, call_inputs, graph, template)
                    torch.npu.synchronize()
                    latencies.append(time.perf_counter() - start)
                # Trace separately: no cache restoration or profiler overhead in baseline.
                call_inputs = restore()
                import torch_npu
                profiler = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output / "traces"), worker_name=f"rank{rank}-depth{depth}"),
                    record_shapes=False, with_stack=False,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1))
                with profiler:
                    with observe(proposer):
                        torch.distributed.barrier()
                        torch.npu.synchronize()
                        _, trace_cycle_s = trace_propose(runner, proposer, call_inputs, graph, template)
                diagnostic = None
                if a["diagnostic_stages"]:
                    call_inputs = restore()
                    with observe(proposer, synchronize=True) as (diagnostic, _):
                        execute(runner, proposer, call_inputs, graph, template)
                if graph and graph_counts != [len(w.concrete_aclgraph_entries) for w in graph_wrappers]:
                    raise ValueError("New ACL graph capture occurred after warmup; timings are invalid")
                if graph and compilation_counter.num_backend_compilations != backend_compilations:
                    raise ValueError("Backend compilation occurred after graph warmup; timings are invalid")
                results.append({"depth": depth, "layer_sequence": layer_sequence(plan["config"], depth),
                                "latencies_s": latencies, "median_s": statistics.median(latencies),
                                "s_per_proposed_token": statistics.median(latencies) / depth,
                                "trace_cycle_s": trace_cycle_s,
                                "trace_s_per_proposed_token": trace_cycle_s / depth,
                                "trace_to_baseline_ratio": trace_cycle_s / statistics.median(latencies),
                                "tokens_shape": list(tokens.shape), "diagnostic_stage_s": diagnostic,
                                "memory_allocated_bytes": torch.npu.memory_allocated(),
                                "memory_reserved_bytes": torch.npu.memory_reserved(),
                                "peak_memory_allocated_bytes": torch.npu.max_memory_allocated(),
                                "graph_entry_counts": graph_counts,
                                "backend_compilations_after_warmup": backend_compilations,
                                "synthetic": record is None})
            write_json(output / "results.json", results)
            torch.distributed.barrier()
            if rank == 0:
                aggregate(plan)
    finally:
        context.__exit__(None, None, None)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def aggregate(plan):
    a = plan["options"]
    output = Path(a["output"])
    ranks = [json.loads((output / f"rank-{i:03d}" / "results.json").read_text())
             for i in range(topology_world_size(plan["topology"]))]
    summary = []
    for index, depth in enumerate(a["depths"]):
        critical = [max(rank[index]["latencies_s"][j] for rank in ranks) for j in range(a["repeats"])]
        seconds = statistics.median(critical)
        batch = plan["total_sequences"]
        row = {"depth": depth, "critical_path_median_s": seconds,
                        "total_sequences": batch, "proposed_tokens_per_s": batch * depth / seconds,
                        "per_rank_median_s": [rank[index]["median_s"] for rank in ranks]}
        have_trace = ["trace_cycle_s" in rank[index] for rank in ranks]
        if any(have_trace) and not all(have_trace):
            raise ValueError("Trace cycle timing is missing on some ranks")
        if all(have_trace):
            trace_times = [rank[index]["trace_cycle_s"] for rank in ranks]
            trace_seconds = max(trace_times)
            row.update(trace_cycle_s=trace_seconds, trace_s_per_proposed_token=trace_seconds / depth,
                       trace_proposed_tokens_per_s=batch * depth / trace_seconds,
                       trace_to_baseline_ratio=trace_seconds / seconds,
                       per_rank_trace_cycle_s=trace_times)
        summary.append(row)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in writer.fieldnames} for row in summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    try:
        run(plan)
    except BaseException as error:
        path = Path(plan["options"]["output"]) / f"failure-rank-{os.getenv('RANK', 'unknown')}.json"
        write_json(path, {"error": type(error).__name__, "message": str(error), "npu_validated": False})
        raise


if __name__ == "__main__":
    main()
