"""Allocate and bind caches without constructing a model runner."""

import math


def compatible_groups(groups):
    """Share metadata only when the backend and full cache spec agree."""
    buckets = []
    for group in groups:
        for bucket in buckets:
            if bucket[0].backend is group.backend and bucket[0].spec == group.spec:
                bucket.append(group)
                break
        else:
            buckets.append([group])
    return buckets


class CacheGroup:
    def __init__(self, name, layer, spec, cfg, device):
        import torch
        from omni_npu.attention.backends.mome import NPUPanguMomeBackend

        self.name, self.layer, self.spec = name, layer, spec
        self.backend = layer.get_attn_backend()
        self.mome = self.backend is NPUPanguMomeBackend
        self.blocks = 2 if self.mome else 1 + math.ceil(cfg.model_config.max_model_len / spec.block_size)
        # Block zero is reserved; all real requests use block one onwards.
        self.raw = torch.zeros(self.blocks * spec.page_size_bytes, dtype=torch.int8, device=device)
        self.cache = self.backend.reshape_kv_cache(self.raw, self.blocks, spec)
        layer.kv_cache = self.cache
        self.block_table = torch.arange(1, self.blocks, dtype=torch.int32, device=device).reshape(1, -1)


class Caches:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, device
        self.groups = []
        for name, layer in cfg.compilation_config.static_forward_context.items():
            getter = getattr(layer, "get_kv_cache_spec", None)
            if getter is None:
                continue
            if getattr(layer, "kv_sharing_target_layer_name", None):
                raise ValueError("CLA cache sharing needs a separate adapter")
            spec = getter(cfg)
            if spec is not None:
                self.groups.append(CacheGroup(name, layer, spec, cfg, device))
        if not self.groups:
            raise ValueError("Model did not register any attention/state caches")
        # Every layer has its own raw allocation, but one batch-one request
        # uses identical logical block IDs in compatible layers.
        for bucket in compatible_groups(self.groups):
            for group in bucket[1:]:
                group.block_table = bucket[0].block_table
        self.bytes = sum(group.raw.numel() for group in self.groups)

    def reset(self):
        for group in self.groups:
            group.raw.zero_()

    def lane(self, names, size):
        return MetadataLane(self, names, size)


class MetadataLane:
    """One persistent set of input tensors/builders per captured call shape."""

    def __init__(self, caches, names, size):
        import torch
        from vllm.v1.attention.backend import CommonAttentionMetadata

        self.cfg, self.size = caches.cfg, size
        self.groups = [group for group in caches.groups if group.name in names]
        if not self.groups:
            raise ValueError("Execution lane has no caches")
        self.positions = torch.empty(size, dtype=torch.int64, device=caches.device)
        self.offsets = torch.arange(size, dtype=torch.int64, device=caches.device)
        self.accepted = torch.empty(1, dtype=torch.int32, device=caches.device)
        self.prompt = torch.empty(1, dtype=torch.int32, device=caches.device)
        self.q_cpu = torch.tensor([0, size], dtype=torch.int32)
        self.q = self.q_cpu.to(caches.device)
        self.seq_cpu = torch.empty(1, dtype=torch.int32)
        self.computed_cpu = torch.empty(1, dtype=torch.int32)
        self.seq = torch.empty(1, dtype=torch.int32, device=caches.device)
        self.entries = []
        for bucket in compatible_groups(self.groups):
            group = bucket[0]
            builder = group.backend.get_builder_cls()(
                group.spec, [item.name for item in bucket], self.cfg, caches.device)
            slots = torch.empty(size, dtype=torch.int64, device=caches.device)
            common = CommonAttentionMetadata(
                query_start_loc=self.q, query_start_loc_cpu=self.q_cpu,
                seq_lens=self.seq, _seq_lens_cpu=self.seq_cpu,
                _num_computed_tokens_cpu=self.computed_cpu,
                seq_lens_cpu_upper_bound=self.seq_cpu,
                num_reqs=1, num_actual_tokens=size, max_query_len=size,
                max_seq_len=1, block_table_tensor=group.block_table,
                slot_mapping=slots, causal=True)
            self.entries.append((bucket, builder, common))

    def prepare(self, start, accepted, prompt, draft_index=None):
        import torch

        self.positions.copy_(self.offsets).add_(start)
        self.seq_cpu.fill_(start + self.size)
        self.computed_cpu.fill_(start)
        self.seq.fill_(start + self.size)
        self.prompt.fill_(prompt)
        if accepted is not None:
            self.accepted.fill_(accepted)
        metadata, slots = {}, {}
        for bucket, builder, common in self.entries:
            group = bucket[0]
            common.max_seq_len = start + self.size
            common._num_computed_tokens_cache = None
            if group.mome:
                common.slot_mapping.fill_(0)
            else:
                # Blocks are contiguous, so no gather or temporary block IDs.
                common.slot_mapping.copy_(self.positions).add_(group.spec.block_size)
            if draft_index is None:
                if group.mome:
                    value = builder.build(0, common,
                        num_accepted_tokens=self.accepted if accepted is not None else None,
                        num_prompt_tokens=self.prompt)
                else:
                    value = builder.build(0, common)
            else:
                extra = {}
                if group.mome:
                    extra = {"num_accepted_tokens": self.accepted if accepted is not None else None,
                             "num_prompt_tokens": self.prompt}
                value = builder.build_for_drafting(common_attn_metadata=common,
                                                    draft_index=draft_index, **extra)
            for item in bucket:
                metadata[item.name] = value
                slots[item.name] = common.slot_mapping
        return metadata, slots
