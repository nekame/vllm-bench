# SPDX-License-Identifier: Apache-2.0
# Portions Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
"""Explicit MoME construction, without OmniInfer's dynamic npumome module."""

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.abstract import MambaBase
from omni_npu.attention.backends.mome import NPUPanguMomeBackend
from omni_npu.layers.mome.mome_rl import ColumnParallelMOMERL
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.v1.layers.attention.npu_mla import NPUDeepseekMLAAttention as MLA
from omni_npu.v1.layers.attention.npu_dsa import NPUDeepseekSparseAttention as DSA
from omni_npu.v1.layers.attention.npu_cla import (
    NPUDeepseekMLACLAAttention as MLACLA,
    NPUDeepseekSparseCLAAttention as DSACLA,
)


class MomeAttention(MambaBase, nn.Module):
    """Three original MoME convolutions with explicitly bound state tensors.

    Adapted from NPUMoMEPatch.MomeAttention in patch_mome_hybrid.py. The
    arithmetic and cache specification are unchanged; plugin hooks, registry
    decoration and synthetic module injection are omitted.
    """

    def __init__(self, *, kernel_size, num_spec_tokens, state_dtypes,
                 state_shapes, quant_config, vllm_config, prefix):
        super().__init__()
        self.prefix = prefix
        self.kernel_size = kernel_size
        self.num_spec_tokens = num_spec_tokens
        self.state_shapes = state_shapes
        self.state_dtypes = state_dtypes
        self.kv_cache = ()
        self.qa_conv = ColumnParallelMOMERL(
            dim=state_shapes[0][0], kernel_width=kernel_size,
            quant_config=quant_config, prefix=f"{prefix}.qa_conv", disable_tp=True)
        self.compresskv_conv = ColumnParallelMOMERL(
            dim=state_shapes[1][0], kernel_width=kernel_size,
            quant_config=quant_config, prefix=f"{prefix}.compresskv_conv", disable_tp=True)
        self.o_conv = ColumnParallelMOMERL(
            dim=state_shapes[2][0], kernel_width=kernel_size,
            quant_config=quant_config, prefix=f"{prefix}.o_conv", disable_tp=True)
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate state layer {prefix}")
        context[prefix] = self

    def get_attn_backend(self):
        return NPUPanguMomeBackend

    @property
    def mamba_type(self):
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        return MambaAttentionBackendEnum.CUSTOM

    def get_state_shape(self):
        return self.state_shapes

    def get_state_dtype(self):
        return self.state_dtypes

    def get_kv_cache_spec(self, cfg):
        from vllm.v1.kv_cache_interface import MomeSpec
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        if cfg.cache_config.enable_prefix_caching:
            raise ValueError("Direct MoME runtime requires prefix caching disabled")
        return MomeSpec(
            shapes=self.state_shapes, dtypes=self.state_dtypes,
            block_size=cfg.cache_config.mamba_block_size,
            page_size_padded=cfg.cache_config.mamba_page_size_padded,
            mamba_type=MambaAttentionBackendEnum.CUSTOM,
            mamba_cache_mode="none", kernel_size=self.kernel_size,
            num_spec_tokens=self.num_spec_tokens,
            num_extra_reserved_blocks=model_extra_config.operator_opt_config.num_extra_reserved_blocks)

    def forward(self, hidden_states, state_indice, is_prefill=False, inplace=False, **kwargs):
        metadata = get_forward_context().attn_metadata[self.prefix]
        metadata = metadata.prefill if is_prefill else metadata.decode
        if metadata is None:
            raise ValueError(f"Missing MoME metadata for {self.prefix}")
        conv = (self.qa_conv, self.compresskv_conv, self.o_conv)[state_indice]
        return conv(hidden_states, self.kv_cache[state_indice], metadata, inplace=inplace)


class DirectMomeMixin:
    def _build_mome_conv(self, fake_num_spec_tokens, config, vllm_config, prefix):
        self.mome_state_shapes = (
            (self.q_lora_rank,), (self.kv_lora_rank,),
            (self.num_heads * self.v_head_dim,),
        )
        self.mome_state_dtypes = (self.dtype,) * 3
        self.kernel_size = config.router_sliding_window
        self.cache_dtype_str = None
        self.conv = MomeAttention(
            kernel_size=self.kernel_size, num_spec_tokens=fake_num_spec_tokens,
            state_dtypes=self.mome_state_dtypes, state_shapes=self.mome_state_shapes,
            quant_config=None, vllm_config=vllm_config, prefix=f"{prefix}.attn.mome")


class NPUDeepseekMLAAttention(DirectMomeMixin, MLA):
    pass


class NPUDeepseekSparseAttention(DirectMomeMixin, DSA):
    pass


class NPUDeepseekMLACLAAttention(DirectMomeMixin, MLACLA):
    pass


class NPUDeepseekSparseCLAAttention(DirectMomeMixin, DSACLA):
    pass
