# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Local MoME metadata correction for the first target step after prefill.

Draft-step synthesis follows omni/attention/backends/mome.py. Operators,
state layout and the rest of metadata construction remain in the base class.
"""

from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder


class MomePrefillMetadataBuilder(NPUMomeAttentionMetadataBuilder):
    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_prompt_tokens=None,
        allow_runner_num_prompt_tokens=True,
        draft_step0=False,
        for_drafting=False,
    ):
        if (num_prompt_tokens is None and allow_runner_num_prompt_tokens
                and self.runner_num_prompt_tokens is not None):
            num_prompt_tokens = self.runner_num_prompt_tokens[:common_attn_metadata.num_reqs]
        if (not for_drafting and not draft_step0 and not self.is_pd_disagg
                and num_accepted_tokens is not None and num_prompt_tokens is not None):
            # A target at the prompt boundary has a prefill tail, not a
            # previously verified speculative window. Preserve caller counts.
            computed = common_attn_metadata.compute_num_computed_tokens()
            num_accepted_tokens = num_accepted_tokens.masked_fill(
                computed == num_prompt_tokens, self.fake_num_spec + 1)
        return super().build(
            common_prefix_len, common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            fast_build=fast_build, num_prompt_tokens=num_prompt_tokens,
            allow_runner_num_prompt_tokens=allow_runner_num_prompt_tokens,
            draft_step0=draft_step0,
        )

    def build_for_drafting(
        self, common_attn_metadata, draft_index,
        num_accepted_tokens=None, num_prompt_tokens=None,
    ):
        draft_step0_synthesized = False
        if draft_index == 0 and num_accepted_tokens is None and num_prompt_tokens is not None:
            import torch

            draft_step0_synthesized = True
            num_accepted_tokens = torch.full_like(
                common_attn_metadata.compute_num_computed_tokens(), self.kernel_width - 1)
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata,
            num_accepted_tokens=num_accepted_tokens, fast_build=True,
            num_prompt_tokens=num_prompt_tokens,
            draft_step0=draft_step0_synthesized, for_drafting=True,
        )
