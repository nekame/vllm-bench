"""Explicit Omni worker adapter, selected only with --fix-mome-prefill-state."""

from omni_npu.attention.backends.mome import NPUMomeAttentionMetadataBuilder
from omni_npu.worker.npu_worker import NPUWorker

from vllm_bench.pangu_infer.mome_prefill import MomePrefillMetadataBuilder


def install_mome_builders(runner):
    count = 0
    for groups in runner.attn_groups:
        for group in groups:
            for index, builder in enumerate(group.metadata_builders):
                if not isinstance(builder, NPUMomeAttentionMetadataBuilder):
                    continue
                if not isinstance(builder, MomePrefillMetadataBuilder):
                    replacement = MomePrefillMetadataBuilder(
                        builder.kv_cache_spec, group.layer_names, runner.vllm_config, runner.device)
                    replacement.runner_num_prompt_tokens = builder.runner_num_prompt_tokens
                    group.metadata_builders[index] = replacement
                count += 1
    if not count:
        raise RuntimeError("MoME prefill correction requested, but no MoME metadata builders were initialized")
    # The proposer keeps its own references to group builders. Rebuild them
    # through the native initializer before any warmup/capture can use them.
    if runner.drafter is not None:
        runner.drafter.initialize_attn_backend(runner.kv_cache_config)
    return count


class MomePrefillWorker(NPUWorker):
    def initialize_from_config(self, kv_cache_config):
        if self.use_v2_model_runner:
            raise ValueError("MoME prefill correction requires the V1 model runner")
        super().initialize_from_config(kv_cache_config)
        self.mome_prefill_builder_count = install_mome_builders(self.model_runner)
        self.mome_prefill_state_fix = True
