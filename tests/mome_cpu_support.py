"""Load original Omni CPU methods with import-only vLLM/NPU stand-ins.

This module never replaces torch or any method under test. Hardware-only
operations fail explicitly if a test accidentally reaches them.
"""

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


MOME_FILE = Path("omni/attention/backends/mome.py")
REFERENCE_FILE = Path("omni/layers/ops/causal_conv1d.py")


def unavailable(*args, **kwargs):
    raise AssertionError("A hardware-only dependency was called by the CPU test")


def import_standins(torch):
    def cdiv(value, divisor):
        return torch.div(value + divisor - 1, divisor, rounding_mode="floor")

    def split(common, decode_threshold):
        lengths = common.query_start_loc_cpu.diff().tolist()
        decodes = sum(length <= decode_threshold for length in lengths)
        if any(length > decode_threshold for length in lengths[:decodes]):
            raise AssertionError("CPU fixture requires decode rows before prefill rows")
        decode_tokens = sum(lengths[:decodes])
        return decodes, len(lengths) - decodes, decode_tokens, sum(lengths) - decode_tokens

    definitions = {
        "vllm": {}, "vllm.config": {"VllmConfig": object},
        "vllm.distributed": {"get_tp_group": unavailable},
        "vllm.logger": {"init_logger": lambda name: SimpleNamespace(info=unavailable)},
        "vllm.utils": {}, "vllm.utils.math_utils": {"cdiv": cdiv},
        "vllm.v1": {}, "vllm.v1.attention": {},
        "vllm.v1.attention.backend": {"AttentionBackend": object,
            "AttentionCGSupport": SimpleNamespace(ALWAYS="ALWAYS"), "CommonAttentionMetadata": object},
        "vllm.v1.attention.backends": {},
        "vllm.v1.attention.backends.gdn_attn": {"GDNAttentionMetadataBuilder": object},
        "vllm.v1.attention.backends.utils": {"PAD_SLOT_ID": -1},
        "vllm.v1.kv_cache_interface": {"AttentionSpec": object},
        "omni_npu": {"envs": SimpleNamespace()}, "omni_npu.attention": {},
        "omni_npu.attention.backends": {},
        "omni_npu.attention.backends.attention": {"NPUAttentionBackendImpl": object},
        "omni_npu.attention.backends.utils": {
            "_maybe_padded_raw_tensor_to_strided_caches": unavailable,
            "register_attention_backend": lambda name: lambda cls: cls,
            "scheme_conv_sp": unavailable, "split_decodes_and_prefills": split},
        "omni_npu.model_config": {}, "omni_npu.model_config.config_loader": {},
        "omni_npu.model_config.config_loader.loader": {"model_extra_config": SimpleNamespace(
            operator_opt_config=SimpleNamespace(enable_mome_sp=False))},
        "omni_custom_ops": {}, "torch_npu": {"npu_scatter_nd_update_": unavailable},
    }
    modules = {}
    for name, attributes in definitions.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    return modules


@contextmanager
def load_original_methods(source, torch, *, name):
    # Import real torch.nn.functional before changing the dependency namespace.
    import torch.nn.functional  # noqa: F401

    with patch.dict(sys.modules, import_standins(torch)), patch.object(sys, "dont_write_bytecode", True):
        loaded = []
        for suffix, relative in (("metadata", MOME_FILE), ("reference", REFERENCE_FILE)):
            module_name = f"_pangu_cpu_{name}_{suffix}"
            spec = importlib.util.spec_from_file_location(module_name, Path(source) / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        yield SimpleNamespace(builder=loaded[0].NPUMomeAttentionMetadataBuilder,
                              convolution=loaded[1].causal_conv1d_ref)


def patched_copy(source, destination, fix_patch):
    for relative in (MOME_FILE, REFERENCE_FILE):
        target = Path(destination) / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(source) / relative, target)
    command = ["git", "apply", "--include=" + MOME_FILE.as_posix(), str(Path(fix_patch).resolve())]
    result = subprocess.run(command, cwd=destination, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Cannot apply the candidate fix to a temporary source copy:\n" + result.stderr)
    return Path(destination)


class Fixture:
    """CPU batch-one inputs; the original builder constructor is not tested."""

    def __init__(self, torch, methods, *, context=8192, depth=7, apc=True, persistent=False, pd=False):
        self.torch, self.methods = torch, methods
        self.context, self.depth, self.apc = context, depth, apc
        self.block_size, self.channels = 128, 16
        self.max_len = context + 512
        self.columns = (self.max_len + self.block_size - 1) // self.block_size if apc else 1
        self.table = torch.arange(1, self.columns + 1, dtype=torch.int32, device="cpu").reshape(1, -1)
        builder = methods.builder.__new__(methods.builder)
        builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=apc))
        builder.compilation_config = SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: persistent))
        builder.num_spec, builder.fake_num_spec = depth, depth
        builder.kernel_width, builder.state_len = 3, depth + 2
        builder.mome_block_size, builder.reorder_batch_threshold = self.block_size, depth + 1
        builder.is_pd_disagg, builder.is_decode_node = pd, pd
        builder.reuse_prefilled_tokens = builder.enable_flashcomm2 = False
        builder.use_full_cuda_graph, builder.decode_cudagraph_max_bs = persistent, depth + 1
        builder.runner_num_prompt_tokens = None
        capacity = depth + 1
        shape = (capacity, self.columns) if apc else (capacity,)
        builder.cache_indices_tensor = torch.zeros(shape, dtype=torch.int32, device="cpu")
        for field in ("num_computed_tokens", "num_accepted_tokens", "block_idx_last_computed_token",
                      "block_idx_first_scheduled_token", "block_idx_last_scheduled_token"):
            setattr(builder, field, torch.zeros(capacity, dtype=torch.int32, device="cpu"))
        self.builder = builder

    def metadata(self, start, count, *, accepted=1, draft_index=None):
        torch = self.torch
        qsl = torch.tensor([0, count], dtype=torch.int32, device="cpu")
        common = SimpleNamespace(num_reqs=1, num_actual_tokens=count, max_query_len=count,
            seq_lens=torch.tensor([start + count], dtype=torch.int32, device="cpu"),
            query_start_loc=qsl, query_start_loc_cpu=qsl, block_table_tensor=self.table,
            compute_num_computed_tokens=lambda: torch.tensor([start], dtype=torch.int32, device="cpu"))
        kwargs = {"num_accepted_tokens": None if accepted is None else torch.tensor([accepted], dtype=torch.int32, device="cpu"),
                  "num_prompt_tokens": torch.tensor([self.context], dtype=torch.int32, device="cpu")}
        if draft_index is None:
            return self.builder.build(0, common, **kwargs)
        return self.builder.build_for_drafting(common, draft_index=draft_index, **kwargs)

    def tensors(self, dtype):
        torch = self.torch
        prompt = (torch.arange(self.context * self.channels, device="cpu").reshape(-1, self.channels) % 17 - 8).to(dtype)
        query = (torch.arange((self.depth + 1) * self.channels, device="cpu").reshape(-1, self.channels) % 13 - 6).to(dtype)
        weight = torch.tensor([[.25, .5, 1.]] * self.channels, dtype=dtype, device="cpu")
        return prompt, query, weight

    def cache_after_prefill(self, prompt, weight, *, poison):
        torch = self.torch
        cache = torch.full((self.columns + 1, self.depth + 2, self.channels), poison, dtype=prompt.dtype, device="cpu")
        prefill = self.metadata(0, self.context)
        logical = int(prefill.block_idx_last_scheduled_token[0]) if self.apc else 0
        physical = int(self.table[0, logical])
        # The source reference writes the valid K-1 history into the state tail.
        self.methods.convolution(prompt, weight, None, None, return_final_states=True,
                                 final_states_out=cache[physical].unsqueeze(0), activation=None)
        return cache

    def forward(self, query, weight, cache, metadata):
        # This adapter follows the documented CANN history-selection contract.
        # The fused NPU operator itself is not executed or emulated here.
        logical = int(metadata.block_idx_last_computed_token[0]) if self.apc else 0
        physical = int(metadata.cache_indices[0, logical] if self.apc else metadata.cache_indices[0])
        offset = int(metadata.num_accepted_tokens[0]) - 1
        history = cache[physical, offset:offset + 2].unsqueeze(0)
        if history.shape[1] != 2:
            raise AssertionError("Metadata selected history outside the CPU state buffer")
        output, _ = self.methods.convolution(query, weight, None, history, activation=None)
        return output
