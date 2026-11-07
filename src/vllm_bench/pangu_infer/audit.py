"""Inspect launch settings and installed source files without loading ML packages."""

import argparse
import hashlib
from importlib import util
import json
import os
from pathlib import Path
import sys


SOURCES = {
    "vllm.v1.worker.gpu_model_runner": ("vllm", "v1/worker/gpu_model_runner.py"),
    "vllm.v1.worker.gpu_input_batch": ("vllm", "v1/worker/gpu_input_batch.py"),
    "vllm.v1.spec_decode.utils": ("vllm", "v1/spec_decode/utils.py"),
    "vllm.v1.engine.llm_engine": ("vllm", "v1/engine/llm_engine.py"),
    "vllm.v1.core.single_type_kv_cache_manager": ("vllm", "v1/core/single_type_kv_cache_manager.py"),
    "omni_npu.worker.npu_model_runner": ("omni_npu", "worker/npu_model_runner.py"),
    "omni_npu.attention.backends.mome": ("omni_npu", "attention/backends/mome.py"),
    "omni_npu.layers.mome.mome_rl": ("omni_npu", "layers/mome/mome_rl.py"),
    "omni_npu.compilation.acl_graph": ("omni_npu", "compilation/acl_graph.py"),
    "vllm_bench.pangu_infer.mome_prefill": ("vllm_bench", "pangu_infer/mome_prefill.py"),
    "vllm_bench.pangu_infer.omni_worker": ("vllm_bench", "pangu_infer/omni_worker.py"),
    "omni_npu.v1.models.pangu.pangu_ultra_moe": ("omni_npu", "v1/models/pangu/pangu_ultra_moe.py"),
    "omni_npu.v1.models.pangu.pangu_ultra_moe_mtp": ("omni_npu", "v1/models/pangu/pangu_ultra_moe_mtp.py"),
    "omni_npu.model_config.config_loader.loader": ("omni_npu", "model_config/config_loader/loader.py"),
    "omni_npu.vllm_patches.patches.common.patch_eagle":
        ("omni_npu", "vllm_patches/patches/common/patch_eagle.py"),
    "omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_scheduler_spec_decode_pad":
        ("omni_npu", "vllm_patches/patches/models/pangu_v2_base/patch_scheduler_spec_decode_pad.py"),
}

ENVIRONMENT = (
    "VLLM_PLUGINS", "OMNI_VLLM_PATCHES_DIR", "OMNI_CUSTOM_MODEL_CONFIG_PATH",
    "OMNI_PANGU_V2_HIGH_THROUGHOUT", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_AOT_COMPILE",
    "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_BATCH_INVARIANT", "OMNI_DRAFT_TOP_K",
    "OMNI_HYBRID_ATTN_GROUP_SIZE", "OMNI_DISABLE_SPEC_DECODE_PAD", "USE_MOE_FORCE_LOAD_BALANCE",
    "HCCL_BUFFSIZE", "HCCL_MOE_DISPATCH_EP_BUFFSIZE", "ASCEND_LAUNCH_BLOCKING",
)


def package_roots():
    roots = {}
    for package in sorted({package for package, _ in SOURCES.values()}):
        try:
            spec = util.find_spec(package)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None and spec.submodule_search_locations:
            roots[package] = Path(next(iter(spec.submodule_search_locations)))
    return roots


def source_fingerprints(roots=None, modules=None):
    """Prefer loaded module paths; otherwise inspect the installed package tree."""
    roots = package_roots() if roots is None else roots
    modules = sys.modules if modules is None else modules
    result = {}
    for module, (package, relative) in SOURCES.items():
        loaded = modules.get(module)
        filename = getattr(loaded, "__file__", None)
        path = None
        if filename:
            path = Path(filename)
        elif package in roots:
            path = roots[package] / relative
        entry = {"loaded": loaded is not None, "path": str(path.resolve()) if path else None,
                 "sha256": None}
        if path is not None:
            try:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                entry["error"] = str(error)
        result[module] = entry
    return result


def runtime_manifest():
    from vllm_bench.pangu_infer.results import versions

    return {"python": sys.executable, "versions": versions(),
            "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
            "sources": source_fingerprints()}


def execution_settings(cfg):
    """Small, explicit subset of effective settings that affect this workload."""
    model, scheduler, cache, compilation = (
        cfg.model_config, cfg.scheduler_config, cfg.cache_config, cfg.compilation_config)
    return {"max_model_len": model.max_model_len, "dtype": str(model.dtype),
            "architectures": getattr(model.hf_config, "architectures", None),
            "is_hybrid": getattr(model, "is_hybrid", None),
            "num_nextn_predict_layers": getattr(model.hf_config, "num_nextn_predict_layers", None),
            "router_sliding_window": getattr(model.hf_config, "router_sliding_window", None),
            "max_num_seqs": scheduler.max_num_seqs,
            "max_num_batched_tokens": scheduler.max_num_batched_tokens,
            "enable_chunked_prefill": scheduler.enable_chunked_prefill,
            "cudagraph_capture_sizes": compilation.cudagraph_capture_sizes,
            "max_cudagraph_capture_size": compilation.max_cudagraph_capture_size,
            "mamba_page_size_padded": cache.mamba_page_size_padded,
            "num_gpu_blocks": getattr(cache, "num_gpu_blocks", None),
            "npugraph_ex_config": cfg.additional_config.get("npugraph_ex_config", {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(runtime_manifest(), ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
