"""Shared child-process environment and exclusive, opt-in instrumentation."""
import os
from pathlib import Path

from vllm_bench.common.paths import SRC_ROOT

RESET_ENV = ["OMNI_HYBRID_ALIGNED_DECODE", "OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD",
             "OMNI_NPU_BENCH_ALIGNED_DECODE_THRESHOLD", "OMNI_DP_ROUND_ROBIN",
             "USE_MOE_FORCE_LOAD_BALANCE", "OMNI_PROFILE_TOKEN_THRESHOLD",
             "PROFILER_TOKEN_THRESHOLD", "OMNI_ENABLE_PREFILL_PROFILER",
             "ENABLE_PREFILL_PROFILER"]

INSTRUMENTATION = {
    "layer_profile": "PANGU12_ANNOTATE_LAYERS",
    "mtp_profile": "PANGU_MTP_CAPTURE_DIR",
    "mtp_debug": "PANGU_SPEC_DEBUG_DIR",
}


def source_pythonpath(env):
    """Make package imports available to all workers, preserving existing entries."""
    entries = [str(SRC_ROOT), *env.get("PYTHONPATH", "").split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(entry for entry in entries if entry))


def instrumentation_pythonpath(env, scenario, command=()):
    """Choose one sitecustomize; reject other hooks before starting a process.

    scenario=None is used by the standalone runtime, which must not inherit a
    recorder or annotation hook. No ML packages are imported here.
    """
    selected = SRC_ROOT / "vllm_bench" / scenario / "instrumentation" if scenario else None
    for name, variable in INSTRUMENTATION.items():
        if name != scenario and env.get(variable):
            raise ValueError(f"Conflicting instrumentation: {variable}; use one sitecustomize per launch")
    if scenario != "layer_profile" and "--annotate-layers" in command:
        raise ValueError("Do not combine --annotate-layers with another sitecustomize instrumentation")
    wrappers = {"vllm_bench.mtp_profile.record_mtp": "mtp_profile",
                "vllm_bench.mtp_debug.debug_spec": "mtp_debug"}
    for module, name in wrappers.items():
        if module in command and name != scenario:
            raise ValueError(f"Do not combine sitecustomize wrappers: {module}")
    for entry in env.get("PYTHONPATH", "").split(os.pathsep):
        if entry:
            path = Path(entry).resolve()
            if (path / "sitecustomize.py").is_file() and path != selected:
                raise ValueError(f"Conflicting sitecustomize on PYTHONPATH: {path}")
    source = source_pythonpath(env)
    return os.pathsep.join([str(selected), source]) if selected else source
