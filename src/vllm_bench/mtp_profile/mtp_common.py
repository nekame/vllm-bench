"""Standard-library-only planning and validation for standalone MTP."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

TOPOLOGIES = {"tp1": {"tp": 1, "dp": 1, "ep": False},
              "tp8": {"tp": 8, "dp": 1, "ep": False},
              "dp8-ep8": {"tp": 1, "dp": 8, "ep": True}}

SUMMARY_FIELDS = ("depth", "critical_path_median_s", "total_sequences", "proposed_tokens_per_s",
                  "trace_cycle_s", "trace_s_per_proposed_token", "trace_proposed_tokens_per_s",
                  "trace_to_baseline_ratio")

OMNI_COMPILE_BACKEND = "omni_npu.compilation.npugraph_ex.NpuGraphExAdaptor"


def topology_world_size(topology):
    return topology["tp"] * topology["dp"]


def topology_name(tp, dp, ep):
    for name, settings in TOPOLOGIES.items():
        if settings == {"tp": tp, "dp": dp, "ep": ep}:
            return name
    raise ValueError(f"Unsupported MTP topology TP={tp}, DP={dp}, EP={ep}; expected one of {list(TOPOLOGIES)}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compilation_input_options(options):
    """Convert resolved runtime compilation options back to constructor inputs.

    vLLM has bool-annotated pass flags whose defaults are None, but rejects
    explicit JSON null for some of those fields. Preserve explicit values.
    """
    result = dict(options)
    passes = result.get("pass_config")
    if isinstance(passes, dict):
        result["pass_config"] = {key: value for key, value in passes.items() if value is not None}
    # CompilationConfig validates mode=3 against constructor backend names before
    # resolving an empty backend through current_platform.get_compile_backend().
    # The resolved OOT class path cannot itself be passed back to that validator.
    if (result.get("mode", 3) in (3, "VLLM_COMPILE")
            and result.get("backend") == OMNI_COMPILE_BACKEND):
        result["backend"] = ""
    return result


def layer_sequence(config, depth):
    start, count = config["num_hidden_layers"], config["num_nextn_predict_layers"]
    if count < 1 or depth < 1:
        raise ValueError("Positive MTP count and depth required")
    # Omni patch_eagle.propose uses multi-MTP only when depth <= n_predict.
    # Longer proposals omit spec_step_idx and the predictor uses head zero.
    return [start + i for i in range(depth)] if count > 1 and depth <= count else [start] * depth


def checkpoint_config(path):
    config = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "mtp_checkpoint_manifest.json").read_text())
    if manifest.get("format") != "pangu-mtp-checkpoint-v1" or manifest.get("target_layers_present") is not False:
        raise ValueError("Use python -m vllm_bench.mtp_profile.prepare_mtp_checkpoint output, not a full or prefix checkpoint")
    if manifest["config_sha256"] != digest(config):
        raise ValueError("MTP config was modified after extraction")
    return config, manifest


def validate_record(meta, *, topology, config, versions=None):
    if meta.get("format") != "pangu-mtp-capture-v1":
        raise ValueError("Unsupported capture format")
    if meta["topology"] != topology:
        raise ValueError("Capture topology differs; TP/DP conversion is not supported")
    if meta["config_sha256"] != digest(config):
        raise ValueError("Capture model config differs from checkpoint")
    if versions and meta["versions"] != versions:
        raise ValueError("Capture library versions differ; use the recording environment")


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def proposer_globals(method):
    function = inspect.unwrap(getattr(method, "__func__", method))
    globals_ = function.__globals__
    if "_forward_draft_model" not in globals_:
        raise RuntimeError("Expected OmniInfer patch_eagle proposer with _forward_draft_model; private API differs")
    return globals_
