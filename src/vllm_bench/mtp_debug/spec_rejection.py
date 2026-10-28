"""Event-triggered target/MoME diagnostics, installed after model loading.

No torch import during Python startup. All snapshots are synchronous and read-only.
Use one in-flight request and eager execution to make cycle association explicit.
"""
import contextvars
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re

from vllm_bench.mtp_debug.spec_debug import values

TARGET = contextvars.ContextVar("pangu_spec_target", default=None)
_RECORDER = None


def snapshot(value, limit=64):
    if value is None:
        return None
    if hasattr(value, "detach"):
        tensor = value.detach()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "values": values(tensor.reshape(-1)[:limit]),
                "truncated": tensor.numel() > limit}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return {"values": list(value[:limit]), "truncated": len(value) > limit}
    return value


META_FIELDS = ("num_reqs", "num_actual_tokens", "num_decodes", "num_prefills",
               "max_query_len", "seq_lens", "query_start_loc", "slot_mapping",
               "block_table_tensor", "num_accepted_tokens", "num_computed_tokens",
               "cache_indices", "block_idx_last_computed_token",
               "block_idx_first_scheduled_token", "block_idx_last_scheduled_token")


def metadata_snapshot(metadata):
    if isinstance(metadata, list):
        return [metadata_snapshot(item) for item in metadata]
    return {key: snapshot(getattr(metadata, key))
            for key in META_FIELDS if hasattr(metadata, key)}


def fingerprint(tensor):
    """Hash exact bytes, with readable samples; no reduction on the NPU."""
    import torch
    # Only active blocks, selected by the caller, reach here.
    if tensor.numel() * tensor.element_size() > 8 * 1024 * 1024:
        return {"shape": list(tensor.shape), "omitted": "over 8 MiB"}
    cpu = tensor.detach().cpu().contiguous()
    raw = cpu.view(torch.uint8).numpy().tobytes()
    flat = cpu.float().reshape(-1)
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sample": flat[:16].tolist()}


class Recorder:
    def __init__(self, output):
        self.output = Path(output)
        self.latest_target = None
        self.previous = None
        self.pending = []
        self.forward_index = 0
        self.cycles = 0
        self.limit = int(os.environ.get("PANGU_SPEC_DEBUG_LIMIT", "1000"))

    def enabled(self):
        return self.cycles < self.limit and (self.output / "ARM").exists()

    def write_window(self, window):
        directory = self.output / "rejections" / f"rank-{window['rank']}-pid-{os.getpid()}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cycle-{window['rejection_cycle']:06d}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(window, ensure_ascii=False) + "\n")
        temporary.replace(path)

    @staticmethod
    def identity(event):
        return (event.get("label"), tuple(event.get("target_forward", {}).get("request_ids", [])))

    def attach(self, event):
        if self.latest_target is None:
            raise RuntimeError("spec-debug: no target forward captured; verify V1 runner hook installation")
        event["target_forward"] = self.latest_target
        self.latest_target = None

    def record(self, event):
        key = self.identity(event)
        for window in self.pending:
            if self.identity(window["rejection"]) == key:
                window["next"] = event
                window["status"] = "complete"
                actual = event["target_forward"]["runner_accepted_gpu"]
                window["next_runner_count_matches_output"] = (
                    actual.get("values") == window["expected_next_runner_count"]
                    if isinstance(actual, dict) and not actual.get("truncated") else None)
            else:
                window["status"] = "request_changed_before_next_verification"
            self.write_window(window)
        self.pending = []
        verification = event.get("verify", {})
        cumulative = verification.get("cu_num_draft_tokens", [])
        proposed = [end - (cumulative[i - 1] if i else 0) for i, end in enumerate(cumulative)]
        accepted = verification.get("accepted", [])
        rejected = [n - a for n, a in zip(proposed, accepted)]
        if any(n > 0 for n in rejected):
            window = {"format": "pangu-rejection-window-v1", "rank": event["rank"],
                      "rejection_cycle": event["cycle"], "rejected_counts": rejected,
                      "expected_next_runner_count": [sum(t != -1 for t in row) for row
                                                     in event["sampler_output_with_padding"]],
                      "status": "awaiting_next_verification",
                      "previous": self.previous if self.previous and self.identity(self.previous) == key else None,
                      "rejection": event, "next": None}
            self.write_window(window)
            self.pending.append(window)
        self.previous = event
        self.cycles += 1


def recorder():
    global _RECORDER
    if _RECORDER is None:
        _RECORDER = Recorder(os.environ["PANGU_SPEC_DEBUG_DIR"])
    return _RECORDER


def selected_rank():
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank == int(os.environ.get("PANGU_SPEC_DEBUG_RANK", "0"))


def install_runner(runner):
    if getattr(runner, "_pangu_rejection_installed", False) or not selected_rank():
        return
    if not getattr(runner.model_config, "enforce_eager", False):
        raise RuntimeError("spec-debug rejection windows require --enforce-eager")
    if runner.cache_config.enable_prefix_caching:
        raise RuntimeError("spec-debug rejection windows require --no-enable-prefix-caching")
    rec = recorder()
    original = runner._model_forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if not rec.enabled():
            return original(*args, **kwargs)
        from vllm.forward_context import get_forward_context
        bound = signature.bind(*args, **kwargs).arguments
        context = get_forward_context()
        batch = runner.input_batch
        count = batch.num_reqs
        if count != 1:
            raise RuntimeError("spec-debug rejection windows require exactly one in-flight request")
        frame = {"forward_index": rec.forward_index,
                 "request_ids": list(batch.req_ids[:count]),
                 "input_ids": snapshot(bound.get("input_ids")),
                 "positions": snapshot(bound.get("positions")),
                 "is_hybrid": runner.model_config.is_hybrid,
                 "use_async_scheduling": getattr(runner, "use_async_scheduling", None),
                 "runner_accepted_gpu": snapshot(runner.num_accepted_tokens.gpu[:count]),
                 "input_batch_accepted_cpu": snapshot(batch.num_accepted_tokens_cpu[:count]),
                 "num_computed_tokens_cpu": snapshot(batch.num_computed_tokens_cpu[:count]),
                 "attention": {name: metadata_snapshot(meta) for name, meta in
                               (context.attn_metadata or {}).items()},
                 "mome_calls": []}
        rec.forward_index += 1
        token = TARGET.set(frame)
        try:
            return original(*args, **kwargs)
        finally:
            TARGET.reset(token)
            rec.latest_target = frame

    runner._model_forward = forward
    layers = os.environ.get("PANGU_SPEC_DEBUG_MOME_LAYERS", "0")
    allowed = None if layers == "all" else {int(x) for x in layers.split(",")}
    hooked = []
    for name, module in runner.get_model().named_modules():
        if type(module).__name__ != "ColumnParallelMOMERL":
            continue
        prefix = getattr(module, "prefix", name)
        match = re.search(r"layers\.(\d+)(?:\.|$)", prefix)
        if allowed is not None and (not match or int(match[1]) not in allowed):
            continue
        install_conv(module, prefix)
        hooked.append(prefix)
    runner._pangu_rejection_installed = True
    manifest = {"format": "pangu-rejection-hooks-v1", "mome_hooks": hooked,
                "is_hybrid": runner.model_config.is_hybrid,
                "n_predict": getattr(getattr(runner, "drafter", None), "n_predict", None),
                "runner_forward": f"{original.__module__}.{original.__qualname__}",
                "note": "Only selected target MoME layers; fingerprints optional; no attention KV payloads.",
                "fingerprint_scope": "Full referenced MoME block, including speculative/unused slots; hash inequality alone is not a correctness failure."}
    rec.output.mkdir(parents=True, exist_ok=True)
    (rec.output / f"hooks-pid-{os.getpid()}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[spec-debug] target hook installed; {len(hooked)} MoME convolutions; pid={os.getpid()}", flush=True)
    if not hooked:
        print("[spec-debug] WARNING: no selected ColumnParallelMOMERL modules found; target metadata is captured, kernel-level MoME metadata is unavailable", flush=True)


def install_conv(module, prefix):
    original = module.forward
    signature = inspect.signature(original)

    @functools.wraps(original)
    def forward(*args, **kwargs):
        frame = TARGET.get()
        if frame is None:
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs).arguments
        meta = bound["mome_metadata"]
        event = {"prefix": prefix, "metadata": metadata_snapshot(meta)}
        cache = bound["conv_states"]
        blocks = []
        if os.environ.get("PANGU_SPEC_DEBUG_STATE_FINGERPRINTS") == "1":
            indices = values(meta.cache_indices.reshape(-1))
            blocks = sorted({int(i) for i in indices if 0 <= int(i) < cache.shape[0]})[:4]
            event["states_before"] = [{"block": i, **fingerprint(cache[i])} for i in blocks]
        frame["mome_calls"].append(event)
        result = original(*args, **kwargs)
        if blocks:
            event["states_after"] = [{"block": i, **fingerprint(cache[i])} for i in blocks]
        return result

    module.forward = forward


def patch_runner(module):
    cls = module.NPUModelRunner
    original = cls.load_model

    @functools.wraps(original)
    def load(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        install_runner(self)
        return result

    cls.load_model = load
