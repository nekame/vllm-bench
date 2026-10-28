"""Small, synchronous greedy-verification traces for OmniInfer V1.

Only enabled after OUTPUT/ARM exists. No model/cache tensors are modified.
CPU copies change timing and may mask a synchronization race.
"""
import contextvars
import functools
import inspect
import json
import os
from pathlib import Path
import time

ACTIVE = contextvars.ContextVar("pangu_spec_debug", default=None)


def values(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return [values(v) for v in value]
    return value


def top2(logits):
    # Limit diagnostic output; supplied logits remain untouched.
    rows = logits[:64].detach().float()
    scores, ids = rows.topk(2, dim=-1)
    scores, ids = values(scores), values(ids)
    return {"shape": list(logits.shape), "rows": [
        {"ids": i, "logits": s, "margin": s[0] - s[1]}
        for i, s in zip(ids, scores)
    ], "truncated": logits.shape[0] > 64}


def expected_greedy(draft, target, bonus, cumulative):
    """Reference prefix acceptance; independent of device implementation."""
    result, counts, start = [], [], 0
    for index, end in enumerate(cumulative):
        count = 0
        while start + count < end and draft[start + count] == target[start + count]:
            count += 1
        tail = target[start + count] if start + count < end else bonus[index]
        result.append(draft[start:start + count] + [tail])
        counts.append(count)
        start = end
    return result, counts


def patch_sampler(module):
    if getattr(module, "_pangu_spec_debug", False):
        return
    for name in ("compute_probs_and_sample", "rejection_greedy_sample_native"):
        if not hasattr(module, name):
            raise RuntimeError(f"spec-debug: unsupported OmniInfer; missing {name}")
    module._pangu_spec_debug = True
    output = Path(os.environ["PANGU_SPEC_DEBUG_DIR"])
    limit = int(os.environ.get("PANGU_SPEC_DEBUG_LIMIT", "1000"))
    selected_rank = int(os.environ.get("PANGU_SPEC_DEBUG_RANK", "0"))
    captured = 0

    original_compute = module.compute_probs_and_sample

    @functools.wraps(original_compute)
    def compute(*args, **kwargs):
        event = ACTIVE.get()
        if event is not None:
            bound = inspect.signature(original_compute).bind(*args, **kwargs)
            event["processed_target_top2"] = top2(bound.arguments["logits"])
        return original_compute(*args, **kwargs)

    module.compute_probs_and_sample = compute
    original_verify = module.rejection_greedy_sample_native

    @functools.wraps(original_verify)
    def verify(*args, **kwargs):
        event = ACTIVE.get()
        if event is None:
            return original_verify(*args, **kwargs)
        bound = inspect.signature(original_verify).bind(*args, **kwargs).arguments
        draft = values(bound["draft_token_ids"])
        target = values(bound["target_argmax"])
        bonus = values(bound["bonus_token_ids"])
        bonus = [b[0] if isinstance(b, list) else b for b in bonus]
        cumulative = values(bound["cu_num_draft_tokens"])
        accepted = original_verify(*args, **kwargs)
        actual = values(bound["output_token_ids"])
        expected, counts = expected_greedy(draft, target, bonus, cumulative)
        event["verify"] = {
            "draft_ids": draft, "target_argmax": target, "bonus_ids": bonus,
            "cu_num_draft_tokens": cumulative, "accepted": values(accepted),
            "output_with_padding": actual, "expected_output": expected,
            "expected_accepted": counts,
            "matches_reference": values(accepted) == counts and all(
                row[:len(ref)] == ref for row, ref in zip(actual, expected)),
        }
        return accepted

    module.rejection_greedy_sample_native = verify
    original_forward = module.NPURejectionSampler.forward

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        nonlocal captured
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != selected_rank or captured >= limit or not (output / "ARM").exists():
            return original_forward(self, *args, **kwargs)
        bound = inspect.signature(original_forward).bind(self, *args, **kwargs).arguments
        sampling = bound["sampling_metadata"]
        if not sampling.all_greedy:
            raise RuntimeError("spec-debug supports only all_greedy requests")
        metadata = bound["metadata"]
        logits = bound["logits"]
        event = {"cycle": captured, "rank": rank, "pid": os.getpid(),
                 "time_ns": time.time_ns(), "temperature": values(sampling.temperature),
                 "num_draft_tokens": values(metadata.num_draft_tokens),
                 "target_logits_indices": values(metadata.target_logits_indices),
                 "bonus_logits_indices": values(metadata.bonus_logits_indices),
                 "raw_target_top2": top2(logits[metadata.target_logits_indices]),
                 "raw_bonus_top2": top2(logits[metadata.bonus_logits_indices])}
        label = output / "LABEL"
        event["label"] = label.read_text().strip() if label.exists() else None
        rejection_recorder = None
        if os.environ.get("PANGU_SPEC_DEBUG_REJECTIONS") == "1":
            from spec_rejection import recorder
            rejection_recorder = recorder()
            rejection_recorder.attach(event)
        token = ACTIVE.set(event)
        try:
            result = original_forward(self, *args, **kwargs)
            event["sampler_output_with_padding"] = values(result.sampled_token_ids)
            return result
        except Exception as exc:
            event["error"] = repr(exc)
            raise
        finally:
            ACTIVE.reset(token)
            if rejection_recorder is not None:
                rejection_recorder.record(event)
            with (output / f"rank-{rank}-pid-{os.getpid()}.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            captured += 1

    module.NPURejectionSampler.forward = forward
    print(f"[spec-debug] sampler hook installed, pid={os.getpid()}, arm={output / 'ARM'}", flush=True)
