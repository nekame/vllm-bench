"""Diagnostic wall times of completed model steps; no logging in the timer."""

import statistics
import time


class ModelTimings:
    def __init__(self):
        self.cycle = None
        self.records = []

    def measure(self, runnable, args, *, model, input_tokens, step=None, head=None):
        import torch

        torch.npu.synchronize()
        started = time.perf_counter_ns()
        result = runnable(*args)
        torch.npu.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        entry = {"cycle": self.cycle, "model": model, "input_tokens": input_tokens,
                 "wall_ms": elapsed_ms}
        if model == "mtp":
            entry.update(step=step, head=head)
        self.records.append(entry)
        return result


def reduce_model_timings(records, device, tp):
    """Called by every rank after decode, never inside a model-call timer."""
    if tp > 1:
        import torch

        counts = torch.tensor([len(records), -len(records)], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.MAX)
        maximum, negative_minimum = counts.tolist()
        if maximum != -negative_minimum:
            raise RuntimeError("Decode model-timing call counts differ between TP ranks: "
                               f"min={-negative_minimum}, max={maximum}")
        if not records:
            return
        times = torch.tensor([entry["wall_ms"] for entry in records], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(times, op=torch.distributed.ReduceOp.MAX)
        values = times.tolist()
    else:
        values = [entry["wall_ms"] for entry in records]
    for entry, value in zip(records, values):
        entry["wall_ms_max_tp"] = value
        del entry["wall_ms"]


def log_model_timings(records, repeat):
    for entry in records:
        predictor = f" step={entry['step']} head={entry['head']}" if entry["model"] == "mtp" else ""
        print(f"[model-time] repeat={repeat} cycle={entry['cycle']} model={entry['model']}"
              f"{predictor} input_tokens={entry['input_tokens']}"
              f" wall_ms_max_tp={entry['wall_ms_max_tp']:.3f}", flush=True)
    if not records:
        print(f"[model-time] repeat={repeat} no_decode_calls", flush=True)


def summarize_model_timings(records):
    groups = {}
    for record in records:
        for entry in record.get("model_timings", []):
            key = (entry["model"], entry["input_tokens"], entry.get("head"))
            groups.setdefault(key, []).append(entry["wall_ms_max_tp"])
    return [{"model": model, "input_tokens": tokens, "head": head, "count": len(values),
             "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}
            for (model, tokens, head), values in groups.items()]


def log_model_timing_summary(summary):
    for group in summary:
        predictor = f" head={group['head']}" if group["model"] == "mtp" else ""
        print(f"[model-time-summary] model={group['model']}{predictor}"
              f" input_tokens={group['input_tokens']} count={group['count']}"
              f" median_ms={group['median_ms']:.3f} min_ms={group['min_ms']:.3f}"
              f" max_ms={group['max_ms']:.3f} aggregation=max_tp", flush=True)
    if not summary:
        print("[model-time-summary] no_decode_calls", flush=True)
