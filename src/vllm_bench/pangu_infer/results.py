"""Portable result files; diagnostic traces never count as benchmark results."""

import csv
from importlib import metadata
from pathlib import Path
import statistics

from vllm_bench.pangu_infer.run import write_json


def versions():
    result = {}
    for name in ("torch", "torch-npu", "vllm", "omni-infer", "npugraph-ex", "safetensors"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def save_result(output, plan, records, effective, details):
    if not records or any(r["decode_seconds"] < 0 or r["decode_tokens"] < 0 for r in records):
        raise ValueError("Invalid decode measurements")
    a = plan["options"]
    tpot = [r["decode_seconds"] / r["decode_tokens"] for r in records if r["decode_tokens"]]
    seconds = sum(r["decode_seconds"] for r in records)
    tokens = sum(r["decode_tokens"] for r in records)
    accepted = [n for r in records for n in r.get("accepted_drafts", [])]
    summary = {"format": "pangu-direct-result-v1", "engine": a["engine"],
               "measurement": "diagnostic" if a["profile"] or a.get("model_timing", False) or a.get("trace_target", False) or a.get("memory_diagnostics", False) else "benchmark",
               "comparison_sha256": plan["comparison_sha256"],
               "prompt_sha256": plan["prompt_sha256"], "model_scope": plan["model_scope"],
               "checkpoint_identity": plan["checkpoint_identity"],
               "options": a, "effective": effective, "details": details,
               "decode_tokens_per_second": tokens / seconds if seconds else 0,
               "median_tpot_ms": statistics.median(tpot) * 1000 if tpot else None,
               "min_tpot_ms": min(tpot) * 1000 if tpot else None,
               "max_tpot_ms": max(tpot) * 1000 if tpot else None,
               "empty_decode_requests": sum(r["decode_tokens"] == 0 for r in records),
               "acceptance_rate": sum(accepted) / (len(accepted) * a["depth"]) if accepted and a["depth"] else None,
               "records": records, "validation": "NPU executed; correctness requires comparison with target-only"}
    if a.get("model_timing", False):
        from vllm_bench.pangu_infer.model_timing import summarize_model_timings
        summary["model_timing_summary"] = summarize_model_timings(records)
    write_json(Path(output) / "result.json", summary)
    with (Path(output) / "summary.csv").open("w", newline="") as handle:
        fields = ["engine", "measurement", "model_scope", "decode_tokens_per_second", "median_tpot_ms", "acceptance_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in fields})
    if plan.get("prompts"):
        from vllm_bench.pangu_infer.prompts import load_tokenizer
        tokenizer = load_tokenizer(a.get("tokenizer") or a["model"])
        with (Path(output) / "generations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "text", "tpot"])
            writer.writeheader()
            for record in records:
                writer.writerow({"id": record["id"],
                                 "text": tokenizer.decode(record["token_ids"], skip_special_tokens=True),
                                 "tpot": record["decode_seconds"] * 1000 / record["decode_tokens"]
                                 if record["decode_tokens"] else None})
    return summary
