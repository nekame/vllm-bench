"""Portable result files; diagnostic traces never count as benchmark results."""

import csv
from importlib import metadata
import json
from pathlib import Path
import statistics

from vllm_bench.pangu_infer.run import write_json


def repeat_consistency(records):
    """Compare repeated greedy requests within each prompt ID."""
    from vllm_bench.pangu_infer.compare import first_difference

    references, differences, checked = {}, [], 0
    for index, record in enumerate(records):
        key = ("id", record["id"]) if "id" in record else ("synthetic",)
        if key not in references:
            references[key] = index
            continue
        reference_index = references[key]
        reference = records[reference_index]
        checked += 1
        difference = first_difference(reference["token_ids"], record["token_ids"])
        if difference is not None:
            differences.append({"reference_record": reference_index, "candidate_record": index,
                                "id": record.get("id"), **difference,
                                "reference_cached_prompt_tokens": reference.get("cached_prompt_tokens"),
                                "candidate_cached_prompt_tokens": record.get("cached_prompt_tokens")})
    return {"checked_pairs": checked, "matches": not differences if checked else None,
            "first_differences": differences}


def versions():
    result = {}
    for name in ("torch", "torch-npu", "vllm", "omni-infer", "npugraph-ex", "safetensors"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def prefix_cache_groups(records, depth, *, greedy):
    """Report actual hit/miss groups without mixing cold and warm requests."""
    groups = {}
    for index, record in enumerate(records):
        cached = record.get("cached_prompt_tokens")
        name = "unknown" if cached is None else "hit" if cached > 0 else "miss"
        groups.setdefault(name, []).append(index)
    result = {}
    for name, indices in groups.items():
        selected = [records[index] for index in indices]
        tpot = [r["decode_seconds"] * 1000 / r["decode_tokens"] for r in selected if r["decode_tokens"]]
        accepted = [n for r in selected for n in r.get("accepted_drafts", [])]
        item = {"record_indices": indices, "requests": len(indices),
                "median_tpot_ms": statistics.median(tpot) if tpot else None,
                "acceptance_rate": sum(accepted) / (len(accepted) * depth) if accepted and depth else None}
        if greedy:
            check = repeat_consistency(selected)
            for difference in check["first_differences"]:
                for key in ("reference_record", "candidate_record"):
                    difference[key] = indices[difference[key]]
            item["repeat_consistency"] = check
        result[name] = item
    return result


def save_result(output, plan, records, effective, details):
    if not records or any(r["decode_seconds"] < 0 or r["decode_tokens"] < 0 for r in records):
        raise ValueError("Invalid decode measurements")
    a = plan["options"]
    tpot = [r["decode_seconds"] / r["decode_tokens"] for r in records if r["decode_tokens"]]
    seconds = sum(r["decode_seconds"] for r in records)
    tokens = sum(r["decode_tokens"] for r in records)
    accepted = [n for r in records for n in r.get("accepted_drafts", [])]
    diagnostic = any((a["profile"], a.get("model_timing", False), a.get("trace_target", False),
                      a.get("memory_diagnostics", False), a.get("trace_mtp_boundary", False),
                      a["engine"] in ("omni", "omni2") and a.get("warmup", 2) == 0))
    summary = {"format": "pangu-direct-result-v1", "engine": a["engine"],
               "measurement": "diagnostic" if diagnostic else "benchmark",
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
    if a.get("temperature", 1) == 0:
        summary["repeat_consistency"] = repeat_consistency(records)
        print("[repeat-check] " + json.dumps(summary["repeat_consistency"]), flush=True)
    if a["engine"] in ("omni", "omni2") and a.get("enable_prefix_caching", False):
        summary["prefix_cache_groups"] = prefix_cache_groups(records, a["depth"], greedy=a.get("temperature", 1) == 0)
        print("[prefix-cache-groups] " + json.dumps(summary["prefix_cache_groups"]), flush=True)
    write_json(Path(output) / "result.json", summary)
    with (Path(output) / "summary.csv").open("w", newline="") as handle:
        fields = ["engine", "measurement", "model_scope", "decode_tokens_per_second", "median_tpot_ms", "acceptance_rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: summary[key] for key in fields})
    if plan.get("prompts"):
        from vllm_bench.pangu_infer.prompts import load_tokenizer
        tokenizer = (load_tokenizer(a.get("tokenizer") or a["model"])
                     if any("text" not in record for record in records) else None)
        with (Path(output) / "generations.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "text", "tpot"])
            writer.writeheader()
            for record in records:
                writer.writerow({"id": record["id"],
                                 "text": record["text"] if "text" in record else tokenizer.decode(
                                     record["token_ids"], skip_special_tokens=True),
                                 "tpot": record["decode_seconds"] * 1000 / record["decode_tokens"]
                                 if record["decode_tokens"] else None})
    return summary
