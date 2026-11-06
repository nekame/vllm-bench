"""Compare measured engines, or check greedy output against target-only."""

import argparse
import json
from pathlib import Path
import sys

from vllm_bench.pangu_infer.run import write_json


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"position": i, "reference": a, "candidate": b}
    if len(left) != len(right):
        return {"position": min(len(left), len(right)), "reference_length": len(left), "candidate_length": len(right)}
    return None


def compare(reference, candidate, *, correctness=False):
    for record in (reference, candidate):
        if record.get("format") != "pangu-direct-result-v1":
            raise ValueError("Unexpected result format")
        if record.get("measurement") not in ("benchmark", "diagnostic"):
            raise ValueError("Unexpected measurement type")
        if not correctness and record["measurement"] != "benchmark":
            raise ValueError("Diagnostic runs cannot be used for benchmark comparison; use --correctness to check tokens")
    ra, ca = reference["options"], candidate["options"]
    if correctness:
        if ra["temperature"] != 0 or ca["temperature"] != 0:
            raise ValueError("Token equality is a greedy correctness test; sampling requires distribution tests")
        for key in ("model", "tp", "max_new_tokens", "respect_eos"):
            if ra[key] != ca[key]:
                raise ValueError(f"Correctness comparison differs in {key}")
        if reference["prompt_sha256"] != candidate["prompt_sha256"]:
            raise ValueError("Different prompt IDs")
        if reference.get("checkpoint_identity") != candidate.get("checkpoint_identity"):
            raise ValueError("Checkpoint changed between runs")
    elif reference["comparison_sha256"] != candidate["comparison_sha256"]:
        raise ValueError("Workloads differ: checkpoint/config/prompt/TP/depth/mode/sampling must match")
    for key in ("operator", "parallel", "block_size", "mamba_block_size"):
        if reference["effective"].get(key) != candidate["effective"].get(key):
            raise ValueError(f"Effective {key} differs; cannot attribute timing differences to the engine")
    if (not correctness and reference["effective"].get("cache_dtype", "auto")
            != candidate["effective"].get("cache_dtype", "auto")):
        raise ValueError("Effective cache_dtype differs; cannot attribute timing differences to the engine")
    if reference["details"].get("hardware") != candidate["details"].get("hardware"):
        raise ValueError("Different hardware")
    if reference["details"].get("versions") != candidate["details"].get("versions"):
        raise ValueError("Different installed runtime versions")
    diffs = [first_difference(left["token_ids"], right["token_ids"])
             for left, right in zip(reference["records"], candidate["records"])]
    if len(reference["records"]) != len(candidate["records"]):
        raise ValueError("Different repeat counts")
    greedy_match = all(diff is None for diff in diffs) if ra["temperature"] == ca["temperature"] == 0 else None
    result = {"format": "pangu-direct-comparison-v1", "correctness_only": correctness,
              "greedy_tokens_match": greedy_match, "first_differences": diffs if greedy_match is not None else None,
              "reference_engine": reference["engine"], "candidate_engine": candidate["engine"],
              "reference_tpot_ms": reference["median_tpot_ms"], "candidate_tpot_ms": candidate["median_tpot_ms"],
              "speedup": None, "interpretation": "Correctness comparison only"}
    if not correctness and greedy_match is not False:
        if not reference["median_tpot_ms"] or not candidate["median_tpot_ms"]:
            raise ValueError("No decode tokens: TPOT/speedup comparison is undefined")
        result["speedup"] = reference["median_tpot_ms"] / candidate["median_tpot_ms"]
        result["interpretation"] = "End-to-end decode difference; use CPU/NPU traces to attribute Python overhead"
        if greedy_match is None:
            result["interpretation"] += "; sampling sequences/acceptance may differ"
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="Run directory or result.json")
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--correctness", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    try:
        def read(path):
            return json.loads((path / "result.json" if path.is_dir() else path).read_text())
        result = compare(read(a.reference), read(a.candidate), correctness=a.correctness)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if a.output:
            if a.output.exists():
                raise ValueError("Output exists")
            write_json(a.output, result)
        return 1 if result["greedy_tokens_match"] is False else 0
    except (ValueError, OSError, KeyError) as error:
        p.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
