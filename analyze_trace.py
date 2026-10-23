#!/usr/bin/env python3
"""Summarize optional CPU layer ranges in a Chrome trace. Not NPU timings."""
import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys


def summarize(data):
    events = data.get("traceEvents", []) if isinstance(data, dict) else data
    groups = {}
    for e in events:
        if (e.get("ph") == "X" and e.get("name", "").startswith("pangu12/layer_")
                and e.get("dur") is not None):
            key = (str(e.get("pid")), str(e.get("tid")), e["name"])
            groups.setdefault(key, []).append(float(e["dur"]))
    return [{"pid": key[0], "tid": key[1], "range": key[2], "calls": len(values),
             "cpu_mean_us": statistics.fmean(values), "cpu_median_us": statistics.median(values),
             "cpu_total_us": sum(values)} for key, values in sorted(groups.items())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path, help="One trace_view.json or Chrome JSON[.gz], one rank per analysis")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    opener = gzip.open if a.trace.suffix == ".gz" else open
    with opener(a.trace, "rt", encoding="utf-8") as f:
        rows = summarize(json.load(f))
    result = {"trace": str(a.trace), "ranges": rows,
              "meaning": "CPU launch ranges; asynchronous NPU kernels can outlive these spans. Do not scale these totals to predict NPU latency."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Found {len(rows)} layer ranges. Saved {a.output}")
    if not rows:
        print("No layer ranges: use --annotate-layers, or inspect the device timeline directly. Graph compilation may hide Python ranges.", file=sys.stderr)


if __name__ == "__main__":
    main()
