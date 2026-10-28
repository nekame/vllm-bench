#!/usr/bin/env python3
"""Summarize rejection-window JSON files without importing torch or vLLM."""
import argparse
import json
from pathlib import Path


def summary(window):
    following = window.get("next") or {}
    frame = following.get("target_forward", {})
    return {"cycle": window["rejection_cycle"], "status": window["status"],
            "label": window["rejection"].get("label"),
            "request_ids": window["rejection"].get("target_forward", {}).get("request_ids"),
            "expected_next_runner_count": window["expected_next_runner_count"],
            "actual_next_runner_count": frame.get("runner_accepted_gpu"),
            "next_runner_count_matches_output": window.get("next_runner_count_matches_output"),
            "is_hybrid": frame.get("is_hybrid"),
            "next_input_ids": frame.get("input_ids"),
            "next_positions": frame.get("positions"),
            "next_mome": [{"prefix": call["prefix"],
                           "num_accepted_tokens": call["metadata"].get("num_accepted_tokens"),
                           "num_computed_tokens": call["metadata"].get("num_computed_tokens")}
                          for call in frame.get("mome_calls", [])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="A window JSON or capture output directory")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("cycle-*.json"))
    if not paths:
        parser.error("No rejection-window files found")
    for path in paths[:args.limit]:
        result = summary(json.loads(path.read_text()))
        print(json.dumps({"file": str(path), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
