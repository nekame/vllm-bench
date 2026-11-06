#!/usr/bin/env python3
"""Print first-verification MoME metadata from saved boundary snapshots."""

import argparse
import json
import math
from pathlib import Path


def verification(data):
    events = data["events"]
    index = next(i for i, event in enumerate(events) if event["event"] == "first_verification")
    if index == 0 or not events[index - 1]["event"].startswith("mtp_"):
        raise ValueError("Expected an MTP snapshot immediately before first verification")
    return events[index], events[index - 1]


def state_comparison(reference, candidate):
    """Compare pre-verification states, before the target overwrites them."""
    verification_events, before_events = zip(*(verification(data) for data in (reference, candidate)))
    states = [event["states"]["target"] for event in before_events]
    metadata = [event["states"]["target"]["metadata"] for event in verification_events]
    if any(not event["states"]["target"]["persistent_decode_fields_current"]
           for event in verification_events):
        raise ValueError("Snapshots do not contain current full-graph decode metadata")
    initial = [int(fields["block_idx_last_computed_token"][0]) for fields in metadata]
    accepted = [int(fields["num_accepted_tokens"][0]) for fields in metadata]
    depths = [sum(event["event"].startswith("mtp_") for event in data["events"])
              for data in (reference, candidate)]
    report = {"reference_record": reference["record"], "candidate_record": candidate["record"],
              "snapshot_events": [event["event"] for event in before_events],
              "initial_state_blocks": initial, "num_accepted_tokens": accepted, "blocks": {}}
    # Include the preceding block's committed tail as well as the selected block.
    for logical in sorted(set(initial + [max(0, block - 1) for block in initial])):
        blocks = [state["blocks"].get(str(logical)) for state in states]
        result = {"available": all(block is not None and block["available"] for block in blocks),
                  "selected_as_initial": [logical == block for block in initial]}
        report["blocks"][str(logical)] = result
        if not result["available"]:
            continue
        result["components"] = []
        if len(blocks[0]["values"]) != len(blocks[1]["values"]):
            raise ValueError("Different numbers of MoME state components")
        for index, (a, b) in enumerate(zip(blocks[0]["values"], blocks[1]["values"])):
            if len(a.shape) != 2 or a.shape != b.shape:
                raise ValueError("Expected equal [state_rows, channels] MoME state shapes")
            rows = [value.tolist() for value in (a, b)]
            maxima = [max(abs(x - y) for x, y in zip(left, right))
                      for left, right in zip(*rows)]
            history = [a.shape[0] - depth for depth in depths]
            selected = [list(range(count - 1, count - 1 + width))
                        for count, width in zip(accepted, history)]
            if any(width <= 0 for width in history) or any(
                    row < 0 or row >= a.shape[0] for indices in selected for row in indices):
                raise ValueError("Accepted history rows fall outside the saved MoME state")
            result["components"].append({
                "component": index, "shape": list(a.shape),
                "row_max_abs_diff": [value if math.isfinite(value) else str(value) for value in maxima],
                "history_rows_from_shape": history,
                "accepted_history_rows": selected,
                "committed_tail_rows": [list(range(a.shape[0] - width, a.shape[0])) for width in history],
            })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Run output directory containing mtp-boundary/")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--records", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--compare-states", action="store_true",
                        help="Compare states before verification; the first record is the reference")
    args = parser.parse_args()

    import torch

    root = args.output / "mtp-boundary" / f"rank-{args.rank}"
    records = []
    for record in args.records:
        path = root / f"record-{record:05d}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        records.append(data)
        event, _ = verification(data)
        state = event["states"]["target"]
        metadata = {key: value.tolist() if value is not None else None
                    for key, value in state["metadata"].items()}
        print(json.dumps({"record": record, "rank": args.rank,
                          "persistent_decode_fields_current": state["persistent_decode_fields_current"],
                          "metadata": metadata}, ensure_ascii=False), flush=True)
    if args.compare_states:
        for candidate in records[1:]:
            print("[state-comparison] " + json.dumps(state_comparison(records[0], candidate),
                                                    ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
