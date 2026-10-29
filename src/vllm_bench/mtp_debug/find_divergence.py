"""Find the earliest recorded token difference and both producing events."""
import argparse
import json
from pathlib import Path

from vllm_bench.mtp_debug.compare_rejections import describe, load_events, requests, timeline
from vllm_bench.mtp_debug.locate_output import locate


def select_request(events, name):
    choices = [(key, items) for key, items in requests(events).items()
               if key[3] == name or key[2][0] == name]
    if len(choices) != 1:
        available = [describe(key, items) for key, items in requests(events).items()]
        raise ValueError(f"Selector {name!r} must match exactly one label or request ID; available: {available}")
    key, items = choices[0]
    return describe(key, items), items


def missing_ranges(known, start, stop):
    """Return gaps in [start, stop), without iterating huge position ranges."""
    cursor, gaps = start, []
    for position in sorted(p for p in known if start <= p < stop):
        if position > cursor:
            gaps.append([cursor, position - 1])
        cursor = position + 1
    if cursor < stop:
        gaps.append([cursor, stop - 1])
    return gaps


def analyze(left, right, context=4):
    frames_a, tokens_a = timeline(left)
    frames_b, tokens_b = timeline(right)
    shared = tokens_a.keys() & tokens_b.keys()
    differences = sorted(p for p in shared if tokens_a[p] != tokens_b[p])
    starts = [min(frames_a), min(frames_b)]
    root = max(starts)
    report = {"format": "pangu-first-divergence-v1", "recording_start_positions": starts,
              "shared_recorded_positions": len(shared), "first_difference": None,
              "notes": ["The prompt/history before recording are not captured; their equality is an external assumption.",
                        "Only committed tokens are compared; rejected draft tokens are excluded.",
                        "Producer cycles and forward boundaries can differ between requests.",
                        "Full MoME block hashes include speculative/unused slots; unequal hashes are not proof of a bug.",
                        "Top-k order with tied logits is not the actual selected token."]}
    if not shared:
        report["status"] = "no_shared_recorded_positions"
        return report
    if not differences:
        report["status"] = "no_difference_in_shared_recorded_tokens"
        report["notes"].append("Missing positions and unrecorded suffixes are not evidence of equal full outputs.")
        return report
    position = differences[0]
    gaps = missing_ranges(shared, root, position)
    prefix_verified = starts[0] == starts[1] and position > root and not gaps
    status = "first_difference_in_contiguous_recorded_prefix" if prefix_verified else "earliest_observed_difference_prefix_incomplete"
    report.update({"status": status, "first_difference": {
        "position": position, "left_token_id": tokens_a[position], "right_token_id": tokens_b[position]},
        "recorded_prefix_complete_before_difference": prefix_verified,
        "same_recorded_prefix_range": [root, position - 1] if prefix_verified and position > root else None,
        "unknown_prefix_ranges": gaps,
        "context": [{"position": p, "left_token_id": tokens_a.get(p),
                     "right_token_id": tokens_b.get(p),
                     "both_recorded": p in shared} for p in range(max(root, position - context), position + 1)],
        "left_producer": locate(left, position), "right_producer": locate(right, position)})
    a, b = report["left_producer"], report["right_producer"]
    if position == root:
        report["notes"].append("The recording already starts with different tokens; the divergence may have happened earlier.")
    if a is None or b is None:
        report["notes"].append("A differing token is known from a later target input, but its producing event is missing on at least one side.")
    else:
        report["same_producer_forward_start"] = a["target_start_position"] == b["target_start_position"]
        report["notes"].append("Compare conditioning prefixes by position. Different input-array lengths can be normal when bonus and correction come from different forward boundaries.")
    return report


def short_producer(producer):
    if producer is None:
        return "producing event not recorded"
    return json.dumps({key: producer[key] for key in (
        "cycle", "token_id", "source", "source_row", "accepted", "actual_selected_id",
        "conditioning_input_ids", "conditioning_positions", "raw_top2", "processed_top2",
        "output_matches_selection")}, ensure_ascii=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Capture directory or JSONL; may also contain rejection windows")
    parser.add_argument("--left", default="request-01", help="Left LABEL or request ID")
    parser.add_argument("--right", default="request-02", help="Right LABEL or request ID")
    parser.add_argument("--context", type=int, default=4, help="Preceding token positions included in the report")
    parser.add_argument("--output", type=Path, help="Write complete JSON to a new file")
    args = parser.parse_args(argv)
    if not 0 <= args.context <= 10000:
        parser.error("--context must be between 0 and 10000")
    try:
        events = load_events(args.path)
        description_a, left = select_request(events, args.left)
        description_b, right = select_request(events, args.right)
        if description_a == description_b:
            raise ValueError("Select two different requests")
        if description_a["rank"] != description_b["rank"]:
            raise ValueError("Both requests must be from the same rank")
        report = analyze(left, right, args.context)
        report.update({"left_request": description_a, "right_request": description_b})
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("status:", report["status"])
        if report["first_difference"]:
            print("first_difference:", json.dumps(report["first_difference"], ensure_ascii=False))
            print("left:", short_producer(report["left_producer"]))
            print("right:", short_producer(report["right_producer"]))
            print("recorded_prefix_complete_before_difference:", report["recorded_prefix_complete_before_difference"])
        if args.output:
            print("report:", args.output)
    except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
