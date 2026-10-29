"""Locate the verification event that emitted a token at an absolute position."""
import argparse
import json
from pathlib import Path

from vllm_bench.mtp_debug.compare_rejections import describe, load_events, requests, timeline


def top_row(event, field, row):
    rows = (event.get(field) or {}).get("rows", [])
    return rows[row] if row < len(rows) else None


def emitted_tokens(item):
    return [token for token in item["event"]["sampler_output_with_padding"][0] if token != -1]


def locate(items, position):
    """Use output spans, never textual occurrences of the position number."""
    frames, _ = timeline(items)
    found = []
    for start, item in sorted(frames.items()):
        output = emitted_tokens(item)
        offset = position - start - 1
        if not 0 <= offset < len(output):
            continue
        event = item["event"]
        verify = event["verify"]
        accepted = verify["accepted"][0]
        proposed = verify["cu_num_draft_tokens"][0]
        if not 0 <= accepted <= proposed or len(output) != accepted + 1:
            raise ValueError(f"Inconsistent accepted/output counts in cycle {event['cycle']}")
        if offset < accepted:
            source, row = "accepted_draft", offset
            selected = verify["draft_ids"][row]
            target_argmax = verify["target_argmax"][row]
        elif accepted < proposed:
            source, row = "target_correction", accepted
            selected = target_argmax = verify["target_argmax"][row]
        else:
            source, row = "bonus", 0
            selected = verify["bonus_ids"][0]
            if isinstance(selected, list):
                selected = selected[0]
            target_argmax = None
        raw_field = "raw_bonus_top2" if source == "bonus" else "raw_target_top2"
        raw = top_row(event, raw_field, row)
        processed = None if source == "bonus" else top_row(event, "processed_target_top2", row)
        context_length = proposed + 1 if source == "bonus" else row + 1
        earlier = [(p, frame) for p, frame in frames.items() if p < start]
        previous = None
        if earlier:
            prev_start, prev = max(earlier, key=lambda pair: pair[0])
            prev_event = prev["event"]
            previous = {"cycle": prev_event["cycle"], "target_start_position": prev_start,
                        "accepted": prev_event["verify"]["accepted"],
                        "output": emitted_tokens(prev), "sources": prev["sources"],
                        "immediately_previous_by_output_position": prev_start + len(emitted_tokens(prev)) == start}
        found.append({"position": position, "token_id": output[offset],
                      "cycle": event["cycle"], "output_index": offset,
                      "target_start_position": start, "output_positions": list(range(start + 1, start + 1 + len(output))),
                      "source": source, "source_row": row,
                      "accepted": accepted, "proposed": proposed,
                      "actual_selected_id": selected, "actual_target_argmax": target_argmax,
                      "output_matches_selection": output[offset] == selected,
                      "accepted_draft_matches_target": selected == target_argmax if source == "accepted_draft" else None,
                      "verify_matches_reference": verify.get("matches_reference"),
                      "draft_ids": verify["draft_ids"], "bonus_ids": verify["bonus_ids"],
                      "target_argmax": verify["target_argmax"],
                      "sampler_output_with_padding": event["sampler_output_with_padding"],
                      "conditioning_input_ids": item["input_ids"][:context_length],
                      "conditioning_positions": item["positions"][:context_length],
                      "raw_top2": raw, "processed_top2": processed,
                      "raw_top2_has_tie": raw["margin"] == 0 if raw else None,
                      "processed_top2_has_tie": processed["margin"] == 0 if processed else None,
                      "sources": item["sources"], "previous_verification": previous,
                      "target_forward": event["target_forward"]})
    if len(found) > 1:
        raise ValueError(f"Multiple events claim output position {position} in one request")
    return found[0] if found else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Capture directory, JSONL or rejection-window JSON")
    parser.add_argument("--position", type=int, required=True,
                        help="Absolute output position, e.g. first_observed_token_difference.path; not token ID")
    parser.add_argument("--label", help="Optionally restrict to one LABEL")
    parser.add_argument("--request", help="Optionally restrict to one request ID")
    parser.add_argument("--output", type=Path, help="Write full JSON to a new file; print a short summary")
    args = parser.parse_args(argv)
    if args.position < 0:
        parser.error("--position must be nonnegative")
    try:
        matches, missing = [], []
        for key, items in requests(load_events(args.path)).items():
            description = describe(key, items)
            if args.label is not None and description["label"] != args.label:
                continue
            if args.request is not None and description["request_id"] != args.request:
                continue
            result = locate(items, args.position)
            if result is None:
                missing.append(description)
            else:
                matches.append({"request": description, **result})
        report = {"format": "pangu-output-position-v1", "position": args.position,
                  "matches": matches, "not_found_in_recorded_outputs": missing,
                  "notes": ["Position is absolute and zero-based, as in captured target positions.",
                            "Overlapping windows are deduplicated. Missing events cannot be reconstructed.",
                            "Top-2 order with tied logits is not the actual argmax; actual_selected_id comes from verifier inputs.",
                            "Bonus processed logits are not captured; raw bonus top-2 may precede processors.",
                            "Full MoME block fingerprints include speculative/unused slots; hash differences alone are not proof of a bug."]}
        if not matches and not missing:
            raise ValueError("No requests match the supplied filters")
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            for match in matches:
                print(json.dumps({key: match[key] for key in
                                  ("request", "position", "token_id", "cycle", "source", "source_row",
                                   "accepted", "actual_selected_id", "output_matches_selection")}, ensure_ascii=False))
            print(f"found: {len(matches)}; not found in recorded outputs: {len(missing)}; report: {args.output}")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
