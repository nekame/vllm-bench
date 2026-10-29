"""Compare recorded requests by position and committed prefix, without ML imports."""
import argparse
import json
import math
from pathlib import Path


PHYSICAL = {"cache_indices", "slot_mapping", "block_table_tensor"}
FRAME_FIELDS = ("is_hybrid", "runner_accepted_gpu", "input_batch_accepted_cpu",
                "num_computed_tokens_cpu", "use_async_scheduling")


def load_events(path):
    """Read JSONL and/or overlapping rejection windows, deduplicating cycles."""
    path = Path(path)
    files = ([path] if path.is_file() else
             sorted(set(path.rglob("rank-*-pid-*.jsonl")) | set(path.rglob("cycle-*.json"))))
    events = {}
    for file in files:
        if file.suffix == ".jsonl":
            documents = [(json.loads(line), f"{file}:{i}") for i, line in
                         enumerate(file.read_text().splitlines(), 1) if line.strip()]
        else:
            documents = [(json.loads(file.read_text()), str(file))]
        for document, source in documents:
            if document.get("format") == "pangu-rejection-window-v1":
                candidates = [(document.get(part), source + "#" + part)
                              for part in ("previous", "rejection", "next")]
            elif "cycle" in document and "verify" in document:
                candidates = [(document, source)]
            else:
                raise ValueError(f"Not a verification event or rejection window: {source}")
            for event, location in candidates:
                if event is None:
                    continue
                ids = event.get("target_forward", {}).get("request_ids", [])
                if len(ids) != 1 or not event.get("verify"):
                    raise ValueError(f"Need a complete single-request event with target_forward: {location}")
                key = (event.get("rank"), event.get("pid"), tuple(ids), event.get("label"), event["cycle"])
                if key in events:
                    if events[key]["event"] != event:
                        raise ValueError(f"Conflicting copies of cycle {event['cycle']}: {location}")
                    events[key]["sources"].append(location)
                else:
                    events[key] = {"event": event, "sources": [location]}
    if not events:
        raise ValueError(f"No recorded verification events found: {path}")
    return list(events.values())


def identity(item):
    event = item["event"]
    return (event.get("rank"), event.get("pid"),
            tuple(event["target_forward"]["request_ids"]), event.get("label"))


def requests(events):
    groups = {}
    for item in events:
        groups.setdefault(identity(item), []).append(item)
    return groups


def describe(key, items):
    return {"rank": key[0], "pid": key[1], "request_id": key[2][0],
            "label": key[3], "events": len(items)}


def select(events, label=None, request_id=None):
    groups = {key: items for key, items in requests(events).items()
              if (label is None or key[3] == label) and (request_id is None or key[2][0] == request_id)}
    if len(groups) != 1:
        choices = [describe(key, items) for key, items in groups.items()]
        raise ValueError(f"Select exactly one request using label/request-id; candidates: {choices}")
    key, items = next(iter(groups.items()))
    return describe(key, items), sorted(items, key=lambda item: item["event"]["cycle"])


def vector(snapshot, field):
    if not isinstance(snapshot, dict) or snapshot.get("truncated") or "values" not in snapshot:
        raise ValueError(f"Missing or truncated {field}")
    shape = snapshot.get("shape")
    if shape is not None and len(shape) != 1:
        raise ValueError(f"Only one-dimensional {field} supported, got {shape}")
    return snapshot["values"]


def timeline(items):
    """Reconstruct observed committed tokens; draft suffix is never committed."""
    frames, tokens = {}, {}

    def commit(position, token):
        if position in tokens and tokens[position] != token:
            raise ValueError(f"Inconsistent committed token at position {position} within one request")
        tokens[position] = token

    for item in items:
        event = item["event"]
        frame = event["target_forward"]
        ids = vector(frame.get("input_ids"), "target input_ids")
        positions = vector(frame.get("positions"), "target positions")
        cumulative = event["verify"]["cu_num_draft_tokens"]
        if len(cumulative) != 1:
            raise ValueError("Only one request per verification is supported")
        length = cumulative[0] + 1
        if len(ids) < length or len(positions) < length:
            raise ValueError("Target input snapshot shorter than verification query")
        positions, ids = positions[:length], ids[:length]
        start = positions[0]
        if positions != list(range(start, start + length)):
            raise ValueError("Target positions must be consecutive")
        if start in frames:
            raise ValueError(f"Repeated target position {start}; cannot associate unambiguously")
        rows = event.get("sampler_output_with_padding", [])
        if len(rows) != 1:
            raise ValueError("Need one sampler output row")
        row = rows[0]
        output = [token for token in row if token != -1]
        if row[:len(output)] != output or any(token < 0 for token in output):
            raise ValueError("Sampler output must be valid tokens followed by -1 padding")
        commit(start, ids[0])
        for offset, token in enumerate(output, 1):
            commit(start + offset, token)
        frames[start] = {**item, "input_ids": ids, "positions": positions}
    return frames, tokens


def logical(value):
    if isinstance(value, dict):
        # Snapshot dtype/shape may differ in CPU/GPU representations; compare values.
        if "values" in value:
            return {"values": value["values"], "truncated": value.get("truncated", False)}
        return {key: logical(item) for key, item in value.items() if key not in PHYSICAL}
    if isinstance(value, list):
        return [logical(item) for item in value]
    return value


def difference(path, left, right):
    return {"path": path, "left": left, "right": right}


def metadata_differences(left, right):
    result = []
    for key in FRAME_FIELDS:
        a, b = logical(left.get(key)), logical(right.get(key))
        if a != b:
            result.append(difference(key, a, b))
    a, b = logical(left.get("attention")), logical(right.get("attention"))
    if a != b:
        # Keep per-layer metadata local to avoid dumping every equal layer.
        if isinstance(a, dict) and isinstance(b, dict):
            for name in sorted(a.keys() | b.keys()):
                if a.get(name) != b.get(name):
                    result.append(difference("attention." + name, a.get(name), b.get(name)))
        else:
            result.append(difference("attention", a, b))
    return result


def close(a, b, atol, rtol):
    return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


def logits_differences(left, right, atol, rtol):
    result = []
    a, b = left["event"], right["event"]
    for field in ("raw_target_top2", "processed_target_top2", "raw_bonus_top2"):
        x, y = a.get(field), b.get(field)
        if x is None or y is None:
            if x != y:
                result.append({"field": field, "status": "missing_on_one_side"})
            continue
        for row, (u, v) in enumerate(zip(x["rows"], y["rows"])):
            context_length = (len(left["input_ids"]) if field == "raw_bonus_top2" else row + 1)
            same_context = (left["input_ids"][:context_length] == right["input_ids"][:context_length]
                            and (field != "raw_bonus_top2" or len(left["input_ids"]) == len(right["input_ids"])))
            if not same_context:
                continue  # Later rows condition on different proposed tokens.
            scores_a, scores_b = dict(zip(u["ids"], u["logits"])), dict(zip(v["ids"], v["logits"]))
            common = scores_a.keys() & scores_b.keys()
            deltas = {str(token): scores_b[token] - scores_a[token] for token in common}
            if u["ids"] != v["ids"] or any(not close(scores_a[t], scores_b[t], atol, rtol) for t in common):
                choice_field = "bonus_ids" if field == "raw_bonus_top2" else "target_argmax"
                choice_row = 0 if choice_field == "bonus_ids" else row
                actual_a = a.get("verify", {}).get(choice_field, [])
                actual_b = b.get("verify", {}).get(choice_field, [])
                selected_a = actual_a[choice_row] if choice_row < len(actual_a) else None
                selected_b = actual_b[choice_row] if choice_row < len(actual_b) else None
                if isinstance(selected_a, list):
                    selected_a = selected_a[0]
                if isinstance(selected_b, list):
                    selected_b = selected_b[0]
                result.append({"field": field, "row": row, "left": u, "right": v,
                               "common_token_logit_deltas": deltas,
                               "top1_changed": u["ids"][0] != v["ids"][0],
                               "top1_changed_means": "topk_first_id_changed_not_actual_argmax",
                               "left_actual_choice": selected_a, "right_actual_choice": selected_b,
                               "actual_choice_changed": selected_a != selected_b
                               if selected_a is not None and selected_b is not None else None,
                               "top2_has_tie": {"left": u["margin"] == 0, "right": v["margin"] == 0}})
    return result


def mome_differences(left, right):
    result, fingerprints, compared = [], [], 0
    calls_a = {call["prefix"]: call for call in left.get("mome_calls", [])}
    calls_b = {call["prefix"]: call for call in right.get("mome_calls", [])}
    for name in sorted(calls_a.keys() | calls_b.keys()):
        a, b = calls_a.get(name), calls_b.get(name)
        if a is None or b is None:
            result.append(difference(name, "present" if a else "missing", "present" if b else "missing"))
            continue
        if logical(a["metadata"]) != logical(b["metadata"]):
            result.append(difference(name + ".metadata", logical(a["metadata"]), logical(b["metadata"])))
        for phase in ("states_before", "states_after"):
            x, y = a.get(phase, []), b.get(phase, [])
            if not x and not y:
                continue
            if len(x) != 1 or len(y) != 1:
                fingerprints.append({"prefix": name, "phase": phase,
                                     "status": "unavailable_or_multiple_blocks_not_mapped"})
                continue
            u, v = x[0], y[0]
            if "sha256" not in u or "sha256" not in v:
                fingerprints.append({"prefix": name, "phase": phase, "status": "fingerprint_omitted"})
                continue
            compared += 1
            if (u["sha256"], u.get("shape"), u.get("dtype")) != (v["sha256"], v.get("shape"), v.get("dtype")):
                fingerprints.append({"prefix": name, "phase": phase, "status": "whole_block_differs",
                                     "left": u, "right": v,
                                     "diagnosis": "Includes speculative/unused slots; not proof of a wrong committed history."})
    return result, fingerprints, compared


def incomplete_snapshots(value, path=""):
    if isinstance(value, dict):
        if value.get("truncated"):
            yield path
        for key, item in value.items():
            yield from incomplete_snapshots(item, path + "." + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from incomplete_snapshots(item, path + f"[{index}]")


def compare(left, right, atol=1e-5, rtol=1e-5):
    frames_a, tokens_a = timeline(left)
    frames_b, tokens_b = timeline(right)
    shared = sorted(tokens_a.keys() & tokens_b.keys())
    first_output = next((difference(str(p), tokens_a[p], tokens_b[p]) for p in shared
                         if tokens_a[p] != tokens_b[p]), None)
    start_a, start_b = min(frames_a), min(frames_b)
    root = max(start_a, start_b)
    # A missing segment cannot be inferred from matching later anchors.
    prefix_end = root - 1
    while prefix_end + 1 in tokens_a and prefix_end + 1 in tokens_b:
        p = prefix_end + 1
        if tokens_a[p] != tokens_b[p]:
            break
        prefix_end = p
    details, skipped, matched = [], [], 0
    for position in sorted(frames_a.keys() & frames_b.keys()):
        if start_a != start_b or position > prefix_end:
            skipped.append({"position": position, "reason": "different_recording_start_or_unknown_or_diverged_prefix"})
            continue
        a, b = frames_a[position], frames_b[position]
        frame_a, frame_b = a["event"]["target_forward"], b["event"]["target_forward"]
        metadata = metadata_differences(frame_a, frame_b)
        mome, fingerprints, fingerprint_count = mome_differences(frame_a, frame_b)
        logits = logits_differences(a, b, atol, rtol)
        inputs_equal = a["input_ids"] == b["input_ids"]
        matched += 1
        details.append({"position": position, "left_cycle": a["event"]["cycle"],
                        "right_cycle": b["event"]["cycle"], "left_sources": a["sources"],
                        "right_sources": b["sources"], "target_input_ids_equal": inputs_equal,
                        "left_input_ids": a["input_ids"], "right_input_ids": b["input_ids"],
                        "metadata_differences": metadata + mome, "logits_differences": logits,
                        "fingerprint_observations": fingerprints,
                        "fingerprints_compared": fingerprint_count,
                        "truncated_snapshots": {"left": list(incomplete_snapshots(frame_a)),
                                                "right": list(incomplete_snapshots(frame_b))}})
    changed = [item for item in details if item["metadata_differences"] or item["logits_differences"]
               or item["fingerprint_observations"] or not item["target_input_ids_equal"]]
    def first(field):
        return next(({"position": item["position"], "left_cycle": item["left_cycle"],
                      "right_cycle": item["right_cycle"]} for item in details if item[field]), None)
    return {"format": "pangu-rejection-comparison-v1", "matched_forwards": matched,
            "recording_start_positions": [start_a, start_b],
            "same_recorded_prefix_through_position": prefix_end,
            "first_observed_token_difference": first_output,
            "first_logits_difference": first("logits_differences"),
            "first_metadata_difference": first("metadata_differences"),
            "first_fingerprint_observation": first("fingerprint_observations"),
            "fingerprint_pairs_compared": sum(item["fingerprints_compared"] for item in details),
            "forwards_with_truncated_metadata": sum(bool(item["truncated_snapshots"]["left"] or
                                                          item["truncated_snapshots"]["right"]) for item in details),
            "unmatched_forward_positions": {"left": sorted(frames_a.keys() - frames_b.keys()),
                                            "right": sorted(frames_b.keys() - frames_a.keys())},
            "skipped": skipped, "differences": changed,
            "limitations": ["The prompt and history before the recording start are not captured; identical earlier input is an external assumption.",
                            "Window-only captures can have gaps; no comparison crosses an unknown or diverged recorded prefix.",
                            "Full MoME block hashes include speculative/unused slots. Hash inequality is not proof of a state-management bug.",
                            "Top-2 cannot measure full-logit differences; deltas compare only identical token IDs.",
                            "Zero compared fingerprint pairs means state equality was not checked.",
                            "Physical cache/slot/block IDs are excluded from semantic metadata comparison."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path, nargs="?", help="Defaults to left for two requests in one capture")
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--left-request")
    parser.add_argument("--right-request")
    parser.add_argument("--list-requests", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, help="Write complete JSON report; stdout remains a brief summary")
    args = parser.parse_args(argv)
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0:
        parser.error("Tolerances must be finite and nonnegative")
    try:
        left = load_events(args.left)
        if args.list_requests:
            for key, items in requests(left).items():
                print(json.dumps(describe(key, items), ensure_ascii=False))
            return
        right = left if args.right is None or args.right == args.left else load_events(args.right)
        description_a, items_a = select(left, args.left_label, args.left_request)
        description_b, items_b = select(right, args.right_label, args.right_request)
        if description_a["rank"] != description_b["rank"]:
            raise ValueError("Select the same rank on both sides")
        if right is left and identity(items_a[0]) == identity(items_b[0]):
            raise ValueError("Selected the same request twice")
        report = compare(items_a, items_b, args.atol, args.rtol)
        report.update({"left": description_a, "right": description_b,
                       "tolerances": {"atol": args.atol, "rtol": args.rtol}})
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        for field in ("matched_forwards", "first_observed_token_difference", "first_logits_difference",
                      "first_metadata_difference", "first_fingerprint_observation", "fingerprint_pairs_compared"):
            print(f"{field}: {json.dumps(report[field], ensure_ascii=False)}")
        print(f"skipped: {len(report['skipped'])}; report: {args.output}")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
