"""Compare exact decode MoME tensors at the first recorded output divergence."""
import argparse
import json
from pathlib import Path

from vllm_bench.mtp_debug.compare_rejections import incomplete_snapshots, load_events, logical, timeline
from vllm_bench.mtp_debug.find_divergence import analyze, missing_ranges, select_request
from vllm_bench.mtp_debug.locate_output import locate
from vllm_bench.mtp_debug.tensor_io import compare_tensors, json_safe
from vllm_bench.mtp_debug.ranks import filter_rank


def capture_root(path):
    path = Path(path).resolve()
    start = path if path.is_dir() else path.parent
    return next((candidate for candidate in (start, *start.parents) if (candidate / "tensors").is_dir()), start)


def compare_states(root, left, right, phase):
    a, b = left.get(phase, []), right.get(phase, [])
    if len(a) != 1 or len(b) != 1:
        return {"status": "unavailable", "reason": "Need one referenced state block per side"}
    a, b = a[0]["tensor"], b[0]["tensor"]
    full = compare_tensors(root, a, root, b)
    result = {"full_block": full, "rows": [],
              "history_mapping": "candidate_only_not_verified_kernel"}
    if full["status"] != "compared":
        return result
    for row in range(a["shape"][0]):
        result["rows"].append({"row": row,
            "left_candidate_history": row in (left.get("candidate_history_rows") or []),
            "right_candidate_history": row in (right.get("candidate_history_rows") or []),
            **compare_tensors(root, a, root, b, rows=(row, row))})
    return result


def compare_activations(root, left, right, field, position):
    a, b = left.get(field), right.get(field)
    result = {"full_tensor": compare_tensors(root, a, root, b), "rows": []}
    pos_a, pos_b = left.get("row_positions", []), right.get("row_positions", [])
    if len(set(pos_a)) != len(pos_a) or len(set(pos_b)) != len(pos_b):
        result["alignment_status"] = "ambiguous_duplicate_positions"
        return result
    for p in sorted(set(pos_a) & set(pos_b)):
        if p >= position:
            continue  # Do not attribute future proposed tokens to this output decision.
        result["rows"].append({"position": p,
            **compare_tensors(root, a, root, b, rows=(pos_a.index(p), pos_b.index(p)))})
    result["alignment_status"] = "aligned_by_absolute_position" if result["rows"] else "no_shared_recorded_rows"
    result["unmatched_positions"] = {"left": sorted(set(pos_a) - set(pos_b)),
                                     "right": sorted(set(pos_b) - set(pos_a))}
    return result


def equal(result):
    return result.get("status") == "compared" and result["exact_equal"]


def unequal(result):
    return result.get("status") == "compared" and not result["exact_equal"]


def compare_call(root, a, b, position, comparable_prefix, same_start):
    left, right = a.get("tensor_capture"), b.get("tensor_capture")
    if not left or not right:
        return {"status": "unavailable", "reason": "Exact tensors absent in old capture or --mome-tensors was disabled"}
    if left.get("format") != "pangu-mome-tensors-v1" or right.get("format") != "pangu-mome-tensors-v1":
        return {"status": "unavailable", "reason": "Unsupported exact tensor capture format"}
    if left.get("status") != "complete" or right.get("status") != "complete":
        return {"status": "unavailable", "left_status": left.get("status"), "right_status": right.get("status"),
                "left_reason": left.get("reason"), "right_reason": right.get("reason")}
    params_a, params_b = left.get("parameters", {}), right.get("parameters", {})
    metadata_a, metadata_b = a.get("metadata"), b.get("metadata")
    metadata_known = (isinstance(metadata_a, dict) and isinstance(metadata_b, dict)
        and all(key in metadata_a and key in metadata_b for key in
                ("num_accepted_tokens", "num_computed_tokens", "query_start_loc", "B_size", "pad_slot_id"))
        and not list(incomplete_snapshots(metadata_a)) and not list(incomplete_snapshots(metadata_b)))
    result = {"status": "compared", "parameters_equal": params_a == params_b,
              "left_y_source": left.get("y_source", "returned_tensor"),
              "right_y_source": right.get("y_source", "returned_tensor"),
              "left_return_kind": left.get("return_kind", "tensor"),
              "right_return_kind": right.get("return_kind", "tensor"),
              "metadata_equal": logical(metadata_a) == logical(metadata_b) if metadata_known else None,
              "left_metadata": metadata_a, "right_metadata": metadata_b,
              "left_parameters": params_a, "right_parameters": params_b,
              "left_candidate_history_rows": left.get("candidate_history_rows"),
              "right_candidate_history_rows": right.get("candidate_history_rows"),
              "state_comparison_context": "same_forward_start" if same_start else "different_forward_starts_not_equivalent_history",
              "states_before": compare_states(root, left, right, "states_before"),
              "states_after": compare_states(root, left, right, "states_after"),
              "x_before": compare_activations(root, left, right, "x_before", position),
              "y_after": compare_activations(root, left, right, "y_after", position)}
    if "x_after" in left or "x_after" in right:
        result["x_after"] = compare_activations(root, left, right, "x_after", position)
    observations = []
    state = result["states_before"].get("full_block", {})
    x = result["x_before"]["full_tensor"]
    if unequal(state):
        observations.append("state_before_differs_active_history_not_proven")
    if any(unequal(row) for row in result["x_before"]["rows"]):
        observations.append("input_activations_differ_at_shared_positions")
    if any(unequal(row) for row in result["y_after"]["rows"]):
        observations.append("output_activations_differ_at_shared_positions")
    known_parameters = bool(params_a.get("weight", {}).get("sha256") if params_a.get("weight") else False)
    known_parameters &= isinstance(params_a.get("on_ascend950"), bool)
    known_parameters &= isinstance(params_a.get("enable_precision_strong_consistency"), bool)
    if (comparable_prefix and same_start and known_parameters and params_a == params_b and result["metadata_equal"] is True
            and equal(state) and equal(x) and "output_activations_differ_at_shared_positions" in observations):
        observations.append("output_differs_with_equal_captured_arguments_requires_kernel_investigation")
    if not params_a == params_b:
        observations.append("different_execution_parameters_or_weights")
    if result["metadata_equal"] is False:
        observations.append("different_convolution_metadata")
    result["observations"] = observations
    result["interpretation"] = "Observations, not proof of a cache or kernel bug; history-row mapping remains unverified."
    available = [result["states_before"].get("full_block", {}), result["states_after"].get("full_block", {}),
                 result["x_before"]["full_tensor"], result["y_after"]["full_tensor"]]
    if "x_after" in result:
        available.append(result["x_after"]["full_tensor"])
    result["all_captured_tensors_exact_equal"] = (all(equal(item) for item in available)
        if all(item.get("status") == "compared" for item in available) else None)
    if any(item.get("status") != "compared" for item in available):
        result["status"] = "partial" if any(item.get("status") == "compared" for item in available) else "unavailable"
        result["reason"] = "one_or_more_payloads_unavailable_or_incompatible"
    return result


def build_report(root, left, right, position=None):
    search = analyze(left, right)
    if position is None:
        difference = search.get("first_difference")
        if not difference:
            return {"format": "pangu-mome-comparison-v1", "status": search["status"], "search": search, "calls": []}
        position = difference["position"]
    a, b = locate(left, position), locate(right, position)
    frames_a, tokens_a = timeline(left)
    frames_b, tokens_b = timeline(right)
    start_a, start_b = min(frames_a), min(frames_b)
    root_position = max(start_a, start_b)
    common = tokens_a.keys() & tokens_b.keys()
    comparable = (start_a == start_b and position > root_position
                  and not missing_ranges(common, root_position, position)
                  and all(tokens_a[p] == tokens_b[p] for p in common if root_position <= p < position))
    report = {"format": "pangu-mome-comparison-v1", "position": position,
              "recorded_prefix_complete_and_equal": comparable,
              "search": {key: value for key, value in search.items() if key not in ("left_producer", "right_producer")},
              "calls": [], "notes": ["Earlier prompt equality is an external assumption.",
                  "All state rows are compared; candidate history rows come from the wrapper contract, not verified kernel internals.",
                  "Physical block IDs do not identify equal history. Different forward starts imply different history boundaries.",
                  "No unchanged-state conclusion is drawn from unavailable tensors. Numerical metrics exclude nonfinite values."]}
    if a is None or b is None:
        report.update(status="producer_not_recorded", left_producer_available=a is not None, right_producer_available=b is not None)
        return report
    report["status"] = "analyzed"
    for key, producer in (("left", a), ("right", b)):
        report[key] = {field: producer[field] for field in ("cycle", "token_id", "source", "source_row",
            "target_start_position", "conditioning_input_ids", "conditioning_positions", "raw_top2", "processed_top2", "sources")}
    calls_a, calls_b = {}, {}
    for target, producer in ((calls_a, a), (calls_b, b)):
        for call in producer["target_forward"].get("mome_calls", []):
            target.setdefault(call["prefix"], []).append(call)
    same_start = a["target_start_position"] == b["target_start_position"]
    for prefix in sorted(calls_a.keys() | calls_b.keys()):
        x, y = calls_a.get(prefix, []), calls_b.get(prefix, [])
        if len(x) != 1 or len(y) != 1:
            result = {"status": "unavailable", "reason": "Missing or ambiguous convolution call"}
        else:
            result = compare_call(root, x[0], y[0], position, comparable, same_start)
        report["calls"].append({"prefix": prefix, **result})
    if not report["calls"]:
        report["status"] = "no_mome_calls_recorded"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--left", default="request-01")
    parser.add_argument("--right", default="request-02")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--position", type=int, help="Defaults to the first recorded committed-token difference")
    parser.add_argument("--capture-root", type=Path, help="Directory containing relative tensor payload paths")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.position is not None and args.position < 0:
        parser.error("--position must be nonnegative")
    try:
        events = filter_rank(load_events(args.path), args.rank)
        description_a, left = select_request(events, args.left)
        description_b, right = select_request(events, args.right)
        if description_a == description_b or description_a["rank"] != description_b["rank"]:
            raise ValueError("Select two different requests on the same rank")
        report = build_report(args.capture_root or capture_root(args.path), left, right, args.position)
        report.update(left_request=description_a, right_request=description_b)
        report = json_safe(report)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print("status:", report["status"], "position:", report.get("position"))
        print("recorded_prefix_complete_and_equal:", report.get("recorded_prefix_complete_and_equal"))
        for call in report["calls"]:
            changed = [row["row"] for row in call.get("states_before", {}).get("rows", []) if unequal(row)]
            print(json.dumps({"prefix": call["prefix"], "status": call["status"],
                              "different_state_rows_before": changed,
                              "all_captured_tensors_exact_equal": call.get("all_captured_tensors_exact_equal"),
                              "observations": call.get("observations", []), "reason": call.get("reason")}, ensure_ascii=False))
        if args.output:
            print("report:", args.output)
        else:
            print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
