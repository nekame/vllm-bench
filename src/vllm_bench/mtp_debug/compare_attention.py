"""Automatic multi-rank HiF8 DSA diagnosis with a small transferable summary."""
import argparse
import json
from pathlib import Path

from vllm_bench.mtp_debug.attention_compare import Reader, compare_layer, compare_payload, stage_map, tensor_row
from vllm_bench.mtp_debug.attention_layout import bytes_difference
from vllm_bench.mtp_debug.compare_mome import capture_root
from vllm_bench.mtp_debug.compare_rejections import load_events, timeline
from vllm_bench.mtp_debug.find_divergence import analyze, select_request
from vllm_bench.mtp_debug.ranks import filter_rank
from vllm_bench.mtp_debug.tensor_io import json_safe


def protocol_checks(items):
    frames, tokens = timeline(items)
    issues, previous = [], None
    for position, item in sorted(frames.items()):
        event, frame = item["event"], item["event"]["target_forward"]
        verify = event["verify"]
        output = [t for t in event["sampler_output_with_padding"][0] if t != -1]
        accepted, proposed = verify["accepted"][0], verify["cu_num_draft_tokens"][0]
        if not 0 <= accepted <= proposed or len(output) != accepted + 1 or verify.get("matches_reference") is False:
            issues.append({"position": position, "code": "verification_invariant"})
        if item["input_ids"][1:] != verify.get("draft_ids"):
            issues.append({"position": position, "code": "draft_input_disagrees_with_verifier"})
        if previous and position == previous[0] + previous[1]:
            actual = (frame.get("runner_accepted_gpu") or {}).get("values")
            if actual is not None and actual != [previous[1]]:
                issues.append({"position": position, "code": "runner_count_mismatch", "actual": actual, "expected": [previous[1]]})
        previous = position, len(output)
    return frames, tokens, issues


def actual_context(frame, tokens, position):
    if position not in frame["positions"]:
        return False
    end = frame["positions"].index(position) + 1
    return all(tokens.get(p) == token for p, token in zip(frame["positions"][:end], frame["input_ids"][:end]))


def prefix_end(a, b, root):
    end = root - 1
    while end + 1 in a and end + 1 in b and a[end + 1] == b[end + 1]:
        end += 1
    return end


def layers(frame):
    trace = frame["event"]["target_forward"].get("attention_trace", {})
    return {entry["layer"]: entry for entry in trace.get("layers", [])}


def state_observations(reader, a, b):
    # Do not promote whole-block differences to a causal stage: history contract unverified.
    ca = {c["prefix"]: c for c in a.get("mome_controls", [])}
    cb = {c["prefix"]: c for c in b.get("mome_controls", [])}
    results = []
    for name in sorted(ca.keys() & cb.keys()):
        for phase in ("states_before", "states_after"):
            x = ca[name].get("tensor_capture", {}).get(phase, [])
            y = cb[name].get("tensor_capture", {}).get(phase, [])
            if len(x) != 1 or len(y) != 1:
                continue
            x, y = x[0]["tensor"], y[0]["tensor"]
            try:
                if reader.read(x) == reader.read(y):
                    continue
                if x["shape"] != y["shape"]:
                    results.append({"prefix": name, "phase": phase, "status": "different_state_layout"})
                    continue
                changed = []
                for row in range(x["shape"][0]):
                    result = compare_payload(reader, x, y, row, row)
                    if result.get("exact_equal") is False:
                        changed.append({"row": row, **result})
                results.append({"prefix": name, "phase": phase, "different_rows": changed,
                                "mapping": "candidate_history_not_verified_kernel"})
            except (ValueError, KeyError, OSError):
                results.append({"prefix": name, "phase": phase, "status": "unavailable"})
    return results


def nonfinite(record):
    if isinstance(record, dict):
        counts = record.get("nonfinite_elements", {})
        if counts.get("left", 0) or counts.get("right", 0):
            return True
        return any(nonfinite(value) for value in record.values())
    return False


def classify_first(record):
    boundary, field = record["boundary"], record["field"]
    if boundary in ("indexer_kernel_output", "attention_indices"):
        return "DSA_SELECTED_INDICES_DIFFER"
    if boundary == "attention_kernel_output":
        return "ATTENTION_OUTPUT_DIFFERS_WITH_EQUAL_SELECTED_INPUTS" if record.get("prior_selected_inputs_equal") else "ATTENTION_OUTPUT_DIFFERS"
    if boundary == "projection_output":
        return "POST_ATTENTION_PROJECTION_DIFFERS"
    if boundary == "tp_reconstruction":
        return "TP_RECONSTRUCTION_MISMATCH"
    if field in ("key_rows", "key_scales"):
        return "INDEXER_HISTORY_DIFFERS" if boundary.startswith("indexer") else "ATTENTION_SELECTED_KV_DIFFERS"
    if boundary.startswith("indexer"):
        return "INDEXER_INPUT_OR_PARAMETERS_DIFFER"
    if boundary == "o_input_after_tp":
        return "O_INPUT_DIFFERS"
    return boundary.upper() + "_DIFFERS"


def gather_inputs(reader, frame, layer, position):
    projection = stage_map(layer).get("projection")
    if not projection or projection.get("status") != "complete":
        raise ValueError("projection missing")
    row = tensor_row(projection, position)
    descriptor = projection["tensors"]["output"]
    raw, shape = reader.row(descriptor, row)
    controls = [c for c in layer.get("mome_controls", []) if c["prefix"].endswith("o_conv")]
    if len(controls) != 1:
        raise ValueError("O input absent/ambiguous")
    control = controls[0].get("tensor_capture", {})
    o_row = tensor_row(control, position)
    o_descriptor = control["x_before"]
    o_raw, o_shape = reader.row(o_descriptor, o_row)
    config = layer["config"]
    tp = frame["event"]["target_forward"]["attention_trace"]["topology"]
    expected = config["num_local_heads"] * config["v_head_dim"]
    if (not config["o_all_gather"] or config["o_gather_size"] != tp["tp_size"]
            or shape != [expected] or o_shape != [expected * tp["tp_size"]]
            or descriptor["dtype"] != o_descriptor["dtype"]):
        raise ValueError("unverified head/layout mapping at TP gather")
    return {"local": raw, "global": o_raw, "dtype": descriptor["dtype"], "tp_rank": tp["tp_rank"], "size": tp["tp_size"]}


def projection_link(reader, layer, position):
    stages = stage_map(layer)
    kernel, projection = stages.get("attention_kernel"), stages.get("projection")
    if not kernel or not projection:
        return {"status": "unavailable", "reason": "kernel/projection stage missing"}
    try:
        return compare_payload(reader, kernel["tensors"]["output"], projection["tensors"]["input"],
                               tensor_row(kernel, position), tensor_row(projection, position))
    except (KeyError, ValueError) as error:
        return {"status": "unavailable", "reason": str(error)}


def compare_capture(root, events, left_name, right_name, expected_ranks=8):
    selected, coverage, protocols = {}, [], []
    for rank in range(expected_ranks):
        try:
            da, a = select_request(filter_rank(events, rank), left_name)
            db, b = select_request(filter_rank(events, rank), right_name)
            if da == db:
                raise ValueError("same request selected twice")
            fa, ta, pa = protocol_checks(a)
            fb, tb, pb = protocol_checks(b)
            protocols.extend({"rank": rank, "side": side, **issue} for side, issues in (("left", pa), ("right", pb)) for issue in issues)
            selected[rank] = (a, b, fa, fb, ta, tb)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            coverage.append({"rank": rank, "reason": "request_or_history_unavailable", "detail": str(error)})
    if not selected:
        return {"format": "pangu-attention-comparison-v1", "result": "INCOMPLETE_CAPTURE", "coverage_issues": coverage,
                "findings": [], "protocol_issues": protocols, "first_boundary": None}
    reference = 0 if 0 in selected else min(selected)
    for rank, (_, _, _, _, ta, tb) in selected.items():
        for side, tokens, ref in (("left", ta, selected[reference][4]), ("right", tb, selected[reference][5])):
            if tokens != ref:
                protocols.append({"rank": rank, "side": side, "code": "committed_tokens_or_coverage_disagree_between_tp_ranks"})
    search = analyze(selected[reference][0], selected[reference][1])
    difference = search.get("first_difference")
    limit = difference["position"] - 1 if difference else None
    comparisons, findings, states, gathers, nonfinites = [], [], [], [], []
    seen_contexts = 0
    for rank, (_, _, fa, fb, ta, tb) in sorted(selected.items()):
        start_a, start_b = min(fa), min(fb)
        if start_a != start_b:
            coverage.append({"rank": rank, "reason": "different_recording_start"})
            continue
        end = prefix_end(ta, tb, start_a)
        if limit is not None:
            if end < limit:
                coverage.append({"rank": rank, "reason": "recorded_prefix_gap_before_token_difference",
                                 "last_equal_contiguous_position": end, "needed_through_position": limit})
            end = min(end, limit)
        # Missing target boundaries are not interchangeable; keep explicit coverage.
        for position in sorted((fa.keys() | fb.keys())):
            if position > end:
                continue
            if position not in fa or position not in fb:
                coverage.append({"rank": rank, "position": position, "reason": "different_forward_boundary_not_compared"})
                continue
            a, b = fa[position], fb[position]
            la, lb = layers(a), layers(b)
            if not la or not lb:
                traces = [x["event"]["target_forward"].get("attention_trace", {}) for x in (a, b)]
                coverage.append({"rank": rank, "position": position, "reason": "attention_trace_absent_or_skipped",
                                 "capture_reasons": [trace.get("reason") for trace in traces]})
                continue
            for layer_id in sorted(la.keys() | lb.keys()):
                if layer_id not in la or layer_id not in lb:
                    coverage.append({"rank": rank, "position": position, "layer": layer_id, "reason": "layer_missing"})
                    continue
                x, y = la[layer_id], lb[layer_id]
                reader = Reader(root)  # Bound memory to one layer/forward pair, not the complete run.
                states.extend({"rank": rank, "position": position, "layer": layer_id, **observation}
                              for observation in state_observations(reader, x, y))
                candidate_positions = sorted(set(a["positions"]) & set(b["positions"]))
                for query_position in candidate_positions:
                    if query_position > end or not actual_context(a, ta, query_position) or not actual_context(b, tb, query_position):
                        continue  # Never diagnose unused rejected draft rows as actual-generation divergence.
                    seen_contexts += 1
                    records = compare_layer(reader, x, y, query_position)
                    configuration_a = {k: v for k, v in x["config"].items() if k != "W_UV"}
                    configuration_b = {k: v for k, v in y["config"].items() if k != "W_UV"}
                    records.insert(0, {"boundary": "execution_parameters", "field": "config", "status": "compared",
                                       "exact_equal": configuration_a == configuration_b,
                                       "left": configuration_a, "right": configuration_b})
                    failures = [r for r in records if r["status"] != "compared" or r.get("coverage_complete") is False]
                    if failures:
                        coverage.append({"rank": rank, "position": query_position, "layer": layer_id,
                            "reason": "incomplete_operands", "fields": [r["boundary"] + "." + r["field"] for r in failures]})
                    for order, record in enumerate(records):
                        if nonfinite(record):
                            nonfinites.append({"rank": rank, "position": query_position, "layer": layer_id,
                                               "boundary": record["boundary"], "field": record["field"]})
                        if record.get("exact_equal") is False:
                            if record["boundary"] == "attention_kernel_output":
                                inputs = [r for r in records if r["boundary"] in
                                          ("execution_parameters", "attention_kernel_parameters", "attention_kernel_input", "attention_indices")]
                                record["prior_selected_inputs_equal"] = bool(inputs) and all(
                                    r.get("status") == "compared" and r.get("exact_equal") is True
                                    and r.get("coverage_complete", True) and not nonfinite(r)
                                    and not r.get("left_future_positions") and not r.get("right_future_positions") for r in inputs)
                            findings.append({"rank": rank, "position": query_position, "layer": layer_id,
                                             "order": order, **record})
                        if record.get("left_future_positions") or record.get("right_future_positions"):
                            coverage.append({"rank": rank, "position": query_position, "reason": "future_sparse_indices_need_mask_contract_check"})
                    comparisons.append({"rank": rank, "position": query_position, "layer": layer_id,
                                        "left_cycle": a["event"]["cycle"], "right_cycle": b["event"]["cycle"],
                                        "records": records})
                    # These are raw bytes used only for within-request gather reconstruction, not JSON.
                    for side, frame, layer in (("left", a, x), ("right", b, y)):
                        link = projection_link(reader, layer, query_position)
                        if link.get("status") != "compared":
                            coverage.append({"rank": rank, "side": side, "position": query_position,
                                             "reason": "kernel_to_projection_link_unavailable", "detail": link.get("reason")})
                        elif link.get("exact_equal") is False:
                            protocols.append({"rank": rank, "side": side, "position": query_position,
                                              "code": "kernel_output_does_not_match_projection_input", "comparison": link})
                        try:
                            g = gather_inputs(reader, frame, layer, query_position)
                            gathers.append({"side": side, "rank": rank, "position": query_position, "layer": layer_id, **g})
                        except (KeyError, ValueError, OSError) as error:
                            coverage.append({"rank": rank, "position": query_position, "layer": layer_id,
                                             "reason": "gather_mapping_unavailable", "detail": str(error)})
    gather_checks = []
    gather_groups = {}
    for item in gathers:
        gather_groups.setdefault((item["side"], item["layer"], item["position"]), []).append(item)
    for (side, layer, position), items in sorted(gather_groups.items()):
        by_tp = {item["tp_rank"]: item for item in items}
        size = items[0]["size"]
        if set(by_tp) != set(range(size)) or len(items) != size or len({i["dtype"] for i in items}) != 1:
            coverage.append({"position": position, "layer": layer, "side": side, "reason": "all_tp_chunks_not_available"})
            continue
        reconstructed = b"".join(by_tp[r]["local"] for r in range(size))
        for item in items:
            result = bytes_difference(reconstructed, item["global"])
            gather_checks.append({"side": side, "rank": item["rank"], "position": position, "layer": layer, **result})
            if not result["exact_equal"]:
                o_order = next((f["order"] for f in findings if f["rank"] == item["rank"] and f["position"] == position
                                and f["layer"] == layer and f["boundary"] == "o_input_after_tp"), 101)
                findings.append({"rank": item["rank"], "position": position, "layer": layer, "order": o_order - 0.5,
                                 "boundary": "tp_reconstruction", "field": side, "status": "compared", **result})
    findings.sort(key=lambda item: (item["position"], item["layer"], item["order"], item["rank"]))
    first = findings[0] if findings else None
    result_code = ("PROTOCOL_INVARIANT_FAILURE" if protocols else "INCOMPLETE_CAPTURE" if coverage or not seen_contexts
                   else "NONFINITE_VALUES_OBSERVED" if nonfinites else classify_first(first) if first else "NO_DIFFERENCE_IN_CAPTURED_ATTENTION")
    return {"format": "pangu-attention-comparison-v1", "result": result_code, "reference_rank": reference,
            "expected_ranks": expected_ranks, "observed_ranks": sorted(selected),
            "first_token_difference": difference, "token_evidence": {
                "left": {k: v for k, v in (search.get("left_producer") or {}).items() if k in
                         ("token_id", "source", "actual_selected_id", "raw_top2", "processed_top2", "raw_top2_has_tie")},
                "right": {k: v for k, v in (search.get("right_producer") or {}).items() if k in
                          ("token_id", "source", "actual_selected_id", "raw_top2", "processed_top2", "raw_top2_has_tie")}},
            "recorded_prefix_complete_before_token_difference": search.get("recorded_prefix_complete_before_difference"),
            "compared_query_contexts": seen_contexts, "first_boundary": first,
            "findings": findings, "comparisons": comparisons, "state_observations": states,
            "tp_gather_checks": gather_checks, "protocol_issues": protocols, "coverage_issues": coverage,
            "nonfinite_observations": nonfinites,
            "notes": ["The first boundary is an observation, not a proven root cause. Indexer and Q/KV paths can run in parallel.",
                      "Rejected speculative rows and post-divergence contexts are excluded; future/padding differences are not causal evidence.",
                      "HiF8 latent values are compared as bytes, not numerically decoded. Physical cache IDs are ignored.",
                      "Only selected KV operands are captured, not unused cache allocation. Earlier prompt equality is an external assumption.",
                      "MoME history-row mapping is unverified and state differences are reported separately."]}


def parameter_differences(left, right, path="", limit=8):
    """Bounded leaf evidence, so parameter failures need no manual JSON search."""
    if left == right or limit < 1:
        return []
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            result.extend(parameter_differences(left.get(key, "<absent>"), right.get(key, "<absent>"),
                                               path + "." + key, limit - len(result)))
            if len(result) >= limit:
                break
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        result = []
        for index, (a, b) in enumerate(zip(left, right)):
            result.extend(parameter_differences(a, b, path + f"[{index}]", limit - len(result)))
            if len(result) >= limit:
                break
        return result
    return [{"path": path.lstrip("."), "left": left, "right": right}]


def summary(report):
    first = report.get("first_boundary")
    return {"result": report["result"], "first_token_difference": report.get("first_token_difference"),
            "first_boundary": {k: first[k] for k in ("rank", "position", "layer", "boundary", "field",
                                                     "max_abs_finite", "relative_l2_finite", "max_ulp_finite", "different_elements", "different_bytes") if k in first} if first else None,
            "parameter_differences": parameter_differences(first.get("left"), first.get("right")) if first else [],
            "expected_ranks": report.get("expected_ranks"), "observed_ranks": report.get("observed_ranks", []),
            "compared_query_contexts": report.get("compared_query_contexts", 0),
            "coverage_issue_count": len(report.get("coverage_issues", [])),
            "first_coverage_issue": next(iter(report.get("coverage_issues", [])), None),
            "protocol_issue_count": len(report.get("protocol_issues", [])),
            "first_protocol_issue": {k: v for k, v in next(iter(report.get("protocol_issues", [])), {}).items() if k != "comparison"},
            "state_observation_count": len(report.get("state_observations", [])),
            "tp_gather_mismatches": sum(not c["exact_equal"] for c in report.get("tp_gather_checks", [])),
            "nonfinite_observation_count": len(report.get("nonfinite_observations", [])),
            "note": "Send result, first_boundary and evidence line; full JSON contains all metrics and limitations."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--left", default="request-01")
    parser.add_argument("--right", default="request-02")
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("attention-comparison.json"))
    args = parser.parse_args(argv)
    if args.expected_ranks < 1:
        parser.error("--expected-ranks must be positive")
    paths = [args.output, args.output.with_suffix(".summary.json"), args.output.with_suffix(".summary.txt")]
    try:
        if any(path.exists() for path in paths):
            raise FileExistsError("Use new output/summary file names")
        report = json_safe(compare_capture(args.capture_root or capture_root(args.path), load_events(args.path), args.left, args.right, args.expected_ranks))
        compact = summary(report)
        evidence = {key: compact[key] for key in ("coverage_issue_count", "first_coverage_issue", "protocol_issue_count",
                                                 "first_protocol_issue", "observed_ranks", "tp_gather_mismatches",
                                                 "nonfinite_observation_count", "parameter_differences")}
        lines = ["result=" + compact["result"], "first_boundary=" + json.dumps(compact["first_boundary"], ensure_ascii=False),
                 "evidence=" + json.dumps(evidence, ensure_ascii=False),
                 "first_token_difference=" + json.dumps(compact["first_token_difference"], ensure_ascii=False),
                 f"state_observations={compact['state_observation_count']}; tp_gather_mismatches={compact['tp_gather_mismatches']}"]
        for path, contents in zip(paths, (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                         json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "\n".join(lines) + "\n")):
            with path.open("x") as stream:
                stream.write(contents)
        print("\n".join(lines))
        print("reports:", *paths)
    except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
