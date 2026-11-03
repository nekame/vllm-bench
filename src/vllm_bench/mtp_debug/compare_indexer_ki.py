"""Compare captured prefill ki, cache writes and first-decode cache retention."""
import argparse
import json
from pathlib import Path

from vllm_bench.mtp_debug.attention_compare import Reader
from vllm_bench.mtp_debug.attention_layout import bytes_difference
from vllm_bench.mtp_debug.tensor_io import compare_bytes, read_tensor
from vllm_bench.mtp_debug.ranks import expected_rank_count


def load_frames(path):
    files = [path] if path.is_file() else sorted(path.rglob("ki-rank-*-pid-*.jsonl"))
    frames = []
    for file in files:
        for line in file.read_text().splitlines():
            if not line.strip():
                continue
            frame = json.loads(line)
            if frame.get("format") != "pangu-indexer-ki-v1":
                raise ValueError(f"Not an indexer ki capture: {file}")
            frames.append(frame)
    if not frames:
        raise ValueError("No ki captures; start a new server capture with --indexer-ki")
    return frames


def select(frames, rank, name):
    choices = {}
    for frame in frames:
        ids = frame.get("request_ids", [])
        if frame.get("rank") == rank and len(ids) == 1 and (name == frame.get("label") or name == ids[0]):
            key = (frame.get("pid"), ids[0], frame.get("label"))
            choices.setdefault(key, []).append(frame)
    if len(choices) != 1:
        raise ValueError(f"rank={rank} selector={name!r} matches {len(choices)} requests")
    return sorted(next(iter(choices.values())), key=lambda frame: frame["forward_index"])


def rows_for(frames, issues, side, prefill_only=True):
    layers = {}
    for frame in frames:
        if prefill_only and frame["phase"] != "prefill":
            continue
        if frame["status"] != "complete":
            failed = next((call for call in frame.get("calls", []) if call.get("status") != "complete"), {})
            issues.append({"rank": frame["rank"], "side": side, "forward": frame["forward_index"], "reason": "incomplete_capture",
                           "status": frame["status"], "detail": frame.get("reason") or failed.get("reason"),
                           "input_shape": failed.get("input_shape"), "forward_row_counts": failed.get("forward_row_counts")})
        for call in frame.get("calls", []):
            if call["status"] != "complete":
                continue
            result = layers.setdefault(call["layer"], {})
            active = call["active_rows"]
            if len(active) != len(call["active_positions"]):
                raise ValueError("Invalid active ki row mapping")
            for output_row, row in enumerate(active):
                position = call["row_positions"][row]
                item = {"token": call["row_input_ids"][row], "frame": frame, "call": call,
                    "ki_before": (call["ki_before"], row), "ki_after": (call["ki_after"], row),
                    "key_after": (call["key_after"], output_row), "scales_after": (call["scales_after"], output_row)}
                if prefill_only and position in result:
                    issues.append({"rank": frame["rank"], "side": side, "position": position, "reason": "duplicate_prefill_position"})
                result[position] = item
    return layers


def compare_rows(reader, left, right, left_field, right_field=None):
    right_field = right_field or left_field
    common = sorted(left.keys() & right.keys())
    result = {"field": left_field, "status": "compared", "compared_rows": len(common), "exact_equal": True,
              "different_rows": 0, "different_bytes": 0, "sampled_positions": [], "first": None}
    if not common or left.keys() != right.keys():
        result.update(status="incomplete_position_coverage", exact_equal=None)
        return result
    for position in common:
        a, ar = left[position][left_field]
        b, br = right[position][right_field]
        try:
            raw_a, shape_a = reader.row(a, ar)
            raw_b, shape_b = reader.row(b, br)
            opaque = a.get("comparison_mode") == "bytes_only" or b.get("comparison_mode") == "bytes_only"
            # QLIv2 views the same four scale bytes as [1,4] at write and
            # [1,2,2] at read. Reshaping does not change their byte order.
            if a["dtype"] != b["dtype"] or len(raw_a) != len(raw_b) or not opaque and shape_a != shape_b:
                raise ValueError("Different row dtype/shape")
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="unavailable", exact_equal=None, reason=str(error), position=position)
            return result
        if raw_a != raw_b:
            result["exact_equal"] = False
            result["different_rows"] += 1
            result["different_bytes"] += sum(x != y for x, y in zip(raw_a, raw_b))
            if len(result["sampled_positions"]) < 16:
                result["sampled_positions"].append(position)
            if result["first"] is None:
                metrics = bytes_difference(raw_a, raw_b) if opaque else compare_bytes(raw_a, raw_b, a["dtype"], shape_a)
                result["first"] = {"position": position, **metrics}
    result["all_different_positions_known"] = result["different_rows"] <= 16
    return result


def retention(reader, frames, prefill, layer, side, issues):
    decodes = [frame for frame in frames if frame["phase"] == "decode"]
    if not decodes:
        issues.append({"side": side, "layer": layer, "reason": "first_decode_missing"})
        return []
    frame = decodes[0]
    reads = [read for read in frame.get("reads", []) if read["layer"] == layer]
    if frame["status"] != "complete" or len(reads) != 1 or reads[0].get("status") != "complete":
        issues.append({"side": side, "layer": layer, "reason": "first_decode_read_or_write_incomplete"})
        return []
    # Account for writes within the first decode before taking the read snapshot.
    current = rows_for([frame], issues, side, prefill_only=False).get(layer, {})
    expected = {position: current.get(position, item) for position, item in prefill.items()}
    read = reads[0]
    observed = {}
    for field, write_field in (("key_rows", "key_after"), ("key_scales", "scales_after")):
        descriptor = read.get(field) or {}
        positions = descriptor.get("logical_positions", [])
        if len(set(positions)) != len(positions):
            issues.append({"side": side, "layer": layer, "reason": "ambiguous_read_positions"})
            return []
        for i, position in enumerate(positions):
            if position in expected:
                observed.setdefault(position, {})[write_field] = (descriptor, i)
    results = []
    for field in ("key_after", "scales_after"):
        if any(field not in item for item in observed.values()):
            issues.append({"side": side, "layer": layer, "reason": "missing_read_operand", "field": field})
            continue
        result = compare_rows(reader, expected, observed, field)
        results.append({"side": side, "layer": layer, "boundary": "cache_retention", **result})
    return results


def compare(root, frames, left_name, right_name, expected_ranks=None):
    expected_ranks = expected_rank_count((frame.get("topology") for frame in frames), expected_ranks)
    issues, records, candidates, pairs = [], [], [], {}
    replay = None
    for rank in range(expected_ranks):
        try:
            left, right = select(frames, rank, left_name), select(frames, rank, right_name)
            if left is right or (left[0]["pid"], left[0]["request_ids"], left[0].get("label")) == (right[0]["pid"], right[0]["request_ids"], right[0].get("label")):
                raise ValueError("Same request selected twice")
            la, lb = rows_for(left, issues, "left"), rows_for(right, issues, "right")
        except (ValueError, KeyError, TypeError, IndexError) as error:
            issues.append({"rank": rank, "reason": "request_or_prefill_unavailable", "detail": str(error)})
            continue
        reader = Reader(root)  # Bound exact payload cache to one rank.
        if not la or not lb:
            issues.append({"rank": rank, "reason": "no_complete_prefill_ki"})
        for layer in sorted(la.keys() | lb.keys()):
            a, b = la.get(layer, {}), lb.get(layer, {})
            positions = sorted(a.keys() & b.keys())
            if (not positions or a.keys() != b.keys() or positions != list(range(positions[-1] + 1))):
                issues.append({"rank": rank, "layer": layer, "reason": "prefill_prefix_not_complete_from_zero"})
                continue
            if any(a[p]["token"] != b[p]["token"] for p in positions):
                issues.append({"rank": rank, "layer": layer, "reason": "prompt_input_ids_differ"})
                continue
            if replay is None:
                replay = (a[positions[0]], b[positions[0]])
            pairs[(rank, layer)] = (a, b)
            if any(a[p]["call"].get("parameters") != b[p]["call"].get("parameters") for p in positions):
                issues.append({"rank": rank, "layer": layer, "reason": "quant_cache_parameters_differ"})
            for field in ("ki_before", "key_after", "scales_after"):
                result = {"rank": rank, "layer": layer, "boundary": "prefill", **compare_rows(reader, a, b, field)}
                records.append(result)
                if result.get("exact_equal") is False:
                    candidates.append(result)
            for side, own, rows in (("left", left, a), ("right", right, b)):
                # Mutation is a within-call invariant; no cross-request conditioning assumption.
                for frame in own:
                    for call in frame.get("calls", []):
                        if call.get("layer") != layer or call.get("status") != "complete":
                            continue
                        try:
                            mutated = reader.read(call["ki_before"]) != reader.read(call["ki_after"])
                            if mutated:
                                candidates.append({"rank": rank, "layer": layer, "side": side, "field": "ki_after",
                                    "boundary": "input_mutated", "first": {"forward": frame["forward_index"]}})
                        except (ValueError, OSError, KeyError) as error:
                            issues.append({"rank": rank, "layer": layer, "reason": "ki_after_unavailable", "detail": str(error)})
                results = retention(reader, own, rows, layer, side, issues)
                records.extend({"rank": rank, **result} for result in results)
                candidates.extend({"rank": rank, **result} for result in results if result.get("exact_equal") is False)
    issues.extend({"rank": r.get("rank"), "layer": r.get("layer"), "field": r["field"], "reason": r["status"],
                   "detail": r.get("reason")} for r in records if r["status"] != "compared")
    priority = lambda r: (0 if r["boundary"] == "input_mutated" else 1 if r["field"] == "ki_before" else
                         2 if r["boundary"] == "prefill" else 3, r.get("first", {}).get("position", -1), r["rank"], r["layer"])
    first = min(candidates, key=priority) if candidates else None
    if first:
        a, b = pairs.get((first["rank"], first["layer"]), ({}, {}))
        position = first.get("first", {}).get("position")
        if position not in a or position not in b:
            position = min(a.keys() & b.keys(), default=None)
        if position in a and position in b:
            replay = a[position], b[position]
    raw = [r for r in records if r["field"] == "ki_before"]
    raw_equal = (False if any(r.get("exact_equal") is False for r in raw) else
                 True if raw and not issues and all(r.get("exact_equal") is True for r in raw) else None)
    codes = {"input_mutated": "KI_MUTATED_BY_CACHE_CALL", "cache_retention": "CACHE_CHANGED_AFTER_WRITE"}
    result = ("INCOMPLETE_CAPTURE" if issues or not raw else
              codes.get(first["boundary"], "KI_INPUT_DIFFERS" if first["field"] == "ki_before" else
                        "CACHE_OUTPUT_DIFFERS_WITH_EQUAL_RECORDED_KI") if first else "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI")
    report = {"format": "pangu-indexer-ki-comparison-v1", "result": result, "raw_ki_exact_equal": raw_equal,
              "expected_ranks": expected_ranks, "first_difference": first, "comparisons": records, "coverage_issues": issues,
              "notes": ["Cross-request raw ki comparison covers recorded prefill, with complete equal input IDs from position zero.",
                        "Chunk boundaries may differ; exact positions are aligned. Different ki already localizes upstream of cache quantization.",
                        "Cache retention compares recorded prefill writes with the first decode read, accounting for captured decode writes.",
                        "No root-cause claim or numerical quantization reference; physical slots/layouts may differ between requests.",
                        "Exact CPU copies synchronize streams and can change reproducibility."]}
    return report, replay


def export_replays(root, output, pair):
    payloads = [read_tensor(root, item["call"]["ki_before"]) for item in pair]
    for side, item, raw in zip(("left", "right"), pair, payloads):
        directory = output / side
        directory.mkdir(parents=True)
        descriptor = item["call"]["ki_before"]
        with (directory / "ki.bin").open("xb") as stream:
            stream.write(raw)
        document = {"format": "pangu-indexer-quant-input-v1", "ki": {**descriptor, "path": "ki.bin"},
                    "capture": {"rank": item["frame"]["rank"], "request_ids": item["frame"]["request_ids"],
                                "forward": item["frame"]["forward_index"], "layer": item["call"]["layer"],
                                "phase": item["frame"]["phase"], "parameters": item["call"].get("parameters"),
                                "cache_layout": item["call"].get("cache_layout"),
                                "row_positions": item["call"]["row_positions"], "slots": item["call"]["slots"]}}
        (directory / "input.json").write_text(json.dumps(document, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--left", default="request-01")
    parser.add_argument("--right", default="request-02")
    parser.add_argument("--expected-ranks", type=int, help="Default: captured TP size, legacy captures: 8")
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("ki-comparison.json"))
    args = parser.parse_args(argv)
    outputs = [args.output, args.output.with_suffix(".summary.json"), args.output.with_suffix(".summary.txt")]
    replays = args.output.parent / (args.output.stem + "-replays")
    try:
        if args.expected_ranks is not None and args.expected_ranks < 1:
            raise ValueError("--expected-ranks must be positive")
        if any(path.exists() for path in [*outputs, replays]):
            raise FileExistsError("Use new report/replay paths")
        root = args.capture_root or (args.path if args.path.is_dir() else args.path.parent)
        report, pair = compare(root, load_frames(args.path), args.left, args.right, args.expected_ranks)
        if pair:
            try:
                export_replays(root, replays, pair)
                report["replays"] = str(replays)
            except (OSError, ValueError, KeyError) as error:
                report["coverage_issues"].append({"reason": "replay_export_unavailable", "detail": str(error)})
                report["result"] = "INCOMPLETE_CAPTURE"
        first = report["first_difference"]
        brief = {k: first[k] for k in ("rank", "layer", "side", "boundary", "field", "different_rows", "different_bytes") if k in first} if first else None
        if first and first.get("first"):
            brief.update({k: v for k, v in first["first"].items() if k in ("position", "forward", "max_abs_finite", "max_ulp_finite")})
        summary = {"result": report["result"], "raw_ki_exact_equal": report["raw_ki_exact_equal"],
                   "first_difference": brief, "coverage_issue_count": len(report["coverage_issues"]),
                   "first_coverage_issue": next(iter(report["coverage_issues"]), None), "replays": report.get("replays")}
        lines = ["result=" + summary["result"], "first_difference=" + json.dumps(brief),
                 "evidence=" + json.dumps({k: summary[k] for k in ("raw_ki_exact_equal", "coverage_issue_count", "first_coverage_issue")})]
        for path, content in zip(outputs, (json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          json.dumps(summary, indent=2, allow_nan=False) + "\n", "\n".join(lines) + "\n")):
            with path.open("x") as stream:
                stream.write(content)
        print("\n".join(lines))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
