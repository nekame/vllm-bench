"""Inspect raw prefill ki across requests AND ranks from an existing capture."""
import argparse
import hashlib
import json
from pathlib import Path

from vllm_bench.mtp_debug.attention_compare import Reader
from vllm_bench.mtp_debug.compare_indexer_ki import rows_for, select
from vllm_bench.mtp_debug.ranks import expected_rank_count
from vllm_bench.mtp_debug.tensor_io import compare_bytes


def load_selected(path, left, right, layer):
    """Stream large decode logs; retain only the two relevant prefill requests."""
    frames, topologies = [], []
    for file in sorted(path.rglob("ki-rank-*-pid-*.jsonl")):
        with file.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                # Our sidecar puts scalar header fields before calls/reads.
                # Avoid materializing large, irrelevant decode cache arrays.
                start = line.find('"calls":')
                header = None
                if start >= 0:
                    try:
                        candidate = json.loads(line[:start].rstrip().rstrip(",") + "}")
                        if {"format", "phase", "label", "request_ids", "topology"} <= candidate.keys():
                            header = candidate
                    except ValueError:
                        pass  # Different JSON key order/format: use the full parser.
                frame = header if header is not None else json.loads(line)
                if frame.get("format") != "pangu-indexer-ki-v1":
                    raise ValueError(f"Unexpected Ki frame: {file}:{number}")
                names = [frame.get("label"), *frame.get("request_ids", [])]
                if left not in names and right not in names:
                    continue
                if frame.get("topology"):
                    topologies.append(frame["topology"])
                if frame.get("phase") != "prefill":
                    continue
                if header is not None:
                    frame = json.loads(line)
                frame.pop("reads", None)
                frame["calls"] = [call for call in frame.get("calls", []) if call.get("layer") == layer]
                frames.append(frame)
    if not frames:
        raise ValueError("No selected prefill Ki frames found")
    return frames, topologies


def ranges(positions):
    result = []
    for pos in positions:
        if result and pos == result[-1][1] + 1:
            result[-1][1] = pos
        else:
            result.append([pos, pos])
    return result


def read_prefix(reader, rows):
    positions = sorted(rows)
    if not positions or positions != list(range(positions[-1] + 1)):
        raise ValueError("Prefill positions must be complete from zero")
    raw, dtype, shape = [], None, None
    for pos in positions:
        descriptor, row = rows[pos]["ki_before"]
        content, tail = reader.row(descriptor, row)
        if dtype is not None and (descriptor["dtype"] != dtype or tail != shape):
            raise ValueError("Ki row layout changes within prefill")
        dtype, shape = descriptor["dtype"], tail
        raw.append(content)
    return {"positions": positions, "tokens": [rows[p]["token"] for p in positions],
            "raw": raw, "dtype": dtype, "shape": shape}


def inspect(root, frames, expected, left_name, right_name, layer):
    issues, changed, groups = [], [], {"left": {}, "right": {}}
    reference = None
    for rank in range(expected):
        prefixes = {}
        for side, name in (("left", left_name), ("right", right_name)):
            try:
                selected = select(frames, rank, name)
                if left_name == right_name:
                    raise ValueError("Choose two different requests")
                local_issues = []
                rows = rows_for(selected, local_issues, side).get(layer, {})
                if local_issues:
                    raise ValueError(str(local_issues[0]))
                prefix = read_prefix(Reader(root), rows)
                # Compare only like inputs and like numeric representations across ranks.
                identity = (prefix["positions"], prefix["tokens"], prefix["dtype"], prefix["shape"])
                if reference is None:
                    reference = identity
                elif reference != identity:
                    raise ValueError("Prompt positions/IDs or Ki dtype/shape differ between selected ranks/requests")
                digest = hashlib.sha256(b"".join(prefix["raw"])).hexdigest()
                groups[side].setdefault(digest, []).append(rank)
                prefixes[side] = prefix
            except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
                issues.append({"rank": rank, "side": side, "reason": str(error)})
        if len(prefixes) != 2:
            continue
        a, b = prefixes["left"], prefixes["right"]
        differing = [p for p, x, y in zip(a["positions"], a["raw"], b["raw"]) if x != y]
        if differing:
            i = a["positions"].index(differing[0])
            metrics = compare_bytes(a["raw"][i], b["raw"][i], a["dtype"], a["shape"])
            changed.append({"rank": rank, "different_rows": len(differing), "ranges": ranges(differing),
                            "first": {"position": differing[0], **metrics}})
    first = min(changed, key=lambda c: (c["first"]["position"], c["rank"])) if changed else None
    brief_first = {"rank": first["rank"], **{k: first["first"][k] for k in
                   ("position", "max_abs_finite", "max_ulp_finite", "relative_l2_finite", "nonfinite_elements")}} if first else None
    summary = {"changed_ranks": [c["rank"] for c in changed],
               "different_rows_by_rank": {str(c["rank"]): c["different_rows"] for c in changed},
               "first": brief_first, "first_rank_ranges": first["ranges"][:4] if first else [],
               "first_rank_ranges_complete": len(first["ranges"]) <= 4 if first else None}
    return {"format": "pangu-ki-rank-inspection-v1", "ki": summary,
            "within_request_groups": {side: list(value.values()) for side, value in groups.items()},
            "coverage": {"expected_ranks": expected, "issue_count": len(issues), "first_issue": next(iter(issues), None)},
            "details": changed, "issues": issues,
            "notes": ["Groups contain ranks with exactly equal recorded raw prefill ki for one request.",
                      "A separate group is an observation; it does not identify a faulty device or prove a collective bug.",
                      "Numerical metrics describe the first different row, not the maximum over all changed rows."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Capture directory, or reproduce output containing capture/")
    parser.add_argument("--left", default="request-01")
    parser.add_argument("--right", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expected-ranks", type=int)
    parser.add_argument("--output", type=Path, help="Optional full report; must be new")
    args = parser.parse_args(argv)
    try:
        if args.left == args.right or args.layer < 0:
            raise ValueError("Choose different requests and a nonnegative layer")
        if args.output and args.output.exists():
            raise FileExistsError("Use a new output file")
        root = args.path / "capture" if (args.path / "capture").is_dir() else args.path
        frames, topologies = load_selected(root, args.left, args.right, args.layer)
        report = inspect(root, frames, expected_rank_count(topologies, args.expected_ranks), args.left, args.right, args.layer)
        from vllm_bench.mtp_debug.compare_indexer_prolog import compare as compare_prolog
        prolog = compare_prolog(root, frames, args.left, args.right, report['coverage']['expected_ranks'], args.layer)
        report['prolog'] = prolog
        operator = root.parent / "server" / "operator_config.json"
        if operator.is_file():
            config = json.loads(operator.read_text())
            report["coverage"]["configured_ena_seq_parallel"] = config.get("model_parallel_config", {}).get("ena_seq_parallel", False)
        if args.output:
            with args.output.open("x") as stream:
                stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for key in ("ki", "within_request_groups", "coverage"):
            print(key + "=" + json.dumps(report[key], allow_nan=False))
        if prolog['status'] != 'not_captured':
            print('prolog=' + json.dumps(prolog['summary'], allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
