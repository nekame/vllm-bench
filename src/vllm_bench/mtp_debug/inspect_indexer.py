"""Extract indexer operand comparisons from a full attention-comparison JSON."""
import argparse
import json
from pathlib import Path


FIELDS = {"query": ("query",), "query_dequant_scale": ("query_dequant_scale", "query_scale"),
          "weights": ("weights",), "key_scales": ("key_scales",)}


def history_locations(record):
    positions = record.get("compared_positions", [])
    size = record.get("left_nbytes")
    if (not positions or type(size) is not int or size <= 0 or size % len(positions)
            or record.get("right_nbytes") != size):
        return {"mapping_status": "unavailable", "reason": "equal byte sizes and compared positions required"}
    width = size // len(positions)
    offsets = record.get("first_different_offsets", [])
    if any(type(offset) is not int or not 0 <= offset < size for offset in offsets):
        return {"mapping_status": "unavailable", "reason": "byte offset outside compared payload"}
    return {"mapping_status": "mapped", "bytes_per_position": width,
            "sampled_different_positions": sorted({positions[offset // width] for offset in offsets}),
            "all_different_positions_known": len(offsets) == record.get("different_bytes"),
            "first_different_position": positions[offsets[0] // width] if offsets else None,
            "first_byte_within_position": offsets[0] % width if offsets else None}


def extract(report, rank=None, position=None, layer=None, history_summary=False):
    first = report.get("first_boundary") or {}
    selected = {name: value if value is not None else first.get(name)
                for name, value in (("rank", rank), ("position", position), ("layer", layer))}
    if any(value is None for value in selected.values()):
        raise ValueError("No first_boundary selector; supply --rank, --position and --layer")
    if not isinstance(report.get("comparisons"), list):
        raise ValueError("Use the full attention-comparison.json, not its summary")
    matches = [item for item in report["comparisons"] if all(item.get(k) == v for k, v in selected.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one comparison for {selected}; found {len(matches)}")
    comparison = matches[0]
    result = {**selected, "left_cycle": comparison.get("left_cycle"), "right_cycle": comparison.get("right_cycle"), "fields": {}}
    parameters = [r for r in comparison.get("records", []) if r.get("boundary") == "indexer_kernel_parameters" and r.get("field") == "parameters"]
    if len(parameters) == 1:
        result["indexer_parameters"] = {side: {key: parameters[0].get(side, {}).get(key) for key in ("operator", "scalars")}
                                        for side in ("left", "right")}
    fields = {"key_rows": ("key_rows",), "key_scales": ("key_scales",)} if history_summary else FIELDS
    for name, aliases in fields.items():
        records = [r for r in comparison.get("records", [])
                   if r.get("boundary") == "indexer_kernel_input" and r.get("field") in aliases]
        if len(records) != 1:
            result["fields"][name] = {"status": "missing" if not records else "ambiguous", "exact_equal": None}
            continue
        record = records[0]
        # Retain diagnostic metrics and reasons, without large history position arrays.
        evidence = {k: record[k] for k in ("field", "status", "reason", "coverage_complete", "different_elements",
                    "different_bytes", "max_abs_finite", "relative_l2_finite", "max_ulp_finite", "nonfinite_elements",
                    "first_differences", "first_different_offsets") if k in record}
        evidence["exact_equal"] = record.get("exact_equal") if record.get("status") == "compared" else None
        if name in ("key_rows", "key_scales") and "different_bytes" in record:
            evidence.update(history_locations(record))
        if history_summary:
            evidence.pop("first_different_offsets", None)
            evidence.pop("first_differences", None)
        result["fields"][name] = evidence
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Full attention-comparison.json")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--history-summary", action="store_true", help="Three short lines: operator context and key/scales logical positions")
    args = parser.parse_args(argv)
    try:
        report = extract(json.loads(args.path.read_text()), args.rank, args.position, args.layer, args.history_summary)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print("context=" + json.dumps({k: v for k, v in report.items() if k != "fields"}, ensure_ascii=False))
    for name, evidence in report["fields"].items():
        print(name + "=" + json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
