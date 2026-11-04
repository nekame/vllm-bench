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


def difference_summary(report, rank=None, position=None, layer=None):
    selected = extract(report, rank, position, layer)
    if not isinstance(report.get("findings"), list):
        raise ValueError("Use the full attention-comparison JSON with findings")
    comparison = next(item for item in report["comparisons"] if all(
        item.get(key) == selected[key] for key in ("rank", "position", "layer")))
    records = comparison.get("records", [])

    def find(boundary, aliases):
        matches = [r for r in records if r.get("boundary") == boundary and r.get("field") in aliases]
        return matches[0] if len(matches) == 1 else {"status": "missing" if not matches else "ambiguous"}

    metadata = find("indexer_kernel_input", ("metadata",))
    header = {key: selected[key] for key in ("rank", "position", "layer")}
    header.update({key: metadata[key] for key in ("status", "exact_equal", "different_elements", "different_bytes") if key in metadata})
    header["first_differences"] = metadata.get("first_differences", [])[:3]
    header["coverage_issue_count"] = len(report.get("coverage_issues", []))
    header["role"] = "raw_operator_metadata_semantics_unverified"
    fields = {
        "parameters": ("indexer_kernel_parameters", ("parameters",)),
        "query": ("indexer_kernel_input", ("query",)),
        "query_scale": ("indexer_kernel_input", ("query_dequant_scale", "query_scale")),
        "weights": ("indexer_kernel_input", ("weights",)),
        "key_rows": ("indexer_kernel_input", ("key_rows",)),
        "key_scales": ("indexer_kernel_input", ("key_scales",)),
        "query_lengths": ("indexer_kernel_input", ("actual_seq_lengths_query", "cu_seqlens_q")),
        "kv_lengths": ("indexer_kernel_input", ("actual_seq_lengths_key", "seqused_k")),
        "indexer_indices": ("indexer_kernel_output", ("indices",)),
        "attention_indices": ("attention_indices", ("sparse_indices",)),
        "attention_output": ("attention_kernel_output", ("output",)),
        "projection_output": ("projection_output", ("output",)),
        "o_input": ("o_input_after_tp", ("x_before",)),
    }
    operands, unavailable = {}, {}
    for name, (boundary, aliases) in fields.items():
        record = find(boundary, aliases)
        complete = record.get("status") == "compared" and record.get("coverage_complete", True)
        operands[name] = record.get("exact_equal") if complete else None
        if operands[name] is None:
            unavailable[name] = record.get("reason", record.get("status"))
        if name in ("indexer_indices", "attention_indices") and operands[name] is not None:
            operands[name + "_same_valid_set"] = True if operands[name] else record.get("same_valid_set")
    if unavailable:
        operands["unavailable"] = unavailable
    # This is an additional view: keep metadata in the original report and
    # never claim equal kernel arguments merely by excluding this operand.
    other = next((r for r in report["findings"] if r.get("exact_equal") is False and not (
        r.get("boundary") == "indexer_kernel_input" and r.get("field") == "metadata")), None)
    brief = {key: other[key] for key in ("rank", "position", "layer", "boundary", "field", "different_elements",
                                       "different_bytes", "max_abs_finite", "max_ulp_finite") if key in other} if other else None
    return {"metadata": header, "operands_equal": operands, "first_non_metadata_difference": brief}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Full attention-comparison.json")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--layer", type=int)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--history-summary", action="store_true", help="Three short lines: operator context and key/scales logical positions")
    modes.add_argument("--difference-summary", action="store_true", help="Metadata sample, operand equality and first difference outside QLI metadata")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.path.read_text())
        if args.difference_summary:
            report = difference_summary(document, args.rank, args.position, args.layer)
            for name, value in report.items():
                print(name + "=" + json.dumps(value, ensure_ascii=False))
            return
        report = extract(document, args.rank, args.position, args.layer, args.history_summary)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print("context=" + json.dumps({k: v for k, v in report.items() if k != "fields"}, ensure_ascii=False))
    for name, evidence in report["fields"].items():
        print(name + "=" + json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
