"""Extract indexer operand comparisons from a full attention-comparison JSON."""
import argparse
import json
from pathlib import Path


FIELDS = {"query": ("query",), "query_dequant_scale": ("query_dequant_scale", "query_scale"),
          "weights": ("weights",), "key_scales": ("key_scales",)}


def extract(report, rank=None, position=None, layer=None):
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
    for name, aliases in FIELDS.items():
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
        result["fields"][name] = evidence
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Full attention-comparison.json")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--layer", type=int)
    args = parser.parse_args(argv)
    try:
        report = extract(json.loads(args.path.read_text()), args.rank, args.position, args.layer)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print("context=" + json.dumps({k: v for k, v in report.items() if k != "fields"}, ensure_ascii=False))
    for name, evidence in report["fields"].items():
        print(name + "=" + json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
