"""Compare target traces on CPU; no inference runtime or NPU is needed."""

import argparse
import json
import math
from pathlib import Path

from vllm_bench.pangu_infer.run import write_json
from vllm_bench.pangu_infer.target_trace import read_manifest


def metadata_differences(reference, candidate, path="", limit=32):
    """Keep bounded, explicit paths to different logical metadata fields."""
    result = []
    if isinstance(reference, dict) and isinstance(candidate, dict):
        for key in sorted(reference.keys() | candidate.keys()):
            field = f"{path}.{key}" if path else key
            if key not in reference or key not in candidate:
                result.append({"path": field, "reference": reference.get(key),
                               "candidate": candidate.get(key), "missing": True})
            else:
                result.extend(metadata_differences(reference[key], candidate[key], field, limit))
            if len(result) >= limit:
                break
    elif reference != candidate:
        result.append({"path": path, "reference": reference, "candidate": candidate})
    return result[:limit]


def tensor_metrics(reference, candidate, rtol, atol):
    import torch

    result = {"reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape),
              "reference_dtype": str(reference.dtype), "candidate_dtype": str(candidate.dtype)}
    if reference.shape != candidate.shape:
        return {**result, "exact": False, "close": False, "max_abs": None, "rmse": None}
    a, b = reference.float(), candidate.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    difference = (a[finite] - b[finite]).abs()
    same_nonfinite = (a == b) | finite
    return {**result, "exact": torch.equal(reference, candidate),
            "close": torch.allclose(a, b, rtol=rtol, atol=atol),
            "nonfinite_mismatches": int((~same_nonfinite).sum().item()),
            "max_abs": float(difference.max().item()) if difference.numel() else None,
            "rmse": float(difference.square().mean().sqrt().item()) if difference.numel() else None}


def logits_metrics(reference, candidate, rtol, atol):
    import torch

    result = tensor_metrics(reference, candidate, rtol, atol)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape != candidate.shape:
        return {**result, "argmax_equal": False, "rows": [], "first_argmax_difference": None}
    rows = []
    first = None
    for row in range(reference.shape[0]):
        a, b = reference[row].float(), candidate[row].float()
        aid, bid = int(a.argmax().item()), int(b.argmax().item())
        if aid != bid and first is None:
            first = row
        entry = {"row": row, "reference_top1": aid, "candidate_top1": bid,
                 "reference_at_candidate_top1": float(a[bid].item()),
                 "candidate_at_reference_top1": float(b[aid].item())}
        for label, logits in (("reference", a), ("candidate", b)):
            values, indices = logits.topk(min(5, logits.numel()))
            entry[f"{label}_top5"] = [{"token_id": int(token), "logit": float(value)}
                                      for token, value in zip(indices.tolist(), values.tolist())]
            entry[f"{label}_margin"] = float((values[0] - values[1]).item()) if values.numel() > 1 else None
            entry[f"{label}_nonfinite"] = int((~torch.isfinite(logits)).sum().item())
        rows.append(entry)
    return {**result, "argmax_equal": first is None, "first_argmax_difference": first, "rows": rows}


def nested_tensors(reference, candidate, rtol, atol, path=""):
    """Candidate may request only a subset of the reference's recorded layers."""
    import torch

    result = {}
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            field = f"{path}.{key}" if path else key
            if not isinstance(reference, dict) or key not in reference:
                result[field] = {"exact": False, "close": False, "missing_reference": True}
            else:
                result.update(nested_tensors(reference[key], value, rtol, atol, field))
    elif isinstance(candidate, list):
        if not isinstance(reference, list) or len(reference) != len(candidate):
            result[path] = {"exact": False, "close": False, "shape_mismatch": True}
        else:
            for index, value in enumerate(candidate):
                result.update(nested_tensors(reference[index], value, rtol, atol, f"{path}.{index}"))
    elif isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor):
        result[path] = tensor_metrics(reference, candidate, rtol, atol)
    else:
        raise ValueError(f"Unexpected layer/state trace value at {path}")
    return result


def compare_traces(reference, candidate, *, rtol=1e-3, atol=1e-3):
    import torch

    reference, rm = read_manifest(reference)
    candidate, cm = read_manifest(candidate)
    rp, cp = rm["plan"], cm["plan"]
    if rp.get("hf_overrides", {}) != cp.get("hf_overrides", {}):
        raise ValueError("Trace workloads differ in hf_overrides")
    for field in ("config", "operator_config", "additional_config", "checkpoint_identity",
                  "prompt_sha256", "max_model_len"):
        if rp[field] != cp[field]:
            raise ValueError(f"Trace workloads differ in {field}")
    for field in ("tp", "depth", "mode", "temperature"):
        if rp["options"][field] != cp["options"][field]:
            raise ValueError(f"Trace workloads differ in {field}")
    for field in ("hardware", "versions"):
        if rm[field] != cm[field]:
            raise ValueError(f"Trace environments differ in {field}")
    for field in ("operator", "parallel", "block_size", "mamba_block_size"):
        if rm["effective"].get(field) != cm["effective"].get(field):
            raise ValueError(f"Effective trace configurations differ in {field}")
    report = {"format": "pangu-target-trace-comparison-v1", "measurement": "diagnostic",
              "reference": str(reference), "candidate": str(candidate),
              "rtol": rtol, "atol": atol, "calls": []}
    first = {"inputs": None, "metadata": None, "logits": None, "argmax": None,
             "layer_output": None, "state_before": None, "state_after": None}
    counts = []
    first_draft = None
    for rank in range(rp["options"]["tp"]):
        rd, cd = reference / f"rank-{rank}", candidate / f"rank-{rank}"
        rc = json.loads((rd / "index.json").read_text())["calls"]
        cc = json.loads((cd / "index.json").read_text())["calls"]
        if rc != cc:
            raise ValueError(f"Trace call counts differ on rank {rank}: {rc} != {cc}")
        counts.append(rc)
        if cp["options"].get("replay_with_mtp", False):
            proposals = json.loads((cd / "draft-proposals.json").read_text())
            if len(proposals) != rc - 1:
                raise ValueError(f"Incomplete draft replay diagnostics on rank {rank}")
            for proposal in proposals:
                row = proposal["first_difference"]
                if row is not None:
                    location = {"target_call": proposal["target_call"], "rank": rank, "row": row,
                                "reference": proposal["reference"][row], "candidate": proposal["candidate"][row]}
                    if first_draft is None or (location["target_call"], rank) < (first_draft["target_call"], first_draft["rank"]):
                        first_draft = location
        for call in range(rc):
            a = torch.load(rd / f"call-{call:05d}.pt", map_location="cpu", weights_only=True)
            b = torch.load(cd / f"call-{call:05d}.pt", map_location="cpu", weights_only=True)
            inputs_equal = (torch.equal(a["input_ids"], b["input_ids"])
                            and torch.equal(a["positions"], b["positions"]))
            metadata = metadata_differences(a["metadata"], b["metadata"])
            logits = logits_metrics(a["logits"], b["logits"], rtol, atol)
            layers = nested_tensors(a["layers"], b["layers"], rtol, atol)
            before = nested_tensors(a["states_before"], b["states_before"], rtol, atol)
            after = nested_tensors(a["states_after"], b["states_after"], rtol, atol)
            entry = {"rank": rank, "target_call": call, "phase": "prefill" if call == 0 else "decode",
                     "input_tokens": int(a["input_ids"].numel()), "inputs_equal": inputs_equal,
                     "metadata_equal": not metadata, "metadata_differences": metadata,
                     "logits": logits, "layer_outputs": layers,
                     "states_before": before, "states_after": after}
            report["calls"].append(entry)
            differences = {"inputs": not inputs_equal, "metadata": bool(metadata),
                           "logits": not logits["exact"], "argmax": not logits["argmax_equal"],
                           "layer_output": any(not item["exact"] for item in layers.values()),
                           "state_before": any(not item["exact"] for item in before.values()),
                           "state_after": any(not item["exact"] for item in after.values())}
            for field, differs in differences.items():
                location = {"target_call": call, "rank": rank}
                if field == "argmax":
                    location["row"] = logits["first_argmax_difference"]
                if field in ("layer_output", "state_before", "state_after") and differs:
                    metrics = {"layer_output": layers, "state_before": before, "state_after": after}[field]
                    location["tensor"] = next(name for name, value in metrics.items() if not value["exact"])
                if differs and (first[field] is None or (call, rank) < (
                        first[field]["target_call"], first[field]["rank"])):
                    first[field] = location
    if not counts or min(counts) < 2 or len(set(counts)) != 1:
        raise ValueError("Traces require equal non-empty call counts on all ranks, including decode")
    report["summary"] = {"calls_per_rank": counts[0], "tp": rp["options"]["tp"],
                         "inputs_equal": first["inputs"] is None,
                         "metadata_equal": first["metadata"] is None,
                         "argmax_equal": first["argmax"] is None,
                         "captured_layers": cp["trace_layers"], "first_differences": first,
                         "replay_with_mtp": cp["options"].get("replay_with_mtp", False),
                         "mtp_pad_to_8": cp["options"].get("mtp_pad_to_8", False),
                         "first_draft_proposal_difference": first_draft,
                         "interpretation": "Target inputs and target acceptance are forced from the source trace. "
                         "Exact differences alone do not establish a correctness bug. Inspect top-1 margins, "
                         "logical metadata and selected layer/state metrics."}
    if first["metadata"] is not None:
        location = first["metadata"]
        entry = next(item for item in report["calls"]
                     if item["target_call"] == location["target_call"] and item["rank"] == location["rank"])
        report["summary"]["first_metadata_detail"] = entry["metadata_differences"][:4]
    if first["argmax"] is not None:
        location = first["argmax"]
        entry = next(item for item in report["calls"]
                     if item["target_call"] == location["target_call"] and item["rank"] == location["rank"])
        row = location["row"]
        report["summary"]["first_argmax_detail"] = (
            entry["logits"]["rows"][row] if row is not None else {"shape_mismatch": True})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--rtol", type=float, default=1e-3)
    p.add_argument("--atol", type=float, default=1e-3)
    a = p.parse_args(argv)
    try:
        if not math.isfinite(a.rtol) or not math.isfinite(a.atol) or a.rtol < 0 or a.atol < 0:
            raise ValueError("Tolerances must be finite and non-negative")
        report = compare_traces(a.reference, a.candidate, rtol=a.rtol, atol=a.atol)
        if a.output:
            if a.output.exists():
                raise ValueError("Output exists")
            write_json(a.output, report)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        return 0 if report["summary"]["argmax_equal"] and report["summary"]["metadata_equal"] and report["summary"]["inputs_equal"] else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
