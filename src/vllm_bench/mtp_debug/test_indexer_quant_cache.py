"""Standalone A5 indexer_quant_cache repeatability and layout tests; no model/vLLM."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback

from vllm_bench.mtp_debug.attention_layout import bytes_difference
from vllm_bench.mtp_debug.tensor_io import TensorWriter, read_tensor, tensor_bytes

CASES = ("shared", "contiguous", "reuse", "poison", "permuted", "shifted", "chunked")
WIDTHS = {"key": 128, "scales": 4}


def layout(rows, block_size, contiguous=False, page_bytes=None):
    pages = math.ceil(rows / block_size) + 3  # Guard pages, including a shifted-page tail.
    stride = page_bytes if page_bytes is not None else block_size * 788  # 656 + 128 + 4.
    if stride < block_size * 788:
        raise ValueError("--page-bytes must hold block_size * 788 bytes")
    result = {"pages": pages, "block_size": block_size, "buffers": {}}
    for name, width in WIDTHS.items():
        buffer = name if contiguous else "raw"
        result["buffers"][buffer] = pages * (block_size * width if contiguous else stride)
        result[name] = {"buffer": buffer, "offset": 0 if contiguous else block_size * (656 if name == "key" else 784),
                        "page_stride": block_size * width if contiguous else stride, "row_stride": width}
    return result


def slots_for(case, rows, spec, seed):
    pages = list(range(1, spec["pages"] - 1))
    if case == "permuted":
        random.Random(seed).shuffle(pages)
    offset = spec["block_size"] - 1 if case == "shifted" else 0
    return [pages[(i + offset) // spec["block_size"]] * spec["block_size"]
            + (i + offset) % spec["block_size"] for i in range(rows)]


def address(spec, name, slot):
    page, row = divmod(slot, spec["block_size"])
    item = spec[name]
    return item["buffer"], item["offset"] + page * item["page_stride"] + row * item["row_stride"]


def write_ranges(spec, slots):
    ranges = {name: [] for name in spec["buffers"]}
    for name, width in WIDTHS.items():
        for slot in slots:
            buffer, start = address(spec, name, slot)
            ranges[buffer].append((start, start + width))
    merged = {}
    for buffer, entries in ranges.items():
        output = []
        for start, end in sorted(entries):
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(end, output[-1][1]))
            else:
                output.append((start, end))
        merged[buffer] = output
    return merged


def snapshot_evidence(snapshot, spec, slots, ranges, fill):
    logical, violations = {}, []
    for name, width in WIDTHS.items():
        parts = []
        for slot in slots:
            buffer, start = address(spec, name, slot)
            parts.append(snapshot[buffer][start:start + width])
        logical[name] = b"".join(parts)
    for buffer, size in spec["buffers"].items():
        raw = snapshot[buffer]
        if len(raw) != size:
            raise ValueError("Snapshot size mismatch")
        cursor = 0
        for start, end in [*ranges[buffer], (size, size)]:
            gap = raw[cursor:start]
            if gap != bytes([fill]) * len(gap):
                first = next(i for i, value in enumerate(gap) if value != fill)
                violations.append({"buffer": buffer, "first_offset": cursor + first,
                                   "expected_byte": fill, "actual_byte": gap[first]})
            cursor = end
    return logical, violations


def difference(left, right, field):
    result = bytes_difference(left, right)
    if not result["exact_equal"]:
        width = WIDTHS[field]
        result["sampled_logical_rows"] = sorted({offset // width for offset in result["first_different_offsets"]})
        result["all_different_rows_known"] = result["different_bytes"] == len(result["first_different_offsets"])
    return result


def experiment(plan, backend, output):
    """Backend seam keeps the trial/guard/classification logic CPU-testable."""
    output = Path(output)
    report = {"format": "pangu-indexer-quant-test-v1", "plan": plan, "runtime": backend.info,
              "result": "PASS_IN_TESTED_CASES", "first_failure": None, "cases": [], "failure_counts": {}}
    reference = None

    def failure(code, case, trial, evidence, actual=None):
        report["failure_counts"][code] = report["failure_counts"].get(code, 0) + 1
        if report["first_failure"] is None:
            report["first_failure"] = {"code": code, "case": case, "trial": trial, **evidence}
            if actual is not None and reference is not None:
                directory = output / "first-failure"
                directory.mkdir()
                for name in WIDTHS:
                    (directory / f"reference-{name}.bin").write_bytes(reference[name])
                    (directory / f"actual-{name}.bin").write_bytes(actual[name])

    for case in plan["cases"]:
        spec = layout(plan["rows"], plan["block_size"], case == "contiguous", plan["page_bytes"])
        slots = slots_for(case, plan["rows"], spec, plan["seed"])
        ranges = write_ranges(spec, slots)
        backend.bind_slots(slots)
        cache, case_reference = None, None
        item = {"case": case, "layout": spec, "slots_sha256": hashlib.sha256(json.dumps(slots).encode()).hexdigest(),
                "trials": []}
        report["cases"].append(item)
        for trial in range(plan["repeats"]):
            fill = (0, 165, 255)[trial % 3] if case == "poison" else 93
            if cache is None or case != "reuse":
                cache = backend.allocate(spec, fill)
            chunk = plan["chunk_size"] if case == "chunked" else plan["rows"]
            for start in range(0, plan["rows"], chunk):
                backend.invoke(cache, start, min(start + chunk, plan["rows"]))
                if plan["sync_each_call"]:
                    backend.synchronize()
            backend.synchronize()
            logical, guards = snapshot_evidence(backend.snapshot(cache), spec, slots, ranges, fill)
            unchanged = backend.input_unchanged()
            if reference is None:
                reference = logical
            if case_reference is None:
                case_reference = logical
            comparisons = {name: difference(reference[name], logical[name], name) for name in WIDTHS}
            repeats = {name: difference(case_reference[name], logical[name], name) for name in WIDTHS}
            item["trials"].append({"trial": trial, "input_unchanged": unchanged, "guard_violations": guards,
                "sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in logical.items()},
                "versus_reference": comparisons, "versus_case_first": repeats})
            if not unchanged:
                failure("INPUT_MUTATED", case, trial, {}, logical)
            if guards:
                failure("OUT_OF_SLOT_WRITE", case, trial, {"guards": guards}, logical)
            for name in WIDTHS:
                if not repeats[name]["exact_equal"]:
                    failure("REPEATED_CASE_DIFFERS", case, trial, {"field": name, **repeats[name]}, logical)
                elif not comparisons[name]["exact_equal"]:
                    failure("SCENARIO_DEPENDENT_OUTPUT", case, trial, {"field": name, **comparisons[name]}, logical)
    for code in ("INPUT_MUTATED", "OUT_OF_SLOT_WRITE", "REPEATED_CASE_DIFFERS", "SCENARIO_DEPENDENT_OUTPUT"):
        if code in report["failure_counts"]:
            report["result"] = code
            break
    report["notes"] = ["Exact byte comparison without an independent numerical quantization reference.",
        "REPEATED_CASE_DIFFERS includes changed allocation/fill in fresh/poison cases; consult case name.",
        "Scenario-dependent output is an observation, not proof of an operator bug.",
        "A synthetic pass does not establish equality of the model's original ki or test model stream races."]
    return report


class NPUBackend:
    def __init__(self, plan, output, input_path):
        import torch
        import torch_npu  # noqa: F401 -- register NPU backend
        from vllm_bench.mtp_debug.mome_tensors import runtime_info
        if sys.byteorder != "little":
            raise RuntimeError("This NPU replay requires a little-endian host")
        torch.npu.set_device(plan["device"])
        operator_module = importlib.import_module("cann_ops_transformer.ops.attention.indexer_quant_cache.indexer_quant_cache")
        self.torch, self.device = torch, plan["device"]
        self.op = torch.ops.cann_ops_transformer.indexer_quant_cache
        if input_path:
            descriptor = input_descriptor(input_path)
            raw = read_tensor(input_path.parent, descriptor)
            cpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(getattr(torch, descriptor["dtype"])).reshape(plan["rows"], 128).clone()
        else:
            generator = torch.Generator(device="cpu").manual_seed(plan["seed"])
            cpu = torch.randn((plan["rows"], 128), generator=generator, dtype=torch.float32)
            cpu = cpu * (cpu.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
            if plan["pattern"] == "mixed":
                peaks = (0., 1. - 1/256, 1., 1. + 1/128, 2. - 1/128, 2., 2. + 1/64, -1., -2.)
                for i in range(min(len(peaks), plan["rows"])):
                    cpu[i].fill_(peaks[i])
            cpu = cpu.to(getattr(torch, plan["dtype"]))
        self.ki = cpu.to(self.device).contiguous()
        self.input_digest = hashlib.sha256(tensor_bytes(cpu)).hexdigest()
        size = cpu.numel() * cpu.element_size()
        writer = TensorWriter(output, os.getpid(), max_tensor_bytes=size, budget_bytes=size)
        descriptor = writer.save(cpu, 0, 0, "ki", row_summaries=False)
        (Path(output) / "input.json").write_text(json.dumps({"format": "pangu-indexer-quant-input-v1", "ki": descriptor}, indent=2) + "\n")
        self.info = {**runtime_info(), "device": torch.npu.get_device_name(torch.npu.current_device()),
                     "operator_schema": str(getattr(self.op, "_schemas", "unavailable")),
                     "operator_module": getattr(operator_module, "__file__", None),
                     "operator_version": getattr(sys.modules.get("cann_ops_transformer"), "__version__", None),
                     "input_sha256": self.input_digest}

    def bind_slots(self, slots):
        self.slots = self.torch.tensor(slots, dtype=self.torch.int32, device=self.device)

    def allocate(self, spec, fill):
        torch = self.torch
        buffers = {name: torch.full((size,), fill, dtype=torch.uint8, device=self.device) for name, size in spec["buffers"].items()}
        views = {}
        for name, width in WIDTHS.items():
            desc = spec[name]
            base = buffers[desc["buffer"]]
            base = base.view(torch.float8_e4m3fn) if name == "key" else base
            views[name] = base.as_strided((spec["pages"], spec["block_size"], 1, width),
                (desc["page_stride"], width, width, 1), storage_offset=desc["offset"])
        return buffers, views

    def invoke(self, cache, start, stop):
        _, views = cache
        self.op(views["key"], views["scales"], self.ki[start:stop], self.slots[start:stop],
                quant_mode="mxfp8", round_scale=True)

    def synchronize(self):
        self.torch.npu.synchronize()

    def snapshot(self, cache):
        return {name: tensor_bytes(tensor) for name, tensor in cache[0].items()}

    def input_unchanged(self):
        return hashlib.sha256(tensor_bytes(self.ki)).hexdigest() == self.input_digest


def input_descriptor(path):
    document = json.loads(path.read_text())
    descriptor = document.get("ki", document)
    shape = descriptor.get("shape", [])
    if (len(shape) not in (2, 3) or shape[-1] != 128 or shape[0] < 1
            or len(shape) == 3 and shape[1] != 1 or descriptor.get("dtype") not in ("bfloat16", "float16", "float32")):
        raise ValueError("Input ki must have shape [T,128] or [T,1,128], dtype bfloat16/float16/float32")
    read_tensor(path.parent, descriptor)  # Integrity validation without ML imports.
    return descriptor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="input.json or exact ki tensor descriptor; overrides tokens/dtype")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--pattern", choices=("random", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--page-bytes", type=int, help="Shared page stride, including padding; default block-size * 788")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--cases", default=",".join(CASES), help="Comma-separated cases; first case supplies reference")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--sync-each-call", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    cases = args.cases.split(",")
    try:
        if min(args.tokens, args.block_size, args.chunk_size, args.repeats) < 1 or args.seed < 0:
            raise ValueError("Counts must be positive; seed must be nonnegative")
        if not cases or len(set(cases)) != len(cases) or set(cases) - set(CASES):
            raise ValueError("Choose unique cases from " + ",".join(CASES))
        if not args.device.startswith("npu:"):
            raise ValueError("--device must be an NPU device such as npu:0")
        descriptor = input_descriptor(args.input) if args.input else None
        plan = {"rows": descriptor["shape"][0] if descriptor else args.tokens, "dtype": descriptor["dtype"] if descriptor else args.dtype,
                "pattern": "recorded" if descriptor else args.pattern, "seed": args.seed, "block_size": args.block_size,
                "page_bytes": args.page_bytes, "chunk_size": args.chunk_size, "repeats": args.repeats,
                "cases": cases, "device": args.device, "sync_each_call": args.sync_each_call,
                "input": str(args.input.resolve()) if args.input else None}
        spec = layout(plan["rows"], args.block_size, page_bytes=args.page_bytes)
        plan["shared_cache_bytes"] = sum(spec["buffers"].values())
        if args.output.exists():
            raise ValueError("Use a new output directory")
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps({"plan": plan, "note": "Dry run; add --execute on A5 with torch_npu and cann_ops_transformer"}, indent=2))
        return 0
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    try:
        backend = NPUBackend(plan, args.output, args.input)
        with backend.torch.inference_mode():
            report = experiment(plan, backend, args.output)
    except Exception as error:
        report = {"format": "pangu-indexer-quant-test-v1", "result": "ERROR", "plan": plan,
                  "error": repr(error), "traceback": traceback.format_exc(), "first_failure": None}
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    first = report["first_failure"]
    brief = None if first is None else {key: first[key] for key in ("code", "case", "trial", "field", "different_bytes") if key in first}
    if first and first.get("sampled_logical_rows"):
        brief["first_logical_row"] = first["sampled_logical_rows"][0]
    if first and first.get("guards"):
        brief["first_guard"] = first["guards"][0]
    lines = ["result=" + report["result"], "first_failure=" + json.dumps(brief),
             "failure_counts=" + json.dumps(report.get("failure_counts", {}))]
    (args.output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if report["result"] == "ERROR":
        print("error=" + report["error"], file=sys.stderr)
    return 0 if report["result"] == "PASS_IN_TESTED_CASES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
