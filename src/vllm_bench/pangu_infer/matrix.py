"""Sequential matrix launcher: two checkpoints, TP1/2/4/8, depth3/7, four modes."""

import argparse
import json
from pathlib import Path
import sys

from vllm_bench.pangu_infer.run import parser as run_parser, make_plan, launch, write_json, MODES
from vllm_bench.pangu_infer.compare import compare


def plans(spec, root):
    """Paths in a matrix file resolve relative to that file, not its launcher."""
    options = spec.get("options", [])
    if not isinstance(options, list) or not all(isinstance(x, str) for x in options):
        raise ValueError("Matrix options must be an array of CLI strings")
    blocked = ("--model", "--operator-config", "--tp", "--depth", "--no-mtp", "--mode", "--engine", "--output", "--execute",
               "--trace-target", "--replay-target", "--replay-with-mtp", "--trace-layers")
    if any(option.split("=")[0] in blocked for option in options):
        raise ValueError("Matrix options cannot override workload axes or execution")
    def resolve(value):
        path = Path(value)
        return str((root / path).resolve() if not path.is_absolute() else path)
    output = Path(resolve(spec["output"]))
    result = []
    for i, model in enumerate(spec["models"]):
        for tp in spec.get("tp", [1, 2, 4, 8]):
            operator = resolve(spec["operator_configs"][str(tp)])
            for depth in spec.get("depths", [3, 7]):
                for mode in spec.get("modes", list(MODES)):
                    for engine in spec.get("engines", ["omni", "direct"]):
                        if engine == "omni" and mode == "compile":
                            continue  # Stock decorator explicitly bypasses this mode.
                        directory = output / f"model-{i}-tp{tp}-mtp{depth}-{mode}-{engine}"
                        args = run_parser().parse_args([
                            *options, "--model", resolve(model), "--operator-config", operator,
                            "--tp", str(tp), "--depth", str(depth), "--mode", mode,
                            "--engine", engine, "--output", str(directory)])
                        for name in ("prompt_ids", "additional_config"):
                            value = getattr(args, name)
                            if value:
                                setattr(args, name, Path(resolve(value)))
                        result.append(make_plan(args))
    return output, result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spec", type=Path, help="JSON matrix file")
    p.add_argument("--no-mtp", action="store_true", help="Run depth 0 only, overriding the JSON depths")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args(argv)
    try:
        spec = json.loads(a.spec.read_text())
        if a.no_mtp:
            spec["depths"] = [0]
        output, items = plans(spec, a.spec.resolve().parent)
        if not items:
            raise ValueError("Empty matrix")
        print(json.dumps({"runs": len(items), "output": str(output),
                          "cases": [item["options"] for item in items],
                          "skipped": "omni compile-only: stock decorator bypasses compilation"}, indent=2))
        if not a.execute:
            return 0
        if output.exists():
            raise ValueError("Matrix output exists")
        output.mkdir(parents=True)
        outcomes, completed = [], {}
        for item in items:
            case = item["options"]
            print(f"Running {case['output']}", flush=True)
            code = launch(item)
            outcomes.append({"output": case["output"], "returncode": code})
            if code == 0:
                result = json.loads((Path(case["output"]) / "result.json").read_text())
                pair = completed.setdefault(item["comparison_sha256"], {})
                pair[case["engine"]] = result
                if {"omni", "direct"} <= pair.keys() and not case["profile"] and not case.get("model_timing", False) and not case.get("memory_diagnostics", False):
                    try:
                        outcomes[-1]["comparison"] = compare(pair["omni"], pair["direct"])
                    except ValueError as error:
                        outcomes[-1]["comparison_error"] = str(error)
            write_json(output / "matrix_results.json", outcomes)
        return 1 if any(item["returncode"] or item.get("comparison_error") or
                        item.get("comparison", {}).get("greedy_tokens_match") is False for item in outcomes) else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        p.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
