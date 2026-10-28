#!/usr/bin/env python3
"""Launch existing server with opt-in greedy-verification diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--rejection-windows", action="store_true",
                        help="Capture target inputs and metadata around every rejection (eager, one request at a time)")
    parser.add_argument("--mome-layers", default="0", help="Target layer indices, comma-separated, or all")
    parser.add_argument("--state-fingerprints", action="store_true",
                        help="Hash active MoME blocks before/after selected target convolutions; adds synchronization")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.limit < 1:
        parser.error("Supply a server command and positive --limit")
    if args.mome_layers != "all":
        try:
            layers = [int(x) for x in args.mome_layers.split(",")]
            if any(x < 0 for x in layers):
                raise ValueError()
        except ValueError:
            parser.error("--mome-layers must be nonnegative integers separated by commas, or all")
    if args.state_fingerprints and not args.rejection_windows:
        parser.error("--state-fingerprints requires --rejection-windows")
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory")
    here = Path(__file__).resolve().parent
    settings = {"PANGU_SPEC_DEBUG_DIR": str(output),
                "PANGU_SPEC_DEBUG_LIMIT": str(args.limit),
                "PYTHONPATH": os.pathsep.join([
                    str(here / "spec_debug_instrumentation"), str(here),
                    os.environ.get("PYTHONPATH", "")])}
    if args.rejection_windows:
        settings.update({"PANGU_SPEC_DEBUG_REJECTIONS": "1",
                         "PANGU_SPEC_DEBUG_MOME_LAYERS": args.mome_layers,
                         "PANGU_SPEC_DEBUG_STATE_FINGERPRINTS": "1" if args.state_fingerprints else "0"})
    print(json.dumps({"command": command, "environment": settings,
                      "arm_after_warmup": f"Create {output / 'ARM'}",
                      "note": "V1 runner only; synchronous copies change timings. Do not combine sitecustomize wrappers."}, indent=2))
    if not args.execute:
        return 0
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update(settings)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
