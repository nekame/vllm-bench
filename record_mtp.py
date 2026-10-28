#!/usr/bin/env python3
"""Print or execute an existing full-model command with opt-in MTP capture."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--skip", type=int, default=2, help="Skip this many proposer calls per rank")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--max-gib-per-snapshot", type=float, default=8)
    p.add_argument("--execute", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- your existing full-model launch command")
    a = p.parse_args(argv)
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or a.skip < 0 or a.count < 1 or a.max_gib_per_snapshot <= 0:
        p.error("Command, positive count/byte limit and nonnegative skip required")
    if "--annotate-layers" in command:
        p.error("Do not combine two sitecustomize instrumentations in this capture launch")
    if a.output.exists():
        p.error("Capture output must not exist")
    settings = {"PANGU_MTP_CAPTURE_DIR": str(a.output.resolve()),
                "PANGU_MTP_CAPTURE_SKIP": str(a.skip), "PANGU_MTP_CAPTURE_COUNT": str(a.count),
                "PANGU_MTP_CAPTURE_MAX_GIB": str(a.max_gib_per_snapshot),
                "PYTHONPATH": os.pathsep.join([str(HERE / "mtp_instrumentation"), str(HERE), os.environ.get("PYTHONPATH", "")])}
    print(json.dumps({"command": command, "environment": settings,
                      "warning": "Full target model runs only through your command; capture changes timings"}, indent=2))
    if not a.execute:
        return 0
    env = os.environ.copy()
    env.update(settings)
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
