#!/usr/bin/env python3
"""Export saved Ascend profiler data outside daemon vLLM workers.

The default is a read-only inspection. --execute runs torch_npu's offline
analysis in this standalone process, without loading the model or inference.
Run in the server container/environment where the data was collected.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
import sys


def trace_files(root: Path):
    result = []
    for pattern in ("trace_view.json", "trace_view.json.gz", "*.pt.trace.json",
                    "*.pt.trace.json.gz", "*.trace.json", "*.trace.json.gz"):
        result.extend(p for p in root.rglob(pattern) if p.is_file() and p.stat().st_size > 0)
    return sorted(set(result))


def is_session(path: Path):
    return path.is_dir() and (
        (path / "FRAMEWORK").is_dir()
        or any(p.is_dir() and p.name.startswith("PROF_") for p in path.iterdir())
        or any(path.glob("profiler_info*.json")))


def sessions_under(root: Path):
    if is_session(root):
        return [root]
    return sorted(p for p in root.rglob("*_ascend_pt") if is_session(p))


def inspect_session(path: Path):
    files = [p for p in path.rglob("*") if p.is_file()]
    return {"path": str(path), "trace_files": [str(p) for p in trace_files(path)],
            "framework_files": sum("FRAMEWORK" in p.relative_to(path).parts for p in files),
            "cann_files": sum(any(part.startswith("PROF_") for part in p.relative_to(path).parts) for p in files),
            "database_files": [str(p) for p in files if p.suffix == ".db"],
            "file_count": len(files)}


def export_sessions(root: Path, execute=False, workers=1):
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Profile directory does not exist: {root}")
    sessions = sessions_under(root)
    if not sessions:
        raise ValueError("No Ascend sessions found. Pass the directory ending in _ascend_pt, "
                         "or its parent. ASCEND_PROFILER_OUTPUT alone is not the raw session root.")
    report = {"root": str(root), "execute": execute,
              "sessions": [inspect_session(p) for p in sessions]}
    print(json.dumps(report, indent=2), flush=True)
    if not execute:
        print("Inspection only. Add --execute to run offline analysis; the model is not loaded.")
        return report

    missing = [s for s in report["sessions"] if not s["trace_files"]]
    if not missing:
        print("All sessions already contain trace files; no re-export needed.")
        return report
    # Import only when explicitly executing, so inspection works on any CPU host.
    profiler = importlib.import_module("torch_npu.profiler")
    analyse = getattr(profiler, "analyse", None)
    if not callable(analyse):
        raise RuntimeError("Installed torch_npu.profiler has no offline analyse API")
    parameters = inspect.signature(analyse).parameters
    accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in parameters.values())
    options = {}
    if "max_process_number" in parameters or accepts_kwargs:
        options["max_process_number"] = workers
    if "export_type" in parameters or accepts_kwargs:
        # A DB-only collection otherwise may stay DB-only during re-analysis.
        options["export_type"] = "text"
    else:
        print("This torch_npu version has no export_type argument; using its saved/default export format.", flush=True)
    failures = []
    for session in missing:
        path = Path(session["path"])
        print(f"Offline export: {path}", flush=True)
        try:
            analyse(str(path), **options)
        except Exception as error:
            failures.append(f"{path}: {type(error).__name__}: {error}")
            continue
        # torch_npu's no_exception_func can log analysis errors without raising.
        files = trace_files(path)
        if not files:
            failures.append(f"{path}: offline analysis returned but produced no trace; "
                            "inspect preceding torch_npu/CANN/msprof messages and preserve the raw data")
        else:
            session["trace_files"] = [str(p) for p in files]
            print("Trace files: " + ", ".join(session["trace_files"]), flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="One *_ascend_pt session or parent traces directory")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Maximum analysis workers per session (sessions processed sequentially)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    export_sessions(args.profile_dir, args.execute, args.workers)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit(f"ERROR: {error}")
