#!/usr/bin/env python3
"""Run numerical MoME boundary tests with CPU PyTorch and a read-only Omni checkout."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni-source", type=Path, default=ROOT.parent / "omniinfer")
    parser.add_argument("--fix-patch", type=Path,
                        default=ROOT / "patches/omniinfer/mome-target-prefill-state.patch")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    for path in (args.omni_source / "omni/attention/backends/mome.py",
                 args.omni_source / "omni/layers/ops/causal_conv1d.py", args.fix_patch):
        if not path.is_file():
            parser.error(f"Required source file not found: {path}")
    if importlib.util.find_spec("torch") is None:
        parser.error("CPU PyTorch is required. Install torch in a separate environment; no vLLM or torch_npu needed.")

    started = time.perf_counter()
    # Keep an installed torch_npu backend from autoloading in this CPU-only process.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "tests"))
    import test_mome_boundary_torch_cpu as checks

    checks.MomeBoundaryTests.omni_source = args.omni_source.resolve()
    checks.MomeBoundaryTests.fix_patch = args.fix_patch.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(checks.MomeBoundaryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"scope": "CPU metadata/reference tests; no model, NPU operator or ACL replay",
              "successful": passed, "torch": torch.__version__, "threads": torch.get_num_threads(),
              "tests_run": result.testsRun, "skipped": len(result.skipped),
              "failures": len(result.failures), "errors": len(result.errors),
              "elapsed_seconds": time.perf_counter() - started,
              "peak_rss_mib": peak_rss / (1024 * 1024 if sys.platform == "darwin" else 1024),
              "omni_source": str(args.omni_source.resolve()),
              "source_sha256": getattr(checks.MomeBoundaryTests, "source_hashes", {}),
              "observations": checks.MomeBoundaryTests.observations}
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
