"""Opt-in import hook; does not import torch before worker initialization."""
import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        if module.__name__ == "omni_npu.sample.rejection_sampler":
            from vllm_bench.mtp_debug.spec_debug import patch_sampler
            patch_sampler(module)
        else:
            from vllm_bench.mtp_debug.spec_rejection import patch_runner
            patch_runner(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        targets = {"omni_npu.sample.rejection_sampler"}
        if os.environ.get("PANGU_SPEC_DEBUG_REJECTIONS") == "1":
            targets.add("omni_npu.worker.npu_model_runner")
        if fullname not in targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("PANGU_SPEC_DEBUG_DIR"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.meta_path.insert(0, Finder())
