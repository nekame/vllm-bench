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
        from spec_debug import patch_sampler
        patch_sampler(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "omni_npu.sample.rejection_sampler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("PANGU_SPEC_DEBUG_DIR"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.meta_path.insert(0, Finder())
