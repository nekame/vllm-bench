"""Enable only with PANGU_MTP_CAPTURE_DIR and this directory on PYTHONPATH."""
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys


class Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.original.exec_module(module)
        from mtp_capture import patch_runner
        patch_runner(module.NPUModelRunner)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "omni_npu.worker.npu_model_runner":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = Loader(spec.loader)
        return spec


if os.environ.get("PANGU_MTP_CAPTURE_DIR"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.meta_path.insert(0, Finder())
