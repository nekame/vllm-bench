"""Optional import hook. Only this new directory is added to PYTHONPATH.

Adds CPU launch ranges, NOT NPU elapsed-time measurements. No OmniInfer source
is modified. Graph capture/compilation may change range visibility.
"""
import functools
import importlib.abc
import importlib.machinery
import os
import sys


class _Loader:
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, "create_module", None)
        return method(spec) if method else None

    def exec_module(self, module):
        self.original.exec_module(module)
        import torch
        cls = module.OpenPanguV2DecoderLayer
        for method in ("forward", "mhc_head"):
            original = getattr(cls, method)
            if getattr(original, "_pangu12_annotation", False):
                continue

            def decorate(fn, label):
                @functools.wraps(fn)
                def wrapped(self, *args, **kwargs):
                    name = f"pangu12/layer_{self.layer_idx:03d}/{label}"
                    with torch.autograd.profiler.record_function(name):
                        return fn(self, *args, **kwargs)
                wrapped._pangu12_annotation = True
                return wrapped

            setattr(cls, method, decorate(original, method))
        print("[pangu12] CPU layer annotations installed", file=sys.stderr, flush=True)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "omni_npu.v1.models.pangu.pangu_v2_moe":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            spec.loader = _Loader(spec.loader)
        return spec


if os.environ.get("PANGU12_ANNOTATE_LAYERS") == "1":
    sys.meta_path.insert(0, _Finder())
