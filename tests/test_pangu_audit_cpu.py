"""Source and effective-setting audit without importing torch or vLLM."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from vllm_bench.common.paths import SRC_ROOT
from vllm_bench.pangu_infer.audit import execution_settings, source_fingerprints


class RuntimeAuditTests(unittest.TestCase):
    def test_loaded_module_path_takes_precedence_over_installed_package_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            installed = root / "installed/attention/backends/mome.py"
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"installed version\n")
            loaded = root / "loaded.py"
            loaded.write_bytes(b"actually loaded version\n")
            module = "omni_npu.attention.backends.mome"
            manifest = source_fingerprints({"omni_npu": root / "installed"},
                                           {module: SimpleNamespace(__file__=str(loaded))})
            self.assertTrue(manifest[module]["loaded"])
            self.assertEqual(manifest[module]["path"], str(loaded))
            self.assertEqual(manifest[module]["sha256"], hashlib.sha256(loaded.read_bytes()).hexdigest())

    def test_unavailable_files_are_explicit_and_do_not_abort_the_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = source_fingerprints({"omni_npu": Path(tmp)}, {})
            self.assertIsNone(manifest["omni_npu.attention.backends.mome"]["sha256"])
            self.assertIn("error", manifest["omni_npu.attention.backends.mome"])
            self.assertIsNone(manifest["vllm.v1.worker.gpu_model_runner"]["path"])

    def test_standalone_audit_does_not_import_installed_ml_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for package in ("torch", "torch_npu", "vllm", "omni_npu"):
                directory = root / package
                directory.mkdir()
                (directory / "__init__.py").write_text('raise RuntimeError("ML package imported")\n')
            mome = root / "omni_npu/attention/backends/mome.py"
            mome.parent.mkdir(parents=True)
            mome.write_text("# source inspected without importing it\n")
            output = root / "manifest.json"
            env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(SRC_ROOT))))
            result = subprocess.run([sys.executable, "-m", "vllm_bench.pangu_infer.audit", "--output", str(output)],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(output.read_text())
            entry = manifest["sources"]["omni_npu.attention.backends.mome"]
            self.assertFalse(entry["loaded"])
            self.assertEqual(entry["sha256"], hashlib.sha256(mome.read_bytes()).hexdigest())

    def test_effective_settings_preserve_capacity_capture_and_static_kernel_values(self):
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=133120, dtype="bfloat16",
                hf_config=SimpleNamespace(architectures=["PanguUltraMoEForCausalLM"])),
            scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=2048,
                enable_chunked_prefill=True),
            cache_config=SimpleNamespace(mamba_page_size_padded=65536, num_gpu_blocks=1000),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[8], max_cudagraph_capture_size=8),
            additional_config={"npugraph_ex_config": {"enable": True, "static_kernel_compile": True}})
        before = deepcopy(cfg)
        settings = execution_settings(cfg)
        self.assertEqual(cfg, before)
        self.assertEqual(settings["max_model_len"], 133120)
        self.assertEqual(settings["cudagraph_capture_sizes"], [8])
        self.assertTrue(settings["npugraph_ex_config"]["static_kernel_compile"])
        json.dumps(settings)


if __name__ == "__main__":
    unittest.main()
