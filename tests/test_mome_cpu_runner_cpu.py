"""Test CPU runner isolation and read-only source handling without ML packages."""

import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from mome_cpu_support import MOME_FILE, REFERENCE_FILE, load_original_methods, patched_copy


ROOT = Path(__file__).resolve().parents[1]


def source_fixture(root):
    texts = {MOME_FILE: "class NPUMomeAttentionMetadataBuilder:\n    marker = 1\n",
             REFERENCE_FILE: "def causal_conv1d_ref():\n    return 'source reference'\n"}
    for relative, text in texts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return texts


class MomeRunnerTests(unittest.TestCase):
    def test_loader_preserves_modules_flags_and_never_writes_upstream_bytecode(self):
        # Import-only dummy modules exercise the loader, not the numerical suite.
        torch, nn, functional = (ModuleType(name) for name in ("torch", "torch.nn", "torch.nn.functional"))
        torch.nn, nn.functional = nn, functional
        original_vllm = ModuleType("vllm")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            texts = source_fixture(source)
            modules = {"torch": torch, "torch.nn": nn, "torch.nn.functional": functional,
                       "vllm": original_vllm}
            with patch.dict(sys.modules, modules), patch.object(sys, "dont_write_bytecode", False):
                with load_original_methods(source, torch, name="loader_smoke") as methods:
                    self.assertEqual(methods.builder.marker, 1)
                    self.assertEqual(methods.convolution(), "source reference")
                    self.assertIsNot(sys.modules["vllm"], original_vllm)
                self.assertIs(sys.modules["vllm"], original_vllm)
                self.assertFalse(sys.dont_write_bytecode)
            self.assertEqual(sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file()),
                             sorted(texts))

    @unittest.skipUnless(shutil.which("git"), "git is required to check a patch on a temporary copy")
    def test_candidate_patch_changes_only_temporary_builder_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            texts = source_fixture(source)
            candidate = root / "candidate.patch"
            candidate.write_text(''.join(difflib.unified_diff(
                texts[MOME_FILE].splitlines(True), texts[MOME_FILE].replace("marker = 1", "marker = 2").splitlines(True),
                fromfile="a/" + MOME_FILE.as_posix(), tofile="b/" + MOME_FILE.as_posix())))
            destination = root / "corrected"
            destination.mkdir()
            patched_copy(source, destination, candidate)
            self.assertIn("marker = 2", (destination / MOME_FILE).read_text())
            for relative, text in texts.items():
                self.assertEqual((source / relative).read_text(), text)
            self.assertEqual((destination / REFERENCE_FILE).read_text(), texts[REFERENCE_FILE])

    def test_explicit_runner_fails_instead_of_reporting_all_skipped_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_fixture(root)
            candidate = root / "candidate.patch"
            candidate.write_text("placeholder; execution must stop before applying it\n")
            output = root / "report.json"
            result = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "scripts/check_mome_cpu.py"),
                "--omni-source", str(root), "--fix-patch", str(candidate), "--output", str(output)],
                cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("CPU PyTorch is required", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
