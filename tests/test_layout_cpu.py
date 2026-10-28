"""CPU checks for portable module launches, worker imports and isolated hooks."""
import contextlib
import io
import json
import os
from pathlib import Path
import pkgutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import vllm_bench
from vllm_bench.common.paths import CONF_ROOT, SRC_ROOT
from vllm_bench.common.runtime_env import INSTRUMENTATION, instrumentation_pythonpath
from vllm_bench.mtp_profile import record_mtp


class PortableLayoutTests(unittest.TestCase):
    def child(self, code, directory, env=None):
        settings = {k: v for k, v in os.environ.items()
                    if k not in INSTRUMENTATION.values()}
        settings["PYTHONPATH"] = str(SRC_ROOT)
        settings.update(env or {})
        result = subprocess.run([sys.executable, "-c", code], cwd=directory,
                                env=settings, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_all_modules_and_cli_help_without_ml_packages(self):
        modules = [item.name for item in pkgutil.walk_packages(vllm_bench.__path__, "vllm_bench.")]
        code = '''
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'omni_npu', 'numpy'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, Block())
for name in MODULES:
    module = importlib.import_module(name)
    if callable(getattr(module, 'main', None)):
        sys.argv = [name, '--help']
        try:
            module.main()
        except SystemExit as error:
            assert error.code == 0, (name, error.code)
assert not any(type(f).__module__.endswith('sitecustomize') for f in sys.meta_path)
'''.replace('MODULES', repr(modules))
        with tempfile.TemporaryDirectory() as directory:
            self.child(code, directory)

    def test_default_suite_and_explicit_relative_paths_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text((CONF_ROOT / "common/reference_config-hf.json").read_text())
            code = '''
from vllm_bench.layer_profile.benchmark import main
import sys
sys.argv = ['benchmark', '--model-dir', 'model', '--output', 'result', '--dry-run']
main()
'''
            result = self.child(code, directory)
            self.assertIn('prefill-1k-c1', result.stdout)
            self.assertFalse((root / "result").exists())

    def test_recorder_environment_imports_package_and_delays_ml_hook(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                record_mtp.main(["--output", str(Path(directory) / "capture"), "--", "true"])
            env = json.loads(stream.getvalue())["environment"]
            worker = Path(directory) / "omni_npu/worker"
            worker.mkdir(parents=True)
            for path in [worker.parent / "__init__.py", worker / "__init__.py"]:
                path.write_text('')
            (worker / "npu_model_runner.py").write_text('class NPUModelRunner: pass\n')
            code = '''
import sys
assert 'sitecustomize' in sys.modules
assert not any(name.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'omni_npu'} for name in sys.modules)
from vllm_bench.mtp_profile import mtp_capture
assert any(type(f).__module__ == 'sitecustomize' for f in sys.meta_path)
from unittest.mock import patch
with patch.object(mtp_capture, 'patch_runner') as attach:
    from omni_npu.worker.npu_model_runner import NPUModelRunner
    attach.assert_called_once_with(NPUModelRunner)
'''
            self.child(code, directory, env)
            self.assertFalse((Path(directory) / "capture").exists())

    def test_diagnostic_hook_installs_after_module_import_with_new_package_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "omni_npu/sample"
            sample.mkdir(parents=True)
            for path in [root / "omni_npu/__init__.py", sample / "__init__.py", sample / "rejection_sampler.py"]:
                path.write_text('')
            env = {"PANGU_SPEC_DEBUG_DIR": str(root / "output")}
            env["PYTHONPATH"] = instrumentation_pythonpath({"PYTHONPATH": directory}, "mtp_debug")
            code = '''
from unittest.mock import patch
from vllm_bench.mtp_debug import spec_debug
with patch.object(spec_debug, 'patch_sampler') as attach:
    import omni_npu.sample.rejection_sampler as sampler
    attach.assert_called_once_with(sampler)
'''
            self.child(code, directory, env)

    def test_conflicting_hooks_fail_before_execution(self):
        for selected in INSTRUMENTATION:
            for other, variable in INSTRUMENTATION.items():
                if other == selected:
                    continue
                with self.subTest(selected=selected, other=other):
                    with self.assertRaisesRegex(ValueError, 'Conflicting instrumentation'):
                        instrumentation_pythonpath({variable: '1'}, selected)
                    other_path = SRC_ROOT / 'vllm_bench' / other / 'instrumentation'
                    with self.assertRaisesRegex(ValueError, 'Conflicting sitecustomize'):
                        instrumentation_pythonpath({'PYTHONPATH': str(other_path)}, selected)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PANGU_SPEC_DEBUG_DIR': directory}), \
                patch.object(subprocess, 'run') as launch, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                record_mtp.main(['--output', str(Path(directory) / 'capture'), '--execute', '--', 'true'])
            launch.assert_not_called()
            self.assertFalse((Path(directory) / 'capture').exists())


if __name__ == '__main__':
    unittest.main()
