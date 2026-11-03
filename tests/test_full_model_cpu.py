"""Full target/MTP launch planning and annotation checks without ML packages."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_cpu import fake_checkpoint
from vllm_bench.common.paths import SRC_ROOT
from vllm_bench.common.runtime_env import instrumentation_pythonpath
from vllm_bench.layer_profile import benchmark, serve


class FullModelTests(unittest.TestCase):
    def prepare(self, root, architecture="OpenPanguV2ForCausalLM", mxfp8=False):
        model = root / "model"
        fake_checkpoint(model, architecture=architecture, mxfp8=mxfp8)
        operator = root / "operators.json"
        operator.write_text(json.dumps({"model_parallel_config": {}, "operator_optimization_config": {}}))
        output = root / "server"
        argv = ["serve", "--model", str(model), "--model-scope", "full",
                "--operator-config", str(operator), "--output", str(output)]
        return argv, model, output

    def test_full_dry_runs_without_ml_imports_or_checkpoint_changes(self):
        code = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'omni_npu', 'numpy'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, Block())
runpy.run_module('vllm_bench.layer_profile.serve', run_name='__main__')
'''
        for architecture in ("OpenPanguV2ForCausalLM", "PanguUltraMoEForCausalLM"):
            for mxfp8 in (False, True):
                for depth in (None, 3, 7):
                    with self.subTest(architecture=architecture, mxfp8=mxfp8, depth=depth), \
                            tempfile.TemporaryDirectory() as tmp:
                        argv, model, output = self.prepare(Path(tmp), architecture, mxfp8)
                        before = {p.name: p.read_bytes() for p in model.iterdir()}
                        if depth:
                            argv += ["--enable-mtp"]
                            if depth != 3:  # Exercise the implicit default of three.
                                argv += ["--num-speculative-tokens", str(depth)]
                        env = {k: v for k, v in os.environ.items() if k != "OMNI_VLLM_PATCHES_DIR"}
                        env["PYTHONPATH"] = str(SRC_ROOT)
                        result = subprocess.run([sys.executable, "-c", code, *argv[1:]],
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        command = shlex.split(next(line.removeprefix("Command: ")
                                                  for line in result.stdout.splitlines()
                                                  if line.startswith("Command: ")))
                        self.assertEqual(command[command.index("serve") + 1], str(model))
                        self.assertNotIn("--hf-overrides", command)
                        self.assertNotIn("--enforce-eager", command)
                        self.assertIn("--no-enable-prefix-caching", command)
                        profiler = json.loads(command[command.index("--profiler-config") + 1])
                        self.assertEqual(profiler["torch_profiler_dir"], str(output / "traces"))
                        self.assertEqual(command[command.index("--data-parallel-size") + 1], "1")
                        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
                        if depth:
                            self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]),
                                             {"method": "mtp", "num_speculative_tokens": depth})
                        else:
                            self.assertNotIn("--speculative-config", command)
                        if mxfp8:
                            self.assertIn("omni_custom_models", result.stdout)
                        self.assertIn("high_throughout" if architecture.startswith("PanguUltra")
                                      else "low_latency", result.stdout)
                        self.assertEqual(before, {p.name: p.read_bytes() for p in model.iterdir()})
                        self.assertFalse(output.exists())

    def test_manifest_and_client_full_model_dry_run(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                argv, model, output = self.prepare(root)
                argv += ["--execute", "--tp", "2"]
                if enabled:
                    argv += ["--enable-mtp", "--num-speculative-tokens", "3"]
                with patch.object(sys, "argv", argv), patch.object(serve.subprocess, "call", return_value=0) as launch, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(serve.main(), 0)
                manifest_path = output / "server_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["command"], launch.call_args.args[0])
                self.assertEqual(manifest["model_scope"], "full")
                self.assertEqual(manifest["arguments"]["enable_mtp"], enabled)
                self.assertEqual(manifest["arguments"]["num_speculative_tokens"], 3 if enabled else None)
                self.assertEqual(manifest["speculative_config"],
                                 {"method": "mtp", "num_speculative_tokens": 3} if enabled else None)
                self.assertEqual(manifest["model_config"], json.loads((model / "config.json").read_text()))
                self.assertEqual(manifest["model_config"]["num_hidden_layers"], 50)
                self.assertEqual(manifest["model_config"]["num_nextn_predict_layers"], 3)
                for mode in ("benchmark", "profile"):
                    client_output = root / mode
                    client = ["benchmark", "--model-dir", str(model), "--server-manifest", str(manifest_path),
                              "--mode", mode, "--select", "decode-8k-c1", "--output", str(client_output), "--dry-run"]
                    with patch.object(sys, "argv", client), contextlib.redirect_stdout(io.StringIO()) as stream:
                        benchmark.main()
                    self.assertIn("decode-8k-c1", stream.getvalue())
                    self.assertFalse(client_output.exists())

    def test_invalid_modes_depths_and_checkpoints_fail_before_launch(self):
        cases = [
            (["--model-scope", "reduced", "--enable-mtp"], {"num_nextn_predict_layers": 0}, None,
             "positive integer num_nextn_predict_layers"),
            (["--model-scope", "reduced"], {}, None, "MTP disabled"),
            (["--num-speculative-tokens", "3"], {}, None, "requires --enable-mtp"),
            (["--enable-mtp", "--num-speculative-tokens", "0"], {}, None, "must be positive"),
            (["--enable-mtp", "--num-speculative-tokens", "-1"], {}, None, "must be positive"),
            ([], {"num_hidden_layers": 0}, None, "positive integer num_hidden_layers"),
            ([], {"num_hidden_layers": True}, None, "positive integer num_hidden_layers"),
            ([], {"architectures": ["OtherModel"]}, None, "supported V2 or Ultra"),
            ([], {}, "extraction_manifest.json", "original target checkpoint"),
            ([], {}, "mtp_checkpoint_manifest.json", "original target checkpoint"),
        ]
        cases += [(["--enable-mtp"], {"num_nextn_predict_layers": value}, None,
                   "positive integer num_nextn_predict_layers") for value in (None, 0, -1, True, "3")]
        for extra, overrides, marker, expected in cases:
            with self.subTest(extra=extra, overrides=overrides, marker=marker), tempfile.TemporaryDirectory() as tmp:
                argv, model, output = self.prepare(Path(tmp))
                config = model / "config.json"
                config.write_text(json.dumps({**json.loads(config.read_text()), **overrides}))
                if marker:
                    (model / marker).write_text("{}")
                with patch.object(sys, "argv", argv + extra + ["--execute"]), \
                        patch.object(serve.subprocess, "call") as launch, \
                        contextlib.redirect_stderr(io.StringIO()) as stream, self.assertRaises(SystemExit) as error:
                    serve.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(expected, stream.getvalue())
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_speculative_config_cannot_be_overridden(self):
        for extra in (["--speculative-config", '{"method":"mtp","num_speculative_tokens":7}'],
                      ['--speculative-config={"method":"mtp","num_speculative_tokens":7}'],
                      ["--speculative_config", '{"method":"mtp","num_speculative_tokens":7}'],
                      ['--speculative_config={"method":"mtp","num_speculative_tokens":7}']):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                argv, _, output = self.prepare(Path(tmp))
                extra_path = Path(tmp) / "extra.json"
                extra_path.write_text(json.dumps(extra))
                with patch.object(sys, "argv", argv + ["--enable-mtp", "--engine-cli-json", str(extra_path), "--execute"]), \
                        patch.object(serve.subprocess, "call") as launch, contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit) as error:
                    serve.main()
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()
                self.assertFalse(output.exists())

    def test_full_decoder_and_mtp_block_annotations_in_child(self):
        for module_name, class_name in (("pangu_v2_moe", "OpenPanguV2DecoderLayer"),
                                        ("pangu_ultra_moe", "OpenPanguDecoderLayer")):
            with self.subTest(module_name=module_name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "omni_npu/v1/models/pangu"
                package.mkdir(parents=True)
                for directory in (package, *list(package.parents)[:3]):
                    (directory / "__init__.py").write_text("")
                (package / (module_name + ".py")).write_text(f'''
class {class_name}:
    def __init__(self, layer_idx): self.layer_idx = layer_idx
    def forward(self, x): return self.mhc_head(x) + 1
    def mhc_head(self, x): return x + 1
''')
                (package / (module_name + "_mtp.py")).write_text(f'''
from .{module_name} import {class_name}
class Predictor:
    def __init__(self, idx): self.mtp_block = {class_name}(idx)
    def forward(self, x): return self.mtp_block.forward(x)
''')
                code = f'''
import contextlib, importlib, sys, types
labels = []
@contextlib.contextmanager
def record_function(name):
    labels.append(name)
    yield
sys.modules['torch'] = types.SimpleNamespace(autograd=types.SimpleNamespace(
    profiler=types.SimpleNamespace(record_function=record_function)))
decoder = getattr(importlib.import_module('omni_npu.v1.models.pangu.{module_name}'), '{class_name}')
predictor = importlib.import_module('omni_npu.v1.models.pangu.{module_name}_mtp').Predictor
for idx in (0, 49):
    assert decoder(idx).forward(10) == 12
for idx in (12, 13, 14, 50, 51, 52):
    assert predictor(idx).forward(10) == 12
assert labels == [f'pangu12/layer_{{idx:03d}}/{{method}}'
                  for idx in (0, 49, 12, 13, 14, 50, 51, 52) for method in ('forward', 'mhc_head')], labels
'''
                env = {**os.environ, "PANGU12_ANNOTATE_LAYERS": "1",
                       "PYTHONPATH": instrumentation_pythonpath({"PYTHONPATH": str(root)}, "layer_profile")}
                result = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("CPU layer annotations installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
