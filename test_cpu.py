#!/usr/bin/env python3
"""CPU-only tests. No ML packages, hardware discovery, or NPU execution."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark
import prepare_checkpoint as extract
import verify_checkpoint
import export_profile

HERE = Path(__file__).resolve().parent


def fake_checkpoint(root, mxfp8=False, architecture="OpenPanguV2ForCausalLM"):
    root.mkdir()
    cfg = json.loads((HERE / "reference_config-hf.json").read_text())
    cfg["architectures"] = [architecture]
    if mxfp8:
        cfg["quantization_config"] = {
            "quant_method": "mxfp8", "ignore": ["lm_head"],
            "enable_mxfp8_comm_quant": True, "weight_block_size": [1, 32]}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "configuration_openpangu_v2.py").write_text("# synthetic asset\n")
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
             "model.merge_mhc_module.phi.weight"]
    names += [f"model.layers.{i}.self_attn.q_a_proj.weight" for i in range(53)]
    names += [f"model.layers.{i}.mlp.experts.0.gate_proj.weight" for i in range(3, 50)]
    payloads = {name: bytes([i % 256, (i + 1) % 256]) for i, name in enumerate(names)}
    descriptors = {name: {"dtype": "BF16", "shape": [1]} for name in names}
    if mxfp8:
        for i, name in enumerate(list(names)):
            if "q_a_proj" not in name and ".experts." not in name:
                continue
            payloads[name] = bytes((i + j) % 256 for j in range(64))
            descriptors[name] = {"dtype": "F8_E4M3", "shape": [2, 32]}
            scale = name.removesuffix(".weight") + ".weight_scale"
            names.append(scale)
            payloads[scale] = bytes([127, 128])
            descriptors[scale] = {"dtype": "F8_E8M0" if i == 4 else "U8", "shape": [2, 1]}
    # Two input shards: verify extraction reads tensors from either source.
    index = {}
    for shard, subset in enumerate((names[::2], names[1::2])):
        header = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for name in subset:
            size = len(payloads[name])
            header[name] = {**descriptors[name], "data_offsets": [cursor, cursor + size]}
            cursor += size
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        filename = f"source-{shard}.safetensors"
        (root / filename).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(payloads[n] for n in subset))
        index.update({name: filename for name in subset})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return payloads


class ExtractionTests(unittest.TestCase):
    def test_mxfp8_bytes_scales_config_and_hashes(self):
        for architecture in ("OpenPanguV2ForCausalLM", "PanguUltraMoEForCausalLM"):
            for override in (False, True):
                with self.subTest(architecture=architecture, structural_override=override), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp) / "source", Path(tmp) / "output"
                    payloads = fake_checkpoint(source, mxfp8=True, architecture=architecture)
                    original = json.loads((source / "config.json").read_text())
                    before = {p.name: p.read_bytes() for p in source.iterdir()}
                    command = ["--source", str(source), "--output", str(output), "--hash-tensors"]
                    if override:
                        structural = {k: v for k, v in original.items() if k != "quantization_config"}
                        path = Path(tmp) / "structural.json"
                        path.write_text(json.dumps(structural))
                        command += ["--config", str(path)]
                    with contextlib.redirect_stdout(io.StringIO()):
                        extract.main(command)
                    cfg = json.loads((output / "config.json").read_text())
                    self.assertEqual(cfg["quantization_config"], original["quantization_config"])
                    self.assertEqual(cfg["architectures"], [architecture])
                    self.assertEqual(cfg["dtype"], "bfloat16")
                    self.assertEqual(cfg["num_hidden_layers"], 12)
                    self.assertEqual(cfg["num_nextn_predict_layers"], 0)
                    original_tensors, actual = extract.discover(source), extract.discover(output)
                    expected = {n for n in payloads if not n.startswith("model.layers.")
                                or int(n.split(".")[2]) < 12}
                    self.assertEqual(set(actual), expected)
                    self.assertTrue(any(t["dtype"] == "F8_E4M3" for t in actual.values()))
                    self.assertTrue(any(n.endswith(".weight_scale") for n in actual))
                    for name, item in actual.items():
                        self.assertEqual(item["dtype"], original_tensors[name]["dtype"])
                        self.assertEqual(item["shape"], original_tensors[name]["shape"])
                        a, b = item["data_offsets"]
                        with item["source"].open("rb") as f:
                            f.seek(item["start"] + a)
                            self.assertEqual(f.read(b - a), payloads[name])
                    self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
                    with patch.object(sys, "argv", ["verify_checkpoint.py", str(output), "--hashes"]), contextlib.redirect_stdout(io.StringIO()):
                        verify_checkpoint.main()

    def test_unsupported_and_conflicting_quantization_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            fake_checkpoint(source, mxfp8=True)
            original = json.loads((source / "config.json").read_text())
            override = Path(tmp) / "override.json"
            changed = json.loads(json.dumps(original))
            changed["quantization_config"]["ignore"] = []
            override.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "quantization_config disagrees"):
                extract.resolve_config(source, override)
            for method in ("compressed-tensors", "hifloat8", "w4a8_mxfp"):
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, "Supported checkpoint formats"):
                    extract.reduced_config({**original, "quantization_config": {"quant_method": method}}, 12)

    def test_fp8_without_metadata_is_not_mislabeled_bf16(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            fake_checkpoint(source, mxfp8=True)
            cfg = json.loads((source / "config.json").read_text())
            cfg.pop("quantization_config")
            (source / "config.json").write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "original MXFP8 quantization_config"):
                extract.main(["--source", str(source), "--output", str(output)])
            self.assertFalse(output.exists())

    def test_mxfp8_server_plan_uses_original_architecture(self):
        for architecture in ("OpenPanguV2ForCausalLM", "PanguUltraMoEForCausalLM"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp) / "model"
                fake_checkpoint(model, mxfp8=True, architecture=architecture)
                cfg = extract.reduced_config(json.loads((model / "config.json").read_text()), 12)
                (model / "config.json").write_text(json.dumps(cfg))
                operator = Path(tmp) / "operators.json"
                operator.write_text(json.dumps({"model_parallel_config": {}, "operator_optimization_config": {}}))
                output = Path(tmp) / "server"
                result = subprocess.run([sys.executable, str(HERE / "serve.py"), "--model", str(model),
                                         "--operator-config", str(operator), "--output", str(output)],
                                        capture_output=True, text=True, env={**{k: v for k, v in os.environ.items() if k != "OMNI_VLLM_PATCHES_DIR"}, "VLLM_PLUGINS": "existing_plugin"})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("omni_custom_models", result.stdout)
                self.assertIn("omni_npu_patches", result.stdout)
                self.assertIn("omni-npu", result.stdout)
                self.assertIn("existing_plugin", result.stdout)
                expected = "high_throughout" if architecture.startswith("PanguUltra") else "low_latency"
                self.assertIn(expected, result.stdout)
                command = shlex.split(next(s.removeprefix("Command: ") for s in result.stdout.splitlines() if s.startswith("Command: ")))
                self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
                self.assertNotIn("--hf-overrides", command)
                self.assertFalse(output.exists())

    def test_prefix_preserves_payloads_and_global_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            payloads = fake_checkpoint(source)
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                extract.main(["--source", str(source), "--output", str(output),
                              "--hash-tensors", "--max-shard-gib", "0.00000002"])
            cfg = json.loads((output / "config.json").read_text())
            self.assertEqual(cfg["num_hidden_layers"], 12)
            self.assertEqual(cfg["num_nextn_predict_layers"], 0)
            self.assertEqual(cfg["dsa_layers"], [0, 3, 6, 9])
            self.assertEqual(cfg["swa_layers"], [1, 2, 4, 5, 7, 8, 10, 11])
            self.assertEqual(cfg["sliding_window_list"], [512] * 8)
            self.assertEqual(cfg["block_post_layernorm_idx"], [0, 4, 9])
            actual = extract.discover(output)
            expected = {n for n in payloads if not n.startswith("model.layers.")
                        or int(n.split(".")[2]) < 12}
            self.assertEqual(set(actual), expected)
            for name, item in actual.items():
                a, b = item["data_offsets"]
                with item["source"].open("rb") as f:
                    f.seek(item["start"] + a)
                    self.assertEqual(f.read(b - a), payloads[name])
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertTrue((output / "configuration_openpangu_v2.py").is_file())

    def test_dry_run_and_missing_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            fake_checkpoint(source)
            with contextlib.redirect_stdout(io.StringIO()):
                extract.main(["--source", str(source), "--output", str(output), "--dry-run"])
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
                extract.main(["--source", str(source), "--output", str(output),
                              "--layer-prefix", "wrong.prefix."])
            self.assertFalse(output.exists())


class _Connection:
    def request(self, method, path, body, headers):
        request = json.loads(body)
        records = [{"choices": [{"token_ids": [1], "text": ""}]},
                   {"choices": [{"token_ids": [2, 3], "text": "", "finish_reason": "length"}]},
                   {"choices": [], "usage": {"prompt_tokens": len(request["prompt"]), "completion_tokens": 3}}]
        if request.get("seed") == -1:
            records = records[:1]  # Truncated stream must fail, never yield misleading metrics.
        raw = b"".join(("data: " + json.dumps(record) + "\r\n\r\n").encode() for record in records)
        if request.get("seed") != -1:
            raw += b"data: [DONE]\r\n\r\n"
        self.response = io.BytesIO(raw)
        self.response.status = 200

    def getresponse(self):
        return self.response

    def close(self):
        pass


class BenchmarkTests(unittest.TestCase):
    def test_stream_counts_batched_tokens_and_rejects_truncation(self):
        with patch.object(benchmark, "connection", side_effect=lambda *args: (_Connection(), "")):
            payload = {"request_id": "test", "prompt": [1, 2], "max_tokens": 3}
            base = "http://test.invalid"
            row = benchmark.stream_request(base, payload, 5)
            self.assertTrue(row["success"], row)
            self.assertFalse(row["single_token_chunks"])
            self.assertEqual(row["usage"]["completion_tokens"], 3)
            row["wave"] = 0
            self.assertEqual(benchmark.summarize([row])["output_tokens"], 3)
            bad = benchmark.stream_request(base, {**payload, "seed": -1}, 5)
            self.assertFalse(bad["success"])

    def test_payload_has_exact_lengths_and_no_special_tokens(self):
        cfg = json.loads((HERE / "reference_config-hf.json").read_text())
        case = {"input_tokens": 1024, "output_tokens": 128}
        a = benchmark.make_payload(case, "pangu12", cfg, 123, 0)
        b = benchmark.make_payload(case, "pangu12", cfg, 123, 0)
        self.assertEqual(a, b)
        self.assertEqual(len(a["prompt"]), 1024)
        self.assertFalse(set(a["prompt"]) & {cfg[k] for k in ("bos_token_id", "eos_token_id", "pad_token_id")})


class ProfileExportTests(unittest.TestCase):
    def test_analyse_in_implementation_module_not_reexported(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "worker_ascend_pt"
            (session / "FRAMEWORK").mkdir(parents=True)
            calls = []

            def analyse(path, max_process_number=4, export_type=None):
                calls.append((path, max_process_number, export_type))
                output = Path(path) / "ASCEND_PROFILER_OUTPUT"
                output.mkdir()
                (output / "trace_view.json").write_text('{"traceEvents": []}')

            modules = {"torch_npu.profiler": SimpleNamespace(),
                       "torch_npu.profiler.profiler": SimpleNamespace(analyse=analyse)}
            with patch.object(export_profile.importlib, "import_module", side_effect=modules.__getitem__) as importer, contextlib.redirect_stdout(io.StringIO()):
                result = export_profile.export_sessions(session, execute=True)
            self.assertEqual([call.args[0] for call in importer.call_args_list], list(modules))
            self.assertEqual(result["analysis_api"], "torch_npu.profiler.profiler.analyse")
            self.assertEqual(calls, [(str(session), 1, "text")])
            self.assertEqual(len(result["sessions"][0]["trace_files"]), 1)

    def test_missing_api_error_reports_checked_paths(self):
        from types import SimpleNamespace
        with patch.object(export_profile.importlib, "import_module", return_value=SimpleNamespace()), patch.object(export_profile.metadata, "version", return_value="test-version"):
            with self.assertRaisesRegex(RuntimeError, "test-version") as error:
                export_profile.resolve_analyse()
        self.assertIn("torch_npu.profiler: no callable analyse", str(error.exception))
        self.assertIn("torch_npu.profiler.profiler: no callable analyse", str(error.exception))

    def test_inspection_does_not_import_ml_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "worker_ascend_pt"
            (session / "FRAMEWORK").mkdir(parents=True)
            (session / "FRAMEWORK" / "torch.op_range").write_bytes(b"synthetic raw data")
            (session / "PROF_001" / "device_0").mkdir(parents=True)
            (session / "PROF_001" / "device_0" / "task.data").write_bytes(b"synthetic raw data")
            with patch.object(export_profile.importlib, "import_module") as importer, contextlib.redirect_stdout(io.StringIO()):
                result = export_profile.export_sessions(root)
            importer.assert_not_called()
            self.assertEqual(len(result["sessions"]), 1)
            self.assertEqual(result["sessions"][0]["framework_files"], 1)
            self.assertEqual(result["sessions"][0]["cann_files"], 1)
            self.assertEqual(result["sessions"][0]["trace_files"], [])

    def test_offline_export_requests_text_and_checks_the_result(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "worker_ascend_pt"
            (session / "FRAMEWORK").mkdir(parents=True)
            calls = []

            def analyse(path, max_process_number=4, export_type=None):
                calls.append((path, max_process_number, export_type))
                target = Path(path) / "ASCEND_PROFILER_OUTPUT"
                target.mkdir()
                (target / "trace_view.json").write_text('{"traceEvents": []}')

            with patch.object(export_profile.importlib, "import_module", return_value=SimpleNamespace(analyse=analyse)), contextlib.redirect_stdout(io.StringIO()):
                result = export_profile.export_sessions(session, execute=True)
            self.assertEqual(calls, [(str(session), 1, "text")])
            self.assertEqual(len(result["sessions"][0]["trace_files"]), 1)

    def test_silently_failed_analysis_is_not_reported_as_success(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp) / "worker_ascend_pt"
            (session / "FRAMEWORK").mkdir(parents=True)
            def analyse(path):
                pass
            with patch.object(export_profile.importlib, "import_module", return_value=SimpleNamespace(analyse=analyse)), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "produced no trace"):
                export_profile.export_sessions(session, execute=True)


if __name__ == "__main__":
    unittest.main()
