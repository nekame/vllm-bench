"""Prefix target plus remapped MTP checkpoint: extraction, verification and launch."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_mtp_cpu import checkpoint
from vllm_bench.layer_profile import benchmark, prepare_checkpoint as prepare, serve, verify_checkpoint


class ReducedMTPTests(unittest.TestCase):
    def extract(self, source, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare.main(["--source", str(source), "--output", str(output),
                                 "--keep-mtp", "--hash-tensors", *extra])

    def test_keep_mtp_preserves_payloads_scales_and_remaps_all_metadata(self):
        for quantized in (False, True):
            for ultra in (False, True):
                with self.subTest(quantized=quantized, ultra=ultra), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp) / "source", Path(tmp) / "reduced"
                    cfg, payloads = checkpoint(source, quantized, ultra)
                    cfg["layer_types"] = [f"type-{i}" for i in range(53)]
                    cfg["indexer_types"] = list(range(53))
                    cfg["index_topk_pattern"] = list(range(53))
                    cfg["block_post_layernorm_idx"] += [50, 52]
                    if quantized:
                        cfg["quantization_config"]["ignore"] += [
                            "model.layers.50.eh_proj", "model.layers.12.mlp",
                            r"model\.layers\.51\.eh_proj", r"model\.layers\.13\.mlp"]
                    (source / "config.json").write_text(json.dumps(cfg))
                    before = {p.name: p.read_bytes() for p in source.iterdir()}
                    self.extract(source, output, "--max-shard-gib", "0.0000001", "--buffer-mib", "1")
                    result = json.loads((output / "config.json").read_text())
                    self.assertEqual(result["architectures"], cfg["architectures"])
                    self.assertEqual(result["num_hidden_layers"], 12)
                    self.assertEqual(result["num_nextn_predict_layers"], 3)
                    self.assertEqual(result["swa_layers"], [1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14])
                    self.assertEqual(result["sliding_window_list"], [512] * 8 + [2048] * 3)
                    self.assertEqual(result["dsa_layers"], [0, 3, 6, 9])
                    self.assertEqual(result["block_post_layernorm_idx"], [0, 4, 9, 12, 14])
                    self.assertEqual(result["layer_types"], [f"type-{i}" for i in [*range(12), 50, 51, 52]])
                    self.assertEqual(result["indexer_types"], [*range(12), 50, 51, 52])
                    self.assertEqual(result["index_topk_pattern"], [*range(12), 50, 51, 52])
                    if quantized:
                        self.assertEqual(result["quantization_config"]["ignore"],
                                         ["lm_head", "model.layers.12.eh_proj", r"model\.layers\.13\.eh_proj"])
                    mapping = {**{i: i for i in range(12)}, 50: 12, 51: 13, 52: 14}
                    expected = {}
                    for name, payload in payloads.items():
                        match = re.match(r"model\.layers\.(\d+)\.", name)
                        if match:
                            index = int(match[1])
                            if index not in mapping:
                                continue
                            name = name[:match.start(1)] + str(mapping[index]) + name[match.end(1):]
                        expected[name] = payload
                    tensors = prepare.discover(output)
                    self.assertEqual(set(tensors), set(expected))
                    original_tensors = prepare.discover(source)
                    for name, item in tensors.items():
                        with item["source"].open("rb") as stream:
                            start, end = item["data_offsets"]
                            stream.seek(item["start"] + start)
                            self.assertEqual(stream.read(end - start), expected[name])
                        source_name = re.sub(r"^model\.layers\.(12|13|14)\.",
                                             lambda m: f"model.layers.{int(m[1]) + 38}.", name)
                        self.assertEqual(item["dtype"], original_tensors[source_name]["dtype"])
                        self.assertEqual(item["shape"], original_tensors[source_name]["shape"])
                    manifest = json.loads((output / "extraction_manifest.json").read_text())
                    self.assertEqual(manifest["retained_layer_mapping"], {str(i): j for i, j in mapping.items()})
                    self.assertEqual(manifest["mtp_layers"], 3)
                    self.assertTrue(manifest["keep_mtp"])
                    self.assertEqual(manifest["tensor_sha256"],
                                     {name: hashlib.sha256(payload).hexdigest() for name, payload in expected.items()})
                    with patch.object(sys, "argv", ["verify", str(output), "--hashes"]), \
                            contextlib.redirect_stdout(io.StringIO()):
                        verify_checkpoint.main()
                    self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_reduced_mtp_serve_and_client(self):
        for quantized in (False, True):
            for ultra in (False, True):
                with self.subTest(quantized=quantized, ultra=ultra), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    source, model = root / "source", root / "reduced"
                    checkpoint(source, quantized, ultra)
                    self.extract(source, model)
                    operator = root / "operators.json"
                    operator.write_text(json.dumps({"model_parallel_config": {}, "operator_optimization_config": {}}))
                    output = root / "server"
                    argv = ["serve", "--model", str(model), "--operator-config", str(operator),
                            "--output", str(output), "--tp", "2", "--enable-mtp", "--execute"]
                    with patch.object(sys, "argv", argv), patch.object(serve.subprocess, "call", return_value=0) as launch, \
                            contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(serve.main(), 0)
                    command = launch.call_args.args[0]
                    self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
                    self.assertEqual(json.loads(command[command.index("--speculative-config") + 1]),
                                     {"method": "mtp", "num_speculative_tokens": 3})
                    manifest_path = output / "server_manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["model_scope"], "reduced")
                    self.assertEqual(manifest["model_config"]["num_hidden_layers"], 12)
                    self.assertEqual(manifest["model_config"]["num_nextn_predict_layers"], 3)
                    for mode in ("benchmark", "profile"):
                        client = ["benchmark", "--model-dir", str(model), "--server-manifest", str(manifest_path),
                                  "--mode", mode, "--select", "decode-8k-c1", "--output", str(root / mode), "--dry-run"]
                        with patch.object(sys, "argv", client), contextlib.redirect_stdout(io.StringIO()):
                            benchmark.main()
                        self.assertFalse((root / mode).exists())

    def test_missing_mtp_metadata_or_weights_fail_without_output(self):
        for suffix in (None, "enorm.weight", "self_attn.q_a_proj.weight", "mlp.experts.0.gate_proj.weight"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, output = root / "source", root / "reduced"
                cfg, _ = checkpoint(source)
                if suffix is None:
                    cfg["num_nextn_predict_layers"] = 0
                    (source / "config.json").write_text(json.dumps(cfg))
                else:
                    tensors = prepare.discover(source)
                    del tensors[f"model.layers.51.{suffix}"]
                    prepare.write_shard(source / "replacement.safetensors", list(tensors), tensors, 1024, None)
                    (source / "model.safetensors").unlink()
                with self.assertRaises(ValueError):
                    self.extract(source, output)
                self.assertFalse(output.exists())

    def test_verifier_rejects_tampered_mapping_config_and_payload(self):
        for kind in ("mapping", "config", "payload"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                source, output = Path(tmp) / "source", Path(tmp) / "reduced"
                checkpoint(source)
                self.extract(source, output)
                if kind == "mapping":
                    path = output / "extraction_manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["retained_layer_mapping"]["50"] = 13
                    path.write_text(json.dumps(manifest))
                elif kind == "config":
                    path = output / "config.json"
                    cfg = json.loads(path.read_text())
                    cfg["sliding_window_list"][-1] = 512
                    path.write_text(json.dumps(cfg))
                else:
                    item = prepare.discover(output)["model.layers.12.enorm.weight"]
                    with item["source"].open("r+b") as stream:
                        stream.seek(item["start"] + item["data_offsets"][0])
                        stream.write(b"\xff\xff")
                with patch.object(sys, "argv", ["verify", str(output), "--hashes"]), self.assertRaises(ValueError):
                    verify_checkpoint.main()

    def test_dry_run_and_other_prefix_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "reduced"
            cfg, _ = checkpoint(source)
            self.extract(source, output, "--dry-run")
            self.assertFalse(output.exists())
            for layers in (6, 9, 12):
                config = prepare.reduced_config(cfg, layers, keep_mtp=True)
                self.assertEqual(config["swa_layers"][-3:], [layers, layers + 1, layers + 2])
                self.assertEqual(config["sliding_window_list"][-3:], [2048] * 3)
            # The original default continues to remove MTP and preserve old manifests.
            with contextlib.redirect_stdout(io.StringIO()):
                prepare.main(["--source", str(source), "--output", str(output)])
            manifest_path = output / "extraction_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("keep_mtp")
            manifest.pop("layer_prefix")
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(sys, "argv", ["verify", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                verify_checkpoint.main()

    def test_config_override_cannot_silently_drop_a_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "reduced"
            cfg, _ = checkpoint(source)
            cfg["num_nextn_predict_layers"] = 2
            override = Path(tmp) / "override.json"
            override.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "cannot change the number of predictors"):
                self.extract(source, output, "--config", str(override))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
