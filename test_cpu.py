#!/usr/bin/env python3
"""CPU-only tests. No ML packages, hardware discovery, or NPU execution."""
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import benchmark
import prepare_checkpoint as extract

HERE = Path(__file__).resolve().parent


def fake_checkpoint(root):
    root.mkdir()
    cfg = json.loads((HERE / "reference_config-hf.json").read_text())
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "configuration_openpangu_v2.py").write_text("# synthetic asset\n")
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
             "model.merge_mhc_module.phi.weight"]
    names += [f"model.layers.{i}.self_attn.q_a_proj.weight" for i in range(53)]
    names += [f"model.layers.{i}.mlp.experts.0.gate_proj.weight" for i in range(3, 50)]
    payloads = {name: bytes([i % 256, (i + 1) % 256]) for i, name in enumerate(names)}
    # Two input shards: verify extraction reads tensors from either source.
    index = {}
    for shard, subset in enumerate((names[::2], names[1::2])):
        header = {"__metadata__": {"format": "pt"}}
        cursor = 0
        for name in subset:
            header[name] = {"dtype": "BF16", "shape": [1], "data_offsets": [cursor, cursor + 2]}
            cursor += 2
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        filename = f"source-{shard}.safetensors"
        (root / filename).write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(payloads[n] for n in subset))
        index.update({name: filename for name in subset})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return payloads


class ExtractionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
