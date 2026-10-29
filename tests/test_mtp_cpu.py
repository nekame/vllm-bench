"""Standalone MTP CPU checks: no torch, vLLM, torch_npu or hardware imports."""
import contextlib
from dataclasses import dataclass
from enum import Enum
import hashlib
import importlib.abc
import io
import inspect
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import MagicMock, patch

from vllm_bench.mtp_profile import mtp_common
from vllm_bench.layer_profile import prepare_checkpoint
from vllm_bench.mtp_profile import prepare_mtp_checkpoint
from vllm_bench.mtp_profile import profile_mtp
from vllm_bench.mtp_profile import record_mtp
from vllm_bench.mtp_profile import mtp_state
from vllm_bench.mtp_profile import mtp_runtime
from vllm_bench.mtp_profile import verify_mtp_checkpoint

HERE = Path(__file__).resolve().parent


def checkpoint(path, quantized=False, ultra=False):
    path.mkdir()
    cfg = json.loads((HERE.parent / "conf/common/reference_config-hf.json").read_text())
    if ultra:
        cfg["architectures"] = ["PanguUltraMoEForCausalLM"]
    if quantized:
        cfg["quantization_config"] = {"quant_method": "mxfp8", "ignore": ["lm_head"]}
    (path / "config.json").write_text(json.dumps(cfg))
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for index in range(53):
        names += [f"model.layers.{index}.self_attn.q_a_proj.weight",
                  f"model.layers.{index}.mlp.experts.0.gate_proj.weight"]
        if index >= 50:
            names += [f"model.layers.{index}.{suffix}" for suffix in
                      ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight",
                       "shared_head.head.weight", "embed_tokens.weight")]
    data, header, cursor = {}, {}, 0
    for i, name in enumerate(names):
        is_fp8 = quantized and ("self_attn" in name or "mlp.experts" in name)
        data[name] = bytes([(i + 1) % 255]) * (64 if is_fp8 else 2)
        header[name] = {"dtype": "F8_E4M3" if is_fp8 else "BF16",
                        "shape": [2, 32] if is_fp8 else [1],
                        "data_offsets": [cursor, cursor + len(data[name])]}
        cursor += len(data[name])
        if is_fp8:
            scale = name.removesuffix(".weight") + ".weight_scale"
            data[scale] = bytes([127, 128])
            header[scale] = {"dtype": "F8_E8M0", "shape": [2, 1], "data_offsets": [cursor, cursor + 2]}
            cursor += 2
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(data.values()))
    return cfg, data


class MTPCheckpointTests(unittest.TestCase):
    def test_bf16_and_mxfp8_preserve_bytes_config_indices_and_scales(self):
        for quantized in (False, True):
            for ultra in (False, True):
                with self.subTest(quantized=quantized, ultra=ultra), tempfile.TemporaryDirectory() as tmp:
                    source, output = Path(tmp) / "source", Path(tmp) / "mtp"
                    cfg, original = checkpoint(source, quantized, ultra)
                    with contextlib.redirect_stdout(io.StringIO()):
                        manifest = prepare_mtp_checkpoint.prepare(source, output, shard_bytes=100, hash_tensors=True)
                    self.assertEqual(json.loads((output / "config.json").read_text()), cfg)
                    self.assertEqual(manifest["mtp_layers"], [50, 51, 52])
                    tensors = prepare_checkpoint.discover(output)
                    self.assertNotIn("model.norm.weight", tensors)
                    self.assertNotIn("model.layers.49.self_attn.q_a_proj.weight", tensors)
                    for name, descriptor in tensors.items():
                        with descriptor["source"].open("rb") as f:
                            begin, end = descriptor["data_offsets"]
                            f.seek(descriptor["start"] + begin)
                            payload = f.read(end - begin)
                        self.assertEqual(payload, original[name])
                        self.assertEqual(manifest["tensor_sha256"][name], hashlib.sha256(payload).hexdigest())
                    if quantized:
                        self.assertEqual(tensors["model.layers.50.self_attn.q_a_proj.weight_scale"]["dtype"], "F8_E8M0")
                    self.assertEqual(mtp_common.checkpoint_config(output)[0], cfg)
                    self.assertTrue(verify_mtp_checkpoint.verify(output, hashes=True)["hashes_verified"])

    def test_verifier_detects_payload_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "mtp"
            checkpoint(source, quantized=True)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_mtp_checkpoint.prepare(source, output, hash_tensors=True)
            item = next(iter(prepare_checkpoint.discover(output).values()))
            with item["source"].open("r+b") as f:
                f.seek(item["start"])
                original = f.read(1)
                f.seek(item["start"])
                f.write(bytes([original[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_mtp_checkpoint.verify(output, hashes=True)

    def test_missing_predictor_and_shared_weights_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            cfg, _ = checkpoint(source)
            tensors = prepare_checkpoint.discover(source)
            for missing in ("lm_head.weight", "model.layers.51.eh_proj.weight"):
                reduced = {k: v for k, v in tensors.items() if k != missing}
                with self.subTest(missing=missing), self.assertRaises(ValueError):
                    prepare_mtp_checkpoint.select_mtp(cfg, reduced)
            cfg["num_nextn_predict_layers"] = 0
            with self.assertRaisesRegex(ValueError, "reduced 12-layer"):
                prepare_mtp_checkpoint.select_mtp(cfg, tensors)

    def test_dry_run_does_not_create_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "mtp"
            checkpoint(source)
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_mtp_checkpoint.prepare(source, output, dry_run=True)
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare_mtp_checkpoint.prepare(source, output)


class PlanningTests(unittest.TestCase):
    def prepare(self, root):
        source, model = root / "source", root / "mtp"
        cfg, _ = checkpoint(source, quantized=True)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_mtp_checkpoint.prepare(source, model)
        operator = root / "operator.json"
        operator.write_text(json.dumps({"model_parallel_config": {}, "operator_optimization_config": {}}))
        return cfg, model, operator

    def test_actual_proposer_head_selection(self):
        cfg = {"num_hidden_layers": 50, "num_nextn_predict_layers": 3}
        self.assertEqual(mtp_common.layer_sequence(cfg, 3), [50, 51, 52])
        self.assertEqual(mtp_common.layer_sequence(cfg, 7), [50] * 7)

    def test_dry_plan_modes_and_topologies_without_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model, operator = self.prepare(root)
            for topology in mtp_common.TOPOLOGIES:
                for execution in ("eager", "compiled", "graph"):
                    output = root / "result"
                    stream = io.StringIO()
                    with patch.object(subprocess, "run") as run, contextlib.redirect_stdout(stream):
                        self.assertEqual(profile_mtp.main(["--model", str(model), "--operator-config", str(operator),
                                                          "--topology", topology, "--execution", execution,
                                                          "--output", str(output)]), 0)
                    run.assert_not_called()
                    self.assertFalse(output.exists())
                    plan = json.loads(stream.getvalue())
                    self.assertEqual(plan["world_size"], 1 if topology == "tp1" else 8)
                    self.assertEqual(plan["total_sequences"], 8 if topology == "dp8-ep8" else 1)
                    self.assertEqual(plan["expected_layers"]["7"], [50] * 7)
                    self.assertFalse(plan["target_weights_loaded"])

    def test_wrong_capture_topology_and_config_fail(self):
        cfg = {"num_hidden_layers": 50}
        meta = {"format": "pangu-mtp-capture-v1", "topology": "tp8", "config_sha256": mtp_common.digest(cfg)}
        with self.assertRaisesRegex(ValueError, "topology"):
            mtp_common.validate_record(meta, topology="dp8-ep8", config=cfg)
        with self.assertRaisesRegex(ValueError, "config differs"):
            mtp_common.validate_record(meta, topology="tp8", config={})

    def test_recorder_dry_run_never_launches_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp) / "capture"
            self.assertEqual(record_mtp.main(["--output", str(output), "--", "python", "launch.py"]), 0)
            self.assertFalse(output.exists())
            run.assert_not_called()

    def test_replay_requires_all_ranks_and_uses_recorded_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, model, operator = self.prepare(root)
            folder = root / "capture"
            for rank in range(8):
                path = folder / f"rank-{rank:03d}"
                path.mkdir(parents=True)
                meta = {"format": "pangu-mtp-capture-v1", "topology": "tp8", "rank": rank,
                        "world_size": 8, "complete": True, "depth": 7,
                        "config_sha256": mtp_common.digest(cfg), "versions": {}, "runtime_limits": {},
                        "additional_config": {}, "input_tokens": 1, "batch_per_dp_rank": 1, "padded_tokens": 1,
                        "compilation_options": {"mode": 3, "cudagraph_mode": "FULL"},
                        "operator_config": json.loads(operator.read_text())}
                (path / "capture.json").write_text(json.dumps(meta))
            args = ["--model", str(model), "--operator-config", str(operator), "--topology", "tp8",
                    "--input", "replay", "--capture", str(folder), "--output", str(root / "result")]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                profile_mtp.main(args)
            self.assertEqual(json.loads(stream.getvalue())["options"]["depths"], [7])
            (folder / "rank-007" / "capture.json").unlink()
            with self.assertRaises(OSError):
                profile_mtp.main(args)

    def test_eager_overrides_compilation_and_graph_rejects_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model, operator = self.prepare(root)
            config = root / "compilation.json"
            config.write_text(json.dumps({"mode": 0, "cudagraph_mode": "FULL"}))
            args = ["--model", str(model), "--operator-config", str(operator), "--topology", "tp8",
                    "--compilation-config", str(config), "--output", str(root / "result")]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile_mtp.main(args + ["--execution", "graph"])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                profile_mtp.main(args + ["--execution", "eager"])
            self.assertEqual(json.loads(stream.getvalue())["compilation_config"], {"mode": 0, "cudagraph_mode": "NONE"})

    def test_depths_launch_as_separate_jobs_and_failure_stops_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model, operator = self.prepare(root)
            jobs = []

            def launch(command, **kwargs):
                self.assertEqual(command[:7], [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                                "--nnodes=1", "--nproc-per-node=8", "--module"])
                self.assertEqual(command[7], "vllm_bench.mtp_profile.mtp_runtime")
                from vllm_bench.common.paths import SRC_ROOT
                self.assertEqual(kwargs["env"]["PYTHONPATH"].split(":")[0], str(SRC_ROOT))
                plan = json.loads(Path(command[-1]).read_text())
                jobs.append(plan["options"]["depths"])
                row = {"depth": plan["options"]["depths"][0], "critical_path_median_s": 0.1,
                       "total_sequences": 1, "proposed_tokens_per_s": 30}
                (Path(plan["options"]["output"]) / "summary.json").write_text(json.dumps([row]))
                return SimpleNamespace(returncode=0)

            args = ["--model", str(model), "--operator-config", str(operator), "--topology", "tp8", "--execute"]
            with patch.object(subprocess, "run", side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(profile_mtp.main(args + ["--output", str(root / "result")]), 0)
            self.assertEqual(jobs, [[3], [7]])
            self.assertEqual([x["depth"] for x in json.loads((root / "result" / "summary.json").read_text())], [3, 7])
            with patch.object(subprocess, "run", return_value=SimpleNamespace(returncode=9)) as run, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(profile_mtp.main(args + ["--output", str(root / "failed")]), 9)
                self.assertEqual(run.call_count, 1)

    def test_imports_and_planning_do_not_require_ml_packages(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'vllm', 'omni_npu', 'numpy'}:
            raise AssertionError('Unexpected ML import: ' + fullname)
sys.meta_path.insert(0, Block())
from vllm_bench.mtp_profile import prepare_mtp_checkpoint, profile_mtp, record_mtp, mtp_common, mtp_state, mtp_capture, mtp_runtime
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=HERE, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tp1_launches_one_worker_without_expert_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, model, operator = self.prepare(root)

            def launch(command, **kwargs):
                self.assertIn("--nproc-per-node=1", command)
                plan = json.loads(Path(command[-1]).read_text())
                self.assertEqual(plan["topology"], {"tp": 1, "dp": 1, "ep": False})
                self.assertEqual(plan["total_sequences"], 1)
                self.assertEqual(plan["world_size"], 1)
                row = {"depth": 3, "critical_path_median_s": 0.1,
                       "total_sequences": 1, "proposed_tokens_per_s": 30}
                (Path(plan["options"]["output"]) / "summary.json").write_text(json.dumps([row]))
                return SimpleNamespace(returncode=0)

            with patch.object(subprocess, "run", side_effect=launch) as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(profile_mtp.main(["--model", str(model), "--operator-config", str(operator),
                                                  "--topology", "tp1", "--depths", "3", "--execute",
                                                  "--output", str(root / "result")]), 0)
            execute.assert_called_once()

    def test_tp1_replay_uses_one_rank_and_rejects_wrong_world_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, model, operator = self.prepare(root)
            folder = root / "capture/rank-000"
            folder.mkdir(parents=True)
            meta = {"format": "pangu-mtp-capture-v1", "topology": "tp1", "rank": 0,
                    "world_size": 1, "complete": True, "depth": 3,
                    "config_sha256": mtp_common.digest(cfg), "versions": {}, "runtime_limits": {},
                    "additional_config": {}, "input_tokens": 1, "batch_per_dp_rank": 1, "padded_tokens": 1,
                    "compilation_options": {"mode": 3, "cudagraph_mode": "FULL"},
                    "operator_config": json.loads(operator.read_text())}
            path = folder / "capture.json"
            path.write_text(json.dumps(meta))
            args = ["--model", str(model), "--operator-config", str(operator), "--topology", "tp1",
                    "--input", "replay", "--capture", str(folder.parent), "--output", str(root / "result")]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                profile_mtp.main(args)
            self.assertEqual(json.loads(stream.getvalue())["options"]["depths"], [3])
            meta["world_size"] = 8
            path.write_text(json.dumps(meta))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                profile_mtp.main(args)

    def test_recorder_topology_detection_and_summary_rank_count(self):
        for name, settings in mtp_common.TOPOLOGIES.items():
            with self.subTest(topology=name), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(mtp_common.topology_name(**settings), name)
                size = 1 if name == "tp1" else 8
                root = Path(tmp)
                for rank in range(size):
                    folder = root / f"rank-{rank:03d}"
                    folder.mkdir()
                    (folder / "results.json").write_text(json.dumps([{
                        "latencies_s": [1 + rank, 3 + rank], "median_s": 2 + rank}]))
                sequences = settings["dp"]
                mtp_runtime.aggregate({"topology": settings, "total_sequences": sequences,
                                       "options": {"output": str(root), "depths": [3], "repeats": 2}})
                summary = json.loads((root / "summary.json").read_text())[0]
                self.assertEqual(summary["critical_path_median_s"], size + 1)
                self.assertEqual(len(summary["per_rank_median_s"]), size)
                self.assertAlmostEqual(summary["proposed_tokens_per_s"], sequences * 3 / (size + 1))
        with self.assertRaisesRegex(ValueError, "Unsupported MTP topology"):
            mtp_common.topology_name(1, 1, True)


class StateTests(unittest.TestCase):
    class Tensor:
        def __init__(self, data, dtype="bf16", device="cpu"):
            self.data, self.dtype, self.device = list(data), dtype, device
            self.shape = (len(data),)

        def numel(self):
            return len(self.data)

        def element_size(self):
            return 2

        def detach(self):
            return self

        def cpu(self):
            return self.to("cpu")

        def clone(self):
            return type(self)(self.data, self.dtype, self.device)

        def to(self, device):
            return type(self)(self.data, self.dtype, str(device))

        def copy_(self, other):
            self.data[:] = other.data

        def zero_(self):
            self.data[:] = [0] * len(self.data)

    def modules(self):
        class Placeholder:
            pass
        return {"torch": SimpleNamespace(Tensor=self.Tensor, dtype=Placeholder, device=Placeholder, Generator=Placeholder),
                "numpy": SimpleNamespace(ndarray=Placeholder, generic=Placeholder)}

    def test_tensor_tree_roundtrip_preserves_values_dtype_and_references(self):
        tensor = self.Tensor([3, 9], device="npu:3")
        with patch.dict(sys.modules, self.modules()):
            encoder = mtp_state.Codec()
            tree = encoder.encode({"inputs": [tensor, tensor], "metadata": (7, None)})
            restored = mtp_state.Codec(encoder.tensors, device="npu:1").decode(tree)
        self.assertEqual(len(encoder.tensors), 1)
        self.assertEqual(restored["inputs"][0].data, [3, 9])
        self.assertEqual(restored["inputs"][0].device, "npu:1")
        self.assertIs(restored["inputs"][0], restored["inputs"][1])
        self.assertEqual(restored["metadata"], (7, None))

    def test_cache_restore_keeps_destination_and_rejects_layout_changes(self):
        destination, source = self.Tensor([0, 0], device="npu:0"), self.Tensor([1, 2])
        with patch.dict(sys.modules, self.modules()):
            mtp_state.copy_tree({"cache": [destination]}, {"cache": [source]})
            self.assertEqual(destination.data, [1, 2])
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                mtp_state.copy_tree(destination, self.Tensor([1]))
            with self.assertRaisesRegex(ValueError, "layout mismatch"):
                mtp_state.copy_tree(destination, self.Tensor([1, 2], dtype="fp8"))

    def test_input_restore_reuses_graph_input_buffers(self):
        tensor = self.Tensor([4, 5], device="npu:0")
        with patch.dict(sys.modules, self.modules()):
            encoder = mtp_state.Codec()
            tree = encoder.encode({"hidden": tensor})
            buffers = {}
            first = mtp_state.Codec(encoder.tensors, "npu:0", reuse_buffers=buffers).decode(tree)
            first["hidden"].data[0] = 99
            second = mtp_state.Codec(encoder.tensors, "npu:0", reuse_buffers=buffers).decode(tree)
        self.assertIs(first["hidden"], second["hidden"])
        self.assertEqual(second["hidden"].data, [4, 5])

    def test_capture_limit_and_unapproved_object_fail_explicitly(self):
        with patch.dict(sys.modules, self.modules()):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                mtp_state.Codec(limit_bytes=1).encode(self.Tensor([1]))
            with self.assertRaisesRegex(TypeError, "Unsupported state class"):
                mtp_state.Codec().encode(SimpleNamespace(callback="unserializable"))
        with self.assertRaisesRegex(ValueError, "Unsupported state type"):
            mtp_state.Codec.resolve("os:system")

    def test_snapshot_without_optional_buffer_and_reject_missing_required_state(self):
        buf = SimpleNamespace(cpu=self.Tensor([1]), gpu=self.Tensor([1]))
        runner = SimpleNamespace(num_accepted_tokens=buf, num_prompt_tokens=buf,
                                 drafter=SimpleNamespace(fix_multi_mtp_kvcache=False))
        state = mtp_state.runner_state(runner, [])
        self.assertEqual(set(state["buffers"]), {"num_accepted_tokens", "num_prompt_tokens"})
        runner.drafter.fix_multi_mtp_kvcache = True
        with self.assertRaisesRegex(ValueError, "requires cu_num_draft_tokens"):
            mtp_state.runner_state(runner, [])

    def test_replay_allocates_missing_optional_buffer_and_restores_counts(self):
        buf = SimpleNamespace(cpu=self.Tensor([0, 0], dtype="int32"),
                              gpu=self.Tensor([0, 0], dtype="int32"))
        buf.copy_to_gpu = lambda: buf.gpu.copy_(buf.cpu)
        runner = SimpleNamespace(max_num_reqs=8, _make_buffer=MagicMock(return_value=buf))
        state = {"buffers": {"cu_num_draft_tokens": {"gpu": self.Tensor([3, 6], dtype="int32"),
                                                       "cpu": self.Tensor([3, 6], dtype="int32")}},
                 "block_tables": []}
        with patch.dict(sys.modules, self.modules()):
            mtp_state.restore_runner(runner, state)
        runner._make_buffer.assert_called_once_with(2, dtype="int32")
        self.assertEqual(runner.cu_num_draft_tokens.gpu.data, [3, 6])
        self.assertEqual(runner.cu_num_draft_tokens.cpu.data, [3, 6])
        self.assertIs(mtp_state.ensure_draft_token_buffer(runner), buf)
        self.assertEqual(runner._make_buffer.call_count, 1)


class SyntheticBufferTests(unittest.TestCase):
    def test_synthetic_bootstrap_with_and_without_multi_mtp_cache_fix(self):
        for enabled in (False, True):
            with self.subTest(fix_multi_mtp_kvcache=enabled):
                proposer = SimpleNamespace(fix_multi_mtp_kvcache=enabled, n_predict=3)
                optional = SimpleNamespace(cpu=MagicMock(), gpu=MagicMock(), copy_to_gpu=MagicMock())
                runner = SimpleNamespace(max_model_len=64, max_num_reqs=2,
                                         kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
                                         num_prompt_tokens=SimpleNamespace(cpu=MagicMock(), copy_to_gpu=MagicMock()),
                                         num_accepted_tokens=SimpleNamespace(cpu=MagicMock(), copy_to_gpu=MagicMock()),
                                         input_batch=SimpleNamespace(target_token_ids_cache=MagicMock(),
                                                                     target_model_hidden_states_cache=MagicMock()),
                                         _make_buffer=MagicMock(return_value=optional))
                torch = SimpleNamespace(int32="int32", npu=SimpleNamespace(synchronize=MagicMock()))
                modules = {"torch": torch, "vllm": ModuleType("vllm"),
                           "vllm.v1": ModuleType("vllm.v1"),
                           "vllm.v1.kv_cache_interface": SimpleNamespace(MambaSpec=type("MambaSpec", (), {}))}
                args = {"batch_per_dp_rank": 1, "context_length": 4, "prefill_chunk": 4, "seed": 1}
                with patch.dict(sys.modules, modules), patch.object(mtp_runtime, "cache_views", return_value={}), \
                        patch.object(mtp_runtime, "synthetic_inputs", return_value={}), patch.object(mtp_runtime, "execute") as execute:
                    mtp_runtime.initialize_synthetic(runner, proposer, args)
                execute.assert_called_once()
                if enabled:
                    runner._make_buffer.assert_called_once_with(2, dtype="int32")
                    self.assertIs(runner.cu_num_draft_tokens, optional)
                    optional.cpu.__getitem__.return_value.fill_.assert_called_with(0)
                else:
                    runner._make_buffer.assert_not_called()
                    self.assertFalse(hasattr(runner, "cu_num_draft_tokens"))


class DiagnosticConfigTests(unittest.TestCase):
    def test_effective_config_writes_dtype_device_and_nested_dataclasses(self):
        class Dtype:
            def __str__(self):
                return "torch.bfloat16"

            def __deepcopy__(self, memo):
                raise AssertionError("Diagnostic serialization must not deepcopy runtime objects")

        class Device:
            def __str__(self):
                return "npu:3"

        class Mode(Enum):
            EAGER = "eager"

        @dataclass
        class Nested:
            device: object
            enabled: bool = False
            count: int = 3

        @dataclass
        class Config:
            dtype: object
            operator_opt_config: object

        dtype = Dtype()
        config = Config(dtype, {"nested": Nested(Device()), "values": (Mode.EAGER, 0.5, None), "path": Path("config.json")})
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"torch": SimpleNamespace(dtype=Dtype, device=Device)}):
            value = {"operator_config": mtp_runtime.diagnostic_config(config)}
            path = Path(tmp) / "effective_config.json"
            mtp_common.write_json(path, value)
            restored = json.loads(path.read_text())
        self.assertEqual(restored, {"operator_config": {"dtype": "torch.bfloat16",
                         "operator_opt_config": {"nested": {"device": "npu:3", "enabled": False, "count": 3},
                                                 "values": ["eager", 0.5, None], "path": "config.json"}}})
        self.assertIs(config.dtype, dtype)

    def test_unknown_objects_and_strict_json_paths_still_fail(self):
        with self.assertRaisesRegex(TypeError, "Unsupported effective-config value"):
            mtp_runtime.diagnostic_config(object())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strict.json"
            with self.assertRaises(TypeError):
                mtp_common.write_json(path, {"unsupported": object()})
            self.assertFalse(path.exists())
        with self.assertRaises(TypeError):
            mtp_common.digest({"unsupported": object()})


class LoaderAdapterTests(unittest.TestCase):
    def test_shared_weights_use_target_globals_and_head_is_tied(self):
        registered = {}

        class PlainEmbedding:
            weight = SimpleNamespace(dtype="bf16")

            def __call__(self, input_ids):
                return input_ids

        class NPUEmbedding:
            def __init__(self, num_embeddings, embedding_dim, *, params_dtype, quant_config, prefix):
                self.settings = (num_embeddings, embedding_dim, params_dtype, quant_config, prefix)
                self.calls = []

            def __call__(self, input_ids, enable_scatter=False):
                self.calls.append(enable_scatter)
                return input_ids

        class Base:
            def __init__(self, *, vllm_config, prefix=""):
                self.received_config = vllm_config
                self.received_prefix = prefix
                self.model = SimpleNamespace(layers={str(i): SimpleNamespace(
                    shared_head=SimpleNamespace(head=object())) for i in (50, 51, 52)})
                self.model.embed_tokens = self.original_embedding = PlainEmbedding()

            def load_weights(self, weights):
                self.loaded = dict(weights)
                return set(self.loaded)

        modules = {}
        for path in ("vllm.model_executor.models", "omni_npu.v1.models.pangu.pangu_v2_moe_mtp",
                     "omni_npu.v1.models.pangu.pangu_ultra_moe_mtp",
                     "omni_npu.v1.layers.vocab_parallel_embedding"):
            parts = path.split(".")
            for i in range(1, len(parts) + 1):
                name = ".".join(parts[:i])
                if name not in modules:
                    modules[name] = ModuleType(name)
                    modules[name].__path__ = []
        modules["vllm.model_executor.models"].ModelRegistry = SimpleNamespace(
            register_model=lambda arch, cls: registered.update({arch: cls}))
        modules["omni_npu.v1.models.pangu.pangu_v2_moe_mtp"].OpenPanguV2MTP = Base
        modules["omni_npu.v1.models.pangu.pangu_ultra_moe_mtp"].OpenPanguMTP = Base
        modules["omni_npu.v1.layers.vocab_parallel_embedding"].NPUVocabParallelEmbedding = NPUEmbedding
        for arch, registered_name in (("OpenPanguV2ForCausalLM", "OpenPanguV2MTPModel"),
                                       ("PanguUltraMoEForCausalLM", "OpenPanguMTPModel")):
            with self.subTest(arch=arch), patch.dict(sys.modules, modules):
                mtp_runtime.install_standalone_loader({"architectures": [arch], "num_hidden_layers": 50})
                cls = registered[registered_name]
                # Exercise vLLM's signature-based argument selection, rather
                # than directly supplying kwargs the real loader might omit.
                parameters = inspect.signature(cls.__init__).parameters
                config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=8, hidden_size=4)),
                                         quant_config=object())
                prefix = "draft"
                kwargs = {"vllm_config": config, "prefix": prefix} if {"vllm_config", "prefix"} <= parameters.keys() else {}
                model = cls(**kwargs)
                self.assertIs(model.received_config, config)
                self.assertEqual(model.received_prefix, prefix)
                if arch == "PanguUltraMoEForCausalLM":
                    self.assertIsInstance(model.model.embed_tokens, NPUEmbedding)
                    self.assertEqual(model.model.embed_tokens.settings,
                                     (8, 4, "bf16", config.quant_config, "draft.model.embed_tokens"))
                    for scatter in (False, True):
                        tokens = [1, 2]
                        self.assertIs(model.model.embed_tokens(tokens, enable_scatter=scatter), tokens)
                    self.assertEqual(model.model.embed_tokens.calls, [False, True])
                    with self.assertRaises(TypeError):
                        model.original_embedding([1], enable_scatter=False)
                else:
                    self.assertIs(model.model.embed_tokens, model.original_embedding)
                heads = [layer.shared_head.head for layer in model.model.layers.values()]
                self.assertTrue(all(head is heads[0] for head in heads))
                head, embedding, scale, local = object(), object(), object(), object()
                model.load_weights(iter([("lm_head.weight", head), ("model.embed_tokens.weight", embedding),
                                         ("lm_head.weight_scale", scale),
                                         ("model.layers.50.shared_head.head.weight", local),
                                         ("model.layers.50.embed_tokens.weight", local)]))
                self.assertIs(model.loaded["model.layers.50.shared_head.head.weight"], head)
                self.assertIs(model.loaded["model.layers.50.embed_tokens.weight"], embedding)
                self.assertIs(model.loaded["model.layers.50.shared_head.head.weight_scale"], scale)


if __name__ == "__main__":
    unittest.main()
