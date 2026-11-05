"""Per-layer page allocations retain cache capacity without hybrid padding."""

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from vllm_bench.pangu_infer.cache import CacheGroup, Caches
from vllm_bench.pangu_infer import run
from vllm_bench.pangu_infer.runtime import engine_options


@dataclass(frozen=True)
class Spec:
    natural_page_bytes: int
    block_size: int = 128
    page_size_padded: int | None = 1024

    @property
    def page_size_bytes(self):
        return self.natural_page_bytes if self.page_size_padded is None else self.page_size_padded


class Backend:
    @classmethod
    def indexes_kv_by_block_stride(cls):
        return True

    @staticmethod
    def reshape_kv_cache(raw, num_blocks, spec):
        return (num_blocks, spec.page_size_bytes)


class MomeBackend(Backend):
    pass


def fixture():
    def layer(backend, page):
        return SimpleNamespace(get_attn_backend=lambda: backend, get_kv_cache_spec=lambda cfg: Spec(page))
    layers = {"target.mla": layer(Backend, 64), "target.mome": layer(MomeBackend, 96),
              "draft_model.dsa": layer(Backend, 128)}
    cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=257),
                          compilation_config=SimpleNamespace(static_forward_context=layers))
    torch = SimpleNamespace(int8="int8", int32="int32",
        zeros=lambda size, **kwargs: SimpleNamespace(numel=lambda: size),
        arange=lambda start, end, **kwargs: SimpleNamespace(reshape=lambda *args: (start, end)))
    modules = {"torch": torch, "omni_npu.attention.backends.mome": SimpleNamespace(NPUPanguMomeBackend=MomeBackend)}
    return cfg, layers, modules


class CompactCacheTests(unittest.TestCase):
    def test_backends_without_block_stride_support_are_rejected(self):
        cfg, layers, modules = fixture()
        with patch.dict("sys.modules", modules), patch.object(Backend, "indexes_kv_by_block_stride", return_value=False):
            with self.assertRaisesRegex(ValueError, "block-stride"):
                CacheGroup("target.mla", layers["target.mla"], Spec(64), cfg, "npu:0", compact=True)

    def test_compaction_changes_only_page_padding_and_preserves_spec(self):
        cfg, layers, modules = fixture()
        for name, layer in layers.items():
            with self.subTest(name=name), patch.dict("sys.modules", modules):
                spec = layer.get_kv_cache_spec(cfg)
                padded = CacheGroup(name, layer, spec, cfg, "npu:0")
                compact = CacheGroup(name, layer, spec, cfg, "npu:0", compact=True)
                self.assertEqual(padded.blocks, compact.blocks)
                self.assertEqual(padded.block_table, compact.block_table)
                self.assertEqual(compact.spec.block_size, spec.block_size)
                self.assertEqual(spec.page_size_bytes, 1024)
                self.assertIsNone(compact.spec.page_size_padded)
                self.assertEqual(compact.raw.numel(), compact.blocks * spec.natural_page_bytes)
                self.assertEqual(compact.padding_saved_bytes, padded.raw.numel() - compact.raw.numel())
                self.assertEqual(layer.kv_cache, (compact.blocks, spec.natural_page_bytes))

    def test_target_and_draft_savings_are_accounted_together(self):
        cfg, layers, modules = fixture()
        with patch.dict("sys.modules", modules):
            padded = Caches(cfg, "npu:0")
            compact = Caches(cfg, "npu:0", compact=True)
        self.assertEqual([group.blocks for group in compact.groups], [4, 2, 4])
        self.assertEqual(compact.bytes, 4 * 64 + 2 * 96 + 4 * 128)
        self.assertEqual(compact.padding_saved_bytes, padded.bytes - compact.bytes)

    def test_cli_is_direct_only_and_not_forwarded_as_engine_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"architectures": [run.ULTRA_ARCHITECTURE],
                "num_hidden_layers": 2, "vocab_size": 32}))
            operator = root / "operator.json"
            operator.write_text('{"model_parallel_config": {}, "operator_optimization_config": {}}')
            args = ["--model", str(root), "--operator-config", str(operator), "--output", str(root / "out"),
                    "--no-mtp", "--compact-kv-cache"]
            plan = run.make_plan(run.parser().parse_args(args))
            self.assertTrue(plan["options"]["compact_kv_cache"])
            self.assertNotIn("compact_kv_cache", engine_options(plan, direct=True))
            with self.assertRaisesRegex(ValueError, "requires --engine direct"):
                run.make_plan(run.parser().parse_args([*args, "--engine", "omni"]))


if __name__ == "__main__":
    unittest.main()
