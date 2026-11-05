"""Memory reporting and first-call diagnostics without NPU execution."""

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from vllm_bench.pangu_infer.execution import Execution
from vllm_bench.pangu_infer.memory import log_memory
from vllm_bench.pangu_infer import run


class MemoryDiagnosticsTests(unittest.TestCase):
    def test_snapshot_reports_device_and_allocator_memory_separately(self):
        mib = 1 << 20
        npu = SimpleNamespace(mem_get_info=Mock(return_value=(525 * mib, 94000 * mib)),
                              memory_allocated=Mock(return_value=88000 * mib),
                              memory_reserved=Mock(return_value=90000 * mib))
        output = io.StringIO()
        with patch.dict("sys.modules", {"torch": SimpleNamespace(npu=npu)}), redirect_stdout(output):
            log_memory("compile_begin", "npu:3", model="mtp", tokens=1)
        record = json.loads(output.getvalue().removeprefix("[memory] "))
        self.assertEqual(record["free_mib"], 525)
        self.assertEqual(record["allocated_mib"], 88000)
        self.assertEqual(record["reserved_unallocated_mib"], 2000)
        self.assertEqual(record["model"], "mtp")
        npu.mem_get_info.assert_called_once_with("npu:3")

    def test_unavailable_memory_api_is_reported_without_raising(self):
        npu = SimpleNamespace(mem_get_info=Mock(side_effect=RuntimeError("device unavailable")))
        output = io.StringIO()
        with patch.dict("sys.modules", {"torch": SimpleNamespace(npu=npu)}), redirect_stdout(output):
            log_memory("model_failed", "npu:3")
        record = json.loads(output.getvalue().removeprefix("[memory] "))
        self.assertEqual(record["diagnostic_error"], "device unavailable")

    def test_diagnostic_call_executes_once_and_reports_only_first_success(self):
        execution = Execution.__new__(Execution)
        execution.calls = 0
        execution.log_memory = Mock()
        result = object()

        def forward(self, *args, **kwargs):
            self.calls += 1
            return result

        with patch.object(Execution, "__call__", forward):
            self.assertIs(execution.diagnostic_call("ids", 8192, 1, 8192), result)
            self.assertIs(execution.diagnostic_call("ids", 8193, 1, 8192), result)
        self.assertEqual(execution.calls, 2)
        self.assertEqual([call.args[0] for call in execution.log_memory.call_args_list],
                         ["model_begin", "model_end"])

    def test_failure_preserves_exception_and_reports_phase_even_after_warmup(self):
        execution = Execution.__new__(Execution)
        execution.calls = 5
        execution.log_memory = Mock()
        error = RuntimeError("allocation failed")

        def forward(self, *args, **kwargs):
            self.memory_phase = "backend_compile"
            raise error

        with patch.object(Execution, "__call__", forward):
            with self.assertRaises(RuntimeError) as caught:
                execution.diagnostic_call("ids", 8192, 1, 8192, draft_index=1)
        self.assertIs(caught.exception, error)
        execution.log_memory.assert_called_once_with(
            "model_failed", phase="backend_compile", start=8192, draft_step=1)

    def test_cli_rejects_unsupported_omni_diagnostics(self):
        args = run.parser().parse_args([
            "--model", "/tmp/model", "--operator-config", "/tmp/operator.json",
            "--output", "/tmp/output", "--engine", "omni", "--memory-diagnostics"])
        with self.assertRaisesRegex(ValueError, "requires --engine direct"):
            run.make_plan(args)


if __name__ == "__main__":
    unittest.main()
