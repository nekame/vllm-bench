"""Launch planning, process lifecycle, fixed payloads and TP2 capture checks."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from attention_fixtures import dataset as attention_data, payload
from test_cpu import _Connection
from test_indexer_ki_cpu import dataset as ki_data
from vllm_bench.common import openai_client
from vllm_bench.mtp_debug import attention_trace, compare_attention, compare_indexer_ki, reproduce, serve
from vllm_bench.mtp_debug.ranks import expected_rank_count


def inputs(root):
    model, operator = root / "model", root / "operator.json"
    model.mkdir()
    model.joinpath("config.json").write_text(json.dumps({"architectures": ["PanguUltraMoEForCausalLM"],
        "num_hidden_layers": 12, "num_nextn_predict_layers": 1, "vocab_size": 256,
        "max_position_embeddings": 32768, "bos_token_id": 0, "eos_token_id": [1, 2], "pad_token_id": 3}))
    operator.write_text(json.dumps({"model_parallel_config": {}, "operator_optimization_config": {
        "use_noncontiguous_kv": True, "use_quant_lightning_indexer_v2": True}}))
    return ["--model", str(model), "--operator-config", str(operator), "--output", str(root / "run")]


class LaunchTests(unittest.TestCase):
    def test_dry_run_no_profiler_exact_defaults_and_no_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"VLLM_TORCH_PROFILER_DIR": "/old/traces"}):
            root = Path(directory)
            args = inputs(root)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(reproduce.main(args), 0)
            plan = json.loads(output.getvalue())
            command = plan["server"]["engine_command"]
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "2")
            self.assertEqual(json.loads(command[command.index("--profiler-config") + 1]), {"profiler": None})
            self.assertEqual(json.loads(command[command.index("--speculative-config") + 1])["num_speculative_tokens"], 1)
            self.assertEqual(command[command.index("--max-model-len") + 1], "8320")
            self.assertIn("--enforce-eager", command)
            self.assertIn("--no-enable-prefix-caching", command)
            self.assertNotIn("torch_profiler_dir", json.dumps(plan))
            options = serve.parser().parse_args(args)
            _, env = serve.build_debug_plan(options)
            self.assertNotIn("VLLM_TORCH_PROFILER_DIR", env)
            self.assertFalse((root / "run").exists())

    def test_bad_operator_config_and_managed_override_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inputs(root)
            extra = root / "extra.json"
            for flags in (["--profiler_config.profiler=torch"], ["--kv_cache_dtype=auto"], ["--no-enforce-eager"],
                          ["--pipeline-parallel-size", "2"], ["--port", "9000"]):
                extra.write_text(json.dumps(flags))
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    reproduce.main([*args, "--engine-cli-json", str(extra)])
            config = json.loads((root / "operator.json").read_text())
            config["operator_optimization_config"]["use_quant_lightning_indexer_v2"] = False
            (root / "operator.json").write_text(json.dumps(config))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                reproduce.main(args)
            self.assertFalse((root / "run").exists())

    def test_prompt_reproducible_and_sse_retains_actual_ids(self):
        config = {"vocab_size": 6, "bos_token_id": 0, "eos_token_id": [1, 2], "pad_token_id": 3}
        a = reproduce.make_prompt(config, 8192, 123)
        self.assertEqual(a, reproduce.make_prompt(config, 8192, 123))
        self.assertEqual(set(a), {4, 5})
        row = openai_client.stream_request("http://test.invalid", {"request_id": "x", "prompt": [4, 5], "max_tokens": 3},
            5, retain_token_ids=True, connection_factory=lambda *args: (_Connection(), ""))
        self.assertTrue(row["success"])
        self.assertEqual(row["output_token_ids"], [1, 2, 3])
        with self.assertRaises(ValueError):
            reproduce.make_prompt({"vocab_size": 1, "eos_token_id": 0}, 1, 0)


class WorkflowTests(unittest.TestCase):
    def test_loopback_server_full_client_capture_analysis_and_shutdown(self):
        # Real child/process group + real SSE transport, with synthetic worker captures.
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        except OSError as error:
            self.skipTest(f"Loopback sockets unavailable: {error}")
        original_popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = inputs(root)
            source = root / "stub_server.py"
            source.write_text('''
import http.server,json,sys
from pathlib import Path
capture=Path(sys.argv[2]); frames=json.loads(Path(sys.argv[3]).read_text())
class Handler(http.server.BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  body=json.dumps({'data':[{'id':'pangu-ki'}]}).encode()
  self.send_response(200); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
 def do_POST(self):
  request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  if (capture/'ARM').exists():
   label=(capture/'LABEL').read_text().strip()
   for frame in frames:
    if frame['label']==label:
     with (capture/('ki-rank-%s-pid-%s.jsonl'%(frame['rank'],frame['pid']))).open('a') as f: f.write(json.dumps(frame)+'\\n')
  events=[{'choices':[{'token_ids':[5,6,7],'finish_reason':'length'}]},
          {'choices':[],'usage':{'prompt_tokens':len(request['prompt']),'completion_tokens':3}}]
  body=''.join('data: '+json.dumps(e)+'\\n\\n' for e in events)+'data: [DONE]\\n\\n'
  data=body.encode(); self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
''')
            children = []
            def spawn(command, **kwargs):
                capture = Path(command[command.index("--capture-output") + 1])
                data = [f for f in ki_data(capture) if f["rank"] < 2]
                prompt = reproduce.make_prompt(json.loads((root / "model/config.json").read_text()), 4, 123)
                for frame in data:
                    frame["topology"] = {"tp_size": 2, "tp_rank": frame["rank"]}
                    for call in frame["calls"]:
                        call["row_input_ids"] = [prompt[p] if p < 4 else 5 for p in call["row_positions"]]
                fixture = root / "stub_frames.json"
                fixture.write_text(json.dumps(data))
                process = original_popen([sys.executable, str(source), str(port), str(capture), str(fixture)], **kwargs)
                children.append(process)
                return process
            with patch.object(reproduce.subprocess, "Popen", side_effect=spawn), contextlib.redirect_stdout(io.StringIO()):
                code = reproduce.main([*args, "--port", str(port), "--input-tokens", "4", "--output-tokens", "3",
                                       "--warmup-requests", "1", "--startup-timeout", "5", "--shutdown-timeout", "1", "--execute"])
            result = json.loads((root / "run/result.json").read_text())
            self.assertEqual(code, 0, result)
            self.assertEqual(result["result"], "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI")
            self.assertTrue(result["evidence"]["raw_ki_exact_equal"])
            self.assertIsNotNone(children[0].poll())
            self.assertTrue((root / "run/comparisons/request-01-vs-request-02-replays/left/input.json").exists())

    def exercise(self, root, fault=None):
        args = inputs(root)
        process = NS(pid=123456, returncode=None, poll=lambda: None)
        seen = []
        capture = root / "run" / "capture"
        def spawn(command, **kwargs):
            self.assertEqual(command[2], "vllm_bench.mtp_debug.serve")
            self.assertTrue(kwargs["start_new_session"])
            self.assertNotIn("VLLM_TORCH_PROFILER_DIR", kwargs["env"])
            capture.mkdir(parents=True)
            return process
        def request(base, body, timeout, **kwargs):
            armed = (capture / "ARM").exists()
            label = (capture / "LABEL").read_text().strip() if armed else "unarmed"
            seen.append((label, copy.deepcopy(body)))
            if fault == "http":
                return {"success": False, "error": "HTTP 500", "output_token_ids": []}
            if fault == "interrupt":
                raise KeyboardInterrupt()
            return {"success": True, "output_token_ids": [5, 6, 8 if fault == "outputs" and label == "request-02" else 7]}
        def analyze(argv):
            if fault == "analysis":
                raise SystemExit(2)
            path = Path(argv[argv.index("--output") + 1])
            summary = {"result": "INCOMPLETE_CAPTURE" if fault == "incomplete" else "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI",
                       "raw_ki_exact_equal": None if fault == "incomplete" else True,
                       "first_difference": None, "coverage_issue_count": int(fault == "incomplete"),
                       "first_coverage_issue": {"reason": "missing_rank"} if fault == "incomplete" else None}
            path.with_suffix(".summary.json").write_text(json.dumps(summary))
        ready = RuntimeError("server exited") if fault == "startup" else {"data": [{"id": "pangu-ki"}]}
        with patch.object(reproduce, "ensure_port_free"), patch.object(reproduce.subprocess, "Popen", side_effect=spawn), \
                patch.object(reproduce, "wait_ready", side_effect=ready if isinstance(ready, Exception) else None, return_value=ready), \
                patch.object(reproduce, "stream_request", side_effect=request), patch.object(reproduce, "stop_owned_group") as stop, \
                patch.object(reproduce.compare_indexer_ki, "main", side_effect=analyze), contextlib.redirect_stdout(io.StringIO()) as output:
            code = reproduce.main([*args, "--input-tokens", "5", "--output-tokens", "3", "--requests", "3", "--execute"])
        stop.assert_called_once_with(process, 30)
        self.assertFalse((capture / "ARM").exists())
        self.assertEqual(len(output.getvalue().splitlines()), 3)
        return code, json.loads((root / "run" / "result.json").read_text()), seen

    def test_warmup_arm_labels_identical_payloads_and_automatic_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, seen = self.exercise(Path(directory))
        self.assertEqual(code, 0)
        self.assertEqual([label for label, body in seen], ["unarmed", "unarmed", "request-01", "request-02", "request-03"])
        normalized = [{k: v for k, v in body.items() if k != "request_id"} for label, body in seen]
        self.assertTrue(all(body == normalized[0] for body in normalized))
        self.assertEqual(len({body["request_id"] for label, body in seen}), 5)
        self.assertEqual(len(report["comparisons"]), 2)
        self.assertTrue(report["evidence"]["outputs_equal"])

    def test_errors_and_interrupt_always_disarm_and_stop(self):
        for fault in ("startup", "http", "analysis", "interrupt", "incomplete"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                code, report, seen = self.exercise(Path(directory), fault)
                self.assertEqual(code, 1)
                self.assertEqual(report["result"], "INCOMPLETE_CAPTURE" if fault == "incomplete" else "ERROR")

    def test_output_difference_is_a_diagnostic_result_not_transport_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report, _ = self.exercise(Path(directory), "outputs")
        self.assertEqual(code, 0)
        self.assertEqual(report["result"], "OUTPUT_TOKENS_DIFFER")
        self.assertEqual(report["first_difference"]["output_index"], 2)

    def test_readiness_exit_timeout_and_occupied_port(self):
        process = NS(poll=lambda: 1, returncode=1)
        with self.assertRaisesRegex(RuntimeError, "exited before readiness"):
            reproduce.wait_ready(process, "http://unused", "pangu-ki", .1)
        clock = iter((0., 2.))
        with patch.object(reproduce.time, "monotonic", side_effect=lambda: next(clock)), self.assertRaises(TimeoutError):
            reproduce.wait_ready(NS(poll=lambda: None), "http://unused", "pangu-ki", 1.)
        with patch.object(reproduce.socket, "socket") as sock:
            sock.return_value.__enter__.return_value.bind.side_effect = OSError("occupied")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                reproduce.ensure_port_free("127.0.0.1", 8000)

    def test_shutdown_escalates_only_its_owned_process_group(self):
        process = subprocess.Popen([sys.executable, "-u", "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(120)"],
                                   start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            reproduce.stop_owned_group(process, .1)
            self.assertEqual(process.returncode, -signal.SIGKILL)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            process.stdout.close()


class TopologyTests(unittest.TestCase):
    def test_runtime_tp2_and_legacy_tp8_inference(self):
        group = NS(world_size=2, rank_in_group=1)
        runner = NS(vllm_config=NS(parallel_config=NS(data_parallel_size=1, pipeline_parallel_size=1)))
        with patch.dict(sys.modules, {"vllm.distributed": NS(get_tp_group=lambda: group)}):
            self.assertEqual(attention_trace.topology(runner)["tp_size"], 2)
            group.world_size = 4
            with self.assertRaises(RuntimeError):
                attention_trace.topology(runner)
        self.assertEqual(expected_rank_count([None, {}]), 8)
        self.assertEqual(expected_rank_count([{"tp_size": 2}]), 2)
        with self.assertRaises(ValueError):
            expected_rank_count([{"tp_size": 2}, {"tp_size": 8}])

    def test_ki_auto_tp2_and_missing_worker_is_not_tp1(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [f for f in ki_data(directory) if f["rank"] < 2]
            for frame in frames:
                frame["topology"] = {"tp_size": 2, "tp_rank": frame["rank"]}
            report, _ = compare_indexer_ki.compare(directory, frames, "request-01", "request-02")
            self.assertEqual(report["result"], "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI")
            report, _ = compare_indexer_ki.compare(directory, [f for f in frames if f["rank"] == 0], "request-01", "request-02")
            self.assertEqual(report["result"], "INCOMPLETE_CAPTURE")
            self.assertEqual(report["expected_ranks"], 2)

    def test_attention_gather_reconstruction_on_two_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [item for item in attention_data(directory) if item["event"]["rank"] < 2]
            for i, item in enumerate(events):
                trace = item["event"]["target_forward"]["attention_trace"]
                trace["topology"]["tp_size"] = 2
                layer = trace["layers"][0]
                layer["config"]["o_gather_size"] = 2
                control = layer["mome_controls"][-1]["tensor_capture"]
                for name in ("x_before", "y_after"):
                    control[name] = payload(directory, f"tp2-{i}-{name}", "float32", [1, 4], [1., 2., 11., 12.])
            report = compare_attention.compare_capture(directory, events, "request-01", "request-02")
            self.assertEqual(report["expected_ranks"], 2)
            self.assertEqual(report["result"], "NO_DIFFERENCE_IN_CAPTURED_ATTENTION", report["coverage_issues"])
            self.assertEqual(len(report["tp_gather_checks"]), 4)
            self.assertTrue(all(item["exact_equal"] for item in report["tp_gather_checks"]))


if __name__ == "__main__":
    unittest.main()
