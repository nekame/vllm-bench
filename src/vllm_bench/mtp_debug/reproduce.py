"""One-command local target launch, fixed requests, Ki comparison and shutdown."""
import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

from vllm_bench.common.openai_client import api, stream_request
from vllm_bench.mtp_debug import compare_indexer_ki, serve

LOG = logging.getLogger(__name__)


def make_prompt(config, length, seed):
    excluded = set()
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = config.get(name)
        excluded.update(v for v in (value if isinstance(value, list) else [value]) if type(v) is int)
    vocab = config["vocab_size"]
    if type(vocab) is not int or vocab < 1 or len({i for i in excluded if type(i) is int and 0 <= i < vocab}) == vocab:
        raise ValueError("No valid vocabulary tokens for the synthetic prompt")
    rng = random.Random(seed)
    tokens = []
    while len(tokens) < length:
        token = rng.randrange(vocab)
        if token not in excluded:
            tokens.append(token)
    return tokens


def ensure_port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError as error:
            raise RuntimeError(f"Port {host}:{port} is unavailable; existing servers will not be stopped") from error


def wait_ready(process, base, name, timeout):
    deadline = time.monotonic() + timeout
    last = "No API response"
    checks = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited before readiness ({process.returncode}); inspect server.log")
        try:
            models = api(base, "/v1/models", min(1., max(.01, deadline - time.monotonic())))
            if name in {item["id"] for item in models["data"]}:
                if process.poll() is not None:
                    raise RuntimeError("Owned server exited during readiness")
                return models
            last = f"Expected served model {name!r} is absent"
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            last = str(error)
        checks += 1
        if checks % 20 == 0:
            LOG.info("Waiting for server API: %s", last)
        time.sleep(min(.25, max(0., deadline - time.monotonic())))
    raise TimeoutError(f"Server readiness timeout: {last}; inspect server.log")


def stop_owned_group(process, timeout):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    deadline = time.monotonic() + timeout
    while True:
        process.poll()  # Reap the group leader; wait for its workers as well.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            return
        time.sleep(.1)


def atomic_label(capture, label):
    temporary = capture / "LABEL.tmp"
    temporary.write_text(label + "\n")
    temporary.replace(capture / "LABEL")


def output_difference(reference, actual):
    for i in range(max(len(reference), len(actual))):
        a = reference[i] if i < len(reference) else None
        b = actual[i] if i < len(actual) else None
        if a != b:
            return {"output_index": i, "left_token_id": a, "right_token_id": b}
    return None


def run(a, server_args, launch_plan, env, prompt):
    output, capture = a.output.resolve(), Path(launch_plan["capture_output"])
    base = f"http://{a.host}:{a.port}"
    process = None
    rows, measured, comparisons = [], [], []
    phase = "startup"
    run_id = uuid.uuid4().hex
    report = {"format": "pangu-ki-reproduction-v1", "result": "ERROR", "first_difference": None,
              "tp": a.tp, "comparisons": comparisons, "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()}
    def send(sink, label):
        payload = {"model": a.name, "prompt": list(prompt), "max_tokens": a.output_tokens, "min_tokens": a.output_tokens,
                   "temperature": 0, "top_p": 1, "top_k": 0, "seed": a.seed, "ignore_eos": True,
                   "add_special_tokens": False, "return_token_ids": True, "stream": True,
                   "stream_options": {"include_usage": True}, "request_id": f"{run_id}-{label}"}
        row = stream_request(base, payload, a.request_timeout, retain_token_ids=True)
        row.update(label=label, phase=phase, prompt_sha256=report["prompt_sha256"])
        row["output_sha256"] = hashlib.sha256(json.dumps(row.get("output_token_ids", []), separators=(",", ":")).encode()).hexdigest()
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        if not row["success"]:
            raise RuntimeError(f"{label}: {row.get('error', 'request failed')}")
        if process.poll() is not None:
            raise RuntimeError(f"Server exited during {label}; inspect server.log")
        return row
    try:
        ensure_port_free(a.host, a.port)
        with (output / "server.log").open("x") as log, (output / "requests.jsonl").open("x") as sink:
            # New session owns the complete debug_spec/vLLM/worker process group.
            process = subprocess.Popen([sys.executable, "-m", "vllm_bench.mtp_debug.serve", *server_args, "--execute"],
                                       env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report["server_pid"] = process.pid
            LOG.info("Server PID %s; startup log: %s", process.pid, output / "server.log")
            report["models_response"] = wait_ready(process, base, a.name, a.startup_timeout)
            if not capture.is_dir() or (capture / "ARM").exists():
                raise RuntimeError("Expected a new, unarmed capture directory after server startup")
            phase = "warmup"
            for i in range(a.warmup_requests):
                send(sink, f"warmup-{i + 1:02d}")
                LOG.info("Warmup %s/%s complete", i + 1, a.warmup_requests)
            phase = "measurement"
            for i in range(a.requests):
                label = f"request-{i + 1:02d}"
                atomic_label(capture, label)
                (capture / "ARM").touch()
                measured.append(send(sink, label))
                LOG.info("%s complete", label)
            (capture / "ARM").unlink(missing_ok=True)
            phase = "comparison"
            folder = output / "comparisons"
            folder.mkdir()
            for row in measured[1:]:
                path = folder / ("request-01-vs-" + row["label"] + ".json")
                with (path.with_suffix(".log")).open("x") as log_result, contextlib.redirect_stdout(log_result), contextlib.redirect_stderr(log_result):
                    try:
                        compare_indexer_ki.main([str(capture), "--left", "request-01", "--right", row["label"],
                                                "--expected-ranks", str(a.tp), "--output", str(path)])
                    except SystemExit as error:
                        raise RuntimeError(f"Ki analysis failed; inspect {path.with_suffix('.log')}") from error
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                summary["right_label"] = row["label"]
                summary["output_difference"] = output_difference(measured[0]["output_token_ids"], row["output_token_ids"])
                comparisons.append(summary)
            incomplete = any(c["result"] == "INCOMPLETE_CAPTURE" for c in comparisons)
            different = next((c for c in comparisons if c["first_difference"] is not None), None)
            output_changed = next((c for c in comparisons if c["output_difference"] is not None), None)
            report["result"] = ("INCOMPLETE_CAPTURE" if incomplete else different["result"] if different else
                                "OUTPUT_TOKENS_DIFFER" if output_changed else "NO_DIFFERENCE_IN_CAPTURED_PREFILL_KI")
            if different:
                report["first_difference"] = {"right_label": different["right_label"], **different["first_difference"]}
            elif output_changed:
                report["first_difference"] = {"right_label": output_changed["right_label"], **output_changed["output_difference"]}
            report["evidence"] = {"tp": a.tp, "compared_pairs": len(comparisons),
                "coverage_issue_count": sum(c["coverage_issue_count"] for c in comparisons),
                "raw_ki_exact_equal": (False if any(c["raw_ki_exact_equal"] is False for c in comparisons) else
                                       True if all(c["raw_ki_exact_equal"] is True for c in comparisons) else None),
                "outputs_equal": output_changed is None,
                "first_coverage_issue": next((c["first_coverage_issue"] for c in comparisons if c["first_coverage_issue"]), None)}
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        report.update(result="ERROR", error=str(error), failed_phase=phase, traceback=traceback.format_exc())
    finally:
        try:
            if capture.is_dir():
                (capture / "ARM").unlink(missing_ok=True)
        except OSError as error:
            report.update(result="ERROR", disarm_error=str(error))
        try:
            stop_owned_group(process, a.shutdown_timeout)
        except Exception as error:
            report.update(result="ERROR", shutdown_error=str(error))
    report["completed_requests"] = len(rows)
    if report["result"] == "ERROR":
        report["evidence"] = {"failed_phase": report.get("failed_phase", phase), "error": report.get("error"),
                              "shutdown_error": report.get("shutdown_error"), "disarm_error": report.get("disarm_error")}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["result=" + report["result"], "first_difference=" + json.dumps(report["first_difference"]),
             "evidence=" + json.dumps(report["evidence"])]
    (output / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if report["result"] in ("ERROR", "INCOMPLETE_CAPTURE") else 0


def main(argv=None):
    p = serve.parser()
    p.description = __doc__
    p.set_defaults(max_model_len=None, max_num_batched_tokens=None)
    p.add_argument("--input-tokens", type=int, default=8192)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--requests", type=int, default=2)
    p.add_argument("--warmup-requests", type=int, default=2)
    p.add_argument("--startup-timeout", type=float, default=1800)
    p.add_argument("--request-timeout", type=float, default=1800)
    p.add_argument("--shutdown-timeout", type=float, default=30)
    a = p.parse_args(argv)
    try:
        if a.requests < 2 or a.warmup_requests < 0 or min(a.input_tokens, a.output_tokens) < 1 or a.seed < 0:
            raise ValueError("Need >=2 requests, >=0 warmups/seed and positive token counts")
        if any(not math.isfinite(t) or t <= 0 for t in (a.startup_timeout, a.request_timeout, a.shutdown_timeout)):
            raise ValueError("Timeouts must be finite and positive")
        if a.host not in ("127.0.0.1", "localhost"):
            raise ValueError("The one-command launcher requires a local loopback host")
        if a.output.exists():
            raise ValueError("Use a new output directory")
        if a.max_model_len is None:
            a.max_model_len = a.input_tokens + a.output_tokens
        if a.max_num_batched_tokens is None:
            a.max_num_batched_tokens = a.input_tokens
        if a.max_model_len < a.input_tokens + a.output_tokens or a.max_num_batched_tokens < a.input_tokens:
            raise ValueError("Server limits must cover the fixed prompt and output")
        # Warmup is unarmed; every measured token requires at most one verification.
        a.limit = max(a.limit, a.requests * (a.output_tokens + 1))
        output = a.output.resolve()
        server_args = ["--model", str(a.model.resolve()), "--operator-config", str(a.operator_config.resolve()),
                       "--output", str(output / "server"), "--capture-output", str((a.capture_output or output / "capture").resolve())]
        for name in ("model_scope", "name", "host", "port", "tp", "num_speculative_tokens", "max_model_len",
                     "max_num_batched_tokens", "gpu_memory_utilization", "kv_cache_memory_bytes", "num_gpu_blocks_override",
                     "patches", "attention_backend", "engine_cli_json", "block_size", "mome_layers", "attention_steps_per_request",
                     "indexer_ki_steps_per_request", "tensor_max_mib", "tensor_budget_mib", "limit"):
            value = getattr(a, name)
            if value is not None:
                server_args += ["--" + name.replace("_", "-"), str(value)]
        if not a.chunked_prefill:
            server_args.append("--no-chunked-prefill")
        if a.enable_expert_parallel:
            server_args.append("--enable-expert-parallel")
        server_options = serve.parser().parse_args(server_args)
        server_options.execute = a.execute
        launch_plan, env = serve.build_debug_plan(server_options)
        if a.input_tokens + a.output_tokens > launch_plan["model_config"]["max_position_embeddings"]:
            raise ValueError("Prompt plus output exceeds checkpoint positional limit")
        prompt = make_prompt(launch_plan["model_config"], a.input_tokens, a.seed)
    except (OSError, ValueError, KeyError, TypeError) as error:
        p.error(str(error))
    manifest = {"format": "pangu-ki-reproduction-plan-v1", "server": launch_plan,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                "prompt_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()}
    if not a.execute:
        print(json.dumps(manifest, indent=2))
        return 0
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "prompt.json").write_text(json.dumps({"prompt_token_ids": prompt, "sha256": manifest["prompt_sha256"]}) + "\n")
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(a, server_args, launch_plan, env, prompt)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raise SystemExit(main())
