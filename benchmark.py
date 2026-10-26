#!/usr/bin/env python3
"""Persistent-server benchmark: exact token-ID inputs and streaming timings.

Uses only stdlib HTTP. Does not import vLLM, torch, or torch_npu. Performance
measurements and profiler runs are separate modes. All latencies are CLIENT
observations, not kernel timings.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import threading
import time
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def connection(base, timeout):
    url = urlsplit(base)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise ValueError("base-url must be http(s), without embedded credentials")
    cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    return cls(url.hostname, url.port, timeout=timeout), url.path.rstrip("/")


def headers():
    result = {"Content-Type": "application/json"}
    key = os.getenv("OPENAI_API_KEY")
    if key:
        result["Authorization"] = "Bearer " + key
    return result


def api(base, path, timeout, method="GET"):
    conn, prefix = connection(base, timeout)
    try:
        conn.request(method, prefix + path, headers=headers())
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            raise RuntimeError(f"{path}: HTTP {response.status}: {body[:2000].decode(errors='replace')}")
        return json.loads(body) if body.strip() else None
    finally:
        conn.close()


def events(response):
    """Parse SSE records, including multi-line data and CRLF."""
    lines = []
    while True:
        raw = response.readline()
        if not raw:
            if lines:
                yield "\n".join(lines)
            return
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))


def stream_request(base, payload, timeout, barrier=None):
    # Serialize large pre-tokenized prompts before measuring request latency.
    body = json.dumps(payload, separators=(",", ":")).encode()
    conn, prefix = connection(base, timeout)
    if barrier:
        barrier.wait(timeout=timeout)
    start = time.perf_counter()
    row = {"request_id": payload["request_id"], "start_perf_s": start,
           "input_length": len(payload["prompt"]), "requested_output_length": payload["max_tokens"],
           "chunks": [], "success": False}
    try:
        conn.request("POST", prefix + "/v1/completions", body=body, headers=headers())
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(2000).decode(errors='replace')}")
        observed = 0
        usage = None
        finish_reason = None
        done = False
        for event in events(response):
            arrival = time.perf_counter() - start
            if event == "[DONE]":
                done = True
                break
            chunk = json.loads(event)
            if "error" in chunk:
                raise RuntimeError(f"Streaming error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                ids = choice.get("token_ids")
                if ids:
                    observed += len(ids)
                    row["chunks"].append({"arrival_s": arrival, "tokens": len(ids)})
                elif choice.get("text") and ids is None:
                    raise RuntimeError("Server omitted delta token_ids; requires return_token_ids support (vLLM 0.25.1)")
                finish_reason = choice.get("finish_reason") or finish_reason
        row["elapsed_s"] = time.perf_counter() - start
        if not done or not usage or not row["chunks"]:
            raise RuntimeError("Incomplete stream, missing usage or missing token IDs")
        if usage["prompt_tokens"] != len(payload["prompt"]):
            raise RuntimeError(f"Prompt length mismatch: usage={usage['prompt_tokens']}, requested={len(payload['prompt'])}")
        if observed != usage["completion_tokens"] or observed != payload["max_tokens"]:
            raise RuntimeError(f"Output length mismatch: streamed={observed}, usage={usage['completion_tokens']}, requested={payload['max_tokens']}")
        first, last = row["chunks"][0]["arrival_s"], row["chunks"][-1]["arrival_s"]
        row.update(success=True, usage=usage, finish_reason=finish_reason,
                   ttft_s=first, last_token_s=last,
                   tpot_s=(last - first) / (observed - 1) if observed > 1 else None,
                   single_token_chunks=all(c["tokens"] == 1 for c in row["chunks"]))
    except Exception as error:
        row.update(error=str(error), elapsed_s=time.perf_counter() - start)
    finally:
        conn.close()
    return row


def distribution(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None

    def quantile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)

    return {"count": len(values), "mean": statistics.fmean(values), "min": values[0],
            "p50": quantile(.5), "p90": quantile(.9), "p99": quantile(.99), "max": values[-1]}


def summarize(rows):
    good = [r for r in rows if r["success"]]
    # Sum disjoint wave wall times, including each wave's complete drain.
    waves = {}
    for r in rows:
        waves.setdefault(r["wave"], []).append(r)
    wall = sum(max(r["start_perf_s"] + r["elapsed_s"] for r in wave)
               - min(r["start_perf_s"] for r in wave) for wave in waves.values())
    out_tokens = sum(r["usage"]["completion_tokens"] for r in good)
    in_tokens = sum(r["usage"]["prompt_tokens"] for r in good)
    gaps = [b["arrival_s"] - a["arrival_s"] for r in good
            for a, b in zip(r["chunks"], r["chunks"][1:])]
    return {"requests": len(rows), "successes": len(good), "failures": len(rows) - len(good),
            "wave_wall_s": wall, "output_tokens": out_tokens, "input_tokens": in_tokens,
            "output_tokens_per_s_e2e": out_tokens / wall if wall else None,
            "input_tokens_per_s_e2e": in_tokens / wall if wall else None,
            "ttft_s": distribution(r["ttft_s"] for r in good),
            "request_latency_s": distribution(r["elapsed_s"] for r in good),
            "tpot_s": distribution(r["tpot_s"] for r in good),
            "stream_chunk_gap_s": distribution(gaps),
            "all_chunks_single_token": bool(good) and all(r["single_token_chunks"] for r in good),
            "note": "E2E throughput includes prefill, HTTP, queueing and wave drain; TPOT uses client token arrivals."}


def validate_suite(data, config):
    cases = data["cases"]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Suite must have nonempty, uniquely named cases")
    for c in cases:
        for key in ("input_tokens", "output_tokens", "concurrency"):
            if not isinstance(c[key], int) or c[key] <= 0:
                raise ValueError(f"Invalid {key} in {c['name']}")
        if c["input_tokens"] + c["output_tokens"] > config["max_position_embeddings"]:
            raise ValueError(f"Case exceeds model positional limit: {c['name']}")
    return cases


def make_payload(case, model, config, seed, number, corpus=None):
    rng = random.Random(seed + number)
    length = case["input_tokens"]
    if corpus:
        tokens = corpus[number % len(corpus)]
        if len(tokens) < length:
            raise ValueError(f"Corpus sample {number % len(corpus)} shorter than {length}")
        prompt = tokens[:length]
    else:
        excluded = {config.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")}
        prompt = []
        while len(prompt) < length:
            token = rng.randrange(config["vocab_size"])
            if token not in excluded:
                prompt.append(token)
    if any(not isinstance(t, int) or not 0 <= t < config["vocab_size"] for t in prompt):
        raise ValueError("Prompt contains invalid token IDs")
    return {"model": model, "prompt": prompt, "max_tokens": case["output_tokens"],
            "min_tokens": case["output_tokens"], "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True},
            "return_token_ids": True, "add_special_tokens": False,
            "seed": seed, "request_id": f"pangu12-{seed}-{number}"}


def run_wave(args, case, config, number, wave, phase, corpus, sink):
    payloads = [make_payload(case, args.model, config, args.seed, number + i, corpus)
                for i in range(case["concurrency"])]
    barrier = threading.Barrier(len(payloads))
    with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(stream_request, args.base_url, payload, args.timeout, barrier)
                   for payload in payloads]
        rows = [f.result() for f in futures]
    for row, payload in zip(rows, payloads):
        row.update(case=case["name"], wave=wave, phase=phase,
                   prompt_sha256=hashlib.sha256(json.dumps(payload["prompt"]).encode()).hexdigest())
        sink.write(json.dumps(row) + "\n")
    sink.flush()
    errors = [r["error"] for r in rows if not r["success"]]
    if errors:
        raise RuntimeError(f"Wave failed, raw results saved: {errors[:3]}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Server root, without /v1")
    p.add_argument("--model", default="pangu12")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--suite", type=Path, default=HERE / "suite.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["benchmark", "profile"], default="benchmark")
    p.add_argument("--select", nargs="+", help="Case names to run; default all in benchmark mode")
    p.add_argument("--warmup-waves", type=int, default=2)
    p.add_argument("--measure-waves", type=int, default=5)
    p.add_argument("--profile-kind", choices=["prefill", "generation"], default="generation")
    p.add_argument("--profile-tokens", type=int, default=8,
                   help="Output length in generation trace; includes prefill and this short generation")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--prompts-jsonl", type=Path, help="Optional corpus: one {prompt_token_ids: [...]} per line")
    p.add_argument("--server-manifest", type=Path, help="serve.py manifest for engine-limit checks and reproducibility")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.warmup_waves < 0 or a.measure_waves < 1 or a.profile_tokens < 1 or a.timeout <= 0:
        p.error("Invalid wave counts, profile tokens or timeout")
    cfg = json.loads((a.model_dir / "config.json").read_text())
    suite = json.loads(a.suite.read_text())
    cases = validate_suite(suite, cfg)
    if a.select:
        unknown = set(a.select) - {c["name"] for c in cases}
        if unknown:
            p.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["name"] in a.select]
    if a.mode == "profile":
        if len(cases) != 1:
            p.error("Profile mode requires --select with exactly one case")
        cases = [{**cases[0], "output_tokens": 1 if a.profile_kind == "prefill" else a.profile_tokens}]
        validate_suite({"cases": cases}, cfg)
    corpus = None
    if a.prompts_jsonl:
        corpus = [json.loads(line)["prompt_token_ids"] for line in a.prompts_jsonl.read_text().splitlines() if line.strip()]
        if not corpus:
            p.error("Prompt corpus is empty")
        if any(len(t) < max(c["input_tokens"] for c in cases) for t in corpus):
            p.error("Every corpus sample must cover the longest selected input")
    server = json.loads(a.server_manifest.read_text()) if a.server_manifest else None
    if server:
        for c in cases:
            limits = server["arguments"]
            if c["input_tokens"] + c["output_tokens"] > limits["max_model_len"]:
                p.error(f"Case exceeds server max-model-len: {c['name']}")
            if c["concurrency"] > limits["max_num_seqs"]:
                p.error(f"Case exceeds max-num-seqs, so not a full active batch: {c['name']}")
        if server["model_config"] != cfg:
            p.error("Server manifest config differs from model-dir/config.json")
    plan = {"mode": a.mode, "cases": cases, "warmup_waves": a.warmup_waves,
            "measure_waves": a.measure_waves, "seed": a.seed,
            "input_source": "corpus" if corpus else "synthetic_uniform_token_ids",
            "profile_window": a.profile_kind if a.mode == "profile" else None,
            "model_config": cfg, "server_manifest": server,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    print(json.dumps({k: v for k, v in plan.items() if k not in {"model_config", "server_manifest"}}, indent=2), flush=True)
    if a.dry_run:
        return 0
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        p.error("Result directory must be empty")
    models = api(a.base_url, "/v1/models", a.timeout)
    if a.model not in {m["id"] for m in models["data"]}:
        p.error("Requested model name is not served")
    card = next(m for m in models["data"] if m["id"] == a.model)
    if card.get("max_model_len"):
        for case in cases:
            if case["input_tokens"] + case["output_tokens"] > card["max_model_len"]:
                p.error(f"Case exceeds advertised server model length: {case['name']}")
    plan["models_response"] = models
    (output / "run_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    summaries = []
    number = 0
    with (output / "requests.jsonl").open("x") as sink:
        for case in cases:
            print(f"Warmup: {case['name']}", flush=True)
            for wave in range(a.warmup_waves):
                run_wave(a, case, cfg, number, wave, "warmup", corpus, sink)
                number += case["concurrency"]
            rows = []
            if a.mode == "profile":
                api(a.base_url, "/start_profile", a.timeout, "POST")
                try:
                    rows = run_wave(a, case, cfg, number, 0, "profile", corpus, sink)
                finally:
                    api(a.base_url, "/stop_profile", a.timeout, "POST")
                print("Profile collection stopped. A successful /stop_profile does not prove trace export. "
                      "On the server, check the traces directory with export_profile.py; "
                      "daemon workers may require standalone offline export (--execute).", flush=True)
                number += case["concurrency"]
            else:
                for wave in range(a.measure_waves):
                    rows += run_wave(a, case, cfg, number, wave, "measurement", corpus, sink)
                    number += case["concurrency"]
            summary = {"case": case, "mode": a.mode, "metrics": summarize(rows)}
            summaries.append(summary)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            with (output / "summary.csv").open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["case", "mode", "input_tokens", "output_tokens", "concurrency",
                                 "successes", "output_tokens_per_s_e2e", "ttft_p50_s", "tpot_p50_s"])
                for item in summaries:
                    c, m = item["case"], item["metrics"]
                    writer.writerow([c["name"], item["mode"], c["input_tokens"], c["output_tokens"],
                                     c["concurrency"], m["successes"], m["output_tokens_per_s_e2e"],
                                     m["ttft_s"]["p50"] if m["ttft_s"] else None,
                                     m["tpot_s"]["p50"] if m["tpot_s"] else None])
            print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        sys.exit(f"ERROR: {error}")
