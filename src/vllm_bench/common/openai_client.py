"""Stdlib completions/SSE client shared by profiling and debug scenarios."""
import http.client
import json
import os
import time
from urllib.parse import urlsplit


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


def stream_request(base, payload, timeout, barrier=None, retain_token_ids=False, *, connection_factory=None):
    # Serialize large pre-tokenized prompts before measuring request latency.
    body = json.dumps(payload, separators=(",", ":")).encode()
    conn, prefix = (connection_factory or connection)(base, timeout)
    if barrier:
        barrier.wait(timeout=timeout)
    start = time.perf_counter()
    row = {"request_id": payload["request_id"], "start_perf_s": start,
           "input_length": len(payload["prompt"]), "requested_output_length": payload["max_tokens"],
           "chunks": [], "success": False}
    if retain_token_ids:
        row["output_token_ids"] = []
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
                    if retain_token_ids:
                        row["output_token_ids"].extend(ids)
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


