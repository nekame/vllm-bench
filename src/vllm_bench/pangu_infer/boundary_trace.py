"""First speculative window snapshots through ordinary logits-processor hooks.

Hooks run outside captured model forwards. CPU copies synchronize execution;
this diagnostic can mask a timing-sensitive failure and is not a benchmark.
"""

import math
from pathlib import Path

from vllm_bench.pangu_infer.run import write_json


def cpu(value):
    return value.detach().cpu().clone()


def logical_blocks(start, context, block_size):
    blocks = [(context - 1) // block_size, context // block_size]
    if start:
        blocks.insert(0, (start - 1) // block_size)
    return list(dict.fromkeys(blocks))


def first_difference(left, right, field=""):
    """Compare logical snapshots; physical allocation IDs may differ normally."""
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return {"field": field, "reference_shape": list(left.shape), "candidate_shape": list(right.shape),
                    "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype)}
        if not torch.equal(left, right):
            delta = float((left.float() - right.float()).abs().max().item())
            return {"field": field, "max_abs_diff": delta if math.isfinite(delta) else str(delta)}
        return None
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return {"field": field, "reference_keys": list(left), "candidate_keys": list(right)}
        for key in left:
            if key in ("physical_block", "allocated_block"):
                continue
            difference = first_difference(left[key], right[key], f"{field}.{key}".lstrip("."))
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return {"field": field, "reference_length": len(left), "candidate_length": len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{field}.{index}".lstrip("."))
            if difference is not None:
                return difference
        return None
    return None if left == right else {"field": field, "reference": left, "candidate": right}


def event_difference(reference, candidate, fields):
    def select(events):
        return [{key: value for key, value in event.items() if key in fields} for event in events]

    difference = first_difference(select(reference), select(candidate))
    if difference is not None:
        prefix = difference["field"].split(".")[0]
        if prefix.isdigit() and int(prefix) < min(len(reference), len(candidate)):
            index = int(prefix)
            difference.update(event=candidate[index]["event"],
                              reference_argmax=reference[index].get("argmax"),
                              candidate_argmax=candidate[index].get("argmax"))
    return difference


class BoundaryRecorder:
    def __init__(self, runner, directory, rank, context, depth):
        self.runner, self.rank, self.context, self.depth = runner, rank, context, depth
        self.directory = Path(directory) / f"rank-{rank}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.active, self.references, self.handles = None, {}, []
        self.layers = {}
        registry = runner.vllm_config.compilation_config.static_forward_context
        draft_names = set(runner.drafter.attn_layer_names)
        mome_names = []
        for name, layer in registry.items():
            getter = getattr(layer, "get_attn_backend", None)
            if getter is None or getter().get_name() != "NPUPanguMome":
                continue
            mome_names.append(name)
            kind = "mtp" if name in draft_names else "target"
            if kind in self.layers:
                continue
            for group_id, groups in enumerate(runner.attn_groups):
                for group in groups:
                    if name in group.layer_names:
                        self.layers[kind] = (name, layer, group_id, group.get_metadata_builder())
        if set(self.layers) != {"target", "mtp"}:
            grouped_names = [name for groups in runner.attn_groups for group in groups
                             for name in group.layer_names]
            raise ValueError("Boundary trace requires target and MTP MoME layers: "
                             f"found={list(self.layers)}, mome={mome_names}, "
                             f"draft={sorted(draft_names)}, grouped={grouped_names}")
        target = runner.get_model()
        draft = runner.get_drafter_model()
        try:
            self.handles.append(target.logits_processor.register_forward_hook(self.target))
            self.handles.append(draft.model.logits_processor.register_forward_hook(self.draft))
        except BaseException:
            self.close()
            raise

    def arm(self, request_id, prompt_key, record):
        if self.active is not None:
            raise RuntimeError("Previous boundary trace request has not finished")
        self.active = {"request_id": request_id, "prompt_key": prompt_key, "record": record}
        self.events, self.target_calls, self.draft_calls, self.start = [], 0, 0, None
        self.hook_calls, self.hook_request_ids = {"target": 0, "mtp": 0}, {}

    def owns_request(self, output, kind):
        if self.active is not None:
            self.hook_calls[kind] += 1
            if kind not in self.hook_request_ids:
                self.hook_request_ids[kind] = list(self.runner.input_batch.req_id_to_index)
        return (self.active is not None and output is not None and output.numel() > 0
                and self.active["request_id"] in self.runner.input_batch.req_id_to_index)

    def target(self, module, args, output):
        if not self.owns_request(output, "target") or self.target_calls >= 2:
            return
        if self.target_calls == 0:
            batch = self.runner.input_batch
            slot = batch.req_id_to_index[self.active["request_id"]]
            starts = self.runner.query_start_loc.np
            scheduled = int(starts[slot + 1] - starts[slot])
            if int(batch.num_computed_tokens_cpu[slot]) + scheduled < self.context:
                # Partial prefill also computes logits, but discards its token.
                return
        event = "initial_target" if self.target_calls == 0 else "first_verification"
        self.events.append(self.snapshot(event, args[1], output))
        self.target_calls += 1

    def draft(self, module, args, output):
        if not self.owns_request(output, "mtp") or self.target_calls != 1 or self.draft_calls >= self.depth:
            return
        self.events.append(self.snapshot(f"mtp_{self.draft_calls}", args[1], output))
        self.draft_calls += 1

    def states(self, event):
        result = {}
        slot = self.runner.input_batch.req_id_to_index[self.active["request_id"]]
        request = self.runner.requests[self.active["request_id"]]
        for kind, (name, layer, group_id, builder) in self.layers.items():
            block_size = builder.kv_cache_spec.block_size
            table = self.runner.input_batch.block_table[group_id].get_device_tensor(slot + 1)
            table = table[slot].detach().cpu().tolist()
            allocated = request.block_ids[group_id]
            blocks = {}
            for logical in logical_blocks(self.start, self.context, block_size):
                entry = {"physical_block": table[logical] if logical < len(table) else None,
                         "allocated_block": allocated[logical] if logical < len(allocated) else None}
                block = entry["physical_block"]
                available = (block is not None and block > 0 and block == entry["allocated_block"]
                             and all(block < cache.shape[0] for cache in layer.kv_cache))
                entry["available"] = available
                if available:
                    entry["values"] = [cpu(cache[block]) for cache in layer.kv_cache]
                blocks[str(logical)] = entry
            # These are persistent decode buffers, stale during cached prefill.
            current = ((kind == "target" and event == "first_verification")
                       or (kind == "mtp" and event.startswith("mtp_") and self.draft_calls > 0))
            current = (current and builder.use_full_cuda_graph
                       and self.runner.input_batch.num_reqs <= builder.decode_cudagraph_max_bs)
            fields = {}
            if current:
                for field in ("num_computed_tokens", "num_accepted_tokens", "block_idx_last_computed_token",
                              "block_idx_first_scheduled_token", "block_idx_last_scheduled_token"):
                    value = getattr(builder, field, None)
                    fields[field] = cpu(value[:1]) if value is not None else None
            result[kind] = {"layer": name, "group": group_id, "block_size": block_size,
                            "persistent_decode_fields_current": current, "metadata": fields, "blocks": blocks}
        return result

    def snapshot(self, event, hidden, logits):
        runner = self.runner
        result = {"event": event}
        if not event.startswith("mtp_"):
            count = int(runner.query_start_loc.np[runner.input_batch.num_reqs])
            result["input_ids"] = cpu(runner.input_ids.gpu[:count])
            result["positions"] = cpu(runner.positions[:count])
            if self.start is None:
                self.start = int(result["positions"][0].item())
        result["lm_head_input"] = cpu(hidden)
        result["logits"] = cpu(logits)
        result["argmax"] = result["logits"].argmax(dim=-1).tolist()
        result["runner_accepted_gpu"] = cpu(runner.num_accepted_tokens.gpu[:1])
        result["runner_accepted_cpu"] = runner.input_batch.num_accepted_tokens_cpu[:1].tolist()
        result["runner_computed_cpu"] = runner.input_batch.num_computed_tokens_cpu[:1].tolist()
        result["states"] = self.states(event)
        return result

    def finish(self):
        import torch

        if self.target_calls != 2 or self.draft_calls != self.depth:
            raise RuntimeError(f"Incomplete boundary trace: target={self.target_calls}, mtp={self.draft_calls}, "
                               f"request_id={self.active['request_id']}, hooks={self.hook_calls}, "
                               f"hook_request_ids={self.hook_request_ids}")
        record, prompt_key = self.active["record"], self.active["prompt_key"]
        torch.save({**self.active, "rank": self.rank, "events": self.events},
                   self.directory / f"record-{record:05d}.pt")
        reference = self.references.get(prompt_key)
        report = {"rank": self.rank, "record": record, "prompt_key": prompt_key,
                  "reference_record": reference[0] if reference else None,
                  "first_compute_difference": None, "first_input_difference": None,
                  "first_state_difference": None,
                  "target_state_changes_during_mtp": []}
        if reference is not None:
            for key, fields in (
                ("first_compute_difference", ("event", "lm_head_input", "logits", "argmax")),
                ("first_input_difference", ("event", "input_ids", "positions", "runner_accepted_gpu")),
                ("first_state_difference", ("event", "states")),
            ):
                report[key] = event_difference(reference[1], self.events, fields)
        initial = self.events[0]["states"]["target"]["blocks"]
        for event in self.events[1:]:
            if not event["event"].startswith("mtp_"):
                continue
            blocks = event["states"]["target"]["blocks"]
            for logical, before in initial.items():
                after = blocks[logical]
                if (before["available"] and after["available"]
                        and before["physical_block"] == after["physical_block"]):
                    difference = first_difference(before["values"], after["values"])
                    if difference is not None:
                        report["target_state_changes_during_mtp"].append(
                            {"event": event["event"], "logical_block": int(logical), **difference})
        if reference is None:
            self.references[prompt_key] = (record, self.events)
        self.active = None
        write_json(self.directory / f"record-{record:05d}.json", report)
        return report

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def install(worker, directory, context, depth):
    import torch

    worker._pangu_boundary_recorder = BoundaryRecorder(
        worker.model_runner, directory, torch.distributed.get_rank(), context, depth)


def arm(worker, request_id, prompt_key, record):
    worker._pangu_boundary_recorder.arm(request_id, prompt_key, record)


def finish(worker):
    return worker._pangu_boundary_recorder.finish()


def close(worker):
    recorder = worker._pangu_boundary_recorder
    try:
        recorder.close()
    finally:
        del worker._pangu_boundary_recorder
