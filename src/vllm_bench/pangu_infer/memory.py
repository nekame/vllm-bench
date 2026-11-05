"""Startup memory diagnostics outside compiled model code."""

import json
import os


def log_memory(stage, device, **context):
    """Report allocator and device memory without masking a model exception."""
    import torch

    record = {"rank": int(os.environ.get("RANK", 0)), "stage": stage, **context}
    try:
        free, total = torch.npu.mem_get_info(device)
        allocated = torch.npu.memory_allocated(device)
        reserved = torch.npu.memory_reserved(device)
        record.update({name + "_mib": round(value / (1 << 20), 2) for name, value in {
            "allocated": allocated, "reserved": reserved,
            "reserved_unallocated": reserved - allocated,
            "free": free, "total": total,
        }.items()})
    except Exception as error:
        record["diagnostic_error"] = str(error)
    print("[memory] " + json.dumps(record), flush=True)
