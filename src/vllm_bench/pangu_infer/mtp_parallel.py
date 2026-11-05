# SPDX-License-Identifier: MIT
# Copyright contributors to the vLLM project
# Portions Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""One MTP replica on the first ranks, with explicitly scoped model groups.

The EP/dispatch scope follows Omni's patch_tensor_parallel_group protocol
(omni/vllm_patches/patches/common/patch_parallel_state.py). No functions are
replaced. This sequential runtime restores every target group on scope exit.
"""

from contextlib import contextmanager
import copy
import dataclasses


class MtpParallel:
    def __init__(self, plan, cfg, device, rank):
        import torch
        from vllm.distributed.parallel_state import get_tp_group, init_model_parallel_group

        a = plan["options"]
        self.tp = a["mtp_tp"]
        self.active = rank < self.tp
        self.device = device
        self.target_group = get_tp_group()
        self.cfg = copy.copy(cfg)
        if self.active:
            self.cfg.parallel_config = dataclasses.replace(cfg.parallel_config, tensor_parallel_size=self.tp)
        # Keep the shared static-forward registry: custom operators resolve both
        # models by their unique layer names. Only parallel settings differ.
        ranks = [list(range(self.tp))] + [[r] for r in range(self.tp, a["tp"])]
        self.groups = []
        # vLLM recognizes EP communicators by the "ep" name. Its unique-name
        # counter distinguishes these groups from the target's groups.
        for name in ("tp", "ep", "moe_dispatch_ep"):
            self.groups.append(init_model_parallel_group(
                ranks, self.target_group.local_rank, "hccl", group_name=name))
        self.tp_group, self.ep_group, self.dispatch_group = self.groups
        self.ids = torch.empty(a["depth"], dtype=torch.int64, device=device)
        self.probs = (torch.empty(a["depth"], plan["config"]["vocab_size"], dtype=torch.float32, device=device)
                      if a["temperature"] != 0 else None)

    @contextmanager
    def context(self):
        from vllm.config import set_current_vllm_config
        from vllm.distributed import parallel_state
        from vllm.distributed.parallel_state import patch_tensor_parallel_group

        if not self.active:
            raise RuntimeError("This rank does not own MTP weights")
        old_ep = parallel_state._EP
        old_dispatch = parallel_state._MOE_DISPATCH_EP
        with patch_tensor_parallel_group(self.tp_group), set_current_vllm_config(self.cfg):
            parallel_state._EP = self.ep_group
            parallel_state._MOE_DISPATCH_EP = self.dispatch_group
            try:
                yield
            finally:
                parallel_state._EP = old_ep
                parallel_state._MOE_DISPATCH_EP = old_dispatch

    def broadcast(self, proposal):
        import torch

        if self.active:
            self.ids.copy_(proposal[0])
            if self.probs is not None:
                self.probs.copy_(proposal[1])
        torch.distributed.broadcast(self.ids, src=0, group=self.target_group.device_group)
        if self.probs is not None:
            torch.distributed.broadcast(self.probs, src=0, group=self.target_group.device_group)
        return self.ids, self.probs

    def close(self):
        for group in reversed(self.groups):
            group.destroy()


def record_idle_mtp(timing, options, count, first_size):
    """Keep TP timing rows aligned; inactive ranks contribute zero to MAX."""
    if timing is None or timing.cycle is None:
        return
    multi = count > 1 and options["depth"] <= count
    for step in range(options["depth"]):
        size = first_size if multi or step == 0 else 8 if options.get("mtp_pad_to_8", False) else 1
        timing.records.append({"cycle": timing.cycle, "model": "mtp", "input_tokens": size,
                               "step": step, "head": step if multi else 0, "wall_ms": 0.0})
