"""Check the runtime's CPU group topology without model or HCCL execution."""

import argparse
from datetime import timedelta
import os
import socket


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp-tp", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds per distributed operation")
    args = parser.parse_args(argv)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.mtp_tp > world or world % args.mtp_tp or args.timeout <= 0:
        parser.error("Launch with torchrun; MTP TP must divide WORLD_SIZE and timeout must be positive")

    import torch
    import torch.distributed as dist

    stage = "init"
    timeout = timedelta(seconds=args.timeout)
    print(f"[gloo-check] rank={rank} pid={os.getpid()} host={socket.gethostname()}"
          f" interface={os.environ.get('GLOO_SOCKET_IFNAME', 'auto')}"
          f" lazy_init={os.environ.get('TORCH_GLOO_LAZY_INIT', 'default')} stage={stage}", flush=True)
    try:
        dist.init_process_group("gloo", init_method="env://", timeout=timeout)
        full = [list(range(world))]
        singletons = [[r] for r in range(world)]
        mtp = [list(range(args.mtp_tp))] + [[r] for r in range(args.mtp_tp, world)]
        topology = [
            ("target_tp", full), ("target_ep", full), ("target_dispatch", full),
            ("replicated_layers", singletons),
        ]
        if args.mtp_tp != world:
            topology += [("mtp_tp", mtp), ("mtp_ep", mtp), ("mtp_dispatch", mtp)]
        for stage, partitions in topology:
            print(f"[gloo-check] rank={rank} stage={stage} begin", flush=True)
            for ranks in partitions:
                group = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
                if rank in ranks:
                    value = torch.tensor([rank + 1], dtype=torch.int64, device="cpu")
                    dist.all_reduce(value, group=group)
                    if value.item() != sum(r + 1 for r in ranks):
                        raise RuntimeError(f"Incorrect reduction in {stage}: {ranks}")
            print(f"[gloo-check] rank={rank} stage={stage} done", flush=True)
        stage = "final_barrier"
        dist.barrier()
        if rank == 0:
            print(f"[gloo-check] PASS world_size={world} mtp_tp={args.mtp_tp}", flush=True)
    except Exception as error:
        print(f"[gloo-check] rank={rank} stage={stage} FAILED: {error}", flush=True)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
