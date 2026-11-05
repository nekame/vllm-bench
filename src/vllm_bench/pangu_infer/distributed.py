"""Host-side statistics on the existing TP CPU process group."""


def reduce_cpu_max(values, tp, *, dtype):
    """Reduce completed-request statistics without initializing another HCCL group."""
    if tp == 1:
        return list(values)
    import torch
    from vllm.distributed import get_tp_group

    tensor = torch.tensor(values, dtype=dtype, device="cpu")
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=get_tp_group().cpu_group)
    return tensor.tolist()
