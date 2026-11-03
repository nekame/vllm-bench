"""Rank selection shared by runtime hooks and offline commands; no ML imports."""
import os
import sys


def current_rank():
    dist = sys.modules.get("torch.distributed")
    return dist.get_rank() if dist is not None and dist.is_initialized() else 0


def parse_ranks(value):
    if value == "all":
        return None
    ranks = {int(part) for part in value.split(",")}
    if not ranks or min(ranks) < 0:
        raise ValueError("Ranks must be nonnegative integers or all")
    return ranks


def rank_selected(rank, selector=None):
    selected = parse_ranks(selector if selector is not None else os.environ.get("PANGU_SPEC_DEBUG_RANK", "0"))
    return selected is None or rank in selected


def filter_rank(events, rank):
    return [item for item in events if item["event"].get("rank") == rank]


def expected_rank_count(topologies, explicit=None):
    """Missing workers never reduce the expected TP; legacy metadata defaults to eight."""
    sizes = {t["tp_size"] for t in topologies if isinstance(t, dict) and "tp_size" in t}
    if len(sizes) > 1 or sizes - {2, 8}:
        raise ValueError("Mixed or unsupported TP metadata; expected TP2 or TP8")
    detected = next(iter(sizes), 8)
    if explicit is not None:
        if explicit < 1:
            raise ValueError("--expected-ranks must be positive")
        if sizes and explicit != detected:
            raise ValueError("--expected-ranks disagrees with captured TP metadata")
        return explicit
    return detected
