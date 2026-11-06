#!/usr/bin/env python3
"""Print first-verification MoME metadata from saved boundary snapshots."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Run output directory containing mtp-boundary/")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--records", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()

    import torch

    root = args.output / "mtp-boundary" / f"rank-{args.rank}"
    for record in args.records:
        path = root / f"record-{record:05d}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        event = next(e for e in data["events"] if e["event"] == "first_verification")
        state = event["states"]["target"]
        metadata = {key: value.tolist() if value is not None else None
                    for key, value in state["metadata"].items()}
        print(json.dumps({"record": record, "rank": args.rank,
                          "persistent_decode_fields_current": state["persistent_decode_fields_current"],
                          "metadata": metadata}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
