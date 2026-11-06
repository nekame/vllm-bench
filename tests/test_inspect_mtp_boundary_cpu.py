"""Saved trace inspection without an inference engine or ML packages."""

from copy import deepcopy
from pathlib import Path
import runpy
import unittest


state_comparison = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/inspect_mtp_boundary.py"))["state_comparison"]


class Tensor:
    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0])) if isinstance(rows[0], list) else (len(rows),)

    def __getitem__(self, index):
        return self.rows[index]

    def tolist(self):
        return self.rows


def trace(record, value):
    blocks = {str(block): {"available": True,
              "values": [Tensor([[0, 0]] * 5) for _ in range(3)]} for block in (63, 64)}
    blocks["64"]["values"][0] = Tensor([[value, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    state = {"blocks": blocks, "persistent_decode_fields_current": True,
             "metadata": {"block_idx_last_computed_token": Tensor([64]),
                          "num_accepted_tokens": Tensor([1])}}
    events = [{"event": name, "states": {"target": deepcopy(state)}}
              for name in ("initial_target", "mtp_0", "mtp_1", "mtp_2", "first_verification")]
    return {"record": record, "events": events}


class InspectBoundaryTests(unittest.TestCase):
    def test_compares_last_draft_snapshot_before_target_overwrites_state(self):
        reference, candidate = trace(0, 1), trace(1, 3)
        candidate["events"][-1]["states"]["target"]["blocks"]["64"]["values"][0] = Tensor([[100, 0]] * 5)
        report = state_comparison(reference, candidate)
        self.assertEqual(report["snapshot_events"], ["mtp_2", "mtp_2"])
        self.assertEqual(report["initial_state_blocks"], [64, 64])
        self.assertEqual(report["blocks"]["63"]["components"][0]["row_max_abs_diff"], [0] * 5)
        component = report["blocks"]["64"]["components"][0]
        self.assertEqual(component["row_max_abs_diff"], [2, 0, 0, 0, 0])
        self.assertEqual(component["accepted_history_rows"], [[0, 1], [0, 1]])
        self.assertEqual(component["committed_tail_rows"], [[3, 4], [3, 4]])

    def test_unavailable_blocks_are_reported_without_reading_values(self):
        reference, candidate = trace(0, 1), trace(1, 3)
        candidate["events"][-2]["states"]["target"]["blocks"]["64"] = {"available": False}
        report = state_comparison(reference, candidate)
        self.assertFalse(report["blocks"]["64"]["available"])
        self.assertNotIn("components", report["blocks"]["64"])

    def test_rejects_stale_metadata_and_invalid_history_rows(self):
        reference, candidate = trace(0, 1), trace(1, 3)
        state = candidate["events"][-1]["states"]["target"]
        state["persistent_decode_fields_current"] = False
        with self.assertRaisesRegex(ValueError, "current full-graph"):
            state_comparison(reference, candidate)
        state["persistent_decode_fields_current"] = True
        state["metadata"]["num_accepted_tokens"] = Tensor([5])
        with self.assertRaisesRegex(ValueError, "outside"):
            state_comparison(reference, candidate)


if __name__ == "__main__":
    unittest.main()
