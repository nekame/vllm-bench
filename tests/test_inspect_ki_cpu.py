"""Raw Ki rank groups distinguish a local observation from a shared changed input."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from test_indexer_ki_cpu import frames
from vllm_bench.mtp_debug.inspect_ki import inspect, load_selected, main


def capture(root, changed_ranks):
    data = []
    for rank in range(2):
        for side in ("left", "right"):
            changed = rank in changed_ranks and side == "right"
            part = frames(root, 3 if changed else rank, side, "ki" if changed else None)
            for frame in part:
                frame.update(rank=rank, pid=rank + 1, topology={"tp_size": 2})
            data.extend(part)
    path = Path(root) / "ki-rank-0-pid-1.jsonl"
    path.write_text("\n".join(json.dumps(frame) for frame in data))
    return path


class KiInspectionTests(unittest.TestCase):
    def test_single_rank_changed_group_first_row_metrics_and_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            capture(directory, {1})
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main([directory, "--right", "request-02"])
            lines = dict(line.split("=", 1) for line in output.getvalue().splitlines())
            self.assertEqual(len(lines), 3)
            ki = json.loads(lines["ki"])
            self.assertEqual(ki["changed_ranks"], [1])
            self.assertEqual(ki["first_rank_ranges"], [[2, 2]])
            self.assertEqual(ki["first"]["max_abs_finite"], .125)
            self.assertEqual(json.loads(lines["within_request_groups"]), {"left": [[0, 1]], "right": [[0], [1]]})
            self.assertEqual(json.loads(lines["coverage"])["issue_count"], 0)

    def test_changed_input_shared_by_both_ranks_is_one_group_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            capture(directory, {0, 1})
            selected, _ = load_selected(Path(directory), "request-01", "request-02", 0)
            report = inspect(directory, selected, 2, "request-01", "request-02", 0)
            self.assertEqual(report["ki"]["changed_ranks"], [0, 1])
            self.assertEqual(report["within_request_groups"], {"left": [[0, 1]], "right": [[0, 1]]})

    def test_missing_payload_is_unknown_and_reordered_json_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = capture(directory, set())
            data = [json.loads(line) for line in path.read_text().splitlines()]
            source = next(frame for frame in data if frame["rank"] == 1 and frame["label"] == "request-02" and frame["phase"] == "prefill")["calls"][0]["ki_before"]
            (Path(directory) / source["path"]).unlink()
            path.write_text("\n".join(json.dumps(frame, sort_keys=True, separators=(",", ":")) for frame in data))
            selected, _ = load_selected(Path(directory), "request-01", "request-02", 0)
            self.assertTrue(all(frame["phase"] == "prefill" for frame in selected))
            report = inspect(directory, selected, 2, "request-01", "request-02", 0)
            self.assertEqual(report["coverage"]["issue_count"], 1)
            self.assertEqual(report["coverage"]["first_issue"]["rank"], 1)
            self.assertEqual(report["within_request_groups"]["right"], [[0]])


if __name__ == "__main__":
    unittest.main()
