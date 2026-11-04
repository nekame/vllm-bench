"""Streaming runtime extraction skips decode bodies and rejects ambiguous/missing flags."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vllm_bench.mtp_debug import inspect_ki_runtime as tool


def frame(phase='prefill'):
    return {'format': 'pangu-indexer-ki-v1', 'rank': 0, 'pid': 123, 'phase': phase,
            'label': 'request-11', 'request_ids': ['exact-id'], 'forward_index': 12, 'calls': [],
            'prolog_calls': [{'layer': 0, 'status': 'complete', 'runtime': {
                'use_rope_fusion_op': True, 'rope_interleave': False, 'ki_stream_enabled': False}}]}


class RuntimeInspectionTests(unittest.TestCase):
    def test_cli_auto_request_small_output_and_optional_source_report(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            capture = run / 'capture'
            capture.mkdir()
            path = capture / 'ki-rank-0-pid-123.jsonl'
            path.write_text(json.dumps(frame('decode')) + '\n' + json.dumps(frame()) + '\n')
            (run / 'result.json').write_text(json.dumps({'prolog': {'right_label': 'request-11'}}))
            output = run / 'runtime.json'
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                tool.main([str(run), '--output', str(output)])
            self.assertEqual(stream.getvalue().splitlines(), ['use_rope_fusion_op=true', 'rope_interleave=false', 'ki_stream_enabled=false'])
            self.assertEqual(json.loads(output.read_text())['matches'][0]['line'], 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main([str(run), '--output', str(output)])

    def test_decode_header_fast_path_and_reordered_prefill(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root, 'ki-rank-0-pid-123.jsonl')
            decode = json.dumps(frame('decode'))
            target = json.dumps(frame(), sort_keys=True, separators=(',', ':'))
            path.write_text('\n'.join([decode] * 100 + [target]))
            original = json.loads
            def loads(text):
                if text == decode:
                    self.fail('Decode body should not be parsed')
                return original(text)
            with patch.object(tool.json, 'loads', side_effect=loads):
                report = tool.extract(path, 'exact-id')
            self.assertEqual(len(report['matches']), 1)

    def test_missing_flags_requests_prolog_and_conflicting_or_ambiguous_records(self):
        for failure in ('flags', 'prolog', 'selector', 'changed', 'ambiguous'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                path = Path(root, 'ki-rank-0-pid-123.jsonl')
                value = frame()
                records = [value]
                if failure == 'flags':
                    del value['prolog_calls'][0]['runtime']['ki_stream_enabled']
                elif failure == 'prolog':
                    del value['prolog_calls']
                elif failure == 'selector':
                    value['label'], value['request_ids'] = 'other', ['other']
                else:
                    other = copy.deepcopy(value)
                    if failure == 'changed':
                        other['prolog_calls'][0]['runtime']['ki_stream_enabled'] = True
                    else:
                        other['pid'] = 456
                    records.append(other)
                path.write_text('\n'.join(json.dumps(item) for item in records))
                with self.assertRaises(ValueError):
                    tool.extract(path, 'request-11')


if __name__ == '__main__':
    unittest.main()
