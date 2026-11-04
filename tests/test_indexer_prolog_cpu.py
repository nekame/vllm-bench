"""Ki producer hooks preserve execution and offline comparisons expose missing data."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from attention_fixtures import payload
from test_indexer_ki_cpu import frames
from test_mome_tensors_cpu import Tensor, encode
from test_tp2_reproduce_cpu import inputs
from vllm_bench.mtp_debug import attention_trace, debug_spec, inspect_ki, reproduce, spec_rejection
from vllm_bench.mtp_debug.compare_indexer_ki import compare as compare_ki
from vllm_bench.mtp_debug.compare_indexer_prolog import compare
from vllm_bench.mtp_debug.indexer_prolog_trace import PROLOG, STAGES, parse_positions, scope_prolog
from vllm_bench.mtp_debug.tensor_io import TensorWriter, numbers, read_tensor


def dataset(root, first_stage='wk_output', changed_rank=1, chunked=False):
    data = []
    for rank in range(2):
        for side in ('left', 'right'):
            own = frames(root, 3 if rank == changed_rank else rank, side,
                         'ki' if first_stage is not None else None, chunked)
            for frame in own:
                frame.update(rank=rank, pid=rank + 1, topology={'tp_size': 2})
                if frame['phase'] != 'prefill':
                    continue
                cache = frame['calls'][0]
                positions = cache['row_positions']
                name = f"prolog-{rank}-{side}-{frame['forward_index']}"
                def p(field, data, shape):
                    return payload(root, name + '-' + field, 'float32', shape, data)
                tensors = {}
                for i, field in enumerate(STAGES):
                    active = first_stage is not None and i >= STAGES.index(first_stage) and field not in ('cos', 'sin')
                    raw = [float(pos) + (.125 if active and rank == changed_rank and side == 'right' and pos == 2 else 0)
                           for pos in positions for _ in range(128)]
                    tensors[field] = p(field, raw, [len(positions), 128])
                params = {'wk.named_parameters.weight': p('weight', [1.], [1]),
                          'k_norm.named_parameters.weight': p('norm-weight', [1.], [1])}
                frame['prolog_calls'] = [{'prefix': cache['prefix'], 'layer': 0, 'status': 'complete',
                    'requested_positions': [0, 4], 'row_positions': positions, 'row_input_ids': cache['row_input_ids'],
                    'tensors': tensors, 'parameters_before': params, 'parameters_after': copy.deepcopy(params),
                    'runtime': {'ena_sp': False, 'ki_stream_enabled': False}}]
            data.extend(own)
    return data


class ComparisonTests(unittest.TestCase):
    def test_first_stage_cross_rank_groups_and_automatic_main_report(self):
        with tempfile.TemporaryDirectory() as root:
            data = dataset(root)
            report = compare(root, data, 'request-01', 'request-02', 2)
            self.assertEqual(report['summary']['first_stage_difference']['field'], 'wk_output')
            self.assertEqual(report['summary']['first_stage_difference']['position'], 2)
            self.assertEqual(report['summary']['coverage_issue_count'], 0)
            self.assertEqual(report['summary']['consistency_difference_count'], 0)
            self.assertEqual(report['within_request_groups']['right']['wk_output'], [[0], [1]])
            self.assertEqual(report['within_request_groups']['right']['wk_input'], [[0, 1]])
            main, _ = compare_ki(root, data, 'request-01', 'request-02', 2)
            self.assertEqual(main['prolog']['0']['summary'], report['summary'])

    def test_chunk_alignment_and_parameter_change_not_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            data = dataset(root, chunked=True)
            result = compare(root, data, 'request-01', 'request-02', 2)
            self.assertEqual(result['summary']['coverage_issue_count'], 0)
            self.assertEqual(result['summary']['first_stage_difference']['field'], 'wk_output')
            # Change every chunk's registered weight on one rank/request.
            for f in data:
                if f['rank'] == 1 and f['label'] == 'request-02' and f['phase'] == 'prefill':
                    c = f['prolog_calls'][0]
                    c['parameters_after'] = {}
            result = compare(root, data, 'request-01', 'request-02', 2)
            self.assertEqual(result['status'], 'incomplete_capture')
            self.assertIn('Parameter snapshots absent', result['summary']['first_coverage_issue']['reason'])

    def test_missing_rank_stage_payload_and_requested_position_are_incomplete(self):
        for missing in ('rank', 'payload', 'position', 'stage'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as root:
                data = dataset(root)
                target = next(f for f in data if f['rank'] == 1 and f['label'] == 'request-02' and f['phase'] == 'prefill')
                c = target['prolog_calls'][0]
                if missing == 'rank':
                    data = [f for f in data if f['rank'] != 1]
                elif missing == 'payload':
                    (Path(root) / c['tensors']['wk_input']['path']).unlink()
                elif missing == 'stage':
                    del c['tensors']['wk_output']
                else:
                    c['requested_positions'] = [0, 5]
                result = compare(root, data, 'request-01', 'request-02', 2)
                self.assertEqual(result['status'], 'incomplete_capture')
                self.assertGreater(result['summary']['coverage_issue_count'], 0)

    def test_invariants_parameter_mutation_and_legacy_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            data = dataset(root)
            target = next(f for f in data if f['rank'] == 1 and f['label'] == 'request-02' and f['phase'] == 'prefill')['prolog_calls'][0]
            target['tensors']['wk_input'] = target['tensors']['wk_output']
            target['parameters_after']['wk.named_parameters.weight'] = payload(root, 'new-weight', 'float32', [1], [2.])
            result = compare(root, data, 'request-01', 'request-02', 2)
            self.assertGreater(result['summary']['parameter_difference_count'], 0)
            self.assertGreater(result['summary']['consistency_difference_count'], 0)
            self.assertEqual(result['summary']['first_stage_difference']['field'], 'wk_input')
            for f in data:
                f.pop('prolog_calls', None)
            self.assertEqual(compare(root, data, 'request-01', 'request-02', 2)['status'], 'not_captured')
            for f in data:
                f['prolog_capture'] = {'positions': '0:4'}
            self.assertEqual(compare(root, data, 'request-01', 'request-02', 2)['status'], 'incomplete_capture')

    def test_inspector_short_output_and_saved_detail(self):
        with tempfile.TemporaryDirectory() as root:
            data = dataset(root)
            Path(root, 'ki-rank-0-pid-1.jsonl').write_text('\n'.join(json.dumps(f) for f in data))
            output = Path(root, 'inspect.json')
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                inspect_ki.main([root, '--right', 'request-02', '--output', str(output)])
            lines = stream.getvalue().splitlines()
            self.assertEqual(len(lines), 4)
            summary = json.loads(lines[-1].split('=', 1)[1])
            self.assertEqual(summary['first_stage_difference']['field'], 'wk_output')
            self.assertIn('within_request_groups', json.loads(output.read_text())['prolog'])


class T(Tensor):
    def reshape(self, n, rest):
        flat = self.flat()
        width = len(flat) // n
        return T([flat[i * width:(i + 1) * width] for i in range(n)], self.dtype)


class Linear:
    def __init__(self):
        self.weight = T([1.])
        self.scale = T([17], 'uint8')
    def named_parameters(self):
        return [('weight', self.weight)]
    def named_buffers(self):
        return [('scale', self.scale)]
    def __call__(self, *a, **k):
        return self.forward(*a, **k)
    def forward(self, x):
        return T(copy.deepcopy(x.data)), None


class Norm(Linear):
    eps = 1e-6
    def forward(self, x):
        x.data[1][0] += 2.
        return x


class Indexer:
    ki_stream = wi_stream = None
    use_rope_fusion_op = False
    config = NS(indexer_rope_interleave=False)
    def __init__(self):
        self.wk, self.k_norm = Linear(), Norm()
        self.calls = 0
    def _apply_rope(self, x, cos, sin):
        x.data[1][0] += 3.
        return x
    def _li_prolog_ext(self, wx, qr, kx, q_cos_sin, k_cos_sin):
        self.calls += 1
        self._apply_rope(T([[0.] * 128] * 4), *q_cos_sin)  # Query RoPE must not be captured as Ki.
        x = self.wk(kx)[0]
        x = self.k_norm(x)
        x = self._apply_rope(x, *k_cos_sin)
        self.result = (wx, qr, x)
        return self.result


class RuntimeTests(unittest.TestCase):
    def test_actual_stages_snapshotted_before_inplace_norm_and_rope_preserve_results(self):
        with tempfile.TemporaryDirectory() as root, patch('vllm_bench.mtp_debug.tensor_io.tensor_bytes', side_effect=encode):
            w = TensorWriter(root, 1)
            indexer = Indexer()
            scope_prolog(indexer, NS(ena_sp=False), 'layers.0.attn', 0, (1, 3))
            frame = {'forward_index': 0, '_indexer_ki': {'status': 'capturing', 'phase': 'prefill',
                     'positions': [0, 1, 2, 3], 'input_ids': [10, 11, 12, 13]}}
            x, cos, sin = T([[float(p)] * 128 for p in range(4)]), T([[1.] * 64] * 4), T([[0.] * 64] * 4)
            token = spec_rejection.TARGET.set(frame)
            try:
                with patch.object(attention_trace, 'writer', return_value=w):
                    result = indexer._li_prolog_ext(x, x, x, (cos, sin), (cos, sin))
            finally:
                spec_rejection.TARGET.reset(token)
            self.assertIs(result, indexer.result)
            self.assertIsNone(PROLOG.get())
            call = frame['_indexer_ki']['prolog_calls'][0]
            self.assertEqual(call['status'], 'complete')
            self.assertEqual(call['row_positions'], [1, 2])
            values = {k: numbers(read_tensor(root, d), d['dtype'])[0] for k, d in call['tensors'].items()}
            self.assertEqual(values['wk_output'], 1.)
            self.assertEqual(values['norm_output'], 3.)
            self.assertEqual(values['rope_output'], 6.)
            self.assertEqual(values['cos'], 1.)
            self.assertEqual(x.data[1][0], 1.)
            self.assertEqual(call['runtime']['norm_eps'], 1e-6)
            self.assertIn('wk.named_buffers.scale', call['parameters_before'])

    def test_budget_unarmed_decode_unsupported_sp_and_exception_context(self):
        with tempfile.TemporaryDirectory() as root, patch('vllm_bench.mtp_debug.tensor_io.tensor_bytes', side_effect=encode):
            w = TensorWriter(root, 1, max_tensor_bytes=1, budget_bytes=1)
            idx = Indexer()
            owner = NS(ena_sp=False)
            scope_prolog(idx, owner, 'layers.0.attn', 0, (1, 3))
            x, c = T([[0.] * 128] * 4), T([[1.] * 64] * 4)
            args = (x, x, x, (c, c), (c, c))
            self.assertIs(idx._li_prolog_ext(*args), idx.result)
            frame = {'forward_index': 0, '_indexer_ki': {'status': 'capturing', 'phase': 'decode',
                     'positions': [0, 1, 2, 3], 'input_ids': [10, 11, 12, 13]}}
            token = spec_rejection.TARGET.set(frame)
            try:
                with patch.object(attention_trace, 'writer', return_value=w):
                    idx._li_prolog_ext(*args)
                    self.assertNotIn('prolog_calls', frame['_indexer_ki'])
                    frame['_indexer_ki']['phase'] = 'prefill'
                    idx._li_prolog_ext(*args)
                    self.assertEqual(frame['_indexer_ki']['prolog_calls'][-1]['status'], 'partial')
                    owner.ena_sp = True
                    idx._li_prolog_ext(*args)
                    self.assertEqual(frame['_indexer_ki']['prolog_calls'][-1]['status'], 'unsupported')
                    owner.ena_sp = False
                    idx.wk.forward = lambda x: (_ for _ in ()).throw(RuntimeError('kernel failed'))
                    with self.assertRaisesRegex(RuntimeError, 'kernel failed'):
                        idx._li_prolog_ext(*args)
                    self.assertIsNone(PROLOG.get())
            finally:
                spec_rejection.TARGET.reset(token)

    def test_cli_opt_in_validation_and_reproduce_forwarding(self):
        for bad in ('1', '-1:3', '2:2', '3:2', 'a:b'):
            with self.assertRaises(ValueError):
                parse_positions(bad)
        self.assertIsNone(parse_positions('all'))
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {}, clear=True):
            args = ['debug_spec', '--output', str(Path(root, 'new')), '--indexer-ki-prolog',
                    '--rejection-windows', '--', 'bash', 'launch.sh']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()) as stream:
                debug_spec.main()
            env = json.loads(stream.getvalue())['environment']
            self.assertEqual(env['PANGU_SPEC_DEBUG_INDEXER_KI_PROLOG'], '1')
            self.assertEqual(env['PANGU_SPEC_DEBUG_INDEXER_KI'], '1')
            args = inputs(Path(root))
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                reproduce.main([*args, '--indexer-ki-prolog', '--indexer-ki-prolog-positions', '100:228'])
            plan = json.loads(stream.getvalue())
            command = plan['server']['command']
            self.assertIn('--indexer-ki-prolog', command)
            self.assertEqual(command[command.index('--indexer-ki-prolog-positions') + 1], '100:228')


if __name__ == '__main__':
    unittest.main()
