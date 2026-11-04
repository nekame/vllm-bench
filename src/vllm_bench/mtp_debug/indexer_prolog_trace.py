"""Opt-in prefill-only Ki producer snapshots; never replace model results."""
import contextvars
import functools
import inspect
import math

from vllm_bench.mtp_debug.attention_trace import save

PROLOG = contextvars.ContextVar('pangu_ki_prolog', default=None)
STAGES = ('x_entry', 'wk_input', 'wk_output', 'norm_input', 'norm_output',
          'rope_input', 'cos', 'sin', 'rope_output')


def parse_positions(value):
    if value == 'all':
        return None
    parts = value.split(':')
    if len(parts) != 2:
        raise ValueError('Ki prolog positions must be all or START:STOP (stop excluded)')
    start, stop = map(int, parts)
    if start < 0 or stop <= start:
        raise ValueError('Ki prolog positions require 0 <= START < STOP')
    return start, stop


def capture(context, name, tensor):
    frame, call = context['frame'], context['call']
    if name in call['tensors']:
        call['issues'].append('duplicate_stage:' + name)
        return
    n = context['rows']
    if tensor is None or not tensor.shape or math.prod(tensor.shape) % n:
        call['tensors'][name] = {'status': 'not_recorded', 'reason': 'incompatible_token_layout'}
        return
    # Basic contiguous slice only: no NPU advanced indexing. Cos/sin BNSD
    # and activations TD/T1D are normalized to one row per forward token.
    if name not in ('cos', 'sin') and tensor.shape[0] != n:
        call['tensors'][name] = {'status': 'not_recorded', 'reason': 'incompatible_token_axis'}
        return
    if name in ('cos', 'sin') and math.prod(tensor.shape[:-1]) != n:
        call['tensors'][name] = {'status': 'not_recorded', 'reason': 'incompatible_rope_token_axis'}
        return
    begin, end = context['slice']
    call['tensors'][name] = save(frame, call, 'prolog-' + name, tensor.reshape(n, -1)[begin:end])


def parameters(context, indexer, moment):
    output = {}
    for prefix in ('wk', 'k_norm'):
        module = getattr(indexer, prefix)
        for kind in ('named_parameters', 'named_buffers'):
            for name, tensor in getattr(module, kind)():
                key = prefix + '.' + kind + '.' + name
                output[key] = save(context['frame'], context['call'],
                                   'prolog-' + moment + '-' + key.replace('.', '-'), tensor)
    context['call']['parameters_' + moment] = output
    if not output or not any(k.startswith('wk.named_parameters.') for k in output):
        context['call']['issues'].append('wk_parameters_unavailable:' + moment)


def scope_prolog(indexer, module, prefix, layer, wanted):
    original = indexer._li_prolog_ext
    signature = inspect.signature(original)

    @functools.wraps(original)
    def prolog(*args, **kwargs):
        from vllm_bench.mtp_debug.spec_rejection import TARGET
        frame = TARGET.get()
        trace = frame.get('_indexer_ki', {}) if frame else {}
        if trace.get('status') != 'capturing' or trace.get('phase') != 'prefill':
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs).arguments
        x = bound['kx']
        positions, ids = trace.get('positions', []), trace.get('input_ids', [])
        frame['_ki_tensor_slot'] = frame.get('_ki_tensor_slot', 20000) + 1
        call = {'prefix': prefix, 'layer': layer, 'slot': frame['_ki_tensor_slot'], 'status': 'capturing',
                'tensors': {}, 'issues': [], 'requested_positions': list(wanted) if wanted else 'all',
                'runtime': {'ena_sp': bool(getattr(module, 'ena_sp', False)),
                            'ki_stream_enabled': indexer.ki_stream is not None,
                            'wi_stream_enabled': indexer.wi_stream is not None,
                            'use_rope_fusion_op': bool(getattr(indexer, 'use_rope_fusion_op', False)),
                            'rope_interleave': bool(getattr(getattr(indexer, 'config', None), 'indexer_rope_interleave', False)),
                            'head_dim': getattr(indexer, 'head_dim', None),
                            'rope_dim': getattr(indexer, 'rope_dim', None),
                            'norm_class': type(indexer.k_norm).__name__,
                            'norm_eps': getattr(indexer.k_norm, 'variance_epsilon', getattr(indexer.k_norm, 'eps', None)),
                            'wk_quant_method': type(getattr(indexer.wk, 'quant_method', None)).__name__}}
        trace.setdefault('prolog_calls', []).append(call)
        if call['runtime']['ena_sp'] or len(positions) != x.shape[0] or len(ids) != len(positions):
            call.update(status='unsupported', reason='Prolog slice requires non-SP complete forward row alignment')
            return original(*args, **kwargs)
        rows = [i for i, pos in enumerate(positions) if wanted is None or wanted[0] <= pos < wanted[1]]
        if not rows:
            call.update(status='outside_requested_positions')
            return original(*args, **kwargs)
        if rows != list(range(rows[0], rows[-1] + 1)):
            call.update(status='unsupported', reason='Selected positions are not contiguous forward rows')
            return original(*args, **kwargs)
        call['row_positions'] = [positions[i] for i in rows]
        call['row_input_ids'] = [ids[i] for i in rows]
        context = {'frame': frame, 'call': call, 'rows': len(positions), 'slice': (rows[0], rows[-1] + 1), 'phase': 'entry'}
        capture(context, 'x_entry', x)
        parameters(context, indexer, 'before')
        token = PROLOG.set(context)
        try:
            result = original(*args, **kwargs)
            parameters(context, indexer, 'after')
            call['status'] = ('complete' if not call['issues'] and all(
                call['tensors'].get(stage, {}).get('status') == 'complete' for stage in STAGES)
                and all(d.get('status') == 'complete' for moment in ('before', 'after')
                        for d in call['parameters_' + moment].values()) else 'partial')
            return result
        except BaseException:
            call['status'] = 'error'
            raise
        finally:
            PROLOG.reset(token)

    def wrap_forward(owner, input_name, output_name, next_phase):
        forward = owner.forward
        @functools.wraps(forward)
        def invoke(*args, **kwargs):
            context = PROLOG.get()
            if context is None:
                return forward(*args, **kwargs)
            value = args[0] if args else kwargs.get('x')
            capture(context, input_name, value)
            result = forward(*args, **kwargs)
            capture(context, output_name, result[0] if isinstance(result, (tuple, list)) else result)
            context['phase'] = next_phase
            return result
        owner.forward = invoke

    rope = indexer._apply_rope
    @functools.wraps(rope)
    def apply_rope(x, cos, sin):
        context = PROLOG.get()
        if context is None or context['phase'] != 'norm_done':
            return rope(x, cos, sin)
        capture(context, 'rope_input', x)
        capture(context, 'cos', cos)
        capture(context, 'sin', sin)
        result = rope(x, cos, sin)
        capture(context, 'rope_output', result)
        context['phase'] = 'rope_done'
        return result

    wrap_forward(indexer.wk, 'wk_input', 'wk_output', 'wk_done')
    wrap_forward(indexer.k_norm, 'norm_input', 'norm_output', 'norm_done')
    indexer._apply_rope = apply_rope
    indexer._li_prolog_ext = prolog
