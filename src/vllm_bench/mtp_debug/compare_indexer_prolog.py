"""Exact offline Ki producer comparison, with explicit slice coverage."""
import hashlib

from vllm_bench.mtp_debug.attention_compare import Reader
from vllm_bench.mtp_debug.compare_indexer_ki import select
from vllm_bench.mtp_debug.indexer_prolog_trace import STAGES
from vllm_bench.mtp_debug.tensor_io import compare_bytes

FIELDS = (*STAGES, 'cache_entry')
ERRORS = (ValueError, KeyError, TypeError, OSError, IndexError)


def signature(reader, descriptors):
    if not descriptors:
        raise ValueError('Parameter snapshots absent')
    return {name: (d['dtype'], tuple(d['shape']), hashlib.sha256(reader.read(d)).hexdigest())
            for name, d in descriptors.items()}


def collect(reader, frames, layer):
    rows, parameters, runtime, requested = {}, None, None, None
    for frame in frames:
        if frame['phase'] != 'prefill':
            continue
        calls = [c for c in frame.get('prolog_calls', []) if c['layer'] == layer]
        if not calls:
            raise ValueError('Prefill Ki prolog capture absent')
        for call in calls:
            wanted = call['requested_positions']
            if requested is not None and requested != wanted:
                raise ValueError('Requested slice changes within request')
            requested = wanted
            if call['status'] == 'outside_requested_positions':
                continue
            if call['status'] != 'complete':
                raise ValueError('Incomplete prolog: ' + str(call.get('reason') or call.get('issues') or call['status']))
            current = {moment: signature(reader, call.get('parameters_' + moment)) for moment in ('before', 'after')}
            if parameters is not None and parameters != current:
                raise ValueError('Parameter snapshots change across prefill chunks; inspect prolog_calls')
            parameters = current
            if runtime is not None and runtime != call['runtime']:
                raise ValueError('Prolog runtime changes across prefill chunks')
            runtime = call['runtime']
            positions, ids = call['row_positions'], call['row_input_ids']
            if len(positions) != len(ids):
                raise ValueError('Prolog IDs/positions length mismatch')
            for i, (pos, token) in enumerate(zip(positions, ids)):
                if pos in rows:
                    raise ValueError('Duplicate prolog position')
                item = {'token': token}
                for field in STAGES:
                    descriptor = call['tensors'][field]
                    if descriptor['shape'][0] != len(positions):
                        raise ValueError('Prolog stage row count mismatch: ' + field)
                    raw, tail = reader.row(descriptor, i)
                    item[field] = (raw, descriptor['dtype'], tail)
                rows[pos] = item
        for call in frame.get('calls', []):
            if call['layer'] != layer:
                continue
            for i in call.get('active_rows', []):
                pos = call['row_positions'][i]
                if pos in rows:
                    raw, tail = reader.row(call['ki_before'], i)
                    rows[pos]['cache_entry'] = (raw, call['ki_before']['dtype'], tail)
    positions = sorted(rows)
    if not positions:
        raise ValueError('Requested Ki prolog slice not recorded')
    expected = list(range(positions[-1] + 1)) if requested == 'all' else list(range(*requested))
    if positions != expected:
        raise ValueError('Requested Ki prolog slice is not completely recorded')
    if any('cache_entry' not in item for item in rows.values()):
        raise ValueError('Matching cache-entry Ki snapshot absent')
    return rows, parameters, runtime


def compare_field(a, b, field):
    if a.keys() != b.keys() or any(a[p]['token'] != b[p]['token'] for p in a):
        raise ValueError('Prolog positions/input IDs differ')
    changed, first = [], None
    for pos in sorted(a):
        x, dtype, shape = a[pos][field]
        y, other_dtype, other_shape = b[pos][field]
        if dtype != other_dtype or shape != other_shape:
            raise ValueError('Prolog row layout differs: ' + field)
        if x != y:
            changed.append(pos)
            if first is None:
                first = {'position': pos, **compare_bytes(x, y, dtype, shape)}
    return {'field': field, 'exact_equal': not changed, 'different_rows': len(changed),
            'sampled_positions': changed[:16], 'first': first}


def compare(root, frames, left, right, expected_ranks, layer=0):
    enabled = any('prolog_calls' in f or f.get('prolog_capture') for f in frames)
    if not enabled:
        return {'status': 'not_captured', 'summary': {'status': 'not_captured'}}
    issues, differences, invariants, parameter_differences, runtime_differences = [], [], [], [], []
    groups = {side: {field: {} for field in (*FIELDS, 'parameters_before', 'parameters_after', 'runtime')}
              for side in ('left', 'right')}
    for rank in range(expected_ranks):
        reader, selected = Reader(root), {}
        for side, name in (('left', left), ('right', right)):
            try:
                selected[side] = collect(reader, select(frames, rank, name), layer)
                rows, params, runtime = selected[side]
                for field in FIELDS:
                    digest = hashlib.sha256()
                    for pos, item in sorted(rows.items()):
                        raw, dtype, shape = item[field]
                        digest.update(repr((pos, item['token'], dtype, shape)).encode())
                        digest.update(raw)
                    groups[side][field].setdefault(digest.hexdigest(), []).append(rank)
                for field, value in (('parameters_before', params['before']), ('parameters_after', params['after']), ('runtime', runtime)):
                    groups[side][field].setdefault(repr(sorted(value.items())), []).append(rank)
                for source, target in (('x_entry', 'wk_input'), ('wk_output', 'norm_input'),
                                       ('norm_output', 'rope_input'), ('rope_output', 'cache_entry')):
                    changed = [p for p, item in rows.items() if item[source] != item[target]]
                    if changed:
                        invariants.append({'rank': rank, 'side': side, 'source': source, 'target': target,
                                           'different_rows': len(changed), 'first_position': min(changed)})
                if params['before'] != params['after']:
                    parameter_differences.append({'rank': rank, 'side': side, 'boundary': 'within_prolog',
                        'fields': sorted(k for k in params['before'].keys() | params['after'].keys()
                                         if params['before'].get(k) != params['after'].get(k))})
            except ERRORS as error:
                selected.pop(side, None)
                issues.append({'rank': rank, 'side': side, 'reason': str(error)})
        if len(selected) != 2:
            continue
        a, pa, ra = selected['left']
        b, pb, rb = selected['right']
        try:
            for field in FIELDS:
                record = compare_field(a, b, field)
                if not record['exact_equal']:
                    differences.append({'rank': rank, 'layer': layer, **record})
            for moment in ('before', 'after'):
                fields = sorted(k for k in pa[moment].keys() | pb[moment].keys() if pa[moment].get(k) != pb[moment].get(k))
                if fields:
                    parameter_differences.append({'rank': rank, 'boundary': 'cross_request_' + moment, 'fields': fields})
            if ra != rb:
                runtime_differences.append({'rank': rank, 'left': ra, 'right': rb})
        except ERRORS as error:
            issues.append({'rank': rank, 'reason': str(error)})
    first = min(differences, key=lambda d: (FIELDS.index(d['field']), d['first']['position'], d['rank'])) if differences else None
    brief = None
    if first:
        brief = {k: first[k] for k in ('rank', 'layer', 'field', 'different_rows')}
        brief.update({k: first['first'][k] for k in ('position', 'max_abs_finite', 'max_ulp_finite', 'relative_l2_finite', 'nonfinite_elements')})
    grouped = {side: {field: list(g.values()) for field, g in fields.items()} for side, fields in groups.items()}
    splits = [{'side': side, 'field': field, 'groups': value}
              for side, fields in grouped.items() for field, value in fields.items() if len(value) > 1]
    status = ('incomplete_capture' if issues else 'differences_observed' if differences or parameter_differences or runtime_differences or invariants or splits
              else 'equal_in_captured_slice')
    summary = {'status': status, 'first_stage_difference': brief,
               'stage_differences': [{'rank': d['rank'], 'field': d['field'], 'different_rows': d['different_rows']} for d in differences],
               'consistency_difference_count': len(invariants), 'first_consistency_difference': next(iter(invariants), None),
               'parameter_difference_count': len(parameter_differences), 'first_parameter_difference': next(iter(parameter_differences), None),
               'runtime_difference_count': len(runtime_differences),
               'within_request_split_count': len(splits), 'first_within_request_split': next(iter(splits), None),
               'first_stage_groups': {side: grouped[side][first['field']] for side in grouped} if first else None,
               'coverage_issue_count': len(issues), 'first_coverage_issue': next(iter(issues), None)}
    return {'status': status, 'summary': summary, 'differences': differences, 'consistency_differences': invariants,
            'parameter_differences': parameter_differences, 'runtime_differences': runtime_differences,
            'within_request_groups': grouped,
            'coverage_issues': issues,
            'notes': ['First stage difference orders recorded stages; it is not proof of a faulty operator.',
                      'Parameter equality covers registered parameters/buffers, not unregistered operator workspaces.',
                      'Coverage and equality apply only to the requested prefill slice; copies synchronize NPU streams.']}
