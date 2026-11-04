"""Read only three Ki RoPE runtime flags from selected prefill JSONL records."""
import argparse
import json
from pathlib import Path

FIELDS = ('use_rope_fusion_op', 'rope_interleave', 'ki_stream_enabled')


def header(line):
    # Native sidecars put these scalars before the large calls/reads arrays.
    start = line.find('"calls":')
    if start >= 0:
        try:
            value = json.loads(line[:start].rstrip().rstrip(',') + '}')
            if {'format', 'rank', 'phase', 'label', 'request_ids'} <= value.keys():
                return value
        except ValueError:
            pass
    return json.loads(line)  # Support reordered/compact external JSONL too.


def extract(path, request, rank=0, layer=0):
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        path = path / 'capture' if (path / 'capture').is_dir() else path
        files = sorted(path.rglob(f'ki-rank-{rank}-pid-*.jsonl'))
    if not files:
        raise ValueError(f'No Ki JSONL files for rank {rank}')
    identities, evidence, flags = set(), [], None
    for file in files:
        with file.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    meta = header(line)
                    if meta.get('format') != 'pangu-indexer-ki-v1':
                        raise ValueError('Unexpected Ki format')
                    if (meta.get('rank') != rank or meta.get('phase') != 'prefill'
                            or request not in [meta.get('label'), *meta.get('request_ids', [])]):
                        continue
                    frame = json.loads(line)
                    calls = [c for c in frame.get('prolog_calls', []) if c.get('layer') == layer]
                    if not calls:
                        raise ValueError(f'No prolog_calls for layer {layer}; capture needs --indexer-ki-prolog')
                    identities.add((frame.get('pid'), tuple(frame.get('request_ids', [])), frame.get('label')))
                    for call in calls:
                        runtime = call.get('runtime', {})
                        current = {key: runtime.get(key) for key in FIELDS}
                        if any(type(value) is not bool for value in current.values()):
                            raise ValueError('Missing/non-boolean runtime flags; unavailable is not false')
                        if flags is not None and current != flags:
                            raise ValueError('Runtime flags differ across selected prefill calls; use a narrower capture/request')
                        flags = current
                        evidence.append({'file': str(file), 'line': number, 'forward_index': frame.get('forward_index'),
                                         'call_status': call.get('status')})
                except (ValueError, KeyError, TypeError) as error:
                    raise ValueError(f'{file}:{number}: {error}') from error
    if flags is None:
        raise ValueError(f'No prefill for request={request!r}, rank={rank}, layer={layer}')
    if len(identities) != 1:
        raise ValueError('Request selector matches multiple processes/requests; select one JSONL file or exact request ID')
    return {'request': request, 'rank': rank, 'layer': layer, 'runtime': flags, 'matches': evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Run output, capture directory, or one Ki JSONL file')
    parser.add_argument('--request', help='Label or exact request ID; default: result.json prolog.right_label')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--output', type=Path, help='Optional new JSON report including source file/line')
    args = parser.parse_args(argv)
    try:
        if min(args.rank, args.layer) < 0:
            raise ValueError('Rank and layer must be nonnegative')
        if args.output and args.output.exists():
            raise FileExistsError('Use a new output file')
        request = args.request
        if request is None:
            directory = args.path.parent if args.path.is_file() else args.path
            result = next((p for p in (directory / 'result.json', directory.parent / 'result.json') if p.is_file()), None)
            if result:
                request = json.loads(result.read_text()).get('prolog', {}).get('right_label')
            if not request:
                raise ValueError('Cannot infer request from result.json; supply --request request-NN')
        report = extract(args.path, request, args.rank, args.layer)
        if args.output:
            with args.output.open('x') as stream:
                stream.write(json.dumps(report, indent=2) + '\n')
        for name, value in report['runtime'].items():
            print(name + '=' + json.dumps(value))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
