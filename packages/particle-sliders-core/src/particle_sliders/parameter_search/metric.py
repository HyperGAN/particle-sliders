"""Stream full native response coordinates; pool before normalizing per example.

JSONL is deliberately dependency-free. Chunks are transport only, never a
projection: each declared tensor must cover [0, prod(shape)) exactly once.
"""
import array
import hashlib
import json
from itertools import chain
import math
from pathlib import Path
import struct

METRIC = 'raw-fp64-operand-first-pooled-per-example-relative-sse-v2'
RAW = 'particle-sliders-raw-response-v2'
MAX_LINE_BYTES = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()



def finite_sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as exc:
        raise ValueError('nonfinite pooled response statistics') from exc
    if not math.isfinite(value):
        raise ValueError('nonfinite pooled response statistics')
    return value


class ExactSum:
    """Streaming exact sum of FP64 terms, rounded once across every chunk.

    Every binary64 value is an integer multiple of 2**-1074. A small integer
    superaccumulator keeps transport boundaries from changing rounding/rank.
    Storage grows logarithmically with the number of coordinates, not linearly.
    """
    def __init__(self):
        self.units = 0
        self.finite = True

    def add(self, value):
        if not math.isfinite(value):
            self.finite = False
            return
        numerator, denominator = value.as_integer_ratio()
        self.units += numerator << (1074 - (denominator.bit_length() - 1))

    def value(self):
        if not self.finite:
            raise ValueError('nonfinite pooled response statistics')
        try:
            value = self.units / (1 << 1074)
        except OverflowError as exc:
            raise ValueError('nonfinite pooled response statistics') from exc
        if not math.isfinite(value):
            raise ValueError('nonfinite pooled response statistics')
        return value

    def diagnostic(self):
        try:
            return self.value()
        except ValueError:
            return None


def diagnostic_mean(values):
    values = list(values)
    return None if any(value is None for value in values) else mean(values)


def mean(values):
    values = list(values)
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # The average of finite errors may be finite even if their sum is not.
        return finite_sum(v / len(values) for v in values)


def _finite_value(value):
    if type(value) not in (int, float):
        raise ValueError('nonnumeric native response coordinate')
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError('nonfinite native response coordinate') from exc
    if not math.isfinite(converted) or type(value) is int and converted != value:
        raise ValueError('nonfinite or inexact float64 native response coordinate')
    return converted

def validate_grid(grid):
    if not isinstance(grid, list) or not grid:
        raise ValueError('nonempty complete example grid required')
    examples = set()
    for example in grid:
        if set(example) != {'example', 'positions'}:
            raise ValueError('grid requires example and positions')
        label = example['example']
        if (not isinstance(label, dict) or not isinstance(label.get('scene'), str)
                or not label['scene'] or 'seed' not in label):
            raise ValueError('example needs a stable scene and seed (and backend path/labels)')
        key = canonical(label)
        if key in examples:
            raise ValueError('duplicate example label')
        examples.add(key)
        positions = set()
        if not example['positions']:
            raise ValueError('example has no declared positions')
        for position in example['positions']:
            if set(position) != {'position', 'shape'} or not isinstance(position['position'], dict):
                raise ValueError('position must retain a structured native label and shape')
            key = canonical(position['position'])
            if key in positions:
                raise ValueError('duplicate position label')
            positions.add(key)
            shape = position['shape']
            if not isinstance(shape, list) or not shape or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError('invalid native tensor shape')
    return grid


def write_raw(path, binding, chunks, *, binary=True):
    """Write bounded chunks, using a compact float64 sidecar by default.

    Products can yield bounded CPU slices from safetensors/NumPy/native tensors;
    they need not materialize the complete response or create giant JSON arrays.
    Publish the containing directory atomically (the runner does this).
    """
    path = Path(path)
    data_path = path.with_suffix('.bin')
    data = open(data_path, 'wb') if binary else None
    try:
        with open(path, 'wb') as stream:
            stream.write(canonical({'schema': RAW, 'metric': METRIC, 'binding': binding}) + b'\n')
            for original in chunks:
                chunk = dict(original)
                if binary:
                    for key in ('student', 'target', 'neutral'):
                        original_values = chunk[key]
                        values = array.array('d', (_finite_value(v) for v in original_values))
                        if not 0 < len(values) <= 65536:
                            raise ValueError('yield bounded chunks of 1..65536 coordinates')
                        # Explicit endian; consumers can rescore on another host.
                        import sys
                        if sys.byteorder != 'little':
                            values.byteswap()
                        chunk[key] = dict(file=data_path.name, offset=data.tell(),
                                          count=len(values), dtype='float64-le')
                        data.write(values.tobytes())
                line = canonical(chunk)
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError('response chunk exceeds transport limit; yield smaller chunks')
                stream.write(line + b'\n')
    finally:
        if data is not None:
            data.close()


def _coordinates(value, path):
    if isinstance(value, list):
        return value
    if (not isinstance(value, dict) or set(value) != {'file', 'offset', 'count', 'dtype'}
            or value['dtype'] != 'float64-le' or type(value['count']) is not int
            or not 0 < value['count'] <= 65536 or type(value['offset']) is not int
            or value['offset'] < 0 or value['offset'] % 8):
        raise ValueError('invalid bounded binary coordinate reference')
    root = Path(path).resolve().parent
    target = (root / value['file']).resolve()
    if target.parent != root or target.name != value['file']:
        raise ValueError('binary coordinates must be a local artifact sidecar')
    with open(target, 'rb') as stream:
        stream.seek(value['offset'])
        data = stream.read(value['count'] * 8)
    if len(data) != value['count'] * 8:
        raise ValueError('truncated binary coordinates')
    return [v[0] for v in struct.iter_unpack('<d', data)]


def _rows(path):
    with open(path, 'rb') as stream:
        while True:
            line = stream.readline(MAX_LINE_BYTES + 2)
            if not line:
                return
            if len(line) > MAX_LINE_BYTES + 1 or not line.endswith(b'\n'):
                raise ValueError('oversized or truncated raw response line')
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError('response rows must be objects')
            yield row


def _score(rows, grid, binding, *, path=None, reference_sha256=None, min_edit_energy=0.):
    """Recompute from raw S,T,N. No worker-provided scalar enters this function.

    Targets are backend-declared *actual* training targets, including signed
    formulations. Fixed aligned hidden/token/logit fields can use this contract;
    unrelated generated waveforms or text strings cannot.
    """
    validate_grid(grid)
    if type(min_edit_energy) not in (int, float) or min_edit_energy != 0:
        raise ValueError('min_edit_energy must be zero; no denominator epsilon or threshold')
    rows = iter(rows)
    if canonical(next(rows, None)) != canonical({'schema': RAW, 'metric': METRIC, 'binding': binding}):
        raise ValueError('raw metric header/binding differs')
    reference = hashlib.sha256()
    results = []
    for example in grid:
        totals = [ExactSum() for _ in range(4)]
        count = 0
        for position in example['positions']:
            shape, label = position['shape'], position['position']
            expected = dict(example=example['example'], position=label, shape=shape)
            reference.update(canonical(expected))
            offset, size = 0, math.prod(shape)
            while offset < size:
                row = next(rows, None)
                if row is None or set(row) != {'example', 'position', 'shape', 'offset', 'student', 'target', 'neutral'}:
                    raise ValueError('missing or malformed raw coordinate chunk')
                if (any(canonical(row[key]) != canonical(value) for key, value in expected.items())
                        or type(row['offset']) is not int or row['offset'] != offset):
                    raise ValueError('raw labels/shape/order/offset differ from complete declared grid')
                arrays = [_coordinates(row[key], path) for key in ('student', 'target', 'neutral')]
                length = len(arrays[0]) if isinstance(arrays[0], list) else 0
                if not length or any(not isinstance(v, list) or len(v) != length for v in arrays) or offset + length > size:
                    raise ValueError('coordinate counts differ or exceed native tensor shape')
                for student, target, neutral in zip(*arrays):
                    student, target, neutral = (_finite_value(v) for v in (student, target, neutral))
                    error, edit, actual = student - target, target - neutral, student - neutral
                    values = (error * error, edit * edit, actual * actual, actual * edit)
                    if not all(math.isfinite(v) for v in values[:2]):
                        raise ValueError('nonfinite squared response statistics')
                    for out, value in zip(totals, values):
                        out.add(value)
                    # Stable reference identity independent of transport chunk size.
                    reference.update(struct.pack('!dd', target, neutral))
                count += length
                offset += length
        sse, energy = (values.value() for values in totals[:2])
        actual_energy, dot = (values.diagnostic() for values in totals[2:])
        if energy <= 0:
            raise ValueError('nonfinite or degenerate whole-example teacher edit')
        error = sse / energy
        if not math.isfinite(error):
            raise ValueError('nonfinite normalized whole-example metric')
        gain = None if dot is None else dot / energy
        if gain is not None and not math.isfinite(gain):
            gain = None
        results.append(dict(example=example['example'], elements=count,
            student_teacher_sse=sse, teacher_edit_energy=energy,
            student_edit_energy=actual_energy, student_teacher_edit_dot=dot,
            error=error, signed_edit_gain=gain))
    if next(rows, None) is not None:
        raise ValueError('extra raw coordinates/examples outside the declared grid')
    actual_reference = reference.hexdigest()
    if reference_sha256 is not None and actual_reference != reference_sha256:
        raise ValueError('frozen teacher/neutral reference differs')
    scenes = {}
    for row in results:
        scenes.setdefault(row['example']['scene'], []).append(row)
    summary = dict(metric=METRIC, error=mean(r['error'] for r in results),
        signed_edit_gain=diagnostic_mean(r['signed_edit_gain'] for r in results),
        examples=results, reference_sha256=actual_reference,
        per_scene={key: dict(error=mean(r['error'] for r in values),
            signed_edit_gain=diagnostic_mean(r['signed_edit_gain'] for r in values))
            for key, values in sorted(scenes.items())})
    summary['diagnostics'] = dict(aligned_better_than_off=all(
        v['error'] < 1 and v['signed_edit_gain'] is not None and v['signed_edit_gain'] > 0
        for v in summary['per_scene'].values()))
    return summary


def score(path, grid, binding, *, reference_sha256=None, min_edit_energy=0.):
    """Rescore retained raw binary/JSONL artifacts without importing a model."""
    return _score(_rows(path), grid, binding, path=path,
                  reference_sha256=reference_sha256, min_edit_energy=min_edit_energy)


def score_chunks(chunks, grid, *, reference_sha256=None, min_edit_energy=0.):
    """Pure bounded-iterator metric API for native safetensors/NumPy adapters.

    Pass full-coordinate CPU lists in each chunk; no JSON or binary conversion
    is needed for this API. The runner additionally retains binary audit data.
    """
    binding = {'source': 'in-memory full-coordinate iterator'}
    header = {'schema': RAW, 'metric': METRIC, 'binding': binding}
    return _score(chain([header], chunks), grid, binding,
                  reference_sha256=reference_sha256, min_edit_energy=min_edit_energy)


def decoded_chunks(path):
    """Stream bounded CPU chunks from retained raw evidence, preserving labels."""
    rows = iter(_rows(path))
    next(rows)
    for original in rows:
        row = dict(original)
        for key in ('student', 'target', 'neutral'):
            row[key] = _coordinates(row[key], path)
        yield row


def algebraic_controls(path, grid, *, reference_sha256, min_edit_energy=0.):
    """Qualify the formula on actual retained T/N; these are NOT native OFF."""
    results = {}
    for name, gain, expected in [('off', 0., 1.), ('teacher', 1., 0.), ('halfway', .5, .25), ('reverse', -1., 4.)]:
        def chunks():
            for original in decoded_chunks(path):
                row = dict(original)
                row['student'] = [n + gain * (t - n) for t, n in zip(row['target'], row['neutral'])]
                yield row
        result = score_chunks(chunks(), grid, reference_sha256=reference_sha256,
                              min_edit_energy=min_edit_energy)
        if any(abs(row['error'] - expected) > 1e-6 for row in result['examples']):
            raise ValueError('algebraic metric qualification failed: ' + name)
        results[name] = result['error']
    return dict(evidence='algebraic controls recomputed from retained full target/neutral coordinates; not trained/native OFF',
                expected={'off': 1., 'teacher': 0., 'halfway': .25, 'reverse': 4.}, observed=results,
                reference_sha256=reference_sha256)
