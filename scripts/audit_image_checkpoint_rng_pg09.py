"""Reviewed offline audit candidate; actual archived packet execution is NOT admitted.

static: ZIP metadata and pickle opcodes only. rng: restricted CPU tensor loading
only after a separately reviewed static receipt. No models, optimizers, CUDA,
unrestricted pickle, custom globals, worker dispatch, or RNG restoration.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import pickletools
import resource
import signal
import time
import zipfile

FIELDS = {'identity', 'step', 'trainable', 'optimizer', 'torch_rng', 'python_rng', 'cuda_rng'}
MAX_PACKET_BYTES = 2 * 1024**3
MAX_PICKLE_BYTES = 16 * 1024**2
MAX_ADDRESS_SPACE = 8 * 1024**3
MIN_AVAILABLE = 16 * 1024**3


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def static_inventory(path):
    """Never execute a pickle opcode or read tensor-storage ZIP members."""
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 20000 or len({i.filename for i in infos}) != len(infos):
            raise ValueError('oversized or duplicate ZIP member table')
        if any(i.flag_bits & 1 or '..' in Path(i.filename).parts for i in infos):
            raise ValueError('encrypted or invalid ZIP member')
        pickles = [i for i in infos if i.filename.endswith('/data.pkl')]
        if len(pickles) != 1 or pickles[0].file_size > MAX_PICKLE_BYTES:
            raise ValueError('exactly one bounded PyTorch data.pkl required')
        data = archive.read(pickles[0])
        counts, globals_found, field_strings = Counter(), set(), set()
        for opcode, arg, _ in pickletools.genops(data):
            counts[opcode.name] += 1
            if opcode.name == 'GLOBAL':
                globals_found.add(arg)
            if isinstance(arg, str) and arg in FIELDS:
                field_strings.add(arg)
        return dict(members=[dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size,
                                  compression=i.compress_type) for i in infos],
            data_pickle_sha256=hashlib.sha256(data).hexdigest(), opcode_counts=dict(counts),
            global_references=sorted(globals_found), expected_field_string_hits=sorted(field_strings),
            stack_global_requires_review=bool(counts['STACK_GLOBAL']),
            semantics='opcode/string inventory only; not proof of object schema or safety',
            pickle_executed=False, tensor_storage_members_read=False)


def rng_schema(path, *, identity, step):
    import torch
    # No allowlist expansion, safe_globals, unrestricted fallback, or CUDA map.
    state = torch.load(str(path), weights_only=True, map_location='cpu', mmap=True)
    if type(state) is not dict or set(state) != FIELDS:
        raise ValueError('checkpoint schema differs from frozen TrainingState save contract')
    if state['identity'] != identity or state['step'] != step:
        raise ValueError('packet logical identity/step mismatch')
    if type(state['trainable']) is not dict or type(state['optimizer']) is not dict:
        raise ValueError('trainable/optimizer dictionary schema required')
    if type(state['cuda_rng']) is not list:
        raise ValueError('CUDA RNG list schema required')
    def rng_record(value):
        if (type(value) is not torch.Tensor or value.device.type != 'cpu' or
                value.dtype != torch.uint8 or value.ndim != 1 or value.numel() > 1024**2):
            raise ValueError('bounded CPU uint8 RNG byte vector required')
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    trainable = {}
    for name, value in state['trainable'].items():
        if type(name) is not str or not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
            raise ValueError('unexpected trainable schema')
        # Shapes only; never inspect, instantiate, apply or hash model values.
        trainable[name] = dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel())
    python_rng = state['python_rng']
    if (type(python_rng) is not tuple or len(python_rng) != 3 or type(python_rng[0]) is not int or
            type(python_rng[1]) is not tuple or len(python_rng[1]) > 10000 or
            not all(type(x) is int for x in python_rng[1]) or
            (python_rng[2] is not None and type(python_rng[2]) is not float)):
        raise ValueError('unexpected Python RNG schema')
    result = dict(identity=identity, step=step, fields=sorted(state), trainable_schema=trainable,
        optimizer_keys=sorted(state['optimizer']), torch_rng=rng_record(state['torch_rng']),
        cuda_rng=[rng_record(x) for x in state['cuda_rng']],
        python_rng=dict(version=python_rng[0], integer_count=len(python_rng[1]),
            sha256=hashlib.sha256(json.dumps(python_rng, separators=(',', ':')).encode()).hexdigest()),
        rng_restored=False, model_instantiated=False, optimizer_instantiated=False,
        native_forward=False, cuda_initialized=torch.cuda.is_initialized())
    if result['cuda_initialized']:
        raise RuntimeError('CPU audit initialized CUDA')
    del state
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--mode', choices=['static', 'rng'], required=True)
    parser.add_argument('--reviewed-static-receipt', type=Path)
    parser.add_argument('--reviewed-static-sha256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise RuntimeError('explicit CUDA-hidden single-thread environment required')
    resource.setrlimit(resource.RLIMIT_AS, (MAX_ADDRESS_SPACE, MAX_ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s audit cap')))
    signal.alarm(120)
    start = time.monotonic()
    if sha_file(args.plan) != args.plan_sha256:
        raise ValueError('audit plan checksum mismatch')
    plan = json.loads(args.plan.read_text())
    if sha_file(__file__) != plan['audit_script_sha256']:
        raise ValueError('audit script checksum mismatch')
    for binding in plan['source_bindings']:
        source = Path(binding['path'])
        if source.suffix in {'.pt','.pth','.bin','.safetensors','.ckpt','.model','.npz','.npy'}:
            raise ValueError('binary packet cannot be a source binding')
        if source.stat().st_size != binding['bytes'] or sha_file(source) != binding['sha256']:
            raise ValueError('audit source/input binding changed')
    reviewed = None
    if args.mode == 'rng':
        if not args.reviewed_static_receipt or not args.reviewed_static_sha256:
            raise ValueError('separately reviewed static receipt required before restricted tensor audit')
        if sha_file(args.reviewed_static_receipt) != args.reviewed_static_sha256:
            raise ValueError('static receipt checksum mismatch')
        reviewed = json.loads(args.reviewed_static_receipt.read_text())
        if reviewed['mode'] != 'static' or reviewed['plan_sha256'] != args.plan_sha256:
            raise ValueError('static receipt plan mismatch')
    if args.mode == 'rng':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    output = dict(mode=args.mode, plan_sha256=args.plan_sha256, packets=[],
                  actual_historical_draw_replay_qualified=False)
    for packet in plan['known_packets']:
        if available_bytes() < MIN_AVAILABLE:
            raise RuntimeError('at least16GiB MemAvailable required')
        path, sidecar = Path(packet['path']), Path(packet['sidecar']['path'])
        if (path.stat().st_size != packet['bytes'] or path.stat().st_size > MAX_PACKET_BYTES or
                sha_file(sidecar) != packet['sidecar']['sha256'] or
                sha_file(path) != packet['declared_sha256']):
            raise ValueError('known checkpoint packet changed')
        item = dict(path=str(path), sha256=packet['declared_sha256'], identity=packet['identity'], step=packet['step'])
        if args.mode == 'static':
            item['static'] = static_inventory(path)
        else:
            matching = [p for p in reviewed['packets'] if p['path'] == str(path) and p['sha256'] == packet['declared_sha256']]
            if len(matching) != 1 or matching[0]['static']['stack_global_requires_review']:
                raise ValueError('static packet review missing or unresolved STACK_GLOBAL')
            item['rng_schema'] = rng_schema(path, identity=packet['identity'], step=packet['step'])
        output['packets'].append(item)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > MAX_ADDRESS_SPACE:
            raise RuntimeError('8GiB RSS cap exceeded')
    output.update(elapsed_seconds=time.monotonic()-start,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(dict(mode=args.mode, packets=len(output['packets']), output=str(args.output))))


if __name__ == '__main__':
    main()
