"""Observe native training through the pinned HyperGAN develop reporting API.

This file is also a standalone CPU-only sidecar. It never imports a model,
changes a training configuration, requests a sample, or owns a checkpoint.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
import uuid

HYPERGAN_COMMIT = '5f368a3770cf5bdae0448995c9f76e29af8049f7'
HYPERGAN_SOURCE_SHA256 = '9c725516409bfbb80f9740c65a435c0b7219be55d0ca88e2c777dc76aea2cb1e'
MAX_LINE_BYTES = 1024 * 1024


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def source_hash(root):
    value = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        value.update(path.relative_to(root).as_posix().encode() + b'\0')
        value.update(bytes.fromhex(file_hash(path)))
    return value.hexdigest()


def load_runtime(path, commit, checksum):
    """Verify actual imported develop bytes, without importing training deps."""
    if commit != HYPERGAN_COMMIT or checksum != HYPERGAN_SOURCE_SHA256:
        raise ValueError('unsupported HyperGAN reporting pin; this adapter is reviewed against develop ' + HYPERGAN_COMMIT)
    parent = Path(path).expanduser().resolve()
    package = parent / 'hypergan'
    if not (package / 'observation_io.py').is_file() or source_hash(package) != checksum:
        raise ValueError('HyperGAN reporting source bytes do not match the pinned develop commit')
    if 'torch' in sys.modules:
        raise RuntimeError('HyperGAN reporter must run in its own process without Torch')
    existing = sys.modules.get('hypergan')
    if existing is not None and Path(existing.__file__).resolve().parent != package:
        raise RuntimeError('another HyperGAN runtime is already imported')
    sys.path.insert(0, str(parent))
    observation = importlib.import_module('hypergan.observation_io')
    metrics = importlib.import_module('hypergan.metrics')
    state = importlib.import_module('hypergan.run_state')
    events = importlib.import_module('hypergan.run_events')
    summary = importlib.import_module('hypergan.summary')
    if Path(observation.__file__).resolve().parent != package or 'torch' in sys.modules:
        raise RuntimeError('reporting imports escaped the pinned CPU-only runtime')
    return SimpleNamespace(ObservationIO=observation.ObservationIO, Throughput=metrics.Throughput,
        digest=metrics.digest, read_catalog=metrics.read_catalog, atomic_json=state.atomic_json,
        run_lock=state.run_lock, read_event_page=events.read_event_page,
        write_summary_json=summary.write_summary_json,
        identity=dict(repository='https://github.com/HyperGAN/HyperGAN', branch='develop',
                      commit=commit, python_source_sha256=checksum, imported_path=str(package)))


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + value)))


def metric_record(row):
    """Map already measured scalars; never infer samples or objective weights."""
    raw = row.get('metrics', row)
    if not isinstance(raw, dict):
        raise ValueError('native metrics must be a JSON object')
    step = row.get('completed_steps', raw.get('step', row.get('step')))
    if type(step) is not int or step < 1:
        raise ValueError('native metric row requires a positive completed step')
    names = {'g_loss': 'loss/g_total', 'd_loss': 'loss/d_total',
             'grad_norm_critic': 'grad_norm/critic', 'grad_norm_generator': 'grad_norm/generator',
             'grad_norm_prior': 'grad_norm/prior', 'critic_real_mean': 'critic/real_mean',
             'critic_fake_mean': 'critic/fake_mean', 'critic_margin': 'critic/margin',
             'critic_accuracy': 'critic/accuracy', 'd_guard_ratio': 'optimizer/d_guard_ratio',
             'particle_rms': 'prior/particle_rms', 'diversity_ratio': 'diversity/ratio'}
    scalars = {}
    for key, value in raw.items():
        if key in ('step', 'completed_steps') or type(value) not in (int, float):
            continue
        if not math.isfinite(value):
            raise ValueError('nonfinite native metric: ' + key)
        name = names.get(key, key if '/' in key else 'native/' + key)
        if name in scalars and scalars[name] != value:
            raise ValueError('conflicting aliases for native metric: ' + name)
        scalars[name] = value
    duration = row.get('seconds')
    if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
        raise ValueError('native update duration must be finite and nonnegative')
    return step, scalars, duration


def metric_path(run):
    # Prefer append-only native logs over a controller's periodically copied view.
    for name in ('artifacts/training/training.jsonl', 'artifacts/native/metrics.jsonl',
                 'artifacts/metrics.jsonl'):
        path = run / name
        if path.is_file():
            return path
    return None


def complete_lines(path, offset, anchor):
    """Read bounded complete rows, preserving a concurrently written tail."""
    with path.open('rb') as stream:
        if path.stat().st_size < offset:
            return  # A copied controller view can briefly be truncated mid-copy.
        stream.seek(max(0, offset - 256))
        if anchor and hashlib.sha256(stream.read(offset - max(0, offset - 256))).hexdigest() != anchor:
            raise ValueError('previously observed native metric bytes changed')
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline(MAX_LINE_BYTES + 1)
            if len(line) > MAX_LINE_BYTES:
                raise ValueError('native metric row exceeds the 1 MiB reporting bound')
            if not line or not line.endswith(b'\n'):
                return
            end = stream.tell()
            stream.seek(max(0, end - 256))
            next_anchor = hashlib.sha256(stream.read(end - max(0, end - 256))).hexdigest()
            stream.seek(end)
            value = json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError('nonfinite metric JSON')))
            if not isinstance(value, dict):
                raise ValueError('native metric row must be an object')
            yield value, dict(path=str(path), offset=end, anchor=next_anchor,
                              row_sha256=hashlib.sha256(line).hexdigest(), start=start)


def checkpoint_record(directory):
    """Recognize existing full-state formats and verify the serialized bytes."""
    directory = Path(directory)
    for marker, binary, step_key, hash_key in (
            ('manifest.json', 'state.pt', 'completed_steps', 'weights_sha256'),
            ('checkpoint.json', 'training.pt', 'completed_steps', 'sha256'),
            ('checkpoint.json', 'training.pt', 'step', 'sha256')):
        if not (directory / marker).is_file() or not (directory / binary).is_file():
            continue
        data = read_json(directory / marker)
        if step_key not in data:
            continue
        step = data[step_key]
        if type(step) is not int or step < 0 or file_hash(directory / binary) != data.get(hash_key):
            raise ValueError('native checkpoint step or bytes do not match its marker: ' + str(directory))
        return dict(path=str(directory.resolve()), step=step, sha256=data[hash_key],
                    marker_sha256=file_hash(directory / marker))
    return None


def terminal_identity(result):
    """Observational enrichment after reporter exit is not a new terminal run."""
    return {key: result.get(key) for key in
            ('status', 'completed_steps', 'target_steps', 'checkpoint', 'adapter', 'last_checkpoint')}


class Reporter:
    def __init__(self, run, api):
        self.run, self.api = Path(run).resolve(), api
        self.output = self.run / 'reporting/hypergan'
        self.request = read_json(self.run / 'request.json')
        self.request_hash = file_hash(self.run / 'request.json')
        self.target = self.request.get('resolved', {}).get('target_steps')
        if type(self.target) is not int or self.target < 1:
            raise ValueError('reporter requires a resolved request with positive target_steps')
        self.run_id = 'particle-' + hashlib.sha256(str(self.run).encode()).hexdigest()[:24]
        self.attempt = 'reporter-' + uuid.uuid4().hex[:20]
        self.sequence = 0
        self.offset, self.anchor, self.source = 0, None, None
        self.observed, self.durable = 0, None
        self.latest, self.catalog = {}, {}
        self.training_seconds = 0.
        self.throughput = api.Throughput()
        self.checkpoints = {}
        self.native_status = 'initializing'
        self.terminal_fingerprint = None
        self.writer = None
        self.started = time.monotonic()

    def recover(self):
        if not (self.output / 'events.jsonl').is_file():
            return
        cursor = None
        while True:
            page = self.api.read_event_page(self.output, cursor=cursor, limit=1000)
            for row in page['events']:
                if row.get('source_request_sha256') != self.request_hash:
                    raise ValueError('reporting history belongs to changed training request bytes')
                if row.get('source_metrics'):
                    source = row['source_metrics']
                    self.source = Path(source['path'])
                    self.offset, self.anchor = source['offset'], source['anchor']
                    self.observed = row['step']
                    self.latest.update(row.get('metrics', {}))
                    duration = row.get('native_step_seconds')
                    if duration is not None:
                        self.training_seconds += duration
                        self.throughput.observe(duration)
                if row.get('native_checkpoint'):
                    item = row['native_checkpoint']
                    self.checkpoints[item['path']] = item
                    if self.durable is None or item['step'] > self.durable['step']:
                        self.durable = item
                if row.get('native_result'):
                    self.native_status = row['native_result']['status']
                    self.terminal_fingerprint = self.api.digest(terminal_identity(row['native_result']))
                revision = row.get('catalog')
                if revision:
                    self.catalog.update(self.api.read_catalog(self.output, revision)['metrics'])
            if not page['has_more']:
                if page.get('partial_tail'):
                    raise ValueError('reporting journal has an interrupted tail; preserve it and select a reviewed recovery')
                return
            cursor = page['cursor']

    def catalog_revision(self, metrics):
        for name in metrics:
            if name not in self.catalog:
                definition = dict(kind='scalar', source=name, label=name, unit='native',
                    scope='observed_native_update', direction='none',
                    numerical_sha256=self.request['resolved'].get('semantic_sha256'),
                    description='Already measured native scalar; reported without recomputing training math.')
                self.catalog[name] = dict(definition, definition_hash=self.api.digest(definition))
        catalog = dict(schema_version=1, preset_version='external-native/v1', metrics=self.catalog)
        revision = self.api.digest(catalog)
        self.api.atomic_json(self.output / 'metrics' / ('catalog-' + revision + '.json'), catalog)
        return revision

    def emit(self, event, *, step=None, metrics=None, **values):
        revision = self.catalog_revision(metrics or {})
        self.sequence += 1
        row = dict(schema_version=2, run_id=self.run_id, stream_id='training',
            stream_generation=self.run_id, attempt_id=self.attempt, sequence=self.sequence,
            step=self.observed if step is None else step, event=event, catalog=revision,
            source_request_sha256=self.request_hash, seconds=time.monotonic()-self.started, **values)
        if metrics is not None:
            row['metrics'] = metrics
        self.writer.append(row, wait=True)

    def manifest(self):
        return dict(schema_version=1, run_id=self.run_id, attempt_id=self.attempt,
            name=self.run.name, status=self.native_status,
            steps=max(self.observed, self.durable['step'] if self.durable else 0),
            observed_step=self.observed, total_steps=self.target,
            last_durable_step=self.durable['step'] if self.durable else None,
            checkpoint_path=self.durable['path'] if self.durable else None,
            config={'training': {'steps': self.target}, 'external_native': self.request['training']},
            external_training=dict(run_dir=str(self.run), request_sha256=self.request_hash,
                control_supported=False, native_checkpoints_owned_by_reporter=False),
            source=self.api.identity, metrics_catalog=self.catalog_revision({}),
            measured_update_seconds=self.training_seconds,
            **({'steps_per_second': self.latest['throughput/steps_per_second']}
               if 'throughput/steps_per_second' in self.latest else {}))

    def publish(self, summary_metrics=None):
        manifest = self.manifest()
        self.writer.publish(manifest, wait=True)
        summary_path = self.output / 'summary.json'
        if summary_metrics is not None or not summary_path.exists():
            self.api.write_summary_json(self.output, manifest=manifest, metrics=summary_metrics or {})
        # HG summary computes a telemetry EMA per call. A status-only poll must
        # not count the same observation again. Retain its canonical metric
        # summary and add read-only references to the native checkpoint owner.
        summary = read_json(summary_path)
        summary.update(status=manifest['status'], step=manifest['steps'], total_steps=self.target,
            observed_step=self.observed, last_durable_step=manifest['last_durable_step'],
            updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            checkpoints=[dict(name=Path(item['path']).name, path=item['path'], step=item['step'],
                sha256=item['sha256'], metrics={}, external=True, format='native-full-state')
                for item in sorted(self.checkpoints.values(), key=lambda value: value['step'])])
        self.api.atomic_json(summary_path, summary)

    def state(self, status, **values):
        self.api.atomic_json(self.output / 'reporter-state.json', dict(status=status,
            run_id=self.run_id, observed_step=self.observed,
            last_durable_step=self.durable['step'] if self.durable else None,
            training_status=self.native_status, source=self.api.identity, **values))

    def scan(self):
        if file_hash(self.run / 'request.json') != self.request_hash:
            raise ValueError('training request changed while reporting')
        path = self.source or metric_path(self.run)
        if path is not None and path.exists():
            for raw, cursor in complete_lines(path, self.offset, self.anchor):
                step, values, duration = metric_record(raw)
                if duration is not None:
                    rate = self.throughput.observe(duration)
                    self.training_seconds += duration
                    values['timing/step_seconds'] = duration
                    values['timing/measured_update_seconds'] = self.training_seconds
                    if rate is not None:
                        values['throughput/steps_per_second'] = rate
                self.observed = step
                self.latest.update(values)
                self.emit('train', metrics=values, source_metrics=cursor, native_step_seconds=duration)
                self.source, self.offset, self.anchor = path, cursor['offset'], cursor['anchor']
                self.publish(summary_metrics=values)
        paths = set()
        artifacts = self.run / 'artifacts'
        if artifacts.exists():
            for marker in ('checkpoint.json', 'manifest.json'):
                paths.update(p.parent for p in artifacts.glob('**/checkpoints/step-*/' + marker))
        result_path = self.run / 'result.json'
        result = read_json(result_path) if result_path.is_file() else None
        if result and result.get('checkpoint'):
            paths.add(Path(result['checkpoint']))
        for path in sorted(paths):
            if str(path.resolve()) in self.checkpoints:
                continue
            record = checkpoint_record(path)
            if record is None:
                continue
            self.checkpoints[record['path']] = record
            if self.durable is None or record['step'] > self.durable['step']:
                self.durable = record
            self.emit('checkpoint', step=record['step'], native_checkpoint=record)
        if result:
            status = result.get('status')
            if status in ('completed', 'failed', 'cancelled', 'checkpointed'):
                if status == 'completed' and (result.get('completed_steps') != self.target or
                        self.durable is None or self.durable['step'] != self.target):
                    raise ValueError('controller completion lacks the requested verified durable checkpoint')
                self.native_status = status
                fingerprint = self.api.digest(terminal_identity(result))
                if fingerprint != self.terminal_fingerprint:
                    self.emit('complete' if status == 'completed' else status, native_result=result)
                    self.terminal_fingerprint = fingerprint
                self.publish()
                return True
        progress_path = self.run / 'progress.json'
        if progress_path.is_file():
            current = read_json(progress_path)
            # A scalar update is stronger evidence than a stale "starting" row.
            self.native_status = 'running' if self.observed else current.get('status', 'initializing')
        elif self.observed:
            self.native_status = 'running'
        self.publish()
        return False

    def follow(self, *, follow=False, interval=.5, stopped=lambda: False):
        self.output.mkdir(parents=True, exist_ok=True)
        with self.api.run_lock(self.output):
            self.recover()
            attempt_dir = self.output / 'attempts' / self.attempt
            attempt_dir.mkdir(parents=True)
            self.writer = self.api.ObservationIO(self.output, attempt_dir)
            try:
                self.emit('reporter_attached', source_run=str(self.run))
                self.publish()
                self.state('ready')
                while True:
                    terminal = self.scan()
                    if terminal or not follow or stopped():
                        self.writer.commit_boundary()
                        self.state('complete', terminal_observed=terminal)
                        return
                    self.state('ready')
                    time.sleep(interval)
            except BaseException as error:
                self.state('failed', error=str(error))
                raise
            finally:
                try:
                    self.writer.close()
                except BaseException as error:
                    self.state('failed', error=str(error))
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--runtime-path', type=Path, required=True)
    parser.add_argument('--runtime-commit', required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args(argv)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''):
        raise RuntimeError('reporter requires CUDA_VISIBLE_DEVICES empty')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    api = load_runtime(args.runtime_path, args.runtime_commit, args.runtime_sha256)
    if args.check:
        print(json.dumps(dict(status='checked', source=api.identity, torch_imported=False)))
        return 0
    stop = [False]
    previous = {number: signal.signal(number, lambda *_: stop.__setitem__(0, True))
                for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        Reporter(args.run, api).follow(follow=args.follow, stopped=lambda: stop[0])
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
