"""One-executor, resumable search. Evidence is published before any nomination."""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

from .backend import Capabilities, Parameter, resolve
from .metric import METRIC, algebraic_controls, canonical, digest, file_hash, mean, score, write_raw
from .space import Space
from ..weight_policy import require_no_weight_ema, require_raw_selection

SCHEMA = 'particle-sliders-raw-parameter-search-v2'


class ResourceLimit(RuntimeError):
    pass


def atomic_json(path, value):
    atomic_file(path, lambda temp: Path(temp).write_bytes(canonical(value) + b'\n'))


def atomic_file(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity():
    return dict(python=sys.version, sources={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})


def ambient_rng():
    h = hashlib.sha256(canonical(random.getstate()))
    torch = sys.modules.get('torch')
    if torch is not None:
        h.update(torch.get_rng_state().numpy().tobytes())
        if torch.cuda.is_initialized():
            for value in torch.cuda.get_rng_state_all():
                h.update(value.cpu().numpy().tobytes())
    numpy = sys.modules.get('numpy')
    if numpy is not None:
        state = numpy.random.get_state()
        h.update(canonical([state[0], state[1].tolist(), *state[2:]]))
    return h.hexdigest()


def config_snapshot(config):
    if not isinstance(config, dict):
        raise ValueError('search config must be an object')
    value = json.loads(canonical(config))
    value.setdefault('min_edit_energy', 0.)
    return value


def validated(config, *, allow_native=False):
    config = config_snapshot(config)
    if set(config) != {'schema', 'backend', 'backend_options', 'fixed', 'space', 'search', 'resources', 'selection', 'min_edit_energy'} or config['schema'] != SCHEMA:
        raise ValueError('unsupported/incomplete search config fields')
    # Refuse weight averaging before resolving/importing any product backend.
    require_raw_selection(config['selection'])
    if 'ema_decay' not in config['backend_options']:
        raise ValueError('declare ema_decay=None explicitly')
    require_no_weight_ema(config['backend_options']['ema_decay'])
    backend = resolve(config['backend'])
    cap = backend.describe(json.loads(canonical(config['backend_options'])))
    if cap.name != config['backend'] or config['selection'] not in cap.selections:
        raise ValueError('backend/raw selection capability differs')
    device = cap.identity['device']
    if device != 'cpu' and not allow_native:
        raise ValueError('native resource allocation requires explicit --allow-native and backend device/admission')
    if not isinstance(device, str) or device != 'cpu' and ':' not in device:
        raise ValueError('declare an explicit device; never default to a native accelerator')
    resources = config['resources']
    if set(resources) != {'max_updates', 'max_trials', 'max_artifact_bytes', 'checkpoint_every'}:
        raise ValueError('all finite resource limits must be explicit')
    if any(type(v) is not int or v <= 0 for v in resources.values()):
        raise ValueError('resource limits must be positive integers')
    search = config['search']
    if set(search) != {'seed', 'initial_candidates', 'refinement_candidates', 'budgets', 'keep_per_cohort', 'training_seeds', 'confirmation_seeds'}:
        raise ValueError('explicit bounded search and independent seed sets required')
    budgets = search['budgets']
    if (not isinstance(budgets, list) or not budgets or any(type(v) is not int or v < 1 for v in budgets)
            or budgets != sorted(set(budgets)) or budgets[-1] > cap.identity['schedule_horizon']):
        raise ValueError('strictly increasing actual-step budgets within frozen schedule horizon required')
    for key in ('seed', 'initial_candidates', 'refinement_candidates', 'keep_per_cohort'):
        if type(search[key]) is not int or search[key] < (0 if key in ('seed', 'refinement_candidates') else 1):
            raise ValueError('invalid bounded search count/seed')
    train, confirm = search['training_seeds'], search['confirmation_seeds']
    if (not isinstance(train, list) or not train or not isinstance(confirm, list) or not confirm
            or any(type(v) is not int or v < 0 for v in train + confirm)
            or len(set(train + confirm)) != len(train + confirm)):
        raise ValueError('nonempty disjoint unique training and independent confirmation seeds required')
    if type(config['min_edit_energy']) not in (int, float) or not math.isfinite(config['min_edit_energy']) or config['min_edit_energy'] != 0:
        raise ValueError('min_edit_energy must be zero; no denominator epsilon or threshold')
    space = Space(cap, config['fixed'], config['space'])
    candidates = space.explore(search['initial_candidates'], search['seed'])
    # An explicit count may exceed the finite discrete product; that is harmless.
    if len(candidates) * len(train) > resources['max_trials']:
        raise ValueError('initial exploration exceeds explicit trial limit')
    return backend, cap, space, candidates


def plan(config, *, allow_native=False):
    # describe/open are separate: inspecting a native plan is allocation-free.
    _, cap, space, candidates = validated(config, allow_native=True)
    return dict(schema=SCHEMA, metric=METRIC, capabilities=cap.as_dict(),
                candidates=[dict(id=digest(p)[:16], cohort=space.cohort(cap.identity, p), parameters=p) for p in candidates],
                resources=config['resources'], budgets=config['search']['budgets'],
                qualification='fixture only' if cap.evidence == 'fixture' else 'native adapter qualification is limited to its declared evidence; media/OFF are diagnostics')


def ordered_rows(values, space):
    """The exact RAW metric is the sole rank; simplicity breaks exact ties only."""
    return sorted(values, key=lambda row: (row['error'], space.simplicity(row['parameters']), row['candidate']))


class Run:
    def __init__(self, config, directory, *, allow_native=False):
        self.config, self.root = config_snapshot(config), Path(directory)
        config = self.config
        self.backend, self.cap, self.space, self.initial = validated(config, allow_native=allow_native)
        self.root.mkdir(parents=True, exist_ok=True)
        self.expected = dict(schema=SCHEMA, config=config, capabilities=self.cap.as_dict(),
                             implementation=source_identity(), metric=METRIC)
        self.expected = json.loads(canonical(self.expected))
        self.identity = digest(self.expected)
        self.progress = None

    @contextmanager
    def lock(self):
        with open(self.root / '.executor.lock', 'a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('another executor owns this run; do not launch a second worker') from exc
            manifest = self.root / 'manifest.json'
            if manifest.exists():
                if read_json(manifest) != self.expected:
                    raise ValueError('resume source/config/frozen cohort differs; create a new run')
            else:
                if any(p.name != '.executor.lock' for p in self.root.iterdir()):
                    raise ValueError('new search directory must be empty')
                atomic_json(manifest, self.expected)
            progress = self.root / 'progress.json'
            self.progress = read_json(progress) if progress.exists() else dict(
                attempted_updates=0, completed_update_calls=0, elapsed_seconds=0., trials={}, initials={})
            # After an ungraceful stop the unaccounted phase (including unknown
            # downtime) is conservatively charged; SIGKILL cannot reset a cap.
            previous = self.progress.get('phase_started_utc')
            if previous is not None:
                self.progress['elapsed_seconds'] += max(0., time.time() - previous)
            self.accounted = time.monotonic()
            self.persist()
            try:
                yield self
            finally:
                self.persist()
                self.progress['phase_started_utc'] = None
                atomic_json(progress, self.progress)

    def persist(self):
        now = time.monotonic()
        self.progress['elapsed_seconds'] += now - self.accounted
        self.accounted = now
        self.progress['phase_started_utc'] = time.time()
        atomic_json(self.root / 'progress.json', self.progress)

    def resources(self, *, next_update=False):
        limits = self.config['resources']
        if next_update and self.progress['attempted_updates'] >= limits['max_updates']:
            raise ResourceLimit('physical update-call limit reached (includes replay/recovery)')
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()) > limits['max_artifact_bytes']:
            raise ResourceLimit('artifact byte limit reached; retained evidence is preserved')

    def verify(self, active, parameters):
        if canonical(active.effective_parameters()) != canonical(parameters):
            raise ValueError('actual effective optimizer/regularizer/particle configuration differs')
        groups = active.optimizer_groups()
        expected_roles = {p.role for p in self.cap.parameters if p.role.endswith('_lr')}
        if (not isinstance(groups, list) or {g.get('role') for g in groups} != expected_roles
                or any(set(g) != {'role', 'base_lr', 'lr', 'parameter_count'}
                    or type(g['parameter_count']) is not int or g['parameter_count'] < 1
                    or any(type(g[k]) not in (int, float) or not math.isfinite(g[k]) or g[k] < 0
                           for k in ('base_lr', 'lr')) for g in groups)):
            raise ValueError('actual participating optimizer group rates/ownership are incomplete')
        absolute_rates = {p.role: parameters[p.name] for p in self.cap.parameters if p.role.endswith('_lr')}
        if any(group['base_lr'] != absolute_rates[group['role']] for group in groups):
            raise ValueError('actual optimizer base rate differs from its advertised absolute candidate rate')
        if canonical(active.frozen_identity()) != canonical(self.expected['capabilities']['identity']):
            raise ValueError('actual teacher/model/precision/schedule/data/coordinates differ')

    def update(self, active, parameters):
        self.resources(next_update=True)
        self.verify(active, parameters)
        previous = active.completed_steps
        # Persist BEFORE invocation: an interrupted update is never undercharged.
        self.progress['attempted_updates'] += 1
        self.persist()
        active.step()
        if active.completed_steps != previous + 1:
            raise ValueError('backend did not complete exactly one actual update')
        self.progress['completed_update_calls'] += 1
        self.verify(active, parameters)
        self.persist()

    def checkpoint(self, active, row, directory):
        step = active.completed_steps
        key = str(step)
        if key in row['checkpoints']:
            receipt = row['checkpoints'][key]
            if (file_hash(self.root / receipt['path']) != receipt['sha256']
                    or active.state_digest() != receipt['state_sha256']):
                raise ValueError('retained full checkpoint/state differs')
            return receipt
        path = directory / f'step-{step:08d}.full'
        atomic_file(path, active.checkpoint)
        receipt = dict(path=str(path.relative_to(self.root)), sha256=file_hash(path),
                       state_sha256=active.state_digest(), completed_steps=step,
                       effective_parameters=active.effective_parameters(), optimizer_groups=active.optimizer_groups())
        row['checkpoints'][key] = receipt
        row['latest'] = key
        self.persist()
        return receipt

    def evaluate(self, active, row, directory, split, selection, *, off=False):
        self.resources()
        self.persist()
        receipt = self.checkpoint(active, row, directory)
        name = f'{active.completed_steps:08d}-{split}-{selection}' + ('-trained-off' if off else '')
        target = directory / name
        binding = dict(run=self.identity, trial=row['id'], candidate=row['candidate'], cohort=row['cohort'],
            seed=row['seed'], completed_steps=active.completed_steps, checkpoint_sha256=receipt['sha256'],
            frozen_identity=digest(self.cap.identity), split=split, selection=selection,
            grid_sha256=digest(self.cap.grids[split]), off=off)
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix='.' + name, dir=directory))
            try:
                before, rng = active.state_digest(), ambient_rng()
                write_raw(temporary / 'responses.jsonl', binding, active.responses(split, selection, off=off))
                if active.state_digest() != before or ambient_rng() != rng:
                    raise ValueError('evaluation advanced training/native/RNG/modes state')
                summary = score(temporary / 'responses.jsonl', self.cap.grids[split], binding,
                    reference_sha256=self.cap.references[split], min_edit_energy=self.config['min_edit_energy'])
                summary['binding'] = binding
                atomic_json(temporary / 'score.json', summary)
                artifacts = {p.name: file_hash(p) for p in temporary.iterdir() if p.is_file()}
                atomic_json(temporary / 'receipt.json', dict(binding=binding, artifacts=artifacts))
                for path in temporary.iterdir():
                    with open(path, 'rb') as stream:
                        os.fsync(stream.fileno())
                os.replace(temporary, target)
            except BaseException as exc:
                # Failed native/raw evidence is useful for diagnosis and must
                # survive interruption; it has no completed evaluation receipt.
                atomic_json(temporary / 'failure.json', dict(error=type(exc).__name__, reason=str(exc), binding=binding))
                failed = temporary.with_name(temporary.name + '.failed')
                os.replace(temporary, failed)
                row.setdefault('failed_artifacts', []).append(str(failed.relative_to(self.root)))
                self.persist()
                raise
        summary = audit_evaluation(self.root, str(target.relative_to(self.root)), self.expected,
                                   expected_binding=binding, checkpoint=receipt)
        row['evaluations'][name] = str(target.relative_to(self.root))
        self.resources()
        self.persist()
        return summary

    def open_trial(self, parameters, seed, cohort, checkpoint=None):
        self.resources()
        self.persist()
        return self.backend.open(json.loads(canonical(self.config['backend_options'])),
            json.loads(canonical(parameters)), seed, cohort, checkpoint=checkpoint)

    def trial(self, parameters, seed, budget, *, confirmation=False):
        candidate = digest(parameters)[:16]
        cohort = self.space.cohort(self.cap.identity, parameters)
        trial_id = f'{candidate}-seed-{seed}'
        directory = self.root / 'trials' / trial_id
        directory.mkdir(parents=True, exist_ok=True)
        row = self.progress['trials'].get(trial_id)
        if row is None:
            if len(self.progress['trials']) >= self.config['resources']['max_trials']:
                raise ResourceLimit('trial count limit reached')
            row = dict(id=trial_id, candidate=candidate, cohort=cohort, parameters=parameters,
                       seed=seed, confirmation=confirmation, checkpoints={}, evaluations={}, latest=None, replay=False)
            self.progress['trials'][trial_id] = row
            self.persist()
        if row['parameters'] != parameters or row['confirmation'] != confirmation:
            raise ValueError('trial identity collision')
        checkpoint = None
        if row['latest'] is not None:
            cp = row['checkpoints'][row['latest']]
            checkpoint = self.root / cp['path']
            if file_hash(checkpoint) != cp['sha256']:
                raise ValueError('full checkpoint checksum differs before resume')
        active = self.open_trial(parameters, seed, cohort, checkpoint=checkpoint)
        try:
            self.verify(active, parameters)
            if checkpoint is not None and (active.completed_steps != int(row['latest']) or active.state_digest() != cp['state_sha256']):
                raise ValueError('restored state/counter differs from retained full checkpoint')
            initial_key = f'{cohort}-seed-{seed}'
            initial = self.progress['initials'].setdefault(initial_key, active.initial_state_sha256)
            if initial != active.initial_state_sha256:
                raise ValueError('initial weights/streams differ inside matched architecture cohort/seed')
            self.checkpoint(active, row, directory)
            if not row['replay']:
                # Qualify exact update+restore+replay for EVERY trial, counting
                # physical calls separately from completed training budget.
                if active.completed_steps == 0:
                    self.update(active, parameters)
                    self.checkpoint(active, row, directory)
                first = row['checkpoints']['1']
                active.close()
                active = self.open_trial(parameters, seed, cohort,
                                           checkpoint=self.root / row['checkpoints']['0']['path'])
                self.update(active, parameters)
                if active.state_digest() != first['state_sha256']:
                    raise ValueError('full checkpoint one-update exact replay failed')
                row['replay'] = True
                row['replay_proof'] = dict(initial_checkpoint_sha256=row['checkpoints']['0']['sha256'],
                    expected_state_sha256=first['state_sha256'], observed_state_sha256=active.state_digest(),
                    physical_update_call=self.progress['attempted_updates'])
                self.persist()
            if active.completed_steps > budget:
                # Later rungs have already run: restore the requested common
                # budget for cached evaluation, never compare different steps.
                active.close()
                active = self.open_trial(parameters, seed, cohort,
                    checkpoint=self.root / row['checkpoints'][str(budget)]['path'])
            while active.completed_steps < budget:
                self.update(active, parameters)
                if active.completed_steps % self.config['resources']['checkpoint_every'] == 0:
                    self.checkpoint(active, row, directory)
            self.checkpoint(active, row, directory)
            primary = self.evaluate(active, row, directory, 'dev', self.config['selection'])
            if confirmation:
                self.evaluate(active, row, directory, 'heldout', self.config['selection'])
            for selection in self.cap.selections:
                self.resources()
                path = directory / f'step-{budget:08d}-{selection}.export'
                previous_export = row.setdefault('exports', {}).get(path.name)
                if previous_export is not None:
                    if not path.exists() or file_hash(path) != previous_export['sha256']:
                        raise ValueError('retained export checksum differs; refusing to re-bless modified bytes')
                else:
                    # Unrecorded crash-orphans are retained, then regenerated
                    # from verified exact state rather than silently adopted.
                    if path.exists():
                        os.replace(path, path.with_name(path.name + '.orphan-' + str(time.time_ns())))
                    before, rng = active.state_digest(), ambient_rng()
                    atomic_file(path, lambda temp, s=selection: active.export(temp, s))
                    if active.state_digest() != before or ambient_rng() != rng:
                        raise ValueError('export advanced training/native/RNG state')
                    row['exports'][path.name] = dict(path=str(path.relative_to(self.root)), sha256=file_hash(path))
            self.resources()
            row['outcome'] = 'completed'
            row.pop('failure', None)
            self.persist()
            return primary
        except BaseException as exc:
            row['outcome'] = 'interrupted' if isinstance(exc, (KeyboardInterrupt, ResourceLimit)) else 'failed'
            row['failure'] = dict(error=type(exc).__name__, reason=str(exc))
            self.persist()
            raise
        finally:
            active.close()

    def rank(self, candidates, budget):
        search = self.config['search']
        groups = {}
        for parameters in candidates:
            values = [self.trial(parameters, seed, budget) for seed in search['training_seeds']]
            row = dict(candidate=digest(parameters)[:16], parameters=parameters,
                cohort=self.space.cohort(self.cap.identity, parameters), completed_steps=budget,
                error=mean(v['error'] for v in values),
                diagnostics=dict(aligned_better_than_off=all(v['diagnostics']['aligned_better_than_off'] for v in values)), seed_errors=[v['error'] for v in values])
            groups.setdefault(row['cohort'], []).append(row)
        return {cohort: ordered_rows(values, self.space) for cohort, values in sorted(groups.items())}

    def execute(self, review=None):
        try:
            review = json.loads(canonical(review))
        except (TypeError, ValueError) as exc:
            review = {'diagnostic_input_error': str(exc)}
        search = self.config['search']
        budget = search['budgets'][0]
        candidates = self.initial
        groups = self.rank(candidates, budget)
        if search['refinement_candidates']:
            nominees = [rows[0]['parameters'] for rows in groups.values()]
            candidates = candidates + self.space.refine(nominees, candidates, search['refinement_candidates'])
            groups = self.rank(candidates, budget)
        rungs = [dict(budget=budget, groups=groups)]
        for budget in search['budgets'][1:]:
            candidates = [row['parameters'] for rows in groups.values() for row in rows[:search['keep_per_cohort']]]
            groups = self.rank(candidates, budget)
            rungs.append(dict(budget=budget, groups=groups))
        nominations = []
        for cohort, rows in groups.items():
            finalist = dict(rows[0])
            params = finalist['parameters']
            confirmation = [self.trial(params, seed, budget, confirmation=True) for seed in search['confirmation_seeds']]
            heldout = []
            trial_ids = [f"{finalist['candidate']}-seed-{seed}" for seed in search['training_seeds'] + search['confirmation_seeds']]
            # Heldout is evaluated only after finalist choice, for independent
            # confirmation seeds. No heldout score feeds exploration/refinement.
            for seed in search['confirmation_seeds']:
                row = self.progress['trials'][f"{finalist['candidate']}-seed-{seed}"]
                key = f"{budget:08d}-heldout-{self.config['selection']}"
                heldout.append(read_json(self.root / row['evaluations'][key] / 'score.json'))
            boundary = self.space.boundaries(params)
            diagnostics = nomination_diagnostics(finalist, confirmation, heldout, boundary, self.cap)
            evidence = {t: evidence_binding(self.root, self.progress['trials'][t], budget) for t in trial_ids}
            diagnostics['matched_media_review'] = media_diagnostic(review, self.identity, cohort, evidence)
            if not all(self.progress['trials'][t]['replay'] for t in trial_ids):
                raise ValueError('missing exact replay for finalist')
            finalist.update(diagnostics=diagnostics, boundary_parameters=boundary, confirmation_errors=[v['error'] for v in confirmation],
                heldout_errors=[v['error'] for v in heldout], evidence=evidence, status='confirmed_raw_metric')
            nominations.append(finalist)
        # Architecture/ablation cohorts retain separate initialization and
        # qualification, but their finalists share this run's frozen response
        # protocol and final budget, so an overall provisional nomination helps
        # choose a complete configuration. No cross-model ranking is implied.
        overall = ordered_rows(nominations, self.space)
        report = dict(schema=SCHEMA, run=self.identity, metric=METRIC, selection=self.config['selection'],
                      rungs=rungs, nominations=nominations,
                      overall_nomination=dict(candidate=overall[0]['candidate'], cohort=overall[0]['cohort'],
                          status=overall[0]['status'],
                          basis='lowest exact RAW relative SSE among cohort finalists at the same frozen target/grid/metric/budget/seeds'),
                      resources=self.progress, media_review=review,
                      qualification='synthetic CPU demonstration only' if self.cap.evidence == 'fixture' else
                      'native evidence is limited to this frozen cohort, target, grid, seeds and budget')
        atomic_json(self.root / 'results.json', report)
        return report


def evidence_binding(root, row, budget):
    evaluations = {k: dict(path=v, receipt=read_json(Path(root) / v / 'receipt.json'))
                   for k, v in row['evaluations'].items() if k.startswith(f'{budget:08d}-')}
    return dict(checkpoint=row['checkpoints'][str(budget)], evaluations=evaluations,
                exports={k: v for k, v in row.get('exports', {}).items() if k.startswith(f'step-{budget:08d}-')})


def nomination_diagnostics(best, confirmation, heldout, boundary, cap):
    return dict(training_aligned_better_than_off=best['diagnostics']['aligned_better_than_off'],
        confirmation_aligned_better_than_off=all(v['diagnostics']['aligned_better_than_off'] for v in confirmation),
        heldout_aligned_better_than_off=all(v['diagnostics']['aligned_better_than_off'] for v in heldout),
        boundary_parameters=boundary, trained_off_available=cap.trained_off, trained_off_measured=False,
        native_evidence=cap.evidence == 'native')


def media_diagnostic(review, run, cohort, evidence):
    try:
        return dict(approved=media_review(review, run, cohort, evidence), error=None)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Optional malformed/unavailable media cannot veto the scalar metric.
        return dict(approved=False, error=str(exc))


def media_review(review, run, cohort, evidence):
    if review is None:
        return False
    if (not isinstance(review, dict) or set(review) != {'run', 'cohorts'}
            or review['run'] != run or not isinstance(review['cohorts'], dict)):
        raise ValueError('media review belongs to another run')
    row = review['cohorts'].get(cohort)
    if row is None:
        return False
    if (set(row) != {'reviewer', 'evidence_sha256', 'matched_controls', 'media_artifacts', 'approved', 'notes'}
            or not isinstance(row['reviewer'], str) or not row['reviewer']
            or row['evidence_sha256'] != digest(evidence)
            or row['matched_controls'] != ['off', 'teacher', 'strength_ladder']
            or not isinstance(row['media_artifacts'], list) or not row['media_artifacts']
            or type(row['approved']) is not bool):
        raise ValueError('media review must bind finalist checkpoints/evaluations and matched outputs')
    # The external human/native review is an explicit attestation, not a scalar
    # score masquerading as perception. Every attached local artifact is hashed.
    for artifact in row['media_artifacts']:
        if set(artifact) != {'path', 'sha256'} or file_hash(artifact['path']) != artifact['sha256']:
            raise ValueError('matched media artifact checksum differs')
    return row['approved']


def audit_evaluation(root, relative, manifest, *, expected_binding=None, checkpoint=None):
    directory = Path(root) / relative
    receipt = read_json(directory / 'receipt.json')
    if set(receipt) != {'binding', 'artifacts'}:
        raise ValueError('invalid metric artifact receipt')
    binding = receipt['binding']
    if expected_binding is not None and binding != expected_binding:
        raise ValueError('evaluation cache binding differs')
    if set(receipt['artifacts']) != {'responses.jsonl', 'responses.bin', 'score.json'}:
        raise ValueError('incomplete raw binary/metadata/metric artifact receipt')
    for name, expected in receipt['artifacts'].items():
        if file_hash(directory / name) != expected:
            raise ValueError('raw/metric artifact checksum differs: ' + name)
    cap, config = manifest['capabilities'], manifest['config']
    if (binding['run'] != digest(manifest) or binding['frozen_identity'] != digest(cap['identity'])
            or binding['grid_sha256'] != digest(cap['grids'][binding['split']])
            or binding['selection'] not in cap['selections']):
        raise ValueError('raw cohort/grid/selection differs from manifest')
    if checkpoint is not None and (binding['checkpoint_sha256'] != checkpoint['sha256']
            or binding['completed_steps'] != checkpoint['completed_steps']
            or file_hash(Path(root) / checkpoint['path']) != checkpoint['sha256']):
        raise ValueError('raw checkpoint/actual completed-step binding differs')
    result = score(directory / 'responses.jsonl', cap['grids'][binding['split']], binding,
                   reference_sha256=cap['references'][binding['split']], min_edit_energy=config['min_edit_energy'])
    result['binding'] = binding
    if result != read_json(directory / 'score.json'):
        raise ValueError('stored metric differs from independent raw recomputation')
    return result


def audit(directory):
    root = Path(directory)
    manifest, progress = read_json(root / 'manifest.json'), read_json(root / 'progress.json')
    if manifest['schema'] != SCHEMA or manifest['metric'] != METRIC:
        raise ValueError('historical search schema/cohort is incompatible; retain its original implementation')
    require_raw_selection(manifest['config']['selection'])
    require_no_weight_ema(manifest['config']['backend_options']['ema_decay'])
    if tuple(manifest['capabilities']['selections']) != ('raw',):
        raise ValueError('only raw search capabilities can be audited')
    external_media_discrepancies = []
    count = 0
    for row in progress['trials'].values():
        for checkpoint in row['checkpoints'].values():
            if file_hash(root / checkpoint['path']) != checkpoint['sha256']:
                raise ValueError('retained full checkpoint checksum differs')
        if row['replay']:
            proof = row.get('replay_proof', {})
            if (set(proof) != {'initial_checkpoint_sha256', 'expected_state_sha256', 'observed_state_sha256', 'physical_update_call'}
                    or proof['initial_checkpoint_sha256'] != row['checkpoints']['0']['sha256']
                    or proof['expected_state_sha256'] != row['checkpoints']['1']['state_sha256']
                    or proof['observed_state_sha256'] != proof['expected_state_sha256']
                    or type(proof['physical_update_call']) is not int
                    or not 1 <= proof['physical_update_call'] <= progress['attempted_updates']):
                raise ValueError('exact update/restore/replay proof differs')
        for receipt in row.get('exports', {}).values():
            if file_hash(root / receipt['path']) != receipt['sha256']:
                raise ValueError('retained raw export checksum differs')
        for relative in row['evaluations'].values():
            binding = read_json(root / relative / 'receipt.json')['binding']
            cp = row['checkpoints'][str(binding['completed_steps'])]
            if (binding['trial'] != row['id'] or binding['candidate'] != row['candidate']
                    or binding['cohort'] != row['cohort'] or binding['seed'] != row['seed']):
                raise ValueError('evaluation belongs to a different trial/cohort/seed')
            audit_evaluation(root, relative, manifest, checkpoint=cp)
            count += 1
    for path in (root / 'qualification').glob('*.json'):
        qualification = read_json(path)
        split = path.stem
        actual = algebraic_controls(root / qualification['source_evaluation'] / 'responses.jsonl',
            manifest['capabilities']['grids'][split], reference_sha256=manifest['capabilities']['references'][split],
            min_edit_energy=manifest['config']['min_edit_energy'])
        actual['source_evaluation'] = qualification['source_evaluation']
        if actual != qualification:
            raise ValueError('algebraic qualification differs from raw target/neutral responses')
    if (root / 'results.json').exists():
        results = read_json(root / 'results.json')
        if results['run'] != digest(manifest) or results['metric'] != METRIC or results['selection'] != manifest['config']['selection']:
            raise ValueError('results metric/run/selection differs')
        config = manifest['config']
        descriptor = dict(manifest['capabilities'])
        descriptor['parameters'] = tuple(Parameter(**p) for p in descriptor['parameters'])
        cap = Capabilities(**descriptor)
        space = Space(cap, config['fixed'], config['space'])
        search = config['search']
        def measured(candidate, seed, budget, split):
            trial_id = f"{digest(candidate)[:16]}-seed-{seed}"
            row = progress['trials'][trial_id]
            if (canonical(row['parameters']) != canonical(candidate) or row['id'] != trial_id
                    or row['candidate'] != digest(candidate)[:16] or row['seed'] != seed
                    or row['cohort'] != space.cohort(cap.identity, candidate)
                    or row['confirmation'] != (seed in search['confirmation_seeds'])):
                raise ValueError('result candidate parameters/cohort/seed differ')
            key = f"{budget:08d}-{split}-{config['selection']}"
            return read_json(root / row['evaluations'][key] / 'score.json')
        def ranked(candidates, budget):
            groups = {}
            for parameters in candidates:
                values = [measured(parameters, seed, budget, 'dev') for seed in search['training_seeds']]
                row = dict(candidate=digest(parameters)[:16], parameters=parameters,
                    cohort=space.cohort(cap.identity, parameters), completed_steps=budget,
                    error=mean(v['error'] for v in values),
                    diagnostics=dict(aligned_better_than_off=all(v['diagnostics']['aligned_better_than_off'] for v in values)), seed_errors=[v['error'] for v in values])
                groups.setdefault(row['cohort'], []).append(row)
            return {cohort: ordered_rows(values, space) for cohort, values in sorted(groups.items())}
        candidates = space.explore(search['initial_candidates'], search['seed'])
        budget = search['budgets'][0]
        groups = ranked(candidates, budget)
        if search['refinement_candidates']:
            nominees = [rows[0]['parameters'] for rows in groups.values()]
            candidates += space.refine(nominees, candidates, search['refinement_candidates'])
            groups = ranked(candidates, budget)
        rungs = [dict(budget=budget, groups=groups)]
        for budget in search['budgets'][1:]:
            candidates = [r['parameters'] for rows in groups.values() for r in rows[:search['keep_per_cohort']]]
            groups = ranked(candidates, budget)
            rungs.append(dict(budget=budget, groups=groups))
        if results['rungs'] != rungs:
            raise ValueError('result proposals/ranking/metrics/completed budgets differ from raw evidence')
        if {r['cohort'] for r in results['nominations']} != set(groups) or len(results['nominations']) != len(groups):
            raise ValueError('incomplete/duplicate cohort nominations')
        for nominee in results['nominations']:
            best = groups[nominee['cohort']][0]
            if any(nominee[k] != best[k] for k in best if k != 'diagnostics'):
                raise ValueError('nominee differs from ranked finalist')
            parameters = nominee['parameters']
            trial_ids = [f"{nominee['candidate']}-seed-{seed}" for seed in
                         search['training_seeds'] + search['confirmation_seeds']]
            evidence = {t: evidence_binding(root, progress['trials'][t], budget) for t in trial_ids}
            if nominee['evidence'] != evidence:
                raise ValueError('nomination raw/checkpoint/export evidence differs')
            confirmation = [measured(parameters, seed, budget, 'dev') for seed in search['confirmation_seeds']]
            heldout = [measured(parameters, seed, budget, 'heldout') for seed in search['confirmation_seeds']]
            if (nominee['confirmation_errors'] != [v['error'] for v in confirmation]
                    or nominee['heldout_errors'] != [v['error'] for v in heldout]):
                raise ValueError('confirmation/heldout metrics differ from raw evidence')
            for trial_id in trial_ids:
                row = progress['trials'][trial_id]
                required = [f"{budget:08d}-dev-{config['selection']}"]
                if row['confirmation']:
                    required.append(f"{budget:08d}-heldout-{config['selection']}")
                if not all(k in row['evaluations'] for k in required):
                    raise ValueError('missing finalist complete-grid evidence')
            boundary = space.boundaries(parameters)
            diagnostics = nomination_diagnostics(best, confirmation, heldout, boundary, cap)
            observed_media = media_diagnostic(results.get('media_review'), digest(manifest), nominee['cohort'], evidence)
            recorded_media = nominee['diagnostics'].get('matched_media_review')
            if observed_media != recorded_media:
                external_media_discrepancies.append(dict(candidate=nominee['candidate'], cohort=nominee['cohort'],
                    recorded=recorded_media, observed=observed_media))
            # External media is mutable diagnostic material, not RAW evidence.
            # Keep the original observation without making its later drift a veto.
            diagnostics['matched_media_review'] = recorded_media
            if not all(progress['trials'][t]['replay'] for t in trial_ids):
                raise ValueError('missing exact replay for finalist')
            if nominee['diagnostics'] != diagnostics or nominee['boundary_parameters'] != boundary or nominee['status'] != 'confirmed_raw_metric':
                raise ValueError('nomination qualification/boundary/status differs from raw evidence')
        overall = ordered_rows(results['nominations'], space)[0]
        expected_overall = dict(candidate=overall['candidate'], cohort=overall['cohort'], status=overall['status'],
            basis='lowest exact RAW relative SSE among cohort finalists at the same frozen target/grid/metric/budget/seeds')
        if results['overall_nomination'] != expected_overall:
            raise ValueError('overall nomination differs from ranked cohort finalists')
    return dict(run=digest(manifest), verified_evaluations=count, verified_trials=len(progress['trials']),
                diagnostics=dict(external_media_discrepancies=external_media_discrepancies),
                status='raw metrics, full checkpoints and exports verified; media/OFF/boundary are diagnostics only')
