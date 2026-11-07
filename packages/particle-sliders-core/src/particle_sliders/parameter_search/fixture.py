"""Two deterministic, genuinely trained CPU fixtures; NEVER native quality evidence.

A vector adapter and a coupled image/audio adapter have different parameter and
response geometries. The tiny quadratic minimax game has participating generator,
critic, and particle optimizers plus an active L2 regularizer. It is not PG math.
"""
import hashlib
import json
import math
from pathlib import Path
import random
import struct

from .backend import Capabilities, Parameter
from ..weight_policy import require_no_weight_ema, require_raw_selection
from .metric import canonical, digest


def _grid(name, split):
    shapes = [('hidden', [1, 4])] if name == 'fixture-vector' else [('image', [1, 1, 2, 2]), ('audio', [1, 3])]
    return [dict(example=dict(scene=scene, seed=seed, path=name + '/' + split),
        positions=[dict(position=dict(timestep=t, surface=surface), shape=shape)
                   for t in (0, 1, 2) for surface, shape in shapes])
        for scene, seed in zip(('room', 'outdoors'), {'train': (1, 2), 'dev': (11, 12), 'heldout': (21, 22)}[split])]


def _chunks(grid, gains, *, off=False):
    for example in grid:
        for position in example['positions']:
            label, shape = position['position'], position['shape']
            surface = label['surface']
            arm = 1 if surface == 'audio' else 0
            neutral = [(example['example']['seed'] % 7 + i) * .1 for i in range(math.prod(shape))]
            # A zero-edit position is valid: other positions in this example
            # still have energy. A weak late edit must not dominate the score.
            amplitude = (1., .01, 0.)[label['timestep']]
            delta = [amplitude * ((-1.) ** i) * (i + 1) * (1.3 if arm else 1.) for i in range(len(neutral))]
            target = [n + d for n, d in zip(neutral, delta)]
            student = neutral if off else [n + gains[arm] * d for n, d in zip(neutral, delta)]
            yield dict(example=example['example'], **position, offset=0,
                       neutral=neutral, target=target, student=student)


def _reference(grid):
    h = hashlib.sha256()
    for chunk in _chunks(grid, [0., 0.]):
        h.update(canonical({k: chunk[k] for k in ('example', 'position', 'shape')}))
        for t, n in zip(chunk['target'], chunk['neutral']):
            h.update(struct.pack('!dd', t, n))
    return h.hexdigest()


class FixtureBackend:
    def __init__(self, name):
        self.name = name

    def describe(self, options):
        if set(options) != {'schedule_horizon', 'precision', 'device', 'ema_decay'}:
            raise ValueError('fixture options require schedule_horizon, precision, device, ema_decay')
        if options['device'] != 'cpu' or options['precision'] != 'float64':
            raise ValueError('fixture is CPU float64 only')
        require_no_weight_ema(options['ema_decay'])
        parameters = (
            Parameter('lr.generator', 'real', 'generator_lr', 'fixture generator gradient step', .0001, 1.),
            Parameter('lr.critic', 'real', 'critic_lr', 'fixture critic ascent step', .0001, 1.),
            Parameter('lr.particle', 'real', 'particle_lr', 'participating particle mean gradient step', .0001, 1.),
            Parameter('regularizer.l2_alpha', 'real', 'regularizer', 'L2 penalty on generator and particle coordinates',
                      0., 1., allow_zero=True, cohort='objective'),
            Parameter('particles.population', 'integer', 'particle_setting', 'number of learned particles per response arm',
                      2, 8, cohort='architecture'),
            Parameter('particles.temperature', 'categorical', 'particle_setting', 'divisor of each particle contribution',
                      choices=(1., 2.)),
        )
        grids = {s: _grid(self.name, s) for s in ('train', 'dev', 'heldout')}
        return Capabilities(self.name, identity=dict(model=self.name + '-raw-v2', teacher='analytic-aligned-reference-v1',
            target='neutral + fixed scene edit; strength=1; neutral conditioning',
            precision='float64', schedule_horizon=options['schedule_horizon'],
            training_data=digest(grids['train']), evaluation_data=digest({s: grids[s] for s in ('dev', 'heldout')}),
            coordinate_contract='full native hidden vector' if self.name == 'fixture-vector' else
                'joint full native image and coupled audio coordinates, unweighted',
            implementation=digest(dict(source=Path(__file__).read_text(), options=options)),
            device='cpu', determinism='local Python Random state retained; fixed schedule horizon'),
            parameters=parameters, grids=grids, references={s: _reference(grids[s]) for s in ('dev', 'heldout')},
            selections=('raw',), trained_off=True, evidence='fixture')

    def open(self, options, parameters, seed, cohort, *, checkpoint=None):
        return FixtureTrial(self.describe(options), options, parameters, seed, cohort, checkpoint)


class FixtureTrial:
    def __init__(self, capabilities, options, parameters, seed, cohort, checkpoint):
        self.capabilities, self.options, self.parameters = capabilities, options, dict(parameters)
        for p in capabilities.parameters:
            p.validate(parameters[p.name])
        self.seed, self.cohort = seed, cohort
        self.rng = random.Random(seed)
        arms = 1 if capabilities.name == 'fixture-vector' else 2
        population = parameters['particles.population']
        self.generator = [self.rng.uniform(-.03, .03) for _ in range(arms)]
        self.critic = [self.rng.uniform(-.03, .03) for _ in range(arms)]
        self.particles = [[self.rng.uniform(-.03, .03) for _ in range(population)] for _ in range(arms)]
        self.completed_steps = 0
        self.initial_state_sha256 = digest(dict(weights=self._weights(), critic=self.critic,
                                                rng=self.rng.getstate()))
        self.closed = False
        if checkpoint is not None:
            packet = json.loads(Path(checkpoint).read_text())
            expected = dict(parameters=self.parameters, seed=seed, cohort=cohort,
                            identity=capabilities.identity, initial_state_sha256=self.initial_state_sha256)
            if packet.get('schema') != 'particle-sliders-raw-fixture-checkpoint-v2' or packet['binding'] != expected:
                raise ValueError('fixture checkpoint config/cohort differs')
            state = packet['state']
            if packet['sha256'] != digest(state):
                raise ValueError('fixture full checkpoint checksum differs')
            if set(state) != {'generator', 'critic', 'particles', 'completed_steps', 'rng'}:
                raise ValueError('fixture checkpoint is not a complete raw-only state')
            self.generator, self.critic, self.particles, self.completed_steps = (
                state[k] for k in ('generator', 'critic', 'particles', 'completed_steps'))
            def tuples(value):
                return tuple(tuples(v) for v in value) if isinstance(value, list) else value
            self.rng.setstate(tuples(state['rng']))

    def _weights(self):
        return dict(generator=list(self.generator), particles=[list(p) for p in self.particles])

    def _state(self):
        return dict(**self._weights(), critic=self.critic,
                    completed_steps=self.completed_steps, rng=self.rng.getstate())

    def state_digest(self):
        return digest(self._state())

    def effective_parameters(self):
        # These exact values are read by step()/responses(), not CLI-only knobs.
        return dict(self.parameters)

    def frozen_identity(self):
        return self.capabilities.identity

    def optimizer_groups(self):
        scale = 1. - .5 * max(0, self.completed_steps - 1) / self.options['schedule_horizon']
        arms = len(self.generator)
        return [dict(role=role, base_lr=self.parameters[name], lr=self.parameters[name] * scale,
                     parameter_count=count)
                for role, name, count in [('generator_lr', 'lr.generator', arms),
                    ('critic_lr', 'lr.critic', arms),
                    ('particle_lr', 'lr.particle', arms * len(self.particles[0]))]]

    def step(self):
        if self.closed or self.completed_steps >= self.options['schedule_horizon']:
            raise ValueError('closed or exhausted fixture')
        p = self.parameters
        # Fixed total horizon is deliberately independent of the short rung.
        scale = 1. - .5 * self.completed_steps / self.options['schedule_horizon']
        for i, bank in enumerate(self.particles):
            temperature = p['particles.temperature']
            gain = self.generator[i] + sum(bank) / len(bank) / temperature
            noise = self.rng.uniform(-.005, .005)
            self.critic[i] += p['lr.critic'] * scale * (1. + noise - gain - self.critic[i])
            self.generator[i] += p['lr.generator'] * scale * (self.critic[i] - p['regularizer.l2_alpha'] * self.generator[i])
            for j, value in enumerate(bank):
                bank[j] += p['lr.particle'] * scale * (
                    self.critic[i] / len(bank) / temperature - p['regularizer.l2_alpha'] * value)
        self.completed_steps += 1

    def responses(self, split, selection, *, off=False):
        require_raw_selection(selection)
        weights = self._weights()
        gains = [g + sum(bank) / len(bank) / self.parameters['particles.temperature']
                 for g, bank in zip(weights['generator'], weights['particles'])]
        if len(gains) == 1:
            gains *= 2
        return _chunks(self.capabilities.grids[split], gains, off=off)

    def checkpoint(self, path):
        state = self._state()
        packet = dict(schema='particle-sliders-raw-fixture-checkpoint-v2', binding=dict(parameters=self.parameters, seed=self.seed, cohort=self.cohort,
            identity=self.capabilities.identity, initial_state_sha256=self.initial_state_sha256),
            state=state, sha256=digest(state))
        Path(path).write_bytes(canonical(packet))

    def export(self, path, selection):
        require_raw_selection(selection)
        Path(path).write_bytes(canonical(dict(schema='particle-sliders-raw-fixture-export-v2',
            selection='raw', weights=self._weights())))

    def close(self):
        self.closed = True
