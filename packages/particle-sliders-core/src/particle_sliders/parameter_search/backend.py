"""Explicit model-owned search bindings, separate from native runtime discovery.

Importing this contract never imports Torch, a product or a GPU runtime.
A search adapter must report actual effective optimizer/config values, not echo
CLI input. Unsupported/disabled/no-op controls must not be advertised.
"""
from dataclasses import asdict, dataclass
from importlib.metadata import entry_points
import math
from typing import Protocol

from .metric import digest, validate_grid
from ..weight_policy import require_no_weight_ema, require_raw_selection

BACKEND_API = 'particle-sliders-raw-search-backend-v2'


@dataclass(frozen=True)
class Parameter:
    name: str
    kind: str  # real, integer, categorical
    role: str  # generator_lr, critic_lr, particle_lr, regularizer, particle_setting
    semantics: str  # actual participating optimizer/module/term and upstream binding
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple = ()  # categorical order also declares simplicity preference
    allow_zero: bool = False
    cohort: str = 'shared'  # architecture, objective or shared

    def validate(self, value):
        if self.kind == 'categorical':
            if not any(type(value) is type(choice) and value == choice for choice in self.choices):
                raise ValueError(self.name + ': unsupported categorical value')
        elif (type(value) not in (int, float) or not math.isfinite(value)
              or self.kind == 'integer' and type(value) is not int
              or value == 0 and not self.allow_zero
              or self.minimum is not None and value < self.minimum
              or self.maximum is not None and value > self.maximum):
            raise ValueError(self.name + ': unsupported numeric value')
        return value

    def __post_init__(self):
        if (not self.name or not self.semantics or self.kind not in ('real', 'integer', 'categorical')
                or self.role not in ('generator_lr', 'critic_lr', 'particle_lr', 'regularizer', 'particle_setting')
                or self.cohort not in ('shared', 'architecture', 'objective')):
            raise ValueError('invalid explicit parameter capability')
        if self.kind == 'categorical' and not self.choices:
            raise ValueError('categorical choices/simplicity order required')
        if self.role.endswith('_lr') and (self.kind != 'real' or self.allow_zero):
            raise ValueError('active optimizer rates must be positive independent real controls')
        if self.kind != 'categorical' and (self.minimum is None or self.maximum is None
                or self.minimum < 0 or not math.isfinite(self.maximum) or self.minimum > self.maximum):
            raise ValueError('finite supported numeric parameter bounds required')
        if self.role == 'regularizer' and self.allow_zero and self.cohort != 'objective':
            raise ValueError('disabling a regularizer must isolate objective cohorts')


@dataclass(frozen=True)
class Capabilities:
    name: str
    identity: dict
    parameters: tuple[Parameter, ...]
    grids: dict
    references: dict
    selections: tuple = ('raw',)
    trained_off: bool = False
    evidence: str = 'fixture'  # native qualification requires actual product evidence
    api: str = BACKEND_API

    def __post_init__(self):
        required = {'model', 'teacher', 'target', 'precision', 'schedule_horizon',
                    'training_data', 'evaluation_data', 'coordinate_contract',
                    'implementation', 'device', 'determinism'}
        if (self.api != BACKEND_API or not self.name or not required <= self.identity.keys()
                or self.evidence not in ('fixture', 'native') or tuple(self.selections) != ('raw',)):
            raise ValueError('incomplete frozen backend identity/capabilities')
        if type(self.identity['schedule_horizon']) is not int or self.identity['schedule_horizon'] < 2:
            raise ValueError('fixed schedule horizon must be >=2')
        if set(self.grids) != {'train', 'dev', 'heldout'} or set(self.references) != {'dev', 'heldout'}:
            raise ValueError('explicit disjoint train/dev/heldout grids and reference hashes required')
        labels = []
        for grid in self.grids.values():
            validate_grid(grid)
            current = {digest(row['example']) for row in grid}
            if any(current & other for other in labels):
                raise ValueError('training, selection and heldout labels overlap')
            labels.append(current)
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError('duplicate/ambiguous parameter capability')
        roles = [p.role for p in self.parameters if p.role.endswith('_lr')]
        if len(set(roles)) != len(roles):
            raise ValueError('one independent absolute rate per participating optimizer role')
        if 'generator_lr' not in roles:
            raise ValueError('actual generator LR capability required')
        digest(asdict(self))

    def as_dict(self):
        return asdict(self)


class Trial(Protocol):
    completed_steps: int
    initial_state_sha256: str

    def effective_parameters(self) -> dict: ...
    def frozen_identity(self) -> dict: ...
    def optimizer_groups(self) -> list[dict]:
        """Actual participating groups: role, base_lr, lr, parameter_count."""
        ...
    def state_digest(self) -> str:
        """Digest training/native/RNG/modes state, excluding incidental diagnostics."""
        ...
    def step(self) -> None: ...
    def checkpoint(self, path) -> None:
        """Full RAW model/bank/critic/optimizers/streams/RNG/horizon/counter."""
        ...
    def responses(self, split, selection, *, off=False):
        """Yield bounded raw chunks; preserve training/native/RNG/modes state."""
        ...
    def finished_outputs(self, split, selection):
        """Optional opt-in: genuine finished media for a qualified failure guard."""
        ...
    def export(self, path, selection) -> None: ...
    def close(self) -> None: ...


class SearchBackend(Protocol):
    def describe(self, options: dict) -> Capabilities: ...
    def open(self, options: dict, parameters: dict, seed: int, cohort: str, *, checkpoint=None) -> Trial: ...


def resolve(name):
    if name in ('fixture-vector', 'fixture-coupled'):
        from .fixture import FixtureBackend
        return FixtureBackend(name)
    matches = list(entry_points(group='particle_sliders.search_backends', name=name))
    if len(matches) != 1:
        raise ValueError(f'expected one installed search backend {name!r}; found {len(matches)}')
    result = matches[0].load()
    if not callable(getattr(result, 'describe', None)) or not callable(getattr(result, 'open', None)):
        raise TypeError('search registration must provide describe/open')
    return result


class LifecycleTrial:
    """Small real bridge to existing image/ResponseSession; no second train loop.

    A product factory creates its SessionConfig/opens the normal lifecycle with
    the exact mapped recipe, and supplies an observer with methods below. This
    is deliberately opt-in: ResponseSession.evaluate(index) alone does not bind
    the teacher, complete grids, effective rates or native media qualification.
    """
    def __init__(self, session, observer):
        require_no_weight_ema(session.config.options['ema_decay'])
        if session.ema.values is not None:
            raise ValueError('Weight EMA state is incompatible with raw search')
        self.session, self.observer = session, observer
        self.initial_state_sha256 = observer.initial_state_sha256

    @property
    def completed_steps(self):
        return self.session.game.completed_steps

    def effective_parameters(self):
        return self.observer.effective_parameters(self.session)

    def frozen_identity(self):
        return self.observer.frozen_identity(self.session)

    def optimizer_groups(self):
        return self.observer.optimizer_groups(self.session)

    def state_digest(self):
        # Existing _hash includes full tensor values/shape/dtype and game state.
        from ..pg09 import _hash
        return _hash(dict(packet=self.session.checkpoint(), native=self.observer.native_state()))

    def step(self):
        self.session.step()

    def checkpoint(self, path):
        self.session.checkpoint(path)

    def responses(self, split, selection, *, off=False):
        require_raw_selection(selection)
        # Use the existing observation transaction and restore native state.
        # OFF is a diagnostic at the trained checkpoint, never a ranking gate.
        return self.observer.chunks(self.session, split, selection, off=off)

    def finished_outputs(self, split, selection):
        require_raw_selection(selection)
        provider = getattr(self.observer, 'finished_outputs', None)
        if not callable(provider):
            from ..candidate_failure import GuardUnmeasured
            raise GuardUnmeasured('product has no finished-output probe; native responses are not media')
        return provider(self.session, split, selection)

    def export(self, path, selection):
        require_raw_selection(selection)
        self.session.export(path, selection=selection)

    def close(self):
        self.session.close()
