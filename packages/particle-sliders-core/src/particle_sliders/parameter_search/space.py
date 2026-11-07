"""Bounded deterministic joint exploration plus local bracket refinement."""
import math
import random

from .metric import digest


class Space:
    def __init__(self, capabilities, fixed, search):
        self.parameters = {p.name: p for p in capabilities.parameters}
        if set(fixed) & set(search) or set(fixed) | set(search) != set(self.parameters):
            raise ValueError('bind every active capability exactly once; unknown/no-op/ambiguous knobs are rejected')
        self.fixed, self.search = dict(fixed), dict(search)
        self.initial = dict(fixed)
        for name, value in fixed.items():
            self.parameters[name].validate(value)
        for name, spec in search.items():
            p = self.parameters[name]
            if not isinstance(spec, dict):
                raise ValueError('search dimension must be explicit')
            if p.kind in ('integer', 'categorical'):
                if set(spec) != {'values', 'initial'} or len(spec['values']) < 2 or len({digest(v) for v in spec['values']}) != len(spec['values']):
                    raise ValueError('discrete search requires unique explicit values and initial')
                for value in spec['values']:
                    p.validate(value)
                if spec['initial'] not in spec['values']:
                    raise ValueError('initial value missing from discrete choices')
            else:
                if (set(spec) != {'low', 'high', 'scale', 'include_zero', 'initial'}
                        or spec['scale'] not in ('log', 'linear') or type(spec['include_zero']) is not bool):
                    raise ValueError('real dimensions require low/high/scale/include_zero/initial')
                low, high = spec['low'], spec['high']
                p.validate(low); p.validate(high)
                if low >= high or spec['scale'] == 'log' and low <= 0:
                    raise ValueError('invalid nonempty positive logarithmic bracket')
                if spec['include_zero'] and not p.allow_zero:
                    raise ValueError('zero is not an implemented capability')
                if spec['initial'] != 0 or not spec['include_zero']:
                    if not low <= spec['initial'] <= high:
                        raise ValueError('initial value outside declared bracket')
            p.validate(spec['initial'])
            self.initial[name] = spec['initial']

    def cohort(self, identity, parameters):
        structural = {name: value for name, value in parameters.items()
                      if self.parameters[name].cohort == 'architecture'}
        # Nonzero strengths share an objective, but disabling a term is an ablation.
        ablations = {name: value == 0 for name, value in parameters.items()
                     if self.parameters[name].cohort == 'objective'}
        return digest(dict(identity=identity, architecture=structural, ablations=ablations))[:16]

    def simplicity(self, parameters):
        def value(name):
            p, v = self.parameters[name], parameters[name]
            return list(p.choices).index(v) if p.kind == 'categorical' else v
        # Prefer smaller/simpler active models/penalties, then lower independent rates.
        names = sorted(self.parameters, key=lambda n: (self.parameters[n].role.endswith('_lr'), n))
        return tuple(value(n) for n in names)

    def _at(self, name, fraction):
        spec = self.search[name]
        if 'values' in spec:
            return spec['values'][min(int(fraction * len(spec['values'])), len(spec['values']) - 1)]
        lo, hi = spec['low'], spec['high']
        return math.exp(math.log(lo) + fraction * math.log(hi / lo)) if spec['scale'] == 'log' else lo + fraction * (hi - lo)

    def explore(self, count, seed):
        joint_reserve = 1 if len(self.search) > 1 else 0
        if count < 1 + 2 * len(self.search) + joint_reserve:
            raise ValueError('initial_candidates must cover baseline, both dimension ends and a joint candidate')
        values, seen = [], set()
        def add(value):
            key = digest(value)
            if key not in seen:
                seen.add(key); values.append(value)
        add(self.initial)
        for name, spec in sorted(self.search.items()):
            endpoints = list(spec['values']) if 'values' in spec else [spec['low'], spec['high']] + ([0.] if spec['include_zero'] else [])
            for value in endpoints:
                add(dict(self.initial, **{name: value}))
        if len(values) + joint_reserve > count:
            raise ValueError('initial_candidates cannot cover all discrete/zero/end values plus joint exploration')
        rng = random.Random(seed)
        # Independent shuffled strata give joint exploration, not a one-rate-only screen.
        names = sorted(self.search)
        strata = {name: rng.sample(range(count), count) for name in names}
        for i in range(count * 16):
            if len(values) >= count or not names:
                break
            candidate = dict(self.initial)
            for name in names:
                fraction = (strata[name][i % count] + rng.random()) / count
                candidate[name] = self._at(name, fraction)
            add(candidate)
        return values[:count]

    def refine(self, nominees, previous, count):
        if count <= 0:
            return []
        seen, results = {digest(p) for p in previous}, []
        for nominee in nominees:
            for name, spec in sorted(self.search.items()):
                if 'values' in spec:
                    neighbors = spec['values']
                elif nominee[name] == 0:
                    neighbors = [spec['low']]
                else:
                    values = sorted({spec['low'], spec['high'], *(p[name] for p in previous if p[name] > 0)})
                    value = nominee[name]
                    lower = max((v for v in values if v < value), default=value)
                    upper = min((v for v in values if v > value), default=value)
                    average = (lambda a, b: math.sqrt(a * b)) if spec['scale'] == 'log' else (lambda a, b: (a + b) / 2)
                    neighbors = [average(lower, value), average(value, upper)]
                for value in neighbors:
                    candidate = dict(nominee, **{name: value})
                    key = digest(candidate)
                    if key not in seen:
                        seen.add(key); results.append(candidate)
                        if len(results) >= count:
                            return results
        return results

    def boundaries(self, parameters):
        boundaries = []
        for name, spec in self.search.items():
            p, value = self.parameters[name], parameters[name]
            if 'low' in spec and value in (spec['low'], spec['high']):
                boundaries.append(name)
            elif p.kind == 'integer':
                lo, hi = min(spec['values']), max(spec['values'])
                if value == lo and lo > p.minimum or value == hi and hi < p.maximum:
                    boundaries.append(name)
            elif p.kind == 'categorical' and set(map(digest, spec['values'])) != set(map(digest, p.choices)):
                boundaries.append(name)  # finite choices remain unexplored
        return boundaries
