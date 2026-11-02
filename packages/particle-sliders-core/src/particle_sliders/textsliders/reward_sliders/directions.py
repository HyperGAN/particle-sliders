"""Training-only within-cell direction fitting and whole-family statistics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import numpy as np
import torch


@dataclass
class RewardTeacher:
    layer: int
    coefficient: float
    residual_norm: float
    direction: torch.Tensor
    center: torch.Tensor
    fitting_families: list[str]

    def hook_kwargs(self):
        return dict(layer=self.layer, coefficient=self.coefficient, residual_norm=self.residual_norm,
                    direction=self.direction)


def fit_direction(observations, features, layer):
    cells = defaultdict(list)
    for obs in observations:
        if obs['split'] != 'train':
            continue
        if obs['status'] != 'complete' or not obs['reward']['valid']:
            raise ValueError('Training direction requires all predeclared baseline takes to be valid')
        cells[obs['cell_hash']].append(obs)
    differences, centers, norms = defaultdict(list), defaultdict(list), defaultdict(list)
    for group in cells.values():
        if len(group) < 2 or len({o['family'] for o in group}) != 1:
            raise ValueError('Need matched seeds within one family and exact cell')
        ordered = sorted(group, key=lambda o: (o['reward']['scalar'], o['seed']))
        family = ordered[0]['family']
        get = lambda obs: features[obs['id']][str(layer)]
        differences[family].append(get(ordered[-1])['mean'] - get(ordered[0])['mean'])
        centers[family].append(torch.stack([get(o)['mean'] for o in group]).mean(0))
        norms[family].append(torch.cat([get(o)['residual_norms'].flatten() for o in group]))
    if not differences:
        raise ValueError('No training families')
    direction = torch.stack([torch.stack(v).mean(0) for v in differences.values()]).mean(0).float()
    if not torch.isfinite(direction).all() or direction.norm() < 1e-8:
        raise ValueError('Degenerate direction')
    direction /= direction.norm()
    center = torch.stack([torch.stack(v).mean(0) for v in centers.values()]).mean(0).float()
    # Equal frame counts/seeds make the empirical median family-balanced.
    lengths = [sum(t.numel() for t in v) for v in norms.values()]
    if len(set(lengths)) != 1:
        raise ValueError('Unequal family frame counts require explicit reweighting')
    residual_norm = float(torch.cat([torch.cat(v) for v in norms.values()]).median())
    return RewardTeacher(layer, 0., residual_norm, direction, center, sorted(differences))


def rank_agreement(observations, features, teacher, split):
    cells = defaultdict(list)
    for obs in observations:
        if obs['split'] == split and obs['status'] == 'complete' and obs['reward']['valid']:
            cells[obs['cell_hash']].append(obs)
    families = defaultdict(list)
    for group in cells.values():
        for a, b in itertools.combinations(group, 2):
            delta_r = a['reward']['scalar']-b['reward']['scalar']
            if delta_r == 0:
                continue
            delta_h = features[a['id']][str(teacher.layer)]['mean']-features[b['id']][str(teacher.layer)]['mean']
            prediction = float(delta_h @ teacher.direction)
            families[a['family']].append(.5 if prediction == 0 else float(prediction*delta_r > 0))
    return dict(family_agreement={f: float(np.mean(v)) for f, v in families.items()},
                mean=float(np.mean([np.mean(v) for v in families.values()])) if families else None,
                pairs=sum(map(len, families.values())))


def paired_statistics(observations, treatment, control='off', bootstrap_seed=2081, draws=10000):
    groups = defaultdict(dict)
    failures = defaultdict(int)
    counts = defaultdict(int)
    for obs in observations:
        arm = obs['arm']['name']
        if arm not in (treatment, control):
            continue
        counts[arm] += 1
        if obs['status'] != 'complete' or not obs.get('reward', {}).get('valid'):
            failures[arm] += 1
            continue
        key = (obs['family'], obs['cell_hash'], obs['seed'])
        if arm in groups[key]:
            raise ValueError('Duplicate treatment in a paired cell')
        groups[key][arm] = obs['reward']['scalar']
    by_family = defaultdict(list)
    for (family, _, _), arms in groups.items():
        if treatment in arms and control in arms:
            by_family[family].append(arms[treatment]-arms[control])
    family_means = np.array([np.mean(v) for v in by_family.values()])
    deltas = [d for v in by_family.values() for d in v]
    result = dict(treatment=treatment, control=control, counts=dict(counts), failures=dict(failures),
                  valid_pairs=len(deltas), family_deltas={f: float(np.mean(v)) for f,v in by_family.items()},
                  mean_delta=None, seed_win_rate=None, family_bootstrap_95=None)
    if deltas:
        rng = np.random.default_rng(bootstrap_seed)
        boot = family_means[rng.integers(len(family_means), size=(draws, len(family_means)))].mean(1)
        result.update(mean_delta=float(family_means.mean()), seed_win_rate=float(np.mean(np.array(deltas)>0)),
                      family_bootstrap_95=np.quantile(boot, [.025, .975]).tolist())
    return result


def choose_dev(observations, arms, expected_pairs=8):
    scores = {}
    for arm in arms:
        rows = [o for o in observations if o['arm']['name'] == arm['name']]
        if len(rows) != expected_pairs or any(o['status'] != 'complete' or not o.get('reward', {}).get('valid') for o in rows):
            scores[arm['name']] = None
        else:
            families = defaultdict(list)
            for row in rows:
                families[row['family']].append(row['reward']['scalar'])
            scores[arm['name']] = float(np.mean([np.mean(v) for v in families.values()]))
    if scores.get('off') is None:
        raise ValueError('Off has unresolved failures')
    eligible = [a for a in arms if scores[a['name']] is not None]
    selected = min(eligible, key=lambda a: (-scores[a['name']], a['name'] != 'off',
                                           a.get('coefficient', 0), a.get('layer', -1)))
    return dict(selected=selected, mean_ce=scores)


def causal_gate(observations, *, min_gain=.02, expected_pairs=16):
    off = paired_statistics(observations, 'positive')
    random = paired_statistics(observations, 'positive', 'random')
    reverse = paired_statistics(observations, 'reversed')
    failures = [o['id'] for o in observations if o['status'] != 'complete' or not o.get('reward', {}).get('valid')]
    passed = (not failures and off['valid_pairs'] == random['valid_pairs'] == expected_pairs
              and off['mean_delta'] >= min_gain and random['mean_delta'] >= min_gain
              and off['seed_win_rate'] > .5)
    return dict(passed=passed, minimum_ce_gain=min_gain, positive_vs_off=off, positive_vs_random=random,
                reversed_vs_off=reverse, failures=failures,
                decision='activation_teacher_supported' if passed else 'failed_or_inconclusive_causal_probe')
