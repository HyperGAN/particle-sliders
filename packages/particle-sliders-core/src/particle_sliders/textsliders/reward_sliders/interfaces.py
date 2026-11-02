"""Reward-independent entry points; changing the reward always creates a new fit."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from typing import Protocol

from .specs import RewardSpec, digest
from .directions import fit_direction, choose_dev, paired_statistics


class RewardScorer(Protocol):
    spec: RewardSpec

    def measure(self, path, *, full_song: bool = False) -> dict:
        """Return valid, scalar, window scores and separate diagnostics."""
        ...


def oriented_observations(spec, observations):
    if spec.orientation not in ('higher','lower'):
        raise ValueError('Reward orientation must be higher or lower')
    sign = 1. if spec.orientation=='higher' else -1.
    expected_spec = digest(asdict(spec))
    copied = deepcopy(observations)
    for row in copied:
        if row['status'] != 'complete':
            continue
        reward = row.get('reward')
        if reward and reward.get('reward_spec_sha256') != expected_spec:
            raise ValueError('Observation was scored under a different reward spec; rescore before fitting')
        value = None if reward is None else reward.get('scalar')
        if not reward or not reward.get('valid') or value is None or not math.isfinite(value) or not spec.valid_range[0]<=value<=spec.valid_range[1]:
            row.update(status='failed',error='Invalid reward retained by generic reward interface')
            continue
        reward['scalar'] = sign*value
    return copied


def fit_reward_teacher(spec, observations, features, layer):
    teacher = fit_direction(oriented_observations(spec, observations),features,layer)
    return teacher, dict(reward_spec_sha256=digest(asdict(spec)),layer=layer,
                         training_families=teacher.fitting_families,
                         policy='new reward requires this new fit and separate causal/LoRA validation')


def select_reward_candidate(spec, observations, arms, expected_pairs=8):
    selected = choose_dev(oriented_observations(spec,observations),arms,expected_pairs)
    return dict(selected=selected['selected'],mean_oriented_reward=selected['mean_ce'],
                reward_spec_sha256=digest(asdict(spec)))


def paired_reward_statistics(spec, observations, treatment, control='off'):
    result = paired_statistics(oriented_observations(spec,observations),treatment,control)
    result.update(reward=spec.identifier,orientation=spec.orientation,
                  interpretation='positive oriented delta favors the declared reward')
    return result
