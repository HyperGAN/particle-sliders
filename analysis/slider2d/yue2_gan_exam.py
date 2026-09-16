"""Run the production YuE2 GAN update against the existing unipolar exam.

CPU only: replace the frozen music model by PairField and its student by the
toy GAN's shared odd/even residual. The loss, critic, calibration, optimizer,
cap, clipping, and update order come directly from yue2_arm_b. The canonical
fixtures have three rows, so each update uses all three (live batches use
four). Scale zero is exact by construction, as it is for multiplier LoRA.
This tests transfer of the game, not native model capacity or audio quality.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from analysis.slider2d.gan import AdvResidual
from analysis.slider2d.formulation_leaderboard import REQUIRED_CELLS, unipolar_hit
from analysis.slider2d.plus_neu_exam import (
    PLUS_NEU_CELLS, score_plus_neu_exam, score_plus_neu_residual,
)
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import yue2_arm_b as game

AUDIT_SEEDS = (0, 1, 7)


class ToySlider(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.scale = 0.

    def delta(self, scale):
        return float(scale) * self.odd + abs(float(scale)) * self.even

    @contextmanager
    def scaled(self, scale):
        if scale not in (0., 1.):
            raise ValueError('Training must remain unipolar')
        previous = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = previous

    def snapshot(self):
        return AdvResidual(self.odd.detach().clone(), self.even.detach().clone())


class ToyBackend:
    def __init__(self, field, network):
        self.field = field
        self.network = network
        # Device/config carrier only; no random model weights or extra RNG draws.
        self.model = nn.Module()
        self.model.register_parameter('device_anchor', nn.Parameter(torch.zeros(()), requires_grad=False))
        self.model.config = SimpleNamespace(hidden_size=field.dim)

    def hidden(self, ids, checkpointing=False):
        neutral = self.field.poles(ids[0])[2]
        return (neutral + self.network.delta(self.network.scale))[None, None]


@isolated_seed('seed')
def run_cell(cell, *, steps=(600, 1200, 3400), seed=0):
    if not steps or min(steps) < 1:
        raise ValueError('Positive step budgets required')
    field = PLUS_NEU_CELLS[cell](seed=seed)
    network = ToySlider(field.dim)
    backend = ToyBackend(field, network)
    rows = []
    for i in range(field.rows):
        positive, _, neutral = field.poles(i)
        rows.append(dict(ids=[i], prefix_len=1, neutral=neutral[None], targets=positive[None]))
    critic, g, d = game.build_game(backend, network, rows)
    checkpoints = []
    history = []
    for step in range(1, max(steps) + 1):
        order = torch.randperm(len(rows)).tolist()
        metrics = game.update(backend, network, critic, g, d,
                              [rows[i] for i in order], step=step, checkpointing=False)
        history.append(dict(step=step, **metrics))
        if step in steps:
            score = score_plus_neu_residual(
                game.RECIPE['name'], field, network.snapshot(),
                teacher='faithful_plus_neu', plus_only=True,
            )
            checkpoints.append(dict(step=step, **score))
    control = score_plus_neu_exam('main_supervised_control', field,
        teacher='faithful_plus_neu', plus_neu=True, steps=400, seed=seed)
    return dict(cell=cell, seed=seed, batch=len(rows), recipe=dict(game.RECIPE),
                checkpoints=checkpoints, control=control, history=history)


def accepted(results, *, steps, seeds=AUDIT_SEEDS):
    """Every requested budget must pass both required cells for every seed."""
    if not steps or not seeds:
        return False
    for seed in seeds:
        for cell in REQUIRED_CELLS:
            matches = [r for r in results if r['seed'] == seed and r['cell'] == cell]
            if len(matches) != 1:
                return False
            result = matches[0]
            if not unipolar_hit(result['control']):
                return False
            for step in steps:
                rows = [r for r in result['checkpoints'] if r['step'] == step]
                if len(rows) != 1 or not unipolar_hit(rows[0]):
                    return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, nargs='+', default=[600, 1200, 3400])
    parser.add_argument('--seeds', type=int, nargs='+', default=list(AUDIT_SEEDS))
    parser.add_argument('--cells', nargs='+', choices=tuple(PLUS_NEU_CELLS), default=['divergent', 'close'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for cell in args.cells:
            result = run_cell(cell, steps=tuple(args.steps), seed=seed)
            results.append(result)
            args.out.write_text(json.dumps(dict(results=results), indent=2, allow_nan=False) + '\n')
            for row in result['checkpoints']:
                print(json.dumps(dict(cell=cell, seed=seed, **{key:row[key] for key in
                    ['step', 'cover', 'off_caption', 'neu_hold', 'hit', 'pole_cos']})), flush=True)
    # An acceptance failure is a failing command, not a green equation test.
    return 0 if accepted(results, steps=args.steps, seeds=args.seeds) else 1


if __name__ == '__main__':
    raise SystemExit(main())
