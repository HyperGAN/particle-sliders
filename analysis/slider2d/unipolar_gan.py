"""Propose-only +/0 conditional RpGAN, scored on the unipolar leaderboard.

Use the same free-origin student as the supervised uni board. Neutral hold
must be learned here; it is not made automatic by removing that parameter.
The shared game is also available to explicitly requested music experiments.
No locked default or Music argv is changed by collecting this arm.
"""
from contextlib import contextmanager

import torch
from torch import nn

from analysis.slider2d.plus_neu_exam import OriginResidual
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import unipolar_gan as game

NAME = 'rpgan_bcap_plus_neu'
PROPOSE_ONLY = True
MERGE_TO_TRAINER = False
TOY_LR = .005


class Student(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.odd = nn.Parameter(torch.zeros(dim))
        self.even = nn.Parameter(torch.zeros(dim))
        self.origin = nn.Parameter(torch.zeros(dim))
        self.scale = 0.

    def delta(self, scale):
        return scale * self.odd + abs(scale) * self.even + self.origin

    @contextmanager
    def scaled(self, scale):
        if scale not in game.SCALES:
            raise ValueError('Only +1 and 0 can be trained')
        previous = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = previous

    def snapshot(self):
        return OriginResidual(self.odd.detach().clone(), self.even.detach().clone(),
                              self.origin.detach().clone())


@isolated_seed('seed')
def fit(field, *, steps=400, seed=0):
    network = Student(field.dim)
    real = torch.stack([field.poles(i)[0] - field.poles(i)[2] for i in range(field.rows)])
    critic, g, d = game.build_game(network, real, lr=TOY_LR)
    def predict(i, scale, checkpointing):
        return network.delta(scale)[None]
    history = []
    for step in range(1, steps + 1):
        history.append(game.update(network, critic, g, d, real, predict,
            step=step, total_steps=steps, checkpointing=False))
    return network.snapshot(), history


def score_bipolar(field, residual):
    """Read the SAME unipolar fit on the existing bipolar continuation gates.

    No negative training and no antipodal gate. This diagnostic demonstrates
    why a unipolar HIT must not be promoted as a bipolar winner.
    """
    from analysis.slider2d.exam import (
        teacher_rollouts, teacher_self_match, rollout_report, exam_verdicts,
    )
    from analysis.slider2d.field import cosine
    head = field.readout()
    positive, negative, corpus = teacher_rollouts(field, head)
    def report(plus, minus):
        return rollout_report(field, plus, minus, readout=head,
            teacher_plus=positive, teacher_minus=negative, corpus=corpus)
    row = report([field.poles(i)[2] + residual.delta(1.) for i in range(field.rows)],
                 [field.poles(i)[2] + residual.delta(-1.) for i in range(field.rows)])
    ceiling = report([field.poles(i)[0] for i in range(field.rows)],
                     [field.poles(i)[1] for i in range(field.rows)])
    row.update(name=NAME, teacher='faithful_plus_neu; trained +/0 only',
        roll_swing_kept=row['roll_swing'] / (abs(ceiling['roll_swing']) + 1e-8),
        roll_match_kept=row['roll_match'] / (teacher_self_match(positive, negative) + 1e-8),
        collapse=cosine(residual.delta(1.), residual.delta(-1.)))
    row['axis'] = exam_verdicts(row)
    row['pass'] = all(value == 'right' for value in row['axis'].values())
    row['reason'] = 'Same unipolar weights; negative endpoint was not trained'
    return row
