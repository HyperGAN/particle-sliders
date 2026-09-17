"""CPU-only native routed-particle game on the unchanged UNI continuation gates.

The frozen backend becomes PairField. One routed low-rank projection starts
at the exact base; target normalization therefore cannot solve the exam at init.
Raw and EMA students are both scored, and BI reads the same unipolar weights.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from analysis.slider2d.plus_neu_exam import PLUS_NEU_CELLS, score_plus_neu_residual
from analysis.slider2d.unipolar_gan import score_bipolar
from analysis.slider2d.formulation_leaderboard import bipolar_hit
from analysis.slider2d.rng import isolated_seed
from conceptmod.textsliders import particle_bridge_gan as shared, yue2_particle_bridge as game


class Student(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.particles = nn.Parameter(torch.randn(128, 4))
        self.down = nn.Linear(dim, 8, bias=False)
        self.up = nn.Linear(8, dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=1)
        nn.init.zeros_(self.up.weight)
        self.bridge = shared.RoutedMLP(8, 8)
        self.scale = 0.

    def forward(self, x):
        return x + self.scale * self.up(self.bridge(self.down(x), self.particles))

    @contextmanager
    def scaled(self, scale):
        previous = self.scale; self.scale = scale
        try: yield
        finally: self.scale = previous


class Backend:
    def __init__(self, field, network):
        self.field, self.network = field, network
        self.model = nn.Module()
        self.model.register_parameter('device_anchor', nn.Parameter(torch.zeros(()), requires_grad=False))
        self.model.config = SimpleNamespace(hidden_size=field.dim)

    def hidden(self, ids, checkpointing=False):
        return self.network(self.field.poles(ids[0])[2])[None,None]


class Snapshot:
    def __init__(self, deltas, factor=1.): self.deltas, self.factor = deltas, factor
    def delta_for_row(self, scale, row): return scale * self.factor * self.deltas[row]


@torch.no_grad()
def score(field, network):
    with network.scaled(1.):
        deltas = [(network(field.poles(i)[2]) - field.poles(i)[2]).clone() for i in range(field.rows)]
    residual = Snapshot(deltas)
    uni = score_plus_neu_residual('particle_bridge', field, residual, teacher='faithful_plus_neu', plus_only=True)
    half = score_plus_neu_residual('half', field, Snapshot(deltas,.5), teacher='faithful_plus_neu', plus_only=True)
    bi = score_bipolar(field, residual)
    return dict(uni=uni, half_cover=half['cover'], bi=bi, bipolar_hit=bipolar_hit(bi))


@isolated_seed('seed')
def run_cell(cell, *, seed=0, steps=(600, 3400, 8000)):
    field = PLUS_NEU_CELLS[cell](seed=seed)
    network = Student(field.dim); backend = Backend(field, network)
    fixed = [dict(ids=[i], prefix_len=1, neutral=field.poles(i)[2][None],
                  targets=field.poles(i)[0][None]) for i in range(field.rows)]
    torch.manual_seed(seed + 1000)
    critic, g, d = game.build_game(backend, network, fixed)
    sampler = shared.BridgeSampler(field.rows, seed); ema = shared.initialize_ema(network)
    result = dict(cell=cell, seed=seed, recipe=dict(game.RECIPE), initial=score(field,network), checkpoints=[])
    peak = 0.
    for step in range(1,max(steps)+1):
        metrics = game.update(backend,network,critic,g,d,fixed,sampler=sampler,step=step,checkpointing=False)
        shared.update_ema(ema,network); peak=max(peak,metrics['g_adv'])
        if step in steps:
            live = score(field,network); state = shared.initialize_ema(network)
            network.load_state_dict(ema); averaged = score(field,network); network.load_state_dict(state)
            result['checkpoints'].append(dict(step=step,live=live,ema=averaged,metrics=metrics,peak_g_adv=peak))
            print(json.dumps(dict(cell=cell,seed=seed,step=step,noise_std=metrics['noise_std'],
                live_hit=live['uni']['hit'],ema_hit=averaged['uni']['hit'],
                cover=averaged['uni']['cover'],off_caption=averaged['uni']['off_caption'],
                neu_hold=averaged['uni']['neu_hold'],peak_g_adv=peak)),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,7])
    p.add_argument('--cells',nargs='+',choices=['divergent','close'],default=['divergent','close'])
    p.add_argument('--steps',type=int,nargs='+',default=[600,3400,8000])
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if min(a.steps)<1:p.error('Positive budgets required')
    torch.set_num_threads(1);results=[];a.out.parent.mkdir(parents=True,exist_ok=True)
    for seed in a.seeds:
        for cell in a.cells:
            results.append(run_cell(cell,seed=seed,steps=tuple(a.steps)))
            a.out.write_text(json.dumps(dict(results=results),indent=2,allow_nan=False)+'\n')
    # Earlier budgets are reported honestly; only the declared final budget decides this audit.
    if not all(r['checkpoints'][-1]['ema']['uni']['hit'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
