"""CPU-only released-v2 game on the unchanged UNI continuation gates.

This arm mirrors the released Hub formulation ``particle-gmix-1600-v2``
(``ntc-ai/yue2-concept-sliders`` v2, final EMA at update 1600) instead of the
bridge-doc MLP-only game in :mod:`analysis.slider2d.yue2_particle_exam`:

- Same core: Rp paired-error game, ``b_cap`` lazy x4, 128x4 particles, VIC,
  LRs 6e-4/9e-4/6e-3, Adam (0, 0.999), EMA 0.995, GAN-only (no MSE).
- v2 head: ``gmix_t8_w48_l1`` critic, paired-edit normalization, noise
  schedule with ``sigma0 = edit_rms / 0.28`` and T=1600 + 1.3xE hold, D/G
  batches of 8 drawn with replacement.

Deliberate fixture gap (documented, not hidden): the native seedbank holds
512 sources (128 seeds x 4 templates x 32-token histories). PairField has 3
rows and no histories, so the toy draws batches of 8 with replacement from
those 3 rows under the same replacement policy. Step budgets are therefore
not equivalent measures of convergence between toy and native.

Propose-only. No Music default, locked ``AdvConfig()``, or live
``--lm_target`` is touched. The frozen backend becomes PairField; one routed
low-rank student starts at the exact base; raw and EMA students are scored,
and BI reads the same unipolar weights as an unscored canary.
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

# Golden Hub numbers for particle-gmix-1600-v2 (catalog.json female record,
# FORMULATION.md, evidence/particle-gmix-1600-v2/metal/{manifest,teacher-audit}.json).
# Tests pin the arm below against this dict; update both together or not at all.
V2_SPEC = dict(
    recipe_name='anneal-routed-particle-error-yue2-v1',
    config_sha256='1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb',
    model_glue_reference='df70ccb2ca8f532bdcc07a343fd12bec77362523',
    generator_objective='paired_error_rpgan_plus_particle_vic',
    critic='gmix',
    critic_tokens=8,
    critic_width=48,
    critic_layers=1,
    critic_heads=4,
    critic_score_bound=8.0,
    g_lr=0.0006,
    d_lr=0.0009,
    particle_lr=0.006,
    betas=(0.0, 0.999),
    schedule='constant',
    ema=0.995,
    parts=128,
    particle_dim=4,
    particle_vic_batch=64,
    particle_vic_target_std=1.0,
    particle_vic_eps=1e-4,
    vicreg_weight=1.0,
    adv_b_cap=1.0,
    adv_reg_kappa=1.0,
    penalty_lazy_k=4,
    penalty_method='autograd',
    penalty_anneal='none',
    cap_coordinates='normalized_paired_error_plus_shared_gaussian',
    target_normalization='paired_edit_per_coordinate_std_median_rms_gain',
    edit_rms_target=1.0,
    edit_noise_ratio=0.28,
    noise_start='edit_rms/edit_noise_ratio',
    noise_floor=0.03,
    noise_decay_steps=1600,
    noise_hold='edit_rms*noise_hold_ratio',
    noise_hold_ratio=1.3,
    adv_batch=8,
    sample_seeds=128,
    history_tokens=32,
    seedbank_sources=512,
    polarity='unipolar',
    lm_target='faithful_plus_neu',
    trained_scales=(1.0,),
    recommended_range=(0.0, 1.0),
    adapter_rank=8,
    adapter_alpha=8.0,
    adapter_width=48,
    router_width=16,
    adv_weight=1.0,
    aux_weights=dict(anchor_weight=0.0, cover_weight=0.0, fm_weight=0.0,
                     end_weight=0.0, lyrichold_weight=0.0, plan_weight=0.0,
                     pole_weight=0.0),
    propose_only=True,
)

V2_CRITIC_CONFIG = dict(tokens=8, width=48, layers=1, heads=4, score_bound=8.0)
V2_NOISE_DECAY_STEPS = 1600
V2_BATCH = 8
V2_LADDER = (600, 1200, 1600, 3400)


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


def build_v2_game(backend, network, fixed):
    """Released-v2 game: gmix critic, paired-edit norm, T=1600 schedule.

    ``fixed`` rows carry ``neutral``/``targets`` so the critic whitens by the
    paired edit (scales from ``std(target-neutral)`` + median-RMS gain), and
    ``noise_start`` is ``edit_rms / 0.28``. The hold (1.3xE) comes from
    :func:`shared.update`; only the horizon is pinned here to the v2 1600.
    """
    critic, g, d = game.build_game(backend, network, fixed,
                                   critic='gmix', critic_config=dict(V2_CRITIC_CONFIG))
    critic.noise_decay_steps = int(V2_NOISE_DECAY_STEPS)
    return critic, g, d


@torch.no_grad()
def score(field, network):
    with network.scaled(1.):
        deltas = [(network(field.poles(i)[2]) - field.poles(i)[2]).clone() for i in range(field.rows)]
    residual = Snapshot(deltas)
    uni = score_plus_neu_residual('particle_gmix_v2', field, residual, teacher='faithful_plus_neu', plus_only=True)
    half = score_plus_neu_residual('half', field, Snapshot(deltas,.5), teacher='faithful_plus_neu', plus_only=True)
    bi = score_bipolar(field, residual)
    return dict(uni=uni, half_cover=half['cover'], bi=bi, bipolar_hit=bipolar_hit(bi))


@isolated_seed('seed')
def run_cell(cell, *, seed=0, steps=V2_LADDER):
    field = PLUS_NEU_CELLS[cell](seed=seed)
    network = Student(field.dim); backend = Backend(field, network)
    fixed = [dict(ids=[i], prefix_len=1, neutral=field.poles(i)[2][None],
                  targets=field.poles(i)[0][None]) for i in range(field.rows)]
    torch.manual_seed(seed + 1000)
    critic, g, d = build_v2_game(backend, network, fixed)
    sampler = shared.BridgeSampler(field.rows, seed, batch_size=V2_BATCH)
    ema = shared.initialize_ema(network)
    result = dict(cell=cell, seed=seed, recipe=dict(game.RECIPE),
                  v2_spec={k: (list(v) if isinstance(v, tuple) else v) for k, v in V2_SPEC.items()},
                  critic_kind=type(critic).__name__,
                  initial=score(field,network), checkpoints=[])
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
    p.add_argument('--steps',type=int,nargs='+',default=list(V2_LADDER))
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
