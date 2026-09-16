"""YuE2 model adapter for the locked Music Arm B transfer (not UNI16).

Pole MSE is summed over +/-; adversarial and end losses average the two.
The toy cover pin maps to this single pole loss. No separate cover term.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch
import torch.nn.functional as F
import yaml

from analysis.slider2d.adv import make_grad_regularizer, rp_d_loss, rp_g_loss
from conceptmod.textsliders.lm_adv import LMDiscriminator, param_grad_norm
from conceptmod.textsliders.slider_targets import lm_faithful_guard_e
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B
from conceptmod.textsliders.yue2_backend import sound_only
from conceptmod.textsliders.yue2_uni import end_margins

# Operational settings are the locked Music transfer card, not toy defaults.
RECIPE = dict(ARM_B, name='music-arm-b-yue2-v1', adv_weight=1., end_weight=1.,
    lyrichold_weight=0., plan_weight=0., anchor_weight=0.,
    critic_hidden=256, critic_layers=2, adv_in='scaled', adv_batch=4,
    g_lr=.0005, d_lr=.00075, betas=[0.,.999], g_weight_decay=1e-6,
    schedule='constant', grad_clip_value=1., trained_scales=[1.,-1.],
    penalty_method='autograd', penalty_lazy_k=1, penalty_anneal='none',
    cap_coordinates='fixed_teacher_rms', cover_mapping='pole_loss_once',
    pole_reduction='sum_poles_mean_rows', end_reduction='mean_poles_mean_rows',
    critic_positions='music_start_only', phase_changes=[],
    prompt_policy='balanced_shuffled_passes', history_policy='fresh_base_each_draw_from_step_1')


def load_prompts(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw,dict) or not isinstance(raw.get('rows'),list) or not raw['rows']:
        raise ValueError('Prompts need a nonempty rows list')
    allowed={'rows','slider_positive','slider_negative','leak_positive','leak_negative',
             'plus_label','minus_label','recommended_range'}
    if set(raw)-allowed: raise ValueError(f'Unsupported prompt metadata: {set(raw)-allowed}')
    meta={k:v for k,v in raw.items() if k!='rows'}
    for pair in [('slider_positive','slider_negative'),('leak_positive','leak_negative')]:
        values=[meta.get(k) for k in pair]
        if any(values) and not all(isinstance(v,str) and v.strip() for v in values):
            raise ValueError('Declared axes require two nonempty captions')
    if meta.get('leak_positive') and not meta.get('slider_positive'):
        raise ValueError('A leak axis requires an independently declared slider axis')
    for v in meta.values():
        if isinstance(v,str): sound_only(v)
    rows=[]
    for item in raw['rows']:
        if not isinstance(item,dict) or set(item)!={'neutral','positive','negative','lyrics'}:
            raise ValueError('Each Arm B row must contain neutral, positive, negative, lyrics only')
        if any(not isinstance(v,str) or not v.strip() for v in item.values()):
            raise ValueError('Prompt values must be nonempty strings')
        row={k:sound_only(v.strip()) for k,v in item.items()}
        if len({row[k] for k in ('neutral','positive','negative')}) != 3:
            raise ValueError('Bipolar Arm B requires three distinct captions')
        rows.append(row)
    return rows,meta


@torch.no_grad()
def prepare(backend,rows,meta,frames,max_seq_len):
    limit=min(max_seq_len,backend.model.config.max_position_embeddings)
    def encode(style,lyrics):
        ids=backend.prefix(style,lyrics)
        if len(ids)+frames>limit: raise ValueError('Prompt and history exceed context limit')
        return ids,backend.hidden(ids)[:,-1].float()
    directions={}
    for name in ('slider','leak'):
        if meta.get(name+'_positive'):
            _,a=encode(meta[name+'_positive'],rows[0]['lyrics'])
            _,b=encode(meta[name+'_negative'],rows[0]['lyrics'])
            directions[name]=a-b
            if not torch.isfinite(a-b).all() or float((a-b).norm())<=1e-8:
                raise ValueError(f'Degenerate declared {name} axis')
    prepared=[]
    for row in rows:
        ids,neu=encode(row['neutral'],row['lyrics'])
        _,pos=encode(row['positive'],row['lyrics'])
        _,neg=encode(row['negative'],row['lyrics'])
        if 'leak' in directions:
            plus,minus=lm_faithful_guard_e(pos,neg,neu,directions['leak'],slider_dir=directions['slider'])
        else:
            # This is the shared Music faithful_guard_e no-leak behavior.
            plus,minus=pos,neg
        prepared.append(dict(prefix=ids,prefix_len=len(ids),neutral=neu.cpu(),
            targets=torch.cat([plus,minus]).cpu(),
            guard_applied=not (torch.equal(plus,pos) and torch.equal(minus,neg)),
            target_shift=float((plus-pos).norm()),
            raw_targets=torch.cat([pos,neg]).cpu()))
    return prepared


def build_game(backend,network,fixed):
    device=next(backend.model.parameters()).device
    critic=LMDiscriminator(backend.model.config.hidden_size,hidden_dim=RECIPE['critic_hidden'],
        n_hidden=RECIPE['critic_layers'],in_mode='scaled').to(device)
    real=torch.cat([r['targets']-r['neutral'] for r in fixed]).to(device)
    critic.calibrate_input_scale(real)
    g=torch.optim.AdamW(network.parameters(),lr=RECIPE['g_lr'],betas=tuple(RECIPE['betas']),
                        weight_decay=RECIPE['g_weight_decay'])
    d=torch.optim.Adam(critic.parameters(),lr=RECIPE['d_lr'],betas=tuple(RECIPE['betas']))
    return critic,g,d


def make_regularizer():
    return make_grad_regularizer(arm='b_cap',coeff=RECIPE['adv_b_cap'],kappa=RECIPE['adv_reg_kappa'],
        norm=RECIPE['adv_norm'],lazy_k=1,target_anneal='none')


def penalty(regularizer,critic,real,fake,step):
    # Both sides are already in the fixed teacher-RMS coordinate system.
    # D.net is the same score as D(raw), without dividing by the scale twice.
    # Zero fakes remain in the batch; use the vendored exact-autograd path.
    scale=critic.input_scale.detach()
    return regularizer.penalty(critic.net,real/scale,fake/scale,step=step)


def forward(backend,row,checkpointing):
    hidden=backend.hidden(row['ids'],checkpointing=checkpointing)
    index=row['prefix_len']-1
    return hidden[:,index].float(),end_margins(backend.model,hidden[:,index:])


def update(backend,network,critic,g,d,rows,*,step,checkpointing=True):
    if not rows: raise ValueError('Empty training batch')
    device=next(critic.parameters()).device
    regularizer=make_regularizer()
    critic.requires_grad_(True)
    d.zero_grad(set_to_none=True)
    real=[]; fake=[]
    with torch.no_grad():
        for row in rows:
            neutral=row['neutral'].to(device)
            for pole,scale in enumerate(RECIPE['trained_scales']):
                with network.scaled(scale):
                    # The D phase only needs the music-start vector.
                    pred=backend.hidden(row['ids'])[:,row['prefix_len']-1].float()
                real.append(row['targets'][pole:pole+1].to(device)-neutral)
                fake.append(pred-neutral)
    real=torch.cat(real); fake=torch.cat(fake)
    cap,stats=penalty(regularizer,critic,real,fake,step)
    d_loss=rp_d_loss(critic(real),critic(fake))+cap
    if not torch.isfinite(d_loss): raise FloatingPointError('Non-finite D loss')
    d_loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in critic.parameters()):
        raise FloatingPointError('Non-finite D gradient')
    d.step()
    d.zero_grad(set_to_none=True)
    critic.requires_grad_(False)
    with torch.no_grad(): real_scores=critic(real)
    g.zero_grad(set_to_none=True)
    totals=dict(g_adv=0.,pole=0.,end=0.,cos_pos=0.,cos_neg=0.)
    for i,row in enumerate(rows):
        neutral=row['neutral'].to(device)
        for pole,scale in enumerate(RECIPE['trained_scales']):
            # Keep scale active during checkpoint recomputation in backward.
            with network.scaled(scale):
                pred,margins=forward(backend,row,checkpointing)
                target=row['targets'][pole:pole+1].to(device)
                adv=rp_g_loss(real_scores[2*i+pole:2*i+pole+1],critic(pred-neutral))
                pin=F.mse_loss(pred,target)
                end=F.mse_loss(margins,row['end_teacher'].to(device))
                loss=(.5*adv+pin+.5*end)/len(rows)
                if not torch.isfinite(loss): raise FloatingPointError('Non-finite G loss')
                loss.backward()
            totals['g_adv']+=float(adv.detach())/(2*len(rows))
            totals['pole']+=float(pin.detach())/len(rows)
            totals['end']+=float(end.detach())/(2*len(rows))
            cos=F.cosine_similarity(pred-neutral,target-neutral,dim=-1).mean()
            totals['cos_pos' if pole==0 else 'cos_neg']+=float(cos.detach())/len(rows)
    norm=param_grad_norm(network.parameters())
    if not math.isfinite(norm): raise FloatingPointError('Non-finite G gradient')
    torch.nn.utils.clip_grad_value_(network.parameters(),RECIPE['grad_clip_value'])
    g.step()
    return dict(totals,loss=totals['g_adv']+totals['pole']+totals['end'],
                d_loss=float(d_loss.detach()),d_pen=float(cap.detach()),grad_norm=norm,
                penalty_center=stats['center'])
