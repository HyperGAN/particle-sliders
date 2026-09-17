"""Unipolar YuE2 RpGAN with the verified Arm B discriminator gradient cap.

Train +1 toward the raw positive caption; scale 0 is the exact base model.
The generator has only the adversarial loss. There are no auxiliary losses.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch
import torch.nn.functional as F
import yaml

from analysis.slider2d.adv import make_grad_regularizer, rp_d_loss, rp_g_loss
from conceptmod.textsliders.lm_adv import LMDiscriminator, param_grad_norm
from conceptmod.textsliders.slider_targets import lm_faithful_plus_neu
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B
from conceptmod.textsliders.yue2_backend import sound_only

# Explicitly retain the GAN and cap, not Arm B's bipolar teacher/auxiliary pins.
RECIPE = dict(ARM_B, name='unipolar-rpgan-bcap-yue2-v3',
    lm_target='faithful_plus_neu', polarity='unipolar', adv_weight=1., end_weight=0.,
    pole_weight=0., cover_weight=0.,
    lyrichold_weight=0., plan_weight=0., anchor_weight=0.,
    critic_hidden=256, critic_layers=2, adv_in='scaled', adv_batch=4,
    g_lr=.0005, d_lr=.00075, betas=[0.,.999], g_weight_decay=1e-6,
    schedule='constant', grad_clip_value=1., trained_scales=[1.],
    zero_behavior='exact_base_by_adapter_scale', recommended_range=[0.,1.],
    penalty_method='autograd', penalty_lazy_k=1, penalty_anneal='none',
    cap_coordinates='fixed_teacher_rms', generator_objective='positive_rpgan_only',
    critic_positions='music_start_only', phase_changes=[],
    prompt_policy='balanced_shuffled_passes', history_policy='none_prompt_states_only')


def load_prompts(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw,dict) or not isinstance(raw.get('rows'),list) or not raw['rows']:
        raise ValueError('Prompts need a nonempty rows list')
    allowed={'rows','plus_label','zero_label','recommended_range'}
    if set(raw)-allowed: raise ValueError(f'Unsupported prompt metadata: {set(raw)-allowed}')
    meta={k:v for k,v in raw.items() if k!='rows'}
    if meta.get('recommended_range',[0,1]) != [0,1]:
        raise ValueError('Unipolar Arm B uses the range [0, 1]')
    for v in meta.values():
        if isinstance(v,str): sound_only(v)
    rows=[]
    for item in raw['rows']:
        if not isinstance(item,dict) or set(item)!={'neutral','positive','lyrics'}:
            raise ValueError('Each unipolar row must contain neutral, positive, lyrics only; no negative teacher')
        if any(not isinstance(v,str) or not v.strip() for v in item.values()):
            raise ValueError('Prompt values must be nonempty strings')
        row={k:sound_only(v.strip()) for k,v in item.items()}
        if row['neutral']==row['positive']:
            raise ValueError('Unipolar Arm B requires distinct neutral and positive captions')
        rows.append(row)
    return rows,meta


@torch.no_grad()
def prepare(backend,rows,meta,max_seq_len):
    limit=min(max_seq_len,backend.model.config.max_position_embeddings)
    def encode(style,lyrics):
        ids=backend.prefix(style,lyrics)
        if len(ids)>limit: raise ValueError('Prompt exceeds context limit')
        return ids,backend.hidden(ids)[:,-1].float()
    prepared=[]
    for row in rows:
        ids,neu=encode(row['neutral'],row['lyrics'])
        _,pos=encode(row['positive'],row['lyrics'])
        # The shared UNI helper ignores its legacy negative argument.
        # Pass the already encoded neutral, never encode an opposite caption.
        plus=lm_faithful_plus_neu(pos,neu,neu)
        prepared.append(dict(prefix=ids,prefix_len=len(ids),neutral=neu.cpu(),
            targets=plus.cpu(),guard_applied=False,target_shift=0.,raw_targets=pos.cpu()))
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


def make_regularizer(grad_arm='b_cap'):
    if grad_arm not in ('b_cap', 'a_r1r2'):
        raise ValueError('Unsupported YuE2 discriminator regularizer')
    return make_grad_regularizer(arm=grad_arm,coeff=RECIPE['adv_b_cap'],kappa=RECIPE['adv_reg_kappa'],
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
    return hidden[:,index].float()


def update(backend,network,critic,g,d,rows,*,step,checkpointing=True,grad_arm='b_cap'):
    if not rows: raise ValueError('Empty training batch')
    device=next(critic.parameters()).device
    regularizer=make_regularizer(grad_arm)
    critic.requires_grad_(True)
    d.zero_grad(set_to_none=True)
    real=[]; fake=[]
    with torch.no_grad():
        for row in rows:
            neutral=row['neutral'].to(device)
            with network.scaled(1.):
                # The D phase only needs the music-start vector.
                pred=backend.hidden(row['ids'])[:,row['prefix_len']-1].float()
            real.append(row['targets'].to(device)-neutral)
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
    totals=dict(g_adv=0.,cos_pos=0.)
    for i,row in enumerate(rows):
        neutral=row['neutral'].to(device)
        # Keep +1 active during checkpoint recomputation in backward.
        with network.scaled(1.):
            pred=forward(backend,row,checkpointing)
            target=row['targets'].to(device)
            adv=rp_g_loss(real_scores[i:i+1],critic(pred-neutral))
            loss=adv/len(rows)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite G loss')
            loss.backward()
        totals['g_adv']+=float(adv.detach())/len(rows)
        cos=F.cosine_similarity(pred-neutral,target-neutral,dim=-1).mean()
        totals['cos_pos']+=float(cos.detach())/len(rows)
    norm=param_grad_norm(network.parameters())
    if not math.isfinite(norm): raise FloatingPointError('Non-finite G gradient')
    torch.nn.utils.clip_grad_value_(network.parameters(),RECIPE['grad_clip_value'])
    g.step()
    return dict(totals,loss=totals['g_adv'],
                d_loss=float(d_loss.detach()),d_pen=float(cap.detach()),grad_norm=norm,
                penalty_center=stats['center'])
