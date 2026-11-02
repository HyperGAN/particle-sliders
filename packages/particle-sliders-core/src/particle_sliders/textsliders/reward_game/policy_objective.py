"""Clipped code-policy surrogate for a frozen batch from one recorded parent.

Code probabilities omit sampler truncation. This is not an exact sampled
waveform likelihood or an unbiased policy-gradient estimator for deployment.
"""
import torch


def selected_frame_logp(semantic_logp,residual_logp,codes):
    codes=codes.to(semantic_logp.device).long()
    semantic=semantic_logp.gather(-1,codes[:,0,None]).squeeze(-1)
    residual=residual_logp.gather(-1,codes[:,1:,None]).squeeze(-1).sum(-1)
    return semantic+residual


def clipped_loss(current,reference,advantage,clip=.2):
    reference=reference.to(current).detach()
    ratio=(current-reference).clamp(-20,20).exp()
    advantage=torch.as_tensor(advantage,device=current.device,dtype=current.dtype)
    return -torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage).mean()


def balanced_advantages(episodes,min_gap=.05):
    """Two parent draws per family; tiny score gaps contribute no policy gradient."""
    from collections import defaultdict
    import math
    groups=defaultdict(list)
    for row in episodes:
        if not math.isfinite(row['ce']):raise ValueError('Nonfinite training reward')
        groups[row['family']].append(row)
    result={}
    for family,rows in groups.items():
        if len(rows)!=2 or len({r['seed'] for r in rows})!=2:raise ValueError('Two distinct parent draws per family required')
        gap=abs(rows[0]['ce']-rows[1]['ce'])
        mean=(rows[0]['ce']+rows[1]['ce'])/2
        for row in rows:
            result[row['id']]=0. if gap<min_gap else (row['ce']-mean)/(gap/2)
    return result
