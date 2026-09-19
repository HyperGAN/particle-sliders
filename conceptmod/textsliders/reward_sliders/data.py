"""Explicit generation-span rows on identical baseline/teacher/student histories."""
from __future__ import annotations

import torch

from .capture import ResidualCapture
from .specs import digest, sha


def end_margins(lm, hidden):
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET, _SEMANTIC_VOCAB_SIZE, _AUDIO_END_TOKEN_ID)
    weights = lm.lm_head.weight
    semantic = hidden @ weights[_AUDIO_CODE_OFFSET:_AUDIO_CODE_OFFSET+_SEMANTIC_VOCAB_SIZE].T
    end = hidden @ weights[_AUDIO_END_TOKEN_ID]
    return end.float()-semantic.float().logsumexp(-1)


def generation_hidden(lm, embeds):
    return lm.model(inputs_embeds=embeds, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.long,
                    device=embeds.device), use_cache=False).last_hidden_state


@torch.no_grad()
def prepare_reward_rows(lm, observations, teacher, apply_styles, *, device, frames=128, stride=4):
    """Both branches are independent rows with the exact same forward geometry.

    Baseline and steered targets use one saved trajectory, never separate seed
    histories aligned by index. Prompt/lyric positions are excluded from real.
    The standard merged style weights remain frozen in all three evaluations.
    """
    rows = []
    for observation in observations:
        if observation['split'] != 'train':
            continue
        if observation['status'] != 'complete' or sha(observation['trajectory']) != observation['trajectory_sha256']:
            raise ValueError('Incomplete or changed training trajectory')
        trajectory = torch.load(observation['trajectory'], map_location='cpu', weights_only=True)
        prompt, feedback = trajectory['prompt_embeds'], trajectory['frame_embeds']
        if prompt.shape[0] != 2 or feedback.shape[0] != 2 or feedback.shape[1] < frames:
            raise ValueError('Expected both CFG rows and sufficient saved feedback')
        boundary = prompt.shape[1]
        apply_styles(observation['family'])
        for branch in range(2):
            embeds = torch.cat([prompt[branch:branch+1], feedback[branch:branch+1, :frames]], 1).to(device)
            baseline = generation_hidden(lm, embeds)
            with ResidualCapture(lm, capture=False, prompt_length=boundary, **teacher.hook_kwargs()):
                positive = generation_hidden(lm, embeds)
            if not torch.equal(positive[:, :boundary], baseline[:, :boundary]):
                raise RuntimeError('Activation teacher changed prompt positions')
            neutral = baseline[:, boundary::stride].float()
            real = positive[:, boundary::stride].float()-neutral
            if not torch.isfinite(real).all() or float(real.norm()) <= 0:
                raise ValueError('Generation teacher residual is zero or nonfinite')
            row = dict(family=observation['family'], observation=observation['id'], branch=branch,
                       seed=observation['seed'], trajectory_sha256=observation['trajectory_sha256'],
                       embeds=embeds.cpu(), boundary=boundary, stride=stride,
                       neutral_span=neutral.cpu(), real=real.cpu(), condition=neutral.cpu(),
                       prompt_teacher=baseline[:, :boundary].cpu(),
                       end_teacher=end_margins(lm, baseline[:, boundary-1::stride]).cpu(),
                       fixture=digest([observation['trajectory_sha256'], teacher.layer, teacher.coefficient,
                                       branch, frames, stride]))
            rows.append(row)
        print(f"PREPARED {observation['id']} both CFG branches, {frames} shared feedback positions", flush=True)
    return rows


def set_scale(network, value):
    for module in network.unet_loras:
        module.multiplier = value


class RewardStudentForward:
    def __init__(self, lm, network, apply_styles, device):
        self.lm, self.network, self.apply_styles, self.device = lm, network, apply_styles, device

    def __call__(self, row, with_policy=False):
        if with_policy:
            raise ValueError('This baseline arm does not enable an extra policy loss')
        self.apply_styles(row['family'])
        set_scale(self.network, 1.)
        hidden = generation_hidden(self.lm, row['embeds'].to(self.device))
        fake = hidden[:, row['boundary']::row['stride']].float()-row['neutral_span'].to(self.device)
        result = dict(fake=fake)
        if torch.is_grad_enabled():
            result['end'] = end_margins(self.lm, hidden[:, row['boundary']-1::row['stride']])
        return result


class FamilySampler:
    """Equal family exposure; homogeneous batches avoid changing style mid-update."""
    def __init__(self, rows, batch=4, seed=7):
        self.groups = {}
        for index, row in enumerate(rows):
            self.groups.setdefault(row['family'], []).append(index)
        if any(len(v) % batch for v in self.groups.values()):
            raise ValueError('Each family must have complete, equal-sized batches')
        if len({len(v) for v in self.groups.values()}) != 1:
            raise ValueError('Training families must receive equal sampling mass')
        self.batch, self.queue = batch, []
        self.generator = torch.Generator().manual_seed(seed)

    def next(self):
        if not self.queue:
            names = sorted(self.groups)
            for j in torch.randperm(len(names), generator=self.generator).tolist():
                group = self.groups[names[j]]
                order = [group[i] for i in torch.randperm(len(group), generator=self.generator).tolist()]
                self.queue.extend([order[k:k+self.batch] for k in range(0,len(order),self.batch)])
        return self.queue.pop(0)

    def state_dict(self):
        return dict(groups={k:list(v) for k,v in self.groups.items()}, batch=self.batch,
                    queue=[list(v) for v in self.queue], rng=self.generator.get_state().clone())

    def load_state_dict(self, state):
        if state['groups'] != self.groups or state['batch'] != self.batch:
            raise ValueError('Family sampler geometry changed')
        self.queue = [list(v) for v in state['queue']]
        self.generator.set_state(state['rng'])
