"""Opt-in routed particle branches in YuE2 AR attention; paired-error GAN + VIC."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from conceptmod.textsliders import particle_bridge_gan as shared
from conceptmod.textsliders.yue2_arm_b import load_prompts, prepare, RECIPE as OLD_RECIPE
from conceptmod.textsliders.yue2_backend import (
    YuE2Slider, _Adapter, architecture, attention_targets, sound_only, UPSTREAM_REVISION,
)

FORMAT = 'conceptmod-yue2-routed-particle-ar-v1'
RECIPE = dict(OLD_RECIPE, name='anneal-routed-particle-error-yue2-v1',
    generator_objective='paired_error_rpgan_plus_particle_vic',
    parts=128, particle_dim=4, vicreg_weight=1., g_lr=.0006, d_lr=.0009,
    particle_lr=.006, g_optimizer='Adam', g_weight_decay=0., grad_clip_value=None,
    critic_hidden=48, critic_layers=3, penalty_lazy_k=4,
    cap_coordinates='normalized_paired_error_plus_shared_gaussian',
    target_normalization='training_target_per_coordinate_sample_std_floor_1e-4',
    noise_start=1., noise_floor=.03, noise_decay_steps=8000,
    ema=.995, adv_batch=64, schedule='constant',
    prompt_policy='independent_D_G_with_replacement', routing='all_examples',
    particle_vic_batch=64, particle_vic_target_std=1., particle_vic_eps=1e-4,
    adapter_format=FORMAT, adapter_rank=8, adapter_width=48, router_width=16,
    generator_initialization='zero_output_low_rank; standard_normal_particles',
    critic_seed_offset=1000,
    model_glue_reference='df70ccb2ca8f532bdcc07a343fd12bec77362523',
    config_sha256='1ef39a623505691b8710cd37cb768452cd79f6666d109614af297e9d270d88bb',
    propose_only=True, merge_to_trainer=False)


class ParticleProjection(_Adapter):
    def __init__(self, name, module, particles, rank, alpha):
        super().__init__(name, module, multiplier=0., lora_dim=rank, alpha=alpha)
        # The cloud belongs to the root exactly once, not to every projection.
        object.__setattr__(self, '_particles', particles)
        self.bridge = shared.RoutedMLP(rank, rank)

    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        features = self.lora_down(x.to(device=self.lora_down.weight.device, dtype=torch.float32))
        routed = self.bridge(features, self._particles)
        delta = self.lora_up(routed).to(device=x.device, dtype=x.dtype)
        return self.org_forward(x) + delta * (self.multiplier * self.scale)


class ParticleSlider(YuE2Slider):
    def __init__(self, model, rank=8, alpha=8.):
        nn.Module.__init__(self)
        if rank != 8 or alpha != 8.:
            raise ValueError('The experimental particle format pins rank/alpha 8')
        targets = attention_targets(model)
        if any(isinstance(getattr(m.forward, '__self__', None), _Adapter) for m in targets.values()):
            raise ValueError('A YuE2 slider is already attached')
        self.rank, self.alpha = rank, float(alpha)
        self.architecture, self.target_names = architecture(model), list(targets)
        model.requires_grad_(False)
        device = next(model.parameters()).device
        self.particles = nn.Parameter(torch.randn(128, 4, device=device))
        self.adapters = nn.ModuleDict()
        for name, module in targets.items():
            key = name.replace('.', '-')
            adapter = ParticleProjection(key, module, self.particles, rank, alpha)
            adapter.apply_to()
            self.adapters[key] = adapter
        self.to(device=device, dtype=torch.float32)

    def save(self, path, metadata, *, state=None):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, format=FORMAT, rank=self.rank, alpha=self.alpha,
            architecture=self.architecture, targets=self.target_names,
            particles=128, particle_dim=4, bridge_width=48, router_width=16,
            upstream_revision=UPSTREAM_REVISION)
        payload = json.dumps(record, sort_keys=True, ensure_ascii=False); sound_only(payload)
        state = self.state_dict() if state is None else state
        if state.keys() != self.state_dict().keys() or any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Invalid particle-slider export')
        save_file({k: v.detach().float().cpu().contiguous() for k, v in state.items()},
                  str(path), metadata={'conceptmod': payload})
        path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')

    @classmethod
    def load(cls, model, path):
        with safe_open(str(path), framework='pt', device='cpu') as f:
            record = json.loads((f.metadata() or {}).get('conceptmod', '{}'))
        sound_only(json.dumps(record, ensure_ascii=False))
        required = dict(format=FORMAT, rank=8, alpha=8., particles=128, particle_dim=4,
                        bridge_width=48, router_width=16)
        if any(record.get(k) != v for k, v in required.items()) or record.get('dummy'):
            raise ValueError('Incompatible routed-particle checkpoint')
        if record.get('architecture') != architecture(model) or record.get('targets') != list(attention_targets(model)):
            raise ValueError('Particle-slider architecture/targets differ')
        state = load_file(str(path), device='cpu')
        expected = {'particles': (128, 4)}
        for name, module in attention_targets(model).items():
            prefix = 'adapters.' + name.replace('.', '-')
            expected.update({prefix + '.lora_down.weight': (8, module.in_features),
                prefix + '.lora_up.weight': (module.out_features, 8), prefix + '.alpha': ()})
            for branch, inputs, outputs, width in [('router', 8, 4, 16), ('net', 12, 8, 48)]:
                sizes = [inputs, width, width, width, outputs]
                for j, (a, b) in enumerate(zip(sizes, sizes[1:])):
                    expected[f'{prefix}.bridge.{branch}.{j*2}.weight'] = (b, a)
                    expected[f'{prefix}.bridge.{branch}.{j*2}.bias'] = (b,)
        if state.keys() != expected.keys() or any(tuple(state[k].shape) != shape for k, shape in expected.items()):
            raise ValueError('Incomplete routed-particle tensors')
        if any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError('Non-finite routed-particle tensors')
        if any(float(v) != 8. for k, v in state.items() if k.endswith('.alpha')):
            raise ValueError('Particle-slider alpha differs')
        network = cls(model); network.load_state_dict(state, strict=True)
        return network, record


def build_game(backend, network, fixed):
    targets = torch.cat([r['targets'] for r in fixed]).to(next(network.parameters()).device)
    return shared.build_game(network, targets)


def update(backend, network, critic, g, d, rows, *, sampler, step, checkpointing=True):
    device = critic.target_mean.device
    targets = torch.cat([r['targets'] for r in rows]).to(device)
    predictions = {}
    def predict(i, phase):
        row = rows[i]
        with network.scaled(1.):
            pred = backend.hidden(row['ids'], checkpointing=checkpointing and phase == 'g')[:, row['prefix_len']-1].float()
        if phase == 'g': predictions[i] = pred.detach()
        return critic.normalize(pred)
    result = shared.update(network, critic, g, d, targets, predict, sampler=sampler, step=step)
    result['cos_pos'] = sum(float(torch.nn.functional.cosine_similarity(
        predictions[i] - rows[i]['neutral'].to(device),
        targets[i:i+1] - rows[i]['neutral'].to(device), dim=-1).mean())
        for i in result['g_rows']) / len(result['g_rows'])
    return result
