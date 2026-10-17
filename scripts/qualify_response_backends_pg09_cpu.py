"""CPU-only adapter/core qualification. No model downloads or GPU admission."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from conceptmod.textsliders.pg09_backends import (  # noqa: E402
    AnimaResponseAdapter, SupraResponseAdapter, MossResponseAdapter,
    FlowSchedule, ResponseRow, load_adapter, save_adapter, response_signature,
    AnimaRuntimeResponseAdapter,
)


def fixture(family):
    torch.manual_seed(9)
    if family == 'anima-particles':
        from types import SimpleNamespace
        from diffusers import CosmosTransformer3DModel, FlowMatchEulerDiscreteScheduler
        from lumen_studio.backends.anima import TurboRuntime
        from lumen_studio.particles import ParticleAdapter, Mixer
        model = CosmosTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=32,
            num_layers=1, mlp_ratio=2, text_embed_dim=8, adaln_lora_dim=4,
            max_size=(2, 4, 4), patch_size=(1, 1, 1), extra_pos_embed_type=None,
        ).to(dtype=torch.bfloat16).eval().requires_grad_(False)
        particle_adapter = ParticleAdapter(model)
        runtime = TurboRuntime.__new__(TurboRuntime)  # No loader, model download or CUDA call.
        runtime.device, runtime.dtype = torch.device('cpu'), torch.bfloat16
        runtime.transformer, runtime.scale = model, 1
        runtime.identity = dict(model='real Cosmos architecture at tiny random dimensions', pretrained=False)
        runtime.pipe = SimpleNamespace(scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000))
        runtime.mixer = Mixer(model)
        runtime.mixer.add('fixture', particle_adapter)
        runtime.encode = lambda prompt: torch.full((1, 2, 8), len(prompt) / 10.)
        adapter = AnimaRuntimeResponseAdapter(runtime, particle_adapter, 'fixture')
        row = ResponseRow('neutral', 'positive longer', '', torch.randn(1, 4, 1, 2, 2),
                          timestep=torch.tensor(500.))
    elif family == 'moss':
        from moss_sliders.lora import Slider
        from moss_sliders.smoke import tiny_model
        model = tiny_model()
        adapter = MossResponseAdapter(model, Slider(model, rank=2))
        contexts = [(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool)) for _ in range(3)]
        row = ResponseRow(*contexts, torch.randn(1, 4, 2, 2), timestep=torch.tensor(500.))
    else:
        if family == 'anima':
            from conceptmod.textsliders.anima_fake import FakeAnimaBackend
            backend = FakeAnimaBackend(rank=2, seed=9, lora_targets='dit')
            owned = {id(p) for p in backend.trainable_parameters()}
            for p in backend.transformer.parameters():
                p.requires_grad_(id(p) in owned)
            adapter = AnimaResponseAdapter(backend)
            schedule = FlowSchedule.linear(2, time_scale=1000.)
        else:
            from conceptmod.textsliders.supra_fake import FakeSupraBackend
            backend = FakeSupraBackend(rank=2, seed=9)
            adapter = SupraResponseAdapter(backend)
            schedule = FlowSchedule.supra(2)
        row = ResponseRow('a woman neutral', 'a woman smiling', '',
                          torch.randn(1, *backend.latent_shape), schedule=schedule, guidance=3.)
    rows = [row, replace(row, latent=row.latent * .9 + .1)]
    if family == 'moss':
        rows += [replace(r, scale=-1.) for r in rows]
    prepared = [adapter.prepare(r) for r in rows]
    conditions = torch.tensor([[float(i), r.scale, r.guidance,
                                float(r.timestep) if r.timestep is not None else r.schedule.timesteps[0]]
                               for i, r in enumerate(rows)])
    return adapter, prepared, conditions


class Critic(torch.nn.Module):
    """Example critic: flatten internally only; PG receives native image fields."""
    def __init__(self, shape, condition_width):
        super().__init__()
        width = torch.empty(shape).numel() + condition_width
        self.net = torch.nn.Sequential(torch.nn.Linear(width, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1))

    def forward(self, response, condition):
        return self.net(torch.cat((response.flatten(1), condition), dim=1)).squeeze(-1)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def qualify(family, directory):
    from particle_sliders.pg09 import ConditionalPG09
    def make():
        backend, rows, conditions = fixture(family)
        critic = Critic(rows[0].target.shape[1:], conditions.shape[1])
        game = ConditionalPG09(backend, rows, critic, conditions=conditions,
                              particles=getattr(backend, 'particles', None), preset='bispf_optimized',
                              response_domain='continuous_field', batch_size=4, total_steps=5)
        return backend, rows, game
    backend, rows, game = make()
    metrics = [game.step(), game.step()]
    state = deepcopy(game.state_dict())
    directory.mkdir(parents=True, exist_ok=True)
    export = directory / f'{family}.safetensors'
    save_adapter(backend, export, model_fingerprint=f'CPU-{family}-seed9')
    samples = [backend.generated(r).detach().clone() for r in rows]
    expected = game.step()
    other, other_rows, restored = make()
    restored.load_state_dict(state)
    assert equal(expected, restored.step())
    assert equal(game.state_dict(), restored.state_dict())
    load_adapter(other, export, model_fingerprint=f'CPU-{family}-seed9')
    assert all(torch.equal(value, other.generated(r)) for value, r in zip(samples, other_rows))
    assert not torch.cuda.is_initialized()
    return dict(backend=backend.metadata(), response_shape=list(rows[0].target.shape),
                response_signature=response_signature(backend, rows, model_fingerprint=f'CPU-{family}-seed9'),
                recipe=asdict(game.recipe), metrics=metrics + [expected],
                objective='PG adversarial + required PG constraints; no MSE',
                actual_particle_bank=hasattr(backend, 'particles'),
                qualification=('existing routed-bank GAN software' if hasattr(backend, 'particles')
                               else 'adapter-GAN software only'),
                exact_resume=True, exact_adapter_reload=True, quality_qualified=False,
                cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['all', 'anima', 'supra', 'moss', 'anima-particles'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU qualification requires an uninitialized CUDA runtime')
    families = ['anima', 'supra', 'moss'] if args.family == 'all' else [args.family]
    receipt = {family: qualify(family, args.output) for family in families}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({family: {k: r[k] for k in ('response_shape', 'exact_resume', 'exact_adapter_reload')}
                      for family, r in receipt.items()}, indent=2))


if __name__ == '__main__':
    main()
