"""CPU-only native audio fixtures and fresh-process replay receipt driver.

YuE2 uses the discovered product's TinyYuE2 + ParticleSlider. Music3 uses an
actual random Qwen3 causal model + the existing Music3 ParticleNetwork. No
pretrained weights, audio decoding, legacy training loop or CUDA are used.
"""
from copy import deepcopy
import argparse
import os
from pathlib import Path

import torch
from torch import nn

from conceptmod.textsliders.pg09_audio import (
    AudioRow, Music3ARResponseAdapter, YuE2ARResponseAdapter, audio_conditions,
)


class AudioCritic(nn.Module):
    def __init__(self, width, condition_width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width + condition_width, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, response, condition):
        return self.net(torch.cat((response, condition), dim=1)).squeeze(-1)


def make_audio(family, *, preset='bispf_optimized', checkpointing=False, width=32,
               pg_source_profile='release', penalty_input_domain='auto', critic_class=AudioCritic):
    from particle_sliders import ConditionalPG09, winning_formulation
    torch.set_num_threads(1)
    torch.manual_seed(23)
    if family == 'yue2':
        native_product = os.environ.get('PG09_YUE2_PRODUCT')
        if native_product:
            from inspect_audio_pg09_assets import bootstrap_yue2_product
            bootstrap_yue2_product(native_product)
        from yue2.backend import YuE2Backend
        from yue2.slider import ParticleSlider
        backend = YuE2Backend(dummy=True)
        if native_product:
            from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
            backend.model = YuE2ForCausalLM(YuE2Config(
                hidden_size=width, num_hidden_layers=2, num_attention_heads=2,
                num_key_value_heads=1, head_dim=16, intermediate_size=64,
                vocab_size=256, max_position_embeddings=256,
                max_latent_frames=64)).eval().requires_grad_(False)
            backend.dummy = False
        elif width != 32:
            raise ValueError('wide YuE2 fixture requires the official native runtime')
        model = backend.model
        slider = ParticleSlider(model, winning_formulation())
        modules = (list(model.model.layers) if native_product else [m for layer in model.model.layers for m in (
            layer.self_attn.q_proj, layer.self_attn.k_proj,
            layer.self_attn.v_proj, layer.self_attn.o_proj)]) if checkpointing else []
        adapter = YuE2ARResponseAdapter(backend, slider,
            model_identity={'fixture': ('official YuE2 random weights' if native_product
                                        else 'product TinyYuE2 random weights'), 'seed': 23},
            pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain,
            checkpoint_modules=modules)
    elif family == 'music3':
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from conceptmod.textsliders.music3_particle_bridge import ParticleNetwork
        model = Qwen3ForCausalLM(Qwen3Config(
            hidden_size=width, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            vocab_size=256, attention_dropout=0., use_cache=False,
            attn_implementation='eager')).eval().requires_grad_(False)
        slider = ParticleNetwork(model)
        adapter = Music3ARResponseAdapter(model, slider,
            model_identity={'fixture': 'native Qwen3 random weights', 'seed': 23},
            pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain,
            checkpoint_modules=list(model.model.layers) if checkpointing else [])
    else:
        raise ValueError(family)
    # Fixed shared histories; distinct neutral/positive captions share each suffix.
    rows = [AudioRow((11, 13, 17, 19), (29, 31, 17, 19), history_tokens=2),
            AudioRow((23, 37, 41, 43, 47), (53, 59, 41, 43, 47), history_tokens=3)]
    prepared = [adapter.prepare(row) for row in rows]
    conditions = audio_conditions(prepared)
    critic = critic_class(width, conditions.shape[1])
    game = ConditionalPG09(adapter, prepared, critic, conditions=conditions,
        particles=adapter.particles, response_domain='continuous_vector',
        preset=preset, batch_size=2, total_steps=4, seed=71,
        pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
    return adapter, model, prepared, game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--state-in', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=('all', 'yue2', 'music3'), default='all')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--pg-source-profile', default='release')
    parser.add_argument('--penalty-input-domain', default='auto')
    args = parser.parse_args()
    previous = torch.load(args.state_in, weights_only=False) if args.state_in else {}
    receipt = {}
    for family in (('yue2', 'music3') if args.family == 'all' else (args.family,)):
        for preset in ('bispf', 'bispf_optimized'):
            key = family + '/' + preset
            adapter, model, fixed, game = make_audio(family, preset=preset, checkpointing=True,
                width=args.width, pg_source_profile=args.pg_source_profile,
                penalty_input_domain=args.penalty_input_domain)
            if previous:
                game.load_state_dict(previous[key]['state'])
            logs = deepcopy(previous[key]['metrics']) if previous else []
            for _ in range(args.steps):
                logs.append(game.step())
            sample = game.evaluate(0, seed=17)
            receipt[key] = dict(state=game.state_dict(), metrics=logs, sample=sample,
                                export=game.export_state_dict(), metadata=adapter.metadata())
    assert not torch.cuda.is_initialized()
    torch.save(receipt, args.output)


if __name__ == '__main__':
    main()
