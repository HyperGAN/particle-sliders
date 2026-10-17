"""Run only with the product-pinned isolated Diffusers/Transformers source overlay."""
from copy import deepcopy

import pytest
import torch
from particle_sliders import winning_formulation
from particle_sliders.pg09 import ConditionalPG09
pytest.importorskip('qwen_image_sliders')
from qwen_image_sliders.adapters import AdapterBank
from qwen_image_sliders.tiny import TinyRuntime

from conceptmod.textsliders.pg09_backends import ResponseRow
from conceptmod.textsliders.pg09_images import QwenImage21ResponseAdapter, image_condition_features
from test_pg09_images import Critic, equal


def fixture(preset):
    torch.set_num_threads(1)
    runtime = TinyRuntime(seed=23)
    stamp = winning_formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = torch.nn.Parameter(torch.randn(stamp.spec['parts'], stamp.spec['particle_dim']) * .02)
    adapter = QwenImage21ResponseAdapter(runtime, bank, particles)
    def encode(prompt):
        value = runtime.encode(prompt)
        return dict(embeds=value['prompt_embeds'], mask=value['prompt_embeds_mask'],
                    image_pad_mask=value['image_pad_mask'])
    row = ResponseRow(encode('a neutral portrait'), encode('a warm portrait'), encode(''),
                      torch.randn(1, 8, 1, 4, 4), timestep=torch.tensor(500.))
    prepared = [adapter.prepare(row), adapter.prepare(row)]
    conditions = image_condition_features(prepared)
    game = ConditionalPG09(adapter, prepared, Critic(prepared[0].target.numel(), conditions.shape[1]),
        conditions=conditions, particles=particles, response_domain='continuous_field',
        preset=preset, total_steps=5, batch_size=2, seed=37)
    return adapter, runtime, prepared, game


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_actual_qwen21_routed_bank_resume_export(preset):
    adapter, runtime, prepared, game = fixture(preset)
    frozen = {n: p.clone() for n, p in runtime.model.named_parameters()}
    seen = []
    handle = adapter.adapter.layers[0].route.register_forward_pre_hook(
        lambda m, args: seen.append(args[1] is adapter.particles))
    before = adapter.particles.detach().clone()
    for _ in range(2):
        game.step()
    handle.remove()
    assert seen and all(seen)
    assert not torch.equal(before, adapter.particles)
    gradient = torch.autograd.grad(adapter.generated(prepared[0]).sum(), adapter.particles)[0]
    assert torch.count_nonzero(gradient)
    assert prepared[0].target.shape == (1, 8, 1, 4, 4)
    for n, p in runtime.model.named_parameters():
        assert torch.equal(frozen[n], p)
    state = deepcopy(game.state_dict())
    metrics = game.step()
    final = deepcopy(game.state_dict())
    game.load_state_dict(state)
    equal(metrics, game.step())
    equal(final, game.state_dict())
    other = fixture(preset)[-1]
    other.load_export_state_dict(game.export_state_dict())
    equal(game.evaluate(0, seed=91), other.evaluate(0, seed=91))
    assert not torch.cuda.is_initialized()
