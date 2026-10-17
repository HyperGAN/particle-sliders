"""CPU software qualification using actual tiny Diffusers models, no Hub loads."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from conceptmod.textsliders.pg09_backends import ResponseRow, FlowSchedule
from conceptmod.textsliders.pg09_images import (
    ImageResponseAdapter, image_lora_parameters, image_condition_features,
)
from particle_sliders.pg09 import ConditionalPG09

FAMILIES = ('sana', 'zimage', 'sd', 'sdxl', 'sd3', 'flux', 'krea2', 'cascade')


def fixture(family, *, checkpoint=False):
    import diffusers as d
    torch.set_num_threads(1)
    torch.manual_seed(23)
    kwargs = {}
    latent = torch.randn(1, 4, 4, 4)
    condition = dict(embeds=torch.randn(1, 3, 16), mask=torch.tensor([[True, True, False]]))
    if family == 'sana':
        model = d.SanaTransformer2DModel(in_channels=4, out_channels=4, num_attention_heads=2,
            attention_head_dim=8, num_layers=1, num_cross_attention_heads=2,
            cross_attention_head_dim=8, cross_attention_dim=16, caption_channels=16,
            sample_size=4, patch_size=1)
    elif family == 'zimage':
        model = d.ZImageTransformer2DModel(in_channels=4, dim=32, n_layers=1,
            n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=16,
            axes_dims=[4, 6, 6], axes_lens=[128, 64, 64])
        condition['embeds'] = condition['embeds'][0]
    elif family in ('sd', 'sdxl'):
        config = dict(sample_size=4, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
            block_out_channels=(16,), layers_per_block=1, norm_num_groups=4,
            cross_attention_dim=16, attention_head_dim=4)
        if family == 'sdxl':
            config.update(addition_embed_type='text_time', addition_time_embed_dim=4,
                          projection_class_embeddings_input_dim=32)
            condition.update(pooled=torch.randn(1, 8), time_ids=torch.tensor([[32., 32., 0., 0., 32., 32.]]))
        model = d.UNet2DConditionModel(**config)
        kwargs['scheduler'] = d.DDIMScheduler(prediction_type='epsilon')
    elif family == 'sd3':
        model = d.SD3Transformer2DModel(sample_size=4, patch_size=2, in_channels=4,
            out_channels=4, num_layers=1, attention_head_dim=8, num_attention_heads=2,
            joint_attention_dim=16, caption_projection_dim=16, pooled_projection_dim=8,
            pos_embed_max_size=4)
        condition['pooled'] = torch.randn(1, 8)
    elif family == 'flux':
        model = d.FluxTransformer2DModel(in_channels=16, out_channels=16, num_layers=1,
            num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
            joint_attention_dim=16, pooled_projection_dim=8, axes_dims_rope=(4, 6, 6))
        condition.update(pooled=torch.randn(1, 8), text_ids=torch.zeros(3, 3),
                         image_ids=d.FluxPipeline._prepare_latent_image_ids(1, 2, 2, 'cpu', torch.float32))
    elif family == 'krea2':
        model = d.Krea2Transformer2DModel(in_channels=16, num_layers=1, attention_head_dim=16,
            num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
            timestep_embed_dim=32, text_hidden_dim=16, num_text_layers=2,
            text_num_attention_heads=2, text_num_key_value_heads=1, text_intermediate_size=32,
            num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(4, 6, 6))
        # Native static geometry methods need only these pipeline properties; no VAE/encoder loading.
        pipe = object.__new__(d.Krea2Pipeline)
        pipe.patch_size = 2
        pipe.vae_scale_factor = 8
        pipe.transformer = model
        kwargs['pipeline'] = pipe
        latent = latent.unsqueeze(2)
        condition['embeds'] = torch.randn(1, 3, 2, 16)
        condition['position_ids'] = d.Krea2Pipeline.prepare_position_ids(3, 2, 2, 'cpu')
    else:
        model = d.StableCascadeUNet(in_channels=4, out_channels=4,
            timestep_ratio_embedding_dim=16, conditioning_dim=16,
            block_out_channels=(16,), num_attention_heads=(2,),
            down_num_layers_per_block=(1,), up_num_layers_per_block=(1,),
            down_blocks_repeat_mappers=(1,), up_blocks_repeat_mappers=(1,),
            block_types_per_layer=(('SDCascadeResBlock','SDCascadeTimestepBlock','SDCascadeAttnBlock'),),
            clip_text_in_channels=16, clip_text_pooled_in_channels=8, clip_image_in_channels=8,
            dropout=0.)
        condition.update(pooled=torch.randn(1, 1, 8), image_embeds=torch.zeros(1, 1, 8))
    model.requires_grad_(False)
    if checkpoint:
        model.enable_gradient_checkpointing()
    target = next(n for n, m in model.named_modules() if isinstance(m, nn.Linear) and
                  (n.endswith('to_q') or n.endswith('q_proj')))
    # Only a native attention projection is adapted, so every optimizer leaf participates.
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, target_modules=[target]))
    if family == 'krea2':
        object.__setattr__(kwargs['pipeline'], 'transformer', model)
    network = image_lora_parameters(model)
    adapter = ImageResponseAdapter(family, model, network, **kwargs)
    positive = deepcopy(condition)
    positive['embeds'] = positive['embeds'] + .15
    negative = deepcopy(condition)
    negative['embeds'] = negative['embeds'] - .1
    row = ResponseRow(condition, positive, negative, latent,
                      timestep=torch.tensor(.5 if family == 'cascade' else 500.), guidance=1.)
    return adapter, row, model


class Critic(nn.Module):
    def __init__(self, size, conditions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(size + conditions, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, response, condition):
        return self.net(torch.cat([response.flatten(1), condition], dim=1)).squeeze(-1)


def game_fixture(family, preset='bispf_optimized', *, checkpoint=False):
    adapter, row, model = (iris_fixture() if family == 'iris' else fixture(family, checkpoint=checkpoint))
    rows = [adapter.prepare(row), adapter.prepare(replace(row, latent=row.latent * .95))]
    conditions = image_condition_features(rows)
    game = ConditionalPG09(adapter, rows, Critic(rows[0].target.numel(), conditions.shape[1]),
        conditions=conditions, particles=(adapter.particles if family == 'iris' else None), response_domain='continuous_field',
        preset=preset, total_steps=5, batch_size=2, seed=37)
    return adapter, row, model, rows, game


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('family', FAMILIES)
def test_native_response_and_frozen_inputs(family):
    adapter, row, model = fixture(family)
    prepared = adapter.prepare(row)
    assert prepared.target.shape == row.latent.shape
    assert not prepared.target.requires_grad and not prepared.neutral.requires_grad
    expected = prepared.row.neutral['embeds'].clone()
    row.neutral['embeds'].add_(17)
    assert torch.equal(expected, prepared.row.neutral['embeds'])
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(.01)
    generated = adapter.generated(prepared)
    grads = torch.autograd.grad(generated.square().sum(), list(adapter.network.parameters()), allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    assert not torch.equal(generated, prepared.neutral)
    assert adapter.network.scale == 1.
    assert all(not p.requires_grad for n, p in model.named_parameters() if 'lora_' not in n)


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('family', FAMILIES)
def test_shared_gan_update_exact_resume_and_export(family, preset):
    adapter, row, model, rows, game = game_fixture(family, preset)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = [p.detach().clone() for p in adapter.network.parameters()]
    first = game.step()
    assert all(torch.isfinite(torch.tensor(v)) for v in first.values() if isinstance(v, (int, float)))
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.network.parameters()))
    saved = deepcopy(game.state_dict())
    metrics = game.step()
    continued = deepcopy(game.state_dict())
    game.load_state_dict(saved)
    equal(metrics, game.step())
    equal(continued, game.state_dict())
    for n, p in model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n])
    export = deepcopy(game.export_state_dict())
    output = game.evaluate(0, seed=91)
    other = game_fixture(family, preset)[-1]
    other.load_export_state_dict(export)
    equal(output, other.evaluate(0, seed=91))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', FAMILIES)
def test_coordinate_rejections_and_conditions(family):
    adapter, row, _ = fixture(family)
    fixed = adapter.prepare(row)
    context = image_condition_features([fixed])
    changed = replace(row, positive={**row.positive, 'embeds': row.positive['embeds'] + .2})
    assert not torch.equal(context, image_condition_features([adapter.prepare(changed)]))
    with pytest.raises(ValueError):
        adapter.prepare(replace(row, latent=row.latent.flatten(1)))
    with pytest.raises(ValueError):
        adapter.prepare(replace(row, timestep=torch.tensor(1001.)))
    with pytest.raises(ValueError):
        adapter.prepare(replace(row, scale=-1.))
    if family in ('sd', 'sdxl', 'cascade'):
        with pytest.raises(ValueError, match='scheduler trajectories'):
            adapter.prepare(replace(row, timestep=None, schedule=FlowSchedule.linear(2, time_scale=1000)))


def iris_fixture():
    pytest.importorskip('iris_sliders')
    from iris_sliders.tiny import TinyRuntime
    from iris_sliders.adapters import AdapterBank, new_particles
    from particle_sliders import winning_formulation
    from conceptmod.textsliders.pg09_images import IrisResponseAdapter
    torch.set_num_threads(1)
    runtime = TinyRuntime(seed=23)
    stamp = winning_formulation()
    bank = AdapterBank(runtime.model, stamp)
    particles = new_particles(stamp)
    adapter = IrisResponseAdapter(runtime, bank, particles)
    def condition(prompt):
        value = runtime.encode(prompt)
        return dict(embeds=value.embeddings, mask=value.mask)
    row = ResponseRow(condition('neutral portrait'), condition('warm portrait'), condition(''),
                      torch.randn(1, 3, 8, 8), timestep=torch.tensor(500.), guidance=3.)
    return adapter, row, runtime.model


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_native_iris_actual_routed_bank(preset):
    adapter, row, model, rows, game = game_fixture('iris', preset)
    assert adapter.particles is adapter.network.particles
    assert sum(p is adapter.particles for p in adapter.network.parameters()) == 1
    # The product's output hooks use the identical particle object in the actual model call.
    observed = []
    handle = adapter.adapter.layers[0].route.register_forward_pre_hook(
        lambda m, args: observed.append(args[1] is adapter.particles))
    before = adapter.particles.detach().clone()
    for _ in range(2):
        game.step()
    handle.remove()
    assert observed and all(observed)
    assert not torch.equal(before, adapter.particles)
    gradient = torch.autograd.grad(adapter.generated(rows[0]).square().sum(), adapter.particles)[0]
    assert torch.count_nonzero(gradient) and torch.isfinite(gradient).all()
    state = deepcopy(game.state_dict())
    metrics = game.step()
    final = deepcopy(game.state_dict())
    game.load_state_dict(state)
    equal(metrics, game.step())
    equal(final, game.state_dict())
    export = game.export_state_dict()
    other = game_fixture('iris', preset)[-1]
    other.load_export_state_dict(export)
    equal(game.evaluate(0, seed=91), other.evaluate(0, seed=91))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['sana', 'zimage', 'sd3', 'flux', 'krea2'])
def test_shared_cfg_render_loop_and_scale_zero(family):
    adapter, row, _ = fixture(family)
    adapter.decoder = lambda x: x
    row = replace(row, timestep=None, schedule=FlowSchedule.linear(2, time_scale=1000), guidance=3.)
    fixed = adapter.prepare(row)
    expected = adapter.generated(fixed).detach()
    assert torch.equal(adapter.render(row, scale=1.), expected)
    assert torch.equal(adapter.render(row, scale=0.), fixed.neutral)
    with pytest.raises(ValueError, match='VAE decoder'):
        adapter.decoder = None
        adapter.render(row)


def test_existing_conceptmod_lora_binding():
    from conceptmod.textsliders.lora import LoRANetwork
    from diffusers import UNet2DConditionModel, DDIMScheduler
    model = UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D',), up_block_types=('CrossAttnUpBlock2D',),
        block_out_channels=(16,), layers_per_block=1, norm_num_groups=4,
        cross_attention_dim=16, attention_head_dim=4).eval().requires_grad_(False)
    lora = LoRANetwork(model, rank=2)
    network = image_lora_parameters(model, lora)
    adapter = ImageResponseAdapter('sd', model, network, scheduler=DDIMScheduler())
    c = dict(embeds=torch.randn(1, 3, 16))
    row = ResponseRow(c, {**c, 'embeds': c['embeds']+.1}, c,
                      torch.randn(1, 4, 4, 4), timestep=torch.tensor(500.))
    value = adapter.generated(adapter.prepare(row))
    grads = torch.autograd.grad(value.sum(), list(network.parameters()), allow_unused=True)
    assert all(g is not None for g in grads)
    assert {id(p) for p in network.parameters()} == {id(p) for p in lora.parameters()}
    before = [m.multiplier for m in lora.unet_loras]
    with network.scaled(.3):
        with network.scaled(0):
            assert all(m.multiplier == 0 for m in lora.unet_loras)
    assert before == [m.multiplier for m in lora.unet_loras]


@pytest.mark.parametrize('family', ['flux', 'krea2'])
def test_bf16_host_uses_native_timestep_rounding(family):
    adapter, row, model = fixture(family)
    model.to(torch.bfloat16)
    for p in adapter.network.parameters():
        p.data = p.data.float()
    observed = []
    handle = model.register_forward_pre_hook(lambda m, args, kwargs: observed.append(kwargs['timestep'].clone()),
                                             with_kwargs=True)
    row = replace(row, timestep=torch.tensor(733.7))
    adapter.prepare(row)
    handle.remove()
    expected = (row.timestep.to(torch.bfloat16) / 1000. if family == 'flux'
                else (row.timestep / 1000.).to(torch.bfloat16))
    assert observed and all(torch.equal(t, expected.reshape(1)) for t in observed)


def test_krea2_product_binds_actual_lora_without_loader():
    pytest.importorskip('krea2')
    from krea2.live import LiveKrea2BboxBackend
    adapter, row, model = fixture('krea2')
    product = object.__new__(LiveKrea2BboxBackend)
    product.pipe = adapter.pipeline
    product.pipe.text_encoder = None
    rebound = ImageResponseAdapter.from_krea2_product(product, model_identity={'fixture':'native tiny Krea2'})
    assert {id(p) for p in rebound.network.parameters()} == {id(p) for p in product.trainable_parameters()}
    assert torch.equal(adapter.generated(adapter.prepare(row)), rebound.generated(rebound.prepare(row)))


@pytest.mark.parametrize('family', ['sana', 'zimage', 'sd3', 'flux', 'krea2'])
def test_native_checkpoint_replays_shared_protected_gradients(family):
    adapter, _, model, _, game = game_fixture(family, checkpoint=True)
    assert model.is_gradient_checkpointing
    before = [p.detach().clone() for p in adapter.network.parameters()]
    for _ in range(2):
        game.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, adapter.network.parameters()))
    assert adapter.network.scale == 1.
