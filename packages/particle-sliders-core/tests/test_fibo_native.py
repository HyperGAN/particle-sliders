"""CPU/mocked Fibo EDIT contracts plus real released PG updates; no model weights."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import nn

from particle_sliders.fibo_native import (
    DEFAULTS, FiboCachedRows, FiboEditResponseAdapter, FiboParticleSlider,
    build_fibo_game, target_modules, write_cache,
)
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_images import image_condition_features
from particle_sliders.pg09_moss_campaign import CampaignFieldCritic

IDENTITY = dict(qualification='CPU fixture only', fixture='bounded Fibo EDIT protocol')


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = [nn.Linear(4, 4, bias=False) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False)])

    def forward(self, value):
        return self.to_out[0](torch.sin(self.to_q(value) + self.to_k(value) + self.to_v(value)))


class Block(nn.Module):
    def __init__(self, cross=False):
        super().__init__()
        self.attention = Attention()
        self.cross_attention = Attention() if cross else None

    def forward(self, value):
        value = value + self.attention(value)
        return value if self.cross_attention is None else value + self.cross_attention(value)


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_layers=1, n_refiner_layers=1, injection_layer_ids=[0],
                                      cap_feat_dim=4, in_channels=4, patch_size=2)
        self.noise_refiner = nn.ModuleList([Block()])
        self.layers = nn.ModuleList([Block(True)])
        self.seen = []

    def forward(self, *, hidden_states, timestep, encoder_hidden_states,
                encoder_attention_mask, context_latents, return_dict):
        self.seen.append(dict(time=timestep.detach().clone(), references=[r.detach().clone() for r in context_latents],
                              mask=None if encoder_attention_mask is None else encoder_attention_mask.clone(),
                              shape=hidden_states.shape))
        assert return_dict is False and context_latents
        mask = (torch.ones(1, encoder_hidden_states.shape[2], dtype=torch.bool)
                if encoder_attention_mask is None else encoder_attention_mask)
        embedding = (encoder_hidden_states * mask[:, None, :, None]).sum((1, 2)) / mask.sum()
        reference = sum((i + 1) * r.mean((2, 3)) for i, r in enumerate(context_latents))
        value = hidden_states.permute(0, 2, 3, 1) + embedding[:, None, None] + reference[:, None, None]
        for module in [*self.noise_refiner, *self.layers]:
            value = module(value)
        return (value.permute(0, 3, 1, 2) * .2 + timestep[:, None, None, None] * .1,)


class Pipeline:
    """Mock orchestration, deliberately not a pretrained/native qualification."""
    def __init__(self):
        with torch.random.fork_rng():
            torch.manual_seed(123)
            self.transformer = Transformer()
        self.scheduler = SimpleNamespace(calls=0)
        self.text_encoder = nn.Linear(1, 1)
        self.vae = nn.Linear(1, 1)
        self.calls = []

    def __call__(self, *, prompt, image, mask, output_type, num_inference_steps, guidance_scale, generator,
                 height=None, width=None):
        assert num_inference_steps == 4 and guidance_scale == 1
        self.calls.append(dict(prompt=prompt, image=image, mask=mask, output_type=output_type))
        embeds = torch.full((1, 2, 3, 4), (sum(prompt.encode()) % 101) / 100.)
        # The last embedding is padded and deliberately huge.
        embeds[:, :, -1] = 10000
        contexts = [torch.full((1, 4, 2 + 2 * i, 2), float(img.getpixel((0, 0))[0]) / 255)
                    for i, img in enumerate(image)]
        if mask is not None:
            contexts[0] = contexts[0] + .05
        latent = torch.randn((1, 4, 2, 2), generator=generator)
        for time, delta in [(1000., -.18), (820., -.32), (500., -.35), (150., -.15)]:
            prediction = self.transformer(hidden_states=latent, timestep=torch.tensor([time / 1000]),
                encoder_hidden_states=embeds, encoder_attention_mask=torch.tensor([[True, True, False]]),
                context_latents=contexts, return_dict=False)[0].float()
            latent = latent.float() + delta * prediction
            self.scheduler.calls += 1
        return SimpleNamespace(images=[latent if output_type == 'latent' else 'native decoder output'])


def fixture():
    pipeline = Pipeline()
    slider = FiboParticleSlider(pipeline.transformer)
    adapter = FiboEditResponseAdapter(pipeline, slider, model_identity=IDENTITY)
    return pipeline, slider, adapter


def references():
    return [Image.new('RGB', (8, 8), (64, 20, 30)), Image.new('RGB', (8, 12), (192, 30, 20))]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_capture_native_defaults_references_mask_time_and_same_state_teachers():
    pipe, slider, adapter = fixture()
    scheduler = pipe.scheduler
    rng = torch.get_rng_state().clone()
    rows = adapter.capture_pair('keep jacket', 'make jacket red', references())
    assert torch.equal(rng, torch.get_rng_state())
    assert pipe.scheduler is scheduler and scheduler.calls == 0
    assert len(rows) == 4 and len(pipe.calls) == 2
    assert [float(p.row.timestep) for p in rows] == [1000., 820., 500., 150.]
    assert len(slider.adapters) == 12
    assert len({id(p) for p in adapter.network.parameters()}) == len(list(slider.parameters()))
    assert adapter.network.particles is slider.particles
    for item in rows:
        assert item.target.shape == item.neutral.shape == (1, 4, 2, 2)
        assert item.row.neutral['context_1'].shape == (1, 4, 4, 2)
        assert item.row.neutral['mask'].tolist() == [[True, True, False]]
        torch.testing.assert_close(item.neutral, adapter.velocity(item.row.neutral, item.row.latent, item.row.timestep, scale=0))
        torch.testing.assert_close(item.target, adapter.velocity(item.row.positive, item.row.latent, item.row.timestep, scale=0))
    assert not torch.equal(rows[0].neutral, rows[0].target)
    assert pipe.transformer.seen[-1]['time'].item() == pytest.approx(.15)
    assert all(not p.requires_grad for component in (pipe.transformer, pipe.vae, pipe.text_encoder) for p in component.parameters())
    mask = Image.new('L', (8, 8), 255)
    masked = adapter.capture_pair('keep', 'red', references()[:1], mask=mask)
    assert pipe.calls[-1]['mask'] is mask and len(masked) == 4


@pytest.mark.parametrize('case', ['no_refs', 'five_refs', 'mask_multi', 'vgl', 'time', 'cfg', 'packed', 'no_context', 'different_reference'])
def test_invalid_edit_contracts_fail_closed(case):
    pipe, slider, adapter = fixture()
    images = references()
    if case in ('no_refs', 'five_refs', 'mask_multi', 'vgl'):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            adapter.capture_pair('{}' if case == 'vgl' else 'keep', 'red',
                [] if case == 'no_refs' else images * 3 if case == 'five_refs' else images,
                mask=Image.new('L', (8, 8)) if case == 'mask_multi' else None)
        return
    row = adapter.capture_pair('keep', 'red', images)[0].row
    if case == 'time':
        row = replace(row, timestep=torch.tensor([1001.]))
    if case == 'cfg':
        row = replace(row, guidance=5.)
    if case == 'packed':
        row = replace(row, latent=row.latent.flatten(2).transpose(1, 2))
    if case == 'no_context':
        row = replace(row, neutral={k: v for k, v in row.neutral.items() if not k.startswith('context_')})
    if case == 'different_reference':
        row = replace(row, positive={**row.positive, 'context_0': row.positive['context_0'] + 1})
    with pytest.raises(ValueError):
        adapter.prepare(row)


def test_strength_zero_nested_scale_and_export_native_render_parity(tmp_path):
    pipe, slider, adapter = fixture()
    with torch.no_grad():
        for module in slider.adapters.values():
            module.lora_up.weight.fill_(.015)
    images = references()
    base = adapter.render_edit('keep', images, scale=0., output_type='latent').images[0]
    full = adapter.render_edit('keep', images, scale=1., output_type='latent').images[0]
    assert not torch.equal(base, full)
    assert torch.equal(base, adapter.render_edit('keep', images, scale=0., output_type='latent').images[0])
    with slider.scaled(.5):
        with slider.scaled(0):
            assert slider.scale == 0
        assert slider.scale == .5
    assert slider.scale == 0
    path = tmp_path / 'adapter.safetensors'
    slider.save(path, model_identity=IDENTITY)
    other_pipe = Pipeline()
    rng = torch.get_rng_state().clone()
    loaded = FiboParticleSlider.load(other_pipe.transformer, path, model_identity=IDENTITY)
    assert torch.equal(rng, torch.get_rng_state())
    other = FiboEditResponseAdapter(other_pipe, loaded, model_identity=IDENTITY)
    for scale in (0., .5, 1.):
        assert torch.equal(adapter.render_edit('keep', images, scale=scale, output_type='latent').images[0],
                           other.render_edit('keep', images, scale=scale, output_type='latent').images[0])
    assert other.render_edit('keep', images).images == ['native decoder output']
    assert pipe.scheduler.calls == other_pipe.scheduler.calls == 0
    assert all(not key.startswith(('transformer.', 'text_encoder.', 'vae.')) for key in loaded.state_dict())


@pytest.mark.parametrize('damage', ['model', 'shape', 'dtype', 'nan', 'extra'])
def test_export_rejects_invalid_state_before_attaching(tmp_path, damage):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    pipe, slider, adapter = fixture()
    path = tmp_path / 'adapter.safetensors'
    slider.save(path, model_identity=IDENTITY)
    with safe_open(str(path), framework='pt') as handle:
        metadata = handle.metadata()
    state = load_file(str(path))
    if damage == 'model':
        metadata['fibo_slider'] = metadata['fibo_slider'].replace('bounded Fibo', 'different Fibo')
    if damage == 'shape':
        state['particles'] = state['particles'][:-1]
    if damage == 'dtype':
        state['particles'] = state['particles'].bfloat16()
    if damage == 'nan':
        state['particles'][0, 0] = float('nan')
    if damage == 'extra':
        state['base_weights'] = torch.ones(1)
    save_file(state, str(path), metadata=metadata)
    candidate = Pipeline().transformer
    before = deepcopy(candidate.state_dict())
    targets = target_modules(candidate)
    forwards = [module.forward for module in targets.values()]
    with pytest.raises(ValueError):
        FiboParticleSlider.load(candidate, path, model_identity=IDENTITY)
    equal(before, candidate.state_dict())
    assert forwards == [module.forward for module in target_modules(candidate).values()]


def game_fixture(tmp_path):
    pipe, slider, backend = fixture()
    rows = backend.capture_pair('keep', 'red', references())
    checksum = write_cache(tmp_path / 'cache', rows, model_identity=IDENTITY)
    provider = FiboCachedRows(tmp_path / 'cache', index_sha256=checksum, model_identity=IDENTITY, capacity=1)
    assert len(provider.cache) == 0
    with torch.random.fork_rng():
        torch.manual_seed(44)
        critic = CampaignFieldCritic(4, provider.conditions([0]).shape[1], width=4)
    game = ConditionalPG09(backend, provider, critic, conditions=None, particles=slider.particles,
                           batch_size=2, total_steps=4, seed=7)
    assert len(provider.cache) == 0
    return pipe, slider, backend, provider, game


def test_actual_pg_dualnorm_losses_regularizers_bank_updates_resume_and_export(tmp_path, monkeypatch):
    pipe, slider, backend, provider, game = game_fixture(tmp_path / 'first')
    assert game.recipe.name == 'bispf_optimized'
    assert game.recipe.optimizer_family == 'dualnorm' and game.opt_g.family == 'dualnorm'
    assert game.recipe.loss == 'non_saturating'
    assert game.recipe.critic_formulation == 'bispf'
    assert game.recipe.prior_regularizer == 'sbnp'
    assert game.spread is not None and game.prior.z is slider.particles
    assert game._signature['generator_arm'] == 'routed_particle_gan'
    before = deepcopy(slider.state_dict())
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', lambda *a, **k: pytest.fail('no MSE objective'))
    metrics = game.step()
    assert metrics['g_loss'] == pytest.approx(metrics['g_adv'] + metrics['prior_regularization'])
    assert metrics['d_loss'] == pytest.approx(metrics['d_adv'] + metrics['d_pen'])
    assert metrics['d_pen'] > 0 and not torch.equal(before['particles'], slider.particles)
    assert len(provider.cache) <= 1
    # Saturated raw-logit critics can legitimately produce zero early G
    # movement; the untouched public optimizer/loss controls that decision.
    game.step()
    checkpoint = game.state_dict()
    expected_metrics, expected_state = game.step(), game.state_dict()
    assert any(not torch.equal(before[k], v) for k, v in slider.state_dict().items() if k != 'particles')
    _, other_slider, other_backend, other_provider, other = game_fixture(tmp_path / 'second')
    other.load_state_dict(checkpoint)
    assert other.step() == expected_metrics
    equal(expected_state, other.state_dict())
    packet = game.export_state_dict()
    _, _, _, _, inference = game_fixture(tmp_path / 'third')
    inference.load_export_state_dict(packet)
    assert torch.equal(game.evaluate(0, seed=7), inference.evaluate(0, seed=7))
    path = tmp_path / 'standalone.safetensors'
    slider.save(path, model_identity=IDENTITY)
    native_pipe = Pipeline()
    loaded = FiboParticleSlider.load(native_pipe.transformer, path, model_identity=IDENTITY)
    native = FiboEditResponseAdapter(native_pipe, loaded, model_identity=IDENTITY)
    for scale in (0., .5, 1.):
        assert torch.equal(backend.render_edit('keep', references(), scale=scale, output_type='latent').images[0],
                           native.render_edit('keep', references(), scale=scale, output_type='latent').images[0])
    with pytest.raises(RuntimeError, match='inference export'):
        inference.step()


def test_lazy_cache_immutable_selected_rows_and_corruption(tmp_path):
    _, _, backend = fixture()
    rows = backend.capture_pair('keep', 'red', references())
    checksum = write_cache(tmp_path / 'cache', rows, model_identity=IDENTITY)
    provider = FiboCachedRows(tmp_path / 'cache', index_sha256=checksum, model_identity=IDENTITY, capacity=1)
    assert provider.cache == {} and len(provider) == 4
    assert len(provider.index['contexts']) == 2
    equal(provider.conditions([0, 2]).cpu(), image_condition_features(rows)[[0, 2]])
    item = provider[0]
    item.target.zero_()
    assert not torch.equal(item.target, provider[0].target)
    provider[1]
    assert list(provider.cache) == [1]
    with (tmp_path / 'cache/row-0.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        provider[0]


def test_condition_features_bind_reference_order_padding_and_raw_time():
    _, _, backend = fixture()
    rows = backend.capture_pair('keep', 'red', references())
    base = image_condition_features(rows)
    flipped = replace(rows[0].row, neutral={**rows[0].row.neutral,
        'context_0': rows[0].row.neutral['context_1'], 'context_1': rows[0].row.neutral['context_0']})
    changed = replace(rows[0], row=flipped)
    assert not torch.equal(base[:1], image_condition_features([changed]))
    assert not torch.equal(base[0], base[1])


def test_shared_builder_stays_lazy_preserves_rng_and_owns_actual_bank(tmp_path):
    _, slider, backend, provider, _ = game_fixture(tmp_path)
    rng = torch.get_rng_state().clone()
    game = build_fibo_game(backend, provider, response_scale=1., total_steps=2)
    assert torch.equal(rng, torch.get_rng_state())
    assert not provider.cache and game.prior.z is slider.particles
    assert game._signature['resolved_penalty_domain'] == 'image'
    assert game._signature['backend']['defaults'] == DEFAULTS
    assert not torch.cuda.is_initialized()
