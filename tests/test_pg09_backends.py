from copy import deepcopy
from dataclasses import asdict, replace
import inspect

import pytest
import torch
from particlegan import get_recipe

from conceptmod.textsliders.anima_fake import FakeAnimaBackend
from conceptmod.textsliders.supra_fake import FakeSupraBackend
from conceptmod.textsliders.pg09_backends import (
    AnimaResponseAdapter, SupraResponseAdapter, MossResponseAdapter,
    FlowSchedule, ResponseRow, checkpoint_with_adapter_scale,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def make(family):
    if family == 'anima':
        backend = FakeAnimaBackend(rank=2, seed=9, lora_targets='dit+conditioner')
        owned = {id(p) for p in backend.trainable_parameters()}
        for p in backend.transformer.parameters():
            p.requires_grad_(id(p) in owned)
        adapter = AnimaResponseAdapter(backend)
        latent = torch.randn(1, *backend.latent_shape)
        row = ResponseRow('a woman neutral', 'a woman smiling', '', latent,
                          schedule=FlowSchedule.linear(2, time_scale=1000), guidance=3.)
        model = backend.transformer
    elif family == 'supra':
        backend = FakeSupraBackend(rank=2, seed=9)
        adapter = SupraResponseAdapter(backend)
        latent = torch.randn(1, *backend.latent_shape)
        row = ResponseRow('a woman neutral', 'a woman smiling', '', latent,
                          schedule=FlowSchedule.supra(2), guidance=3.)
        model = backend.transformer
    else:
        pytest.importorskip('moss_sliders')
        from moss_sliders.smoke import tiny_model
        from moss_sliders.lora import Slider
        torch.manual_seed(9)
        model = tiny_model()
        slider = Slider(model, rank=2)
        adapter = MossResponseAdapter(model, slider)
        neutral = (torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
        positive = (torch.randn(1, 3, 16), neutral[1])
        negative = (torch.randn(1, 2, 16), torch.ones(1, 2, dtype=torch.bool))
        latent = torch.randn(1, model.config.latent_channels, 2, 2)
        row = ResponseRow(neutral, positive, negative, latent, timestep=torch.tensor(500.))
    return adapter, row, model


def make_game(family):
    adapter, row, model = make(family)
    rows = [row, replace(row, latent=row.latent * .9 + .1)]
    if family == 'moss':
        rows += [replace(r, scale=-1.) for r in rows]
    fixed = [adapter.prepare(r) for r in rows]
    core = pytest.importorskip('particle_sliders.pg09', reason='shared native-field core implementation pending', exc_type=ImportError)
    conditions = torch.tensor([[float(i), r.row.scale, r.row.guidance] for i, r in enumerate(fixed)])
    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(fixed[0].target.numel() + 3, 16),
                                           torch.nn.SiLU(), torch.nn.Linear(16, 1))
        def forward(self, response, condition):
            return self.net(torch.cat([response.flatten(1), condition], dim=1)).squeeze(-1)
    critic = Critic()
    game = core.ConditionalPG09(adapter, fixed, critic, conditions=conditions,
                               response_domain='continuous_field', particles=None,
                               preset='bispf_optimized', total_steps=4, batch_size=4)
    return adapter, row, model, fixed, game, None


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('family', ['anima', 'supra', 'moss'])
def test_adversarial_core_owns_updates_only_adapter_and_frozen_base(family, monkeypatch):
    adapter, row, model, fixed, game, predict = make_game(family)
    owned = {id(p) for p in adapter.network.parameters()}
    base = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in owned}
    before = deepcopy(adapter.network.state_dict())
    assert asdict(game.recipe) == asdict(get_recipe('bispf_optimized', total_steps=4, batch_size=4))
    assert {id(p) for group in game.opt_g.param_groups for p in group['params']} == owned
    assert all(not r.target.requires_grad and not r.neutral.requires_grad for r in fixed)
    def forbidden(*a, **kw):
        raise AssertionError('MSE is forbidden on this adversarial training path')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    for _ in range(3):
        metrics = game.step()
        assert metrics['g_loss'] == metrics['g_adv']
        assert metrics['prior_regularization'] == 0
        assert all(torch.isfinite(torch.tensor(metrics[k])) for k in ('d_loss', 'g_loss', 'd_pen'))
    assert any(not torch.equal(before[n], p) for n, p in adapter.network.state_dict().items())
    assert all(torch.equal(base[n], p) for n, p in model.named_parameters() if n in base)
    zero = adapter._response(row, scale=0)
    assert torch.equal(zero, fixed[0].neutral)
    assert not torch.equal(adapter.generated(fixed[0]), zero)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('family', ['anima', 'supra', 'moss'])
def test_exact_resume_and_adapter_export_reload(family):
    adapter, row, model, fixed, game, predict = make_game(family)
    game.step()
    state = game.state_dict()
    sample = adapter.generated(fixed[0]).detach().clone()
    metrics = game.step()
    expected = game.state_dict()
    other, _, _, other_fixed, other_game, other_predict = make_game(family)
    other_game.load_state_dict(state)
    assert torch.equal(sample, other.generated(other_fixed[0]))
    assert metrics == other_game.step()
    equal(expected, other_game.state_dict())
    # Adapter-only reload never includes the pretrained host's weights.
    exported = deepcopy(adapter.network.state_dict())
    assert all(k.startswith('weights.') for k in exported)
    other.network.load_state_dict(exported, strict=True)
    assert torch.equal(adapter.generated(fixed[0]), other.generated(other_fixed[0]))


@pytest.mark.parametrize('family', ['anima', 'supra', 'moss'])
def test_cfg_is_conditional_only_and_render_reuses_response(family):
    adapter, row, model = make(family)
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(.02)
    guided = replace(row, guidance=3.)
    time = row.timestep if row.timestep is not None else row.latent.new_tensor(row.schedule.timesteps[0])
    c0 = adapter.velocity(row.neutral, row.latent, time, scale=0.)
    c1 = adapter.velocity(row.neutral, row.latent, time, scale=1.)
    u0 = adapter.velocity(row.negative, row.latent, time, scale=0.)
    torch.testing.assert_close(adapter.cfg(guided, row.latent, time, scale=1.), u0 + 3 * (c1 - u0))
    torch.testing.assert_close(adapter.cfg(guided, row.latent, time, scale=1.) -
                               adapter.cfg(guided, row.latent, time, scale=0.),
                               3 * (c1 - c0), atol=1e-6, rtol=1e-4)
    schedule = row.schedule or FlowSchedule.linear(2, time_scale=1000)
    rendering = replace(guided, timestep=None, schedule=schedule)
    adapter.decoder = lambda latent: latent.clone()
    assert torch.equal(adapter.render(rendering), adapter._response(rendering, scale=1.))
    assert adapter.metadata()['cfg_adapter_branch'] == 'conditional_only'
    assert adapter.metadata()['negative_layer_skip'] is False


def test_moss_checkpoint_recomputation_captures_each_signed_row():
    adapter, row, model = make('moss')
    # Exercise nonzero residuals in both directions and nested frozen CFG calls.
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(.02)
    rows = [row, replace(row, scale=-1.)]
    for checkpointing in (False, True):
        adapter.network.zero_grad(set_to_none=True)
        if checkpointing:
            for block in model.layers:
                checkpoint_with_adapter_scale(block, adapter.network)
        values = [adapter.generated(adapter.prepare(r)) for r in rows]
        # The scale contexts have exited before this batch backward.
        sum(v.square().mean() for v in values).backward()
        gradients = [p.grad.clone() for p in adapter.network.parameters()]
        if not checkpointing:
            reference = gradients
        else:
            for a, b in zip(reference, gradients):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert adapter.slider.scale == 0.


def test_supra_live_scale_is_not_reset_by_legacy_predict_v():
    from conceptmod.textsliders.train_lora_supra import LiveSupraBackend
    fake = FakeSupraBackend(rank=2, seed=9)
    backend = LiveSupraBackend(fake.transformer, fake.loras, fake.device, fake.lora_spec)
    backend.encode_fn = fake.encode_text
    adapter = SupraResponseAdapter(backend)
    with torch.no_grad():
        for lora in fake.loras:
            lora.up.weight.fill_(.1)
    z = torch.randn(1, *backend.latent_shape)
    t = torch.tensor(.5)
    positive = adapter.velocity('a woman', z, t, scale=1.)
    zero = adapter.velocity('a woman', z, t, scale=0.)
    negative = adapter.velocity('a woman', z, t, scale=-1.)
    assert not torch.equal(zero, positive)
    assert not torch.equal(negative, positive)
    assert all(lora.multiplier == 1. for lora in fake.loras)
    with pytest.raises(ValueError, match='sigma'):
        adapter.velocity('a woman', z, torch.tensor(500.))


def test_nested_local_and_peft_scale_restoration_and_live_conditioner():
    from peft import LoraConfig, get_peft_model
    from conceptmod.textsliders.pg09_backends import _lora_scale_context
    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(3, 3)),
                           LoraConfig(r=2, lora_alpha=4, target_modules=['0']))
    layer = model.base_model.model[0]
    old = dict(layer.scaling)
    context = _lora_scale_context([('model', model)])
    with context(-.5):
        assert layer.scaling == {k: v * -.5 for k, v in old.items()}
        with context(0.):
            assert all(v == 0 for v in layer.scaling.values())
        assert layer.scaling == {k: v * -.5 for k, v in old.items()}
    assert layer.scaling == old
    adapter, row, _ = make('anima')
    names = adapter.network.names
    fixed = adapter.prepare(row)
    adapter.generated(fixed).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in
               zip(names, adapter.network.parameters()) if 'conditioner' in n)
    assert 'mse_loss' not in inspect.getsource(type(adapter))


@pytest.mark.parametrize('family', ['anima', 'supra', 'moss'])
def test_portable_export_validation_and_response_signature(family, tmp_path):
    from conceptmod.textsliders.pg09_backends import save_adapter, load_adapter, response_signature
    adapter, row, model = make(family)
    fixed = adapter.prepare(row)
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.add_(.02)
    expected = adapter.generated(fixed).detach().clone()
    path = tmp_path / 'adapter.safetensors'
    save_adapter(adapter, path, model_fingerprint='verified-fixture-model')
    other, _, _ = make(family)
    before = deepcopy(other.network.state_dict())
    with pytest.raises(ValueError, match='frozen model'):
        load_adapter(other, path, model_fingerprint='wrong-frozen-base')
    equal(before, other.network.state_dict())
    load_adapter(other, path, model_fingerprint='verified-fixture-model')
    assert torch.equal(expected, other.generated(other.prepare(row)))
    signature = response_signature(adapter, [fixed], model_fingerprint='verified-fixture-model')
    changed = replace(fixed, row=replace(fixed.row, guidance=2.))
    assert signature != response_signature(adapter, [changed], model_fingerprint='verified-fixture-model')
    assert fixed.target.ndim in (4, 5)  # Native shape reaches PG, never a fake image reshape.


def test_moss_signed_target_snapshot_and_official_scheduler_parity():
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
    adapter, row, model = make('moss')
    positive = adapter.prepare(row)
    negative = adapter.prepare(replace(row, scale=-1.))
    torch.testing.assert_close(positive.target + negative.target, 2 * positive.neutral)
    snapshot = adapter.generated(positive).detach().clone()
    row.neutral[0].zero_()
    row.latent.zero_()
    assert torch.equal(snapshot, adapter.generated(positive))
    row = positive.row
    scheduler = MOSSImageScheduler()
    scheduler.set_timesteps(4, device='cpu')
    schedule = FlowSchedule.from_scheduler(scheduler)
    rendering = replace(row, timestep=None, schedule=schedule)
    expected = row.latent.clone()
    for time in scheduler.timesteps:
        v = adapter.velocity(row.neutral, expected, time, scale=0.)
        expected = scheduler.step(v, time, expected, return_dict=False)[0]
    torch.testing.assert_close(expected, adapter._response(rendering, scale=0.), rtol=0, atol=0)
    # Native pipeline agrees at scale0 with identical CFG and no negative skips.
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    with adapter.slider.at(0):
        native = pipe(prompt_embeds=row.neutral[0], prompt_attention_mask=row.neutral[1],
                      height=32, width=32, num_inference_steps=4, guidance_scale=1.,
                      uncond_skip_layers=[], output_type='latent', latents=row.latent.clone()).images
    torch.testing.assert_close(native, expected, rtol=0, atol=0)


def test_supra_native_increasing_time_and_runtime_bridge():
    from types import SimpleNamespace
    from conceptmod.textsliders.pg09_backends import SupraRuntimeResponseAdapter
    fake = FakeSupraBackend(rank=2, seed=9)
    runtime = SimpleNamespace(model=fake.transformer, device=fake.device,
                              parameters=fake.trainable_parameters,
                              encode=lambda prompt: (fake.encode_text(prompt)[0], None))
    adapter = SupraRuntimeResponseAdapter(runtime)
    z = torch.randn(1, *fake.latent_shape)
    row = ResponseRow('neutral', 'smiling', '', z, schedule=FlowSchedule.supra(4))
    expected = z.clone()
    for i in range(4):
        expected = expected + fake.predict_v('neutral', expected, torch.tensor(i / 4), frozen=True) / 4
    assert torch.equal(expected, adapter._response(row, scale=0.))
    assert row.schedule.timesteps == (0., .25, .5, .75)
    assert row.schedule.deltas == (.25, .25, .25, .25)
