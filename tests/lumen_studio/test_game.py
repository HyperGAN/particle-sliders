import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from lumen_studio.cache import fit_normalization, prepare_targets
from lumen_studio.contracts import digest
from lumen_studio.dataset import compile_manifest
from lumen_studio.game import Game
from lumen_studio.metrics import make_fixtures, measure
from lumen_studio.particles import ParticleAdapter, Mixer
from lumen_studio.training import Trainer
from lumen_studio.vendor.reference import particle_vic, rp_d_loss, rp_g_loss, noise_std
from lumen_studio.vendor.grad_regularizers import GradRegularizer
from conftest import TinyRuntime, tiny_cache


def test_loss_and_vic_gradients_against_equations():
    torch.manual_seed(3)
    r, f = torch.randn(7, requires_grad=True), torch.randn(7, requires_grad=True)
    for impl, expected in ((rp_d_loss(r, f), torch.logaddexp(torch.zeros_like(r), f-r).mean()),
                           (rp_g_loss(r, f), torch.logaddexp(torch.zeros_like(r), r-f).mean())):
        assert torch.allclose(impl, expected)
        a = torch.autograd.grad(impl, (r, f), retain_graph=True)
        b = torch.autograd.grad(expected, (r, f), retain_graph=True)
        assert all(torch.allclose(x, y) for x, y in zip(a, b))
    p = torch.randn(64, 4, requires_grad=True)
    cov = torch.cov(p.T)
    expected = torch.relu(1 - (p.var(0) + 1e-4).sqrt()).mean() + (cov - cov.diag().diag()).square().sum() / 4
    actual = particle_vic(p)
    assert torch.allclose(actual, expected)
    assert torch.allclose(torch.autograd.grad(actual, p)[0], torch.autograd.grad(expected, p)[0])


@pytest.mark.parametrize("step", [1, 3, 4, 8])
def test_exact_lazy_cap_active_and_inactive(step):
    critic = nn.Linear(5, 1, bias=False)
    critic.weight.data.fill_(2.)
    real, fake = torch.randn(8, 5), torch.randn(8, 5)
    cap, _ = GradRegularizer(arm="b_cap", coeff=1, kappa=1, lazy_k=4).penalty(critic, real, fake, step=step)
    expected = 4 * torch.relu((critic.weight.square().sum() + 1e-12).sqrt() - 1).square()
    if step % 4:
        assert cap == 0 and not cap.requires_grad
    else:
        assert torch.allclose(cap, expected)
        assert torch.allclose(torch.autograd.grad(cap, critic.weight)[0], torch.autograd.grad(expected, critic.weight)[0])


def test_mixer_zero_sum_order_export_reload(tiny, tmp_path):
    a, b = ParticleAdapter(tiny.transformer), ParticleAdapter(tiny.transformer)
    x = torch.randn(2, 3, 4)
    original = tiny.transformer.to_q(x)
    tiny.mixer.add("moonlit", b)
    tiny.mixer.add("candlelit", a)
    for adapter in (a, b):
        for branch in adapter.branches:
            nn.init.normal_(branch.up.weight, std=.05)
    assert torch.equal(original, tiny.transformer.to_q(x))
    tiny.mixer.set_mix(dict(candlelit=1, moonlit=2), .75)
    expected = original + .25 * a.branches[0](x, a.particles) + .5 * b.branches[0](x, b.particles)
    actual = tiny.transformer.to_q(x)
    assert torch.allclose(actual, expected, atol=1e-7)
    tiny.mixer.adapters = dict(reversed(list(tiny.mixer.adapters.items())))
    assert torch.equal(actual, tiny.transformer.to_q(x))
    tiny.mixer.set_mix(dict(candlelit=1, moonlit=2), 0)
    assert torch.equal(original, tiny.transformer.to_q(x))
    path = tmp_path / "test.safetensors"
    a.export(path, model_identity=tiny.identity, normalization={"source":"test"}, manifest_hash="b"*64, step=100, ema=a.state_dict())
    c = ParticleAdapter(tiny.transformer)
    c.load_export(path, model_identity=tiny.identity)
    assert all(torch.equal(a.state_dict()[k], v) for k, v in c.state_dict().items())
    with pytest.raises(ValueError, match="different model"):
        c.load_export(path, model_identity={"model":"wrong"})


def test_frozen_base_and_exact_resume(tiny, tmp_path):
    cache = tiny_cache(tmp_path / "cache", tiny)
    before = copy.deepcopy(tiny.transformer.state_dict())
    trainer = Trainer(tiny, cache, tmp_path / "run", "candlelit")
    for _ in range(4):
        trainer.update()
    trainer.save()
    expected_log = trainer.update()
    expected = copy.deepcopy(trainer.adapter.state_dict())
    ema = copy.deepcopy(trainer.ema)
    for name, value in tiny.transformer.state_dict().items():
        assert torch.equal(value, before[name])
    assert expected_log["particle_gan_grad_norm"] > 0
    assert expected_log["d_rows"] != expected_log["g_rows"]
    other = TinyRuntime()
    resumed = Trainer(other, cache, tmp_path / "run", "candlelit")
    actual_log = resumed.update()
    assert actual_log["d_rows"] == expected_log["d_rows"]
    assert actual_log["g_rows"] == expected_log["g_rows"]
    for k, v in resumed.adapter.state_dict().items():
        assert torch.equal(v, expected[k]), k
        assert torch.equal(resumed.ema[k], ema[k]), k


def test_microbatch_update_parity(tmp_path):
    states = []
    for micro in (1, 2, 4, 8):
        runtime = TinyRuntime()
        cache = tiny_cache(tmp_path / f"cache{micro}", runtime)
        trainer = Trainer(runtime, cache, tmp_path / f"run{micro}", "candlelit", microbatch=micro)
        for _ in range(4):
            trainer.update()
        states.append(copy.deepcopy(trainer.adapter.state_dict()))
    for state in states[1:]:
        for key, value in state.items():
            assert torch.allclose(value, states[0][key], atol=2e-5, rtol=2e-4), key


def test_normalization_training_only_and_timestep_sensitive(tiny, tmp_path):
    cache = tiny_cache(tmp_path / "cache", tiny)
    norm = fit_normalization(cache)
    assert norm["scale"].shape == (10, 16)
    assert norm["scale"].dtype == torch.float32
    cache.index["identity"]["split"] = "dev"
    with pytest.raises(ValueError, match="training"):
        fit_normalization(cache)
    cache.index["identity"]["split"] = "train"
    shard, position = cache.offsets[0]
    cache.offsets[0] = shard, 1
    with pytest.raises(ValueError, match="timesteps"):
        fit_normalization(cache)


def test_both_teachers_evaluated_at_same_state_all_ten_steps(tiny, tmp_path):
    manifest = compile_manifest("dev")
    manifest["rows"] = manifest["rows"][:1]
    manifest["sha256"] = digest({k:v for k,v in manifest.items() if k != "sha256"})
    prepare_targets(tiny, manifest, "candlelit", tmp_path / "targets/candlelit/dev")
    assert len(tiny.calls) == 76  # Initial shared state reused on the second trajectory.
    for a, b in zip(tiny.calls[::2], tiny.calls[1::2]):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert not torch.equal(a[2], b[2])


def test_target_preparation_survives_adapter_render_handoff_without_teacher_changes(tmp_path):
    from scripts.anima_campaign import TargetRuntime
    from lumen_studio.api import Generation, generation_payloads
    from lumen_studio.cache import TargetCache
    from lumen_studio.coordinator import Coordinator
    from conftest import TinyRuntime
    manifest = compile_manifest('dev')
    manifest['rows'] = manifest['rows'][:1]
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    baseline_runtime = TinyRuntime()
    baseline_path = tmp_path / 'baseline/targets/candlelit/dev'
    prepare_targets(baseline_runtime, manifest, 'candlelit', baseline_path)
    worker = Coordinator(tmp_path / 'studio', tmp_path / 'model', gpu='cpu', runtime_factory=TinyRuntime)
    adapter = ParticleAdapter(baseline_runtime.transformer)
    with torch.no_grad():
        for branch in adapter.branches:
            branch.up.weight.fill_(.2)
    path = tmp_path / 'ema.safetensors'
    adapter.export(path, model_identity=baseline_runtime.identity, normalization={'test': True},
                   manifest_hash=manifest['sha256'], step=100, ema=adapter.state_dict())
    sha = worker.store.register_checkpoint(path, 'candlelit')
    worker.store.enqueue(generation_payloads(Generation(prompt='adapter render between frozen shards',
        seed=7, mix={'candlelit': 1}, checkpoints={'candlelit': sha}), worker.store, baseline_runtime.identity))
    handoffs = []
    def handoff(done, total):
        if done == 1:
            worker.release_runtime()
            assert worker.render_one()
            assert worker.runtime.mixer.adapters
            worker.release_runtime()
            handoffs.append(done)
    actual_path = tmp_path / 'handoff/targets/candlelit/dev'
    prepare_targets(TargetRuntime(worker), manifest, 'candlelit', actual_path, progress=handoff)
    baseline, actual = TargetCache(baseline_path), TargetCache(actual_path)
    assert handoffs == [1] and len(actual) == len(baseline) == 40
    assert not worker.runtime.mixer.adapters
    for i in range(len(actual)):
        for key in ('latent', 'neutral', 'positive', 'embedding'):
            assert torch.equal(actual[i][key], baseline[i][key])
    worker.release_runtime()
    baseline_runtime.close()


def test_spatial_edits_and_corrupted_targets_are_detected():
    # Zero spatial mean must not erase the edit signal.
    teacher = torch.tensor([[1., -1., 1., -1.], [-1., 1., -1., 1.]])
    fixtures = make_fixtures(4, 2)
    good = measure(teacher, teacher, torch.ones_like(teacher), fixtures)
    bad = measure(-teacher, teacher, torch.ones_like(teacher), fixtures)
    assert good["residual_rms"] == 0
    assert bad["residual_rms"] == 2
    assert bad["edit_cosine"] == pytest.approx(-1)
    assert bad["game_swd"]["1.0"] > 0


@pytest.mark.parametrize("corruption", ["swapped_targets", "shuffled_timesteps", "wrong_neutral"])
def test_fixed_probe_rejects_teacher_pairing_corruption(tiny, tmp_path, corruption):
    cache = tiny_cache(tmp_path / "targets", tiny)
    normalization = fit_normalization(cache)
    records = [cache[i] for i in range(len(cache))]
    neutral = torch.stack([r["neutral"].flatten() for r in records])
    positive = torch.stack([r["positive"].flatten() for r in records])
    correct_edit = positive - neutral
    scale = torch.stack([normalization["scale"][r["position"]] for r in records])
    fixtures = make_fixtures(correct_edit.shape[1], len(records))
    assert measure(correct_edit, correct_edit, scale, fixtures)["residual_rms"] == 0
    if corruption == "swapped_targets":
        # Twenty records per seed: each target comes from a different latent.
        wrong_edit = positive.roll(20, 0) - neutral
    elif corruption == "wrong_neutral":
        wrong_edit = positive - neutral.roll(20, 0)
    else:
        wrong = []
        with torch.no_grad():
            for record in records:
                timestep = 1000 - ((record["position"] + 1) % 10) * 100
                # Both teachers still share z/t, but t is wrong for this fixture.
                vp = tiny.predict(record["latent"], timestep, tiny.encode("positive"))
                vn = tiny.predict(record["latent"], timestep, tiny.encode("neutral"))
                wrong.append((vp - vn).flatten())
        wrong_edit = torch.stack(wrong)
    result = measure(correct_edit, wrong_edit, scale, fixtures)
    assert result["residual_rms"] > .01
    assert result["game_swd"]["1.0"] > 0


def test_pilot_uses_full_noise_horizon():
    assert noise_std(199, start=3.5, decay_steps=1600, hold=1) > 1
    assert noise_std(1599, start=3.5, decay_steps=1600, hold=1) == 1


def test_resume_verifier_checks_active_cap_and_studio_all_timesteps(tiny, tmp_path):
    import json
    cache = tiny_cache(tmp_path / 'cache', tiny)
    trainer = Trainer(tiny, cache, tmp_path / 'run', 'candlelit')
    for _ in range(3):
        trainer.update()
    expected = copy.deepcopy(trainer.adapter.state_dict())
    report = trainer.verify_resume()
    assert report['passed'] and report['checked_update'] == 4
    assert report['studio_prediction_max_abs_by_timestep'] == [0] * 10
    assert trainer.step == 3
    assert all(torch.equal(v, expected[k]) for k, v in trainer.adapter.state_dict().items())
    updates = [json.loads(line) for line in (trainer.directory / 'updates.jsonl').read_text().splitlines()]
    assert [u['step'] for u in updates] == [1, 2, 3]
