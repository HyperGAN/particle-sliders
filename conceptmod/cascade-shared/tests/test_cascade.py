from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from particle_sliders.lifecycle import SessionConfig, CheckpointError, UnsupportedOperation
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference
from particle_sliders.fibo_hndl import HNDLFieldCritic, verified_builder
from particle_sliders.field_critic import FieldCritic
from particle_sliders_cascade import backend as b

torch.set_num_threads(1)


def config(**options):
    return SessionConfig(b.NAME, 'cascade-new-cpu-test', dummy=True, steps=3, ema_decay=.9, **options)


def snapshot(model):
    return ([m.training for m in model.modules()], [(p.requires_grad, p.grad) for p in model.parameters()],
            [len(m._forward_hooks) for m in model.modules()])


def test_installed_identity():
    assert 'site-packages' in str(Path(b.__file__))
    import particle_sliders.pg09
    assert 'site-packages' in particle_sliders.pg09.__file__
    assert verified_builder()[1]['hndl_version'] == '0.10.0'
    assert b.PG_VERSION == '0.9.4'


def test_hndl_lora_outputs_vjps():
    graph = b.LoRAGraph(nn.Linear(5, 7), 3)
    nn.init.normal_(graph.graph['up'].weight)
    x = torch.randn(2, 4, 5, requires_grad=True)
    actual = graph.delta(x)
    reference = torch.nn.functional.linear(torch.nn.functional.linear(x, graph.graph['down'].weight),
                                            graph.graph['up'].weight)
    torch.testing.assert_close(actual, reference)
    cotangent = torch.randn_like(actual)
    leaves = [x, *graph.parameters()]
    for got, wanted in zip(torch.autograd.grad(actual, leaves, cotangent, retain_graph=True),
                           torch.autograd.grad(reference, leaves, cotangent)):
        torch.testing.assert_close(got, wanted)


def test_hndl_critic_second_derivatives():
    actual = HNDLFieldCritic(4, 6, (4, 4), width=8)
    reference = FieldCritic(4, 6, width=8)
    pairs = [('field0', reference.field[0]), ('field2', reference.field[2]),
             ('condition', reference.condition), ('score0', reference.score[0]), ('score2', reference.score[2])]
    for name, layer in pairs:
        layer.load_state_dict(actual.graph[name].state_dict())
    x = torch.randn(2, 4, 4, 4, requires_grad=True)
    context = torch.randn(2, 6, requires_grad=True)
    left, right = actual(x, context), reference(x, context)
    torch.testing.assert_close(left, right)
    ga = torch.autograd.grad(left.sum(), (x, context), create_graph=True)
    gr = torch.autograd.grad(right.sum(), (x, context), create_graph=True)
    for a, r in zip(ga, gr):
        torch.testing.assert_close(a, r)
    actual_parameters = [p for name, _ in pairs for p in actual.graph[name].parameters()]
    reference_parameters = [p for _, layer in pairs for p in layer.parameters()]
    ha = torch.autograd.grad(sum(v.square().sum() for v in ga), [x, context, *actual_parameters], allow_unused=True)
    hr = torch.autograd.grad(sum(v.square().sum() for v in gr), [x, context, *reference_parameters], allow_unused=True)
    for a, r in zip(ha, hr):
        if a is None or r is None:
            assert a is r
        else:
            torch.testing.assert_close(a, r, atol=1e-7, rtol=1e-5)


def test_native_response_and_strength_restore():
    model, rows = b.tiny_model(), b.fixture_rows()
    model.train()
    next(model.parameters()).grad = torch.ones_like(next(model.parameters()))
    before = snapshot(model)
    with open_response(config(), model=model, rows=rows) as session:
        with torch.no_grad():
            native = model(sample=rows[0].latent, timestep_ratio=rows[0].timestep.reshape(1),
                clip_text=rows[0].neutral['embeds'], clip_text_pooled=rows[0].neutral['pooled'],
                clip_img=rows[0].neutral['image_embeds']).sample
        torch.testing.assert_close(session.evaluate(strength=0), native, rtol=0, atol=0)
        original = [p.clone() for p in session.game.network.parameters()]
        rng = torch.get_rng_state().clone()
        stats = session.step()
        assert stats['step'] == 1 and stats['d_pen'] >= 0
        assert any(not torch.equal(a, p) for a, p in zip(original, session.game.network.parameters()))
        zero = session.evaluate(strength=0)
        positive, negative = session.evaluate(strength=1), session.evaluate(strength=-1)
        assert not torch.equal(positive, zero) and not torch.equal(negative, positive)
        assert torch.equal(torch.get_rng_state(), rng)
    assert snapshot(model) == before


def test_exact_resume_ema_native_only(tmp_path, monkeypatch):
    c = config()
    with open_response(c) as full:
        full.train()
        wanted = full.checkpoint()
    with open_response(c) as first:
        first.train(stop_after=1)
        first.checkpoint(tmp_path / 'resume.pt')
    with open_response(c, resume=tmp_path / 'resume.pt') as resumed:
        resumed.train()
        assert b._hash(resumed.checkpoint()) == b._hash(wanted)
        exported = resumed.export_inference(selection='ema')
        with torch.no_grad():
            saved = deepcopy(resumed.game.network.state_dict())
            resumed.game.network.load_state_dict(resumed.ema.values)
            expected = resumed.evaluate(strength=.5)
            resumed.game.network.load_state_dict(saved)
    def forbidden(*args, **kwargs):
        raise AssertionError('training object constructed during native inference')
    monkeypatch.setattr(b, 'ConditionalPG09', forbidden)
    monkeypatch.setattr(b, 'HNDLFieldCritic', forbidden)
    monkeypatch.setattr(b, 'fixture_rows', forbidden)
    with open_inference(exported, operation='response') as inference:
        assert not hasattr(inference, 'game')
        # Independent caller supplied row, no training cache in artifact.
        row = FIXTURE()[0]
        torch.testing.assert_close(inference.response(row=row, strength=.5), expected, rtol=0, atol=0)
        with pytest.raises(UnsupportedOperation):
            inference.train()


FIXTURE = b.fixture_rows


def test_preflight_and_failure_cleanup(monkeypatch):
    model, rows = b.tiny_model(), b.fixture_rows()
    before = snapshot(model)
    with pytest.raises(UnsupportedOperation):
        open_response(config(), operation='render', model=model, rows=rows)
    with pytest.raises(ValueError):
        open_response(config(), model=model, rows=[replace(rows[0], timestep=torch.tensor(100.)), rows[1]])
    assert snapshot(model) == before
    def fail(*args, **kwargs):
        raise RuntimeError('critic allocation failed')
    monkeypatch.setattr(b, 'HNDLFieldCritic', fail)
    with pytest.raises(RuntimeError, match='critic allocation'):
        open_response(config(), model=model, rows=rows)
    assert snapshot(model) == before


def test_rejection_rollback_and_failed_update(monkeypatch):
    model = b.tiny_model()
    before = snapshot(model)
    session = open_response(config(), model=model)
    session.step()
    state = session.checkpoint()
    corrupted = deepcopy(state)
    corrupted['payload']['game']['completed_steps'] = 99
    corrupted['sha256'] = b._hash({k: v for k, v in corrupted.items() if k != 'sha256'})
    with pytest.raises(CheckpointError):
        session.restore(corrupted)
    assert b._hash(session.checkpoint()) == b._hash(state)
    def fail(*args, **kwargs):
        raise RuntimeError('native forward failed')
    monkeypatch.setattr(model, 'forward', fail)
    with pytest.raises(RuntimeError, match='native forward'):
        session.step()
    assert session.closed and snapshot(model) == before
    session.close()


def test_cross_cohort_rejects_before_model_allocation(tmp_path, monkeypatch):
    with open_response(config()) as session:
        session.checkpoint(tmp_path / 'checkpoint.pt')
    monkeypatch.setattr(b, 'load_model', lambda *args: pytest.fail('allocated before rejection'))
    with pytest.raises(CheckpointError):
        open_response(SessionConfig(b.NAME, 'different', dummy=True, steps=3, ema_decay=.9),
                      resume=tmp_path / 'checkpoint.pt')


@pytest.mark.parametrize('stage', ['graph', 'teacher', 'game', 'callback'])
def test_every_partial_failure_unwinds(stage, monkeypatch):
    model = b.tiny_model()
    model.train()
    before = snapshot(model)
    calls = 0
    original = b.LoRAGraph
    def graph(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected graph')
        return original(*args, **kwargs)
    def fail(*args, **kwargs):
        raise RuntimeError('injected ' + stage)
    if stage == 'graph':
        monkeypatch.setattr(b, 'LoRAGraph', graph)
    elif stage == 'teacher':
        monkeypatch.setattr(b.NativeResponse, 'prepare', fail)
    elif stage == 'game':
        monkeypatch.setattr(b, 'ConditionalPG09', fail)
    with pytest.raises(RuntimeError, match='injected'):
        with open_response(config(), model=model) as session:
            session.train(on_step=fail)
    assert snapshot(model) == before


def test_local_staging_and_cache_identity(tmp_path, monkeypatch):
    from dataclasses import asdict
    model_dir = tmp_path / 'prior'
    b.tiny_model().save_pretrained(model_dir, safe_serialization=True)
    options = dict(model_dir=str(model_dir), shape=[1, 4, 4, 4], steps=2)
    model = b.load_model(options)
    cache = tmp_path / 'frozen.pt'
    torch.save(dict(model_identity=b.model_identity(model), rows=[asdict(r) for r in FIXTURE()]), cache)
    options.update(cache_file=str(cache), cache_sha256=b.hashlib.sha256(cache.read_bytes()).hexdigest())
    with open_response(SessionConfig(b.NAME, 'staged-tiny', **options)) as session:
        session.step()
        artifact = session.export_inference()
    assert 'cache_file' not in artifact['specification']['configuration']
    with open_inference(artifact, options={'model_dir': str(model_dir)}, operation='response') as native:
        assert native.response(row=FIXTURE()[0]).shape == (1, 4, 4, 4)
    changed = b.load_model(options)
    with torch.no_grad():
        next(changed.parameters()).add_(.1)
    before = snapshot(changed)
    monkeypatch.setattr(b, 'load_model', lambda *args: changed)
    with pytest.raises(ValueError, match='before adapter attachment'):
        open_inference(artifact, options={'model_dir': str(model_dir)}, operation='response')
    assert snapshot(changed) == before
    with pytest.raises(ValueError, match='different Stage C base'):
        open_response(SessionConfig(b.NAME, 'wrong-cache-base', **options))
    assert snapshot(changed) == before
