"""Installed shared/HG/HNDL CPU integration; never imports pretrained weights."""
from copy import deepcopy
from importlib.metadata import distribution, PackageNotFoundError
import ast
import json
import math
import os
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

try:
    distribution('tinyqwen-shared-preview')
except PackageNotFoundError:
    pytest.skip('separate TinyQwen shared preview wheel is not installed', allow_module_level=True)

from conceptmod.textsliders import tiny_llm_shared as product
from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend, DummyTinyCausalLM
from particle_sliders.fibo_hndl import HNDLParticleProjection
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, UnsupportedOperation
from particle_sliders.pg09 import _hash
from particle_sliders.pg09_audio import AudioResponseCritic
from particle_sliders.response import open_response

ROWS = [dict(neutral_ids=[1, 2], positive_ids=[1, 3]),
        dict(neutral_ids=[4, 5], positive_ids=[4, 6])]


@pytest.fixture(autouse=True)
def cpu_bounded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def session(**options):
    return open_response(product.config(fixture=True, steps=4, ema_decay=.9, **options), rows=ROWS)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def test_installed_wheel_and_registry_no_optional_model_imports():
    entries = {(ep.group, ep.name) for ep in distribution('tinyqwen-shared-preview').entry_points}
    assert ('particle_sliders.response_backends', product.NAME) in entries
    assert ('particle_sliders.inference_backends', product.NAME) in entries
    if os.environ.get('TINYQWEN_EXPECT_INSTALLED') == '1':
        assert '/site-packages/' in product.__file__
    assert not torch.cuda.is_initialized()
    assert 'transformers' not in sys.modules
    assert 'conceptmod.textsliders.particle_bridge_gan' not in sys.modules
    builder, identity = product.verify_runtime()
    assert builder.__module__ == 'hypergan.hndl_networks'
    assert identity['hndl_version'] == '0.10.0'
    assert identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'


def test_real_hndl_modules_critic_output_vjp_and_second_derivatives():
    torch.manual_seed(11)
    reference = AudioResponseCritic(32, 36).double()
    actual = product.TinyQwenHNDLCritic(32, 36).double()
    assert isinstance(actual.graph['hidden'], nn.Linear)
    assert isinstance(actual.graph['score'], nn.Linear)
    for name, layer in [('hidden', reference.net[0]), ('score', reference.net[2])]:
        actual.graph[name].load_state_dict(layer.state_dict())
    x = torch.randn(2, 32, dtype=torch.float64, requires_grad=True)
    c = torch.randn(2, 36, dtype=torch.float64, requires_grad=True)
    outputs = [owner(x, c) for owner in (reference, actual)]
    torch.testing.assert_close(*outputs, rtol=0, atol=0)
    derivatives = []
    for owner, out in zip((reference, actual), outputs):
        gx, gc = torch.autograd.grad(out.sum(), (x, c), create_graph=True)
        # Squared input-gradient functional probes penalty-relevant mixed Hessians,
        # without inventing or duplicating the shared penalty formula.
        second = torch.autograd.grad((gx.square().sum() + gc.square().sum()),
                                     (x, c, *owner.parameters()), allow_unused=True)
        derivatives.append((gx, gc, second))
    for a, b in zip(derivatives[0][:2], derivatives[1][:2]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(derivatives[0][2], derivatives[1][2]):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert 'condition_shape' in actual.extra_repr()
    assert product.CRITIC_GRAPH in json.loads(actual.extra_repr())['plan']



def test_actual_shared_bispf_penalty_and_parameter_second_derivatives():
    torch.manual_seed(13)
    reference = AudioResponseCritic(32,36)
    actual = product.TinyQwenHNDLCritic(32,36)
    actual.graph['hidden'].load_state_dict(reference.net[0].state_dict())
    actual.graph['score'].load_state_dict(reference.net[2].state_dict())
    real, fake, condition = torch.randn(2,32), torch.randn(2,32), torch.randn(2,36)
    results = []
    with session() as s:
        for critic in (reference,actual):
            optimizer = s.game.recipe.make_critic_optimizer(critic)
            penalty = s.game.recipe.make_critic_penalty(optimizer)
            value = penalty(critic,real,fake,condition)
            assert value.item() > 0
            gradients = torch.autograd.grad(value,tuple(critic.parameters()),allow_unused=True)
            results.append((value,gradients))
    torch.testing.assert_close(results[0][0],results[1][0],rtol=0,atol=0)
    for a,b in zip(results[0][1],results[1][1]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_projection_matches_historical_architecture_outputs_input_and_bank_vjp():
    # Import historical code only for this comparison, never on the modern runtime path.
    root = Path(os.environ.get('TINYQWEN_SOURCE_ROOT', Path(__file__).resolve().parents[1]))
    bridge = ast.parse((root/'conceptmod/textsliders/particle_bridge_gan.py').read_text())
    projection = ast.parse((root/'conceptmod/textsliders/tiny_llm_particle.py').read_text())
    # Execute the exact archived definitions, without importing its unrelated game stack.
    namespace = dict(nn=nn, torch=torch, math=math)
    selected = [node for node in bridge.body if getattr(node,'name',None) in ('mlp','RoutedMLP')]
    exec(compile(ast.Module(body=selected,type_ignores=[]), '<archived bridge>', 'exec'),namespace)
    namespace['shared'] = SimpleNamespace(RoutedMLP=namespace['RoutedMLP'])
    selected = [node for node in projection.body if getattr(node,'name',None) == 'TinyParticleProjection']
    exec(compile(ast.Module(body=selected,type_ignores=[]), '<archived projection>', 'exec'),namespace)
    TinyParticleProjection = namespace['TinyParticleProjection']
    torch.manual_seed(12)
    host = nn.Linear(32, 16, bias=False).double().requires_grad_(False)
    old_bank = nn.Parameter(torch.randn(128, 4, dtype=torch.float64))
    bank = nn.Parameter(old_bank.detach().clone())
    old = TinyParticleProjection('old', deepcopy(host), old_bank, 8, 8).double()
    new = HNDLParticleProjection('modern', deepcopy(host), bank, 8, 8, attach=False).double()
    with torch.no_grad():
        old.lora_up.weight.normal_(std=.1)
    new.graph['down'].load_state_dict(old.lora_down.state_dict())
    new.graph['up'].load_state_dict(old.lora_up.state_dict())
    for branch, prefix in [(old.bridge.router, 'router'), (old.bridge.net, 'bridge')]:
        for index in (0, 2, 4, 6):
            new.graph[prefix + str(index)].load_state_dict(branch[index].state_dict())
    old.multiplier = new.multiplier = .5
    x = torch.randn(2, 3, 32, dtype=torch.float64, requires_grad=True)
    left, right = old(x), new(x)
    torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    v = torch.randn_like(left)
    a = torch.autograd.grad((left * v).sum(), (x, old_bank))
    b = torch.autograd.grad((right * v).sum(), (x, bank))
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-12)


def test_actual_shared_update_bank_and_native_gradient_ownership_zero_and_close():
    with session() as s:
        native = s.game.backend
        model = native.backend.model
        slider = native.slider
        assert len(slider.adapters) == 8
        assert all(isinstance(a, HNDLParticleProjection) for a in slider.adapters.values())
        assert all(any(n.endswith('.'+leaf) for leaf in ('q_proj','k_proj','v_proj','o_proj'))
                   for n in slider.target_names)
        assert len([p for p in slider.parameters() if p is slider.particles]) == 1
        assert all(not p.requires_grad for p in model.parameters())
        base = deepcopy(model.state_dict())
        initial = deepcopy(slider.state_dict())
        zero = s.evaluate(strength=0)
        with slider.scaled(.5):
            assert all(a.multiplier == .5 for a in slider.adapters.values())
            with slider.scaled(0):
                assert all(a.multiplier == 0 for a in slider.adapters.values())
            assert all(a.multiplier == .5 for a in slider.adapters.values())
        s.train()
        assert not torch.equal(initial['particles'], slider.particles)
        assert any(not torch.equal(initial[k], v) for k, v in slider.state_dict().items() if k != 'particles')
        equal(model.state_dict(), base)
        assert all(p.grad is None for p in model.parameters())
        equal(s.evaluate(strength=0), zero)
        assert not torch.equal(s.evaluate(strength=1), zero)
        # Native response backpropagates through q/k/v/o to one real bank.
        s.game.network.zero_grad(set_to_none=True)
        with s.game.network.scaled(1):
            native.generated(s.game.rows[0]).square().sum().backward()
        assert slider.particles.grad is not None
        assert torch.count_nonzero(slider.particles.grad)
    assert slider._closed
    assert all(module.forward == original for module, original, _ in slider._hosts)
    slider.close()


def test_exact_split_resume_and_raw_ema_inference_without_training_rows(tmp_path, monkeypatch):
    with session() as whole:
        whole.train()
        expected = whole.checkpoint()
    with session() as split:
        split.train(stop_after=2)
        path = tmp_path / 'resume.pt'
        split.checkpoint(path)
    with open_response(product.config(fixture=True,steps=4,ema_decay=.9),resume=path,rows=ROWS) as resumed:
        resumed.train()
        equal(resumed.checkpoint(), expected)
        for selection in ('raw', 'ema'):
            artifact = resumed.export_inference(selection=selection)
            if selection == 'raw':
                probes = {v:resumed.evaluate(strength=v) for v in (0.,.5,1.)}
            else:
                original = deepcopy(resumed.game.network.state_dict())
                resumed.game.network.load_state_dict(resumed.ema.values)
                probes = {v:resumed.evaluate(strength=v) for v in (0.,.5,1.)}
                resumed.game.network.load_state_dict(original)
            # Poison training and critic construction: native-only inference must stand alone.
            with monkeypatch.context() as m:
                m.setattr(product, 'prepare', lambda *a, **k: pytest.fail('training factory called'))
                m.setattr(product, 'TinyQwenHNDLCritic', lambda *a, **k: pytest.fail('critic allocated'))
                m.setattr(product, 'ConditionalPG094', lambda *a, **k: pytest.fail('game allocated'))
                with open_inference(artifact, operation='response') as inference:
                    assert not hasattr(inference, 'game')
                    for v in (0.,.5,1.):
                        equal(inference.response(ids=ROWS[0]['neutral_ids'],strength=v), probes[v])
                    with pytest.raises(UnsupportedOperation):
                        inference.train()


@pytest.mark.parametrize('option,value', [('pg_version','0.9.3'),('penalty_input_domain','continuous'),
    ('pg_source_profile','unqualified'),('base_precision','bfloat16'),('fixture','true'),('model','different')])
def test_changed_profile_rejected_before_native_construction(option, value, monkeypatch):
    monkeypatch.setattr(product, '_native', lambda *a: pytest.fail('model allocated'))
    options = dict(fixture=True)
    options[option] = value
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(product.config(**options), rows=ROWS)


def test_native_wide_refused_before_loader_or_hooks(monkeypatch):
    monkeypatch.setattr(product, 'TinyLLMBackend', lambda **k: pytest.fail('model loaded'))
    with pytest.raises(ValueError,match='170|continuous|width'):
        open_response(product.config(), rows=ROWS)
    model = DummyTinyCausalLM(hidden_size=176, num_hidden_layers=1, intermediate_size=16)
    before = [(m,m.forward) for m in model.modules() if isinstance(m,nn.Linear)]
    flags = [p.requires_grad for p in model.parameters()]
    with pytest.raises(ValueError,match='170|continuous|width'):
        product.TinyQwenHNDLSlider(model)
    assert all(m.forward == f for m,f in before)
    assert [p.requires_grad for p in model.parameters()] == flags


def test_checkpoint_source_model_shape_and_historical_rejection_before_factory(tmp_path, monkeypatch):
    with session() as s:
        packet = s.checkpoint()
        before = deepcopy(s.game.network.state_dict())
        for key, value in [('cohort','historical-rp-vic'),('backend','bonsai')]:
            bad = deepcopy(packet)
            bad['manifest']['config'][key] = value
            bad['sha256'] = _hash({k:v for k,v in bad.items() if k != 'sha256'})
            with pytest.raises(CheckpointError):
                s.restore(bad)
            equal(s.game.network.state_dict(), before)
        for key in ('implementation_sources','architectures','backend','response_shape'):
            bad = deepcopy(packet)
            bad['payload']['game']['config'][key] = {}
            bad['sha256'] = _hash({k:v for k,v in bad.items() if k != 'sha256'})
            with pytest.raises(CheckpointError):
                s.restore(bad)
            equal(s.game.network.state_dict(), before)
        bad = deepcopy(packet)
        bad['payload']['game']['network']['weights.0'] = torch.zeros(127,4)
        bad['sha256'] = _hash({k:v for k,v in bad.items() if k != 'sha256'})
        with pytest.raises(CheckpointError):
            s.restore(bad)
        equal(s.game.network.state_dict(), before)
        artifact = s.export_inference()
    # Model/source/shape identity changes refused before any parameter load; hooks rolled back.
    attach_calls = []
    original_slider = product.TinyQwenHNDLSlider
    def track_slider(model, *, attach=True):
        attach_calls.append(attach)
        return original_slider(model, attach=attach)
    monkeypatch.setattr(product, 'TinyQwenHNDLSlider', track_slider)
    for key in ('model','sources','network_schema'):
        bad = deepcopy(artifact)
        bad['specification']['identity'][key] = {}
        bad['sha256'] = _hash({k:v for k,v in bad.items() if k != 'sha256'})
        with pytest.raises((ValueError,CheckpointError)):
            open_inference(bad, operation='response')
    assert attach_calls == [False, False, False]
    legacy = tmp_path/'legacy.pt'
    torch.save({'format':'conceptmod-tiny-llm-particle-v1'},legacy)
    monkeypatch.setattr(product, '_native', lambda *a: pytest.fail('model allocated'))
    with pytest.raises(CheckpointError):
        open_response(product.config(fixture=True), resume=legacy, rows=ROWS)


def test_failed_graph_construction_and_failed_prepare_restore_borrowed_state(monkeypatch):
    backend = TinyLLMBackend(dummy=True)
    backend.model.train().requires_grad_(True)
    forwards = [(m,m.forward) for m in backend.model.modules() if isinstance(m,nn.Linear)]
    original, identity = product.verify_runtime()
    calls = []
    def broken(*a,**k):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError('broken graph')
        return original(*a,**k)
    with monkeypatch.context() as m:
        m.setattr(product,'verify_runtime',lambda:(broken,identity))
        with pytest.raises(ValueError,match='broken graph'):
            product.TinyQwenHNDLSlider(backend.model)
    assert all(m.forward == f for m,f in forwards)
    assert backend.model.training and all(p.requires_grad for p in backend.model.parameters())
    slider = product.TinyQwenHNDLSlider(backend.model)
    with pytest.raises(ValueError,match='already wrapped'):
        product.TinyQwenHNDLSlider(backend.model)
    slider.close()
    assert backend.model.training and all(p.requires_grad for p in backend.model.parameters())
    assert all(m.forward == f for m,f in forwards)
    observed = []
    native_factory = product._native
    def capture(opts):
        native = native_factory(opts)
        observed.append(native)
        return native
    monkeypatch.setattr(product,'_native',capture)
    with pytest.raises(ValueError):
        session_bad = open_response(product.config(fixture=True),rows=[ROWS[0]])
        session_bad.close()
    assert observed[0].slider._closed


@pytest.mark.parametrize('core_file',['pg09.py','pg094_release.json','fibo_projection.hndl'])
def test_core_and_builder_tamper_rejected_before_hooks(monkeypatch,core_file):
    # Alter only hash observations; no installed shared files are edited.
    original = product._digest
    with monkeypatch.context() as m:
        m.setattr(product,'_digest',lambda p:'0'*64 if str(p).endswith('/'+core_file) else original(p))
        with pytest.raises(ValueError,match='source closure'):
            product.verify_runtime()
    import particle_sliders.fibo_hndl as hndl
    digest = hndl._sha
    with monkeypatch.context() as m:
        m.setattr(hndl,'_sha',lambda p:'0'*64 if str(p).endswith('hndl_networks.py') else digest(p))
        with pytest.raises(ValueError,match='builder closure'):
            product.verify_runtime()


def test_rng_observation_capabilities_and_invalid_inputs():
    before = torch.get_rng_state().clone()
    with session() as s:
        equal(torch.get_rng_state(), before)
        s.train(stop_after=1)
        after_update = torch.get_rng_state().clone()
        zero = s.evaluate(strength=0)
        s.evaluate(strength=.5)
        equal(s.evaluate(strength=0),zero)
        equal(torch.get_rng_state(), after_update)
        artifact = s.export_inference()
        assert set(artifact) == {'schema','specification','origin','selection','network','sha256'}
        for strength in (-1,float('nan'),1.1):
            with pytest.raises(UnsupportedOperation):
                s.evaluate(strength=strength)
        with pytest.raises(UnsupportedOperation):
            s.render()
    with open_inference(artifact,operation='response') as inference:
        for ids in ([],[-1],[256],['text']):
            with pytest.raises(ValueError):
                inference.response(ids=ids)


def test_close_conflict_preserves_ownership_and_can_retry():
    backend = TinyLLMBackend(dummy=True)
    slider = product.TinyQwenHNDLSlider(backend.model)
    module, _, adapter = slider._hosts[0]
    wrapped = module.forward
    replacement = lambda x:x
    module.forward = replacement
    with pytest.raises(RuntimeError,match='ownership changed'):
        slider.close()
    assert module.forward is replacement
    assert all(m.forward == a.forward for m,_,a in slider._hosts[1:])
    module.forward = wrapped
    slider.close()
    slider.close()
    with pytest.raises(RuntimeError,match='closed'):
        with slider.scaled(1):
            pass
