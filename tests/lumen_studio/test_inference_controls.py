import pytest
import torch
from fastapi.testclient import TestClient

from conftest import TinyRuntime
from lumen_studio.api import create_app
from lumen_studio.contracts import normalized_strengths
from lumen_studio.coordinator import Coordinator
from lumen_studio.inference_controls import inference_strengths, set_inference_mix
from lumen_studio.particles import ParticleAdapter


def test_extended_budget_preserves_pinned_range_and_rejects_invalid_input():
    mix = dict(candlelit=2., moonlit=3., theatrical=5.)
    for energy in (0., .25, .5, 1.):
        assert inference_strengths(mix, energy) == normalized_strengths(mix, energy)
    for energy in (2., 3., 10.):
        values = inference_strengths(mix, energy)
        assert sum(values.values()) == pytest.approx(energy)
        assert values['candlelit'] == pytest.approx(.2 * energy)
        assert not any(inference_strengths({}, energy).values())
    assert inference_strengths(dict(candlelit=1e308, moonlit=1e308), 10)['candlelit'] == 5.
    for energy in (-.1, 10.01, float('inf'), float('nan')):
        with pytest.raises(ValueError):
            inference_strengths(mix, energy)


def test_renderer_applies_tenfold_branch_and_retains_exact_off(tiny):
    adapter = ParticleAdapter(tiny.transformer)
    for branch in adapter.branches:
        torch.nn.init.normal_(branch.up.weight, std=.1)
    tiny.mixer.add('candlelit', adapter)
    inputs = torch.randn(1, 3, 4)
    base = tiny.transformer.to_q(inputs)
    set_inference_mix(tiny.mixer, dict(candlelit=1), 1)
    unit = tiny.transformer.to_q(inputs)
    set_inference_mix(tiny.mixer, dict(candlelit=1), 10)
    boosted = tiny.transformer.to_q(inputs)
    torch.testing.assert_close(boosted - base, 10 * (unit - base), rtol=1e-5, atol=1e-6)
    assert not torch.equal(boosted, unit)
    set_inference_mix(tiny.mixer, dict(candlelit=1), 0)
    assert torch.equal(tiny.transformer.to_q(inputs), base)
    with pytest.raises(ValueError, match='checkpoint'):
        set_inference_mix(tiny.mixer, dict(moonlit=1), 10)


def test_extended_energy_queue_render_rerender_and_comparison(tmp_path):
    captured = []

    class Runtime(TinyRuntime):
        def render(self, *args, **kwargs):
            captured.append(dict(self.mixer.strengths))
            return super().render(*args, **kwargs)

    source = Runtime()
    adapter = ParticleAdapter(source.transformer)
    checkpoint = tmp_path / 'ema.safetensors'
    adapter.export(checkpoint, model_identity=source.identity, normalization={'test': True},
                   manifest_hash='a' * 64, step=200, ema=adapter.state_dict())
    studio = tmp_path / 'studio'
    app = create_app(studio, model_identity=source.identity)
    sha = app.state.store.register_checkpoint(checkpoint, 'candlelit')
    client = TestClient(app)
    request = dict(prompt='a fully clothed adult woman', seed=42, energy=10,
                   mix=dict(candlelit=1), checkpoints=dict(candlelit=sha))
    response = client.post('/api/comparisons', json=request)
    assert response.status_code == 202
    off, on = response.json()['ids']
    worker = Coordinator(studio, tmp_path / 'model', gpu='cpu', runtime_factory=Runtime)
    try:
        worker.render_one(); worker.render_one()
        assert captured[-2]['candlelit'] == 0 and captured[-1]['candlelit'] == 10
        assert app.state.store.image(on)['metadata']['strengths']['candlelit'] == 10
        assert client.get(f'/api/images/{off}/comparison').json()['id'] == on
        assert client.get(f'/api/images/{on}/comparison').json()['id'] == off
        result = client.post(f'/api/images/{on}/rerender')
        assert result.status_code == 202
        assert app.state.store.job(result.json()['ids'][0])['payload']['energy'] == 10
        sweep = client.post('/api/comparisons', json=dict(request, mode='sweep')).json()['ids']
        assert [app.state.store.job(i)['payload']['energy'] for i in sweep] == [0, 2.5, 5, 10]
        assert client.post('/api/generate', json=dict(request, energy=10.01)).status_code == 422
    finally:
        worker.release_runtime()
        source.close()
