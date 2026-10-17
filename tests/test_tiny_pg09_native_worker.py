import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tiny_native_worker', ROOT / 'scripts/verify_tiny_pg09_native.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Tokens:
    eos_token_id = 1

    def encode(self, text, **kwargs):
        return [2 + ord(c) % 90 for c in text][:16]

    def decode(self, ids, **kwargs):
        return ','.join(map(str, [int(i) for i in ids]))


def native_fixture():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = Qwen3Config(hidden_size=64, intermediate_size=96, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                         vocab_size=128, max_position_embeddings=128,
                         bos_token_id=0, eos_token_id=1, pad_token_id=1,
                         attention_dropout=0., tie_word_embeddings=True)
    backend = TinyLLMBackend.__new__(TinyLLMBackend)
    backend.model = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    backend.device, backend.dummy = torch.device('cpu'), False
    backend.tokenizer = Tokens()
    return backend


def test_real_random_qwen_attention_generation_game_and_reload(monkeypatch):
    from particle_sliders.pg09 import ConditionalPG09
    from conceptmod.textsliders.tiny_llm_particle import TinyParticleSlider
    from conceptmod.textsliders.pg09_text_video import TinyQwenResponseAdapter
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic
    backend = native_fixture()
    data = dict(train=[dict(neutral='garden', positive='rose garden'),
                       dict(neutral='street', positive='rainy street')],
                validation=[dict(neutral='sea', positive='calm sea')], final_holdout=[])
    controls = dict(worker.GENERATION, max_new_tokens=3)
    baseline = worker.generate_text(backend, None, 'garden', 0, controls)
    base = worker.base_digest(backend.model)
    slider = TinyParticleSlider(backend.model).float()
    assert len(slider.target_names) == 4
    assert slider.adapters['lora_tiny-model-layers-0-self_attn-q_proj'].lora_up.out_features == 128
    adapter = TinyQwenResponseAdapter(backend, slider, model_identity={'fixture': 'random HF Qwen3'},
                        pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    selected = worker.evaluation_rows(backend, data)
    fixed = {split: [adapter.prepare(row) for row, _ in rows] for split, rows in selected.items()}
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([1., 0.])))[None]
                            for r in fixed['train']])
    critic = ConditionalReadoutCritic(64, conditions.shape[1])
    game = ConditionalPG09(adapter, fixed['train'], critic, particles=adapter.particles,
                          conditions=conditions, **worker.DOMAIN, total_steps=1200,
                          preset='bispf_optimized', seed=7, batch_size=2)
    bank = slider.particles.detach().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('MSE optimized')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    first = worker.sample(adapter, fixed, [], ['garden'], controls)
    worker.equal(first['text'][0], baseline)
    for _ in range(2):
        metrics = game.step()
        assert metrics['g_loss'] == pytest.approx(metrics['g_adv'] + metrics['prior_regularization'])
        assert metrics['d_loss'] == pytest.approx(metrics['d_adv'] + metrics['d_pen'])
    assert not torch.equal(bank, slider.particles)
    saved = game.state_dict()
    rng = torch.get_rng_state().clone()
    sample = worker.sample(adapter, fixed, [], ['garden'], controls)
    worker.equal(saved, game.state_dict())
    worker.equal(rng, torch.get_rng_state())
    worker.equal(sample['text'][0], baseline)
    assert sample['diagnostics']['train'][0]['edit_rms'] > 0
    export = game.export_state_dict()
    with torch.no_grad():
        for p in adapter.network.parameters():
            p.zero_()
    game.load_export_state_dict(export)
    worker.equal(sample, worker.sample(adapter, fixed, [], ['garden'], controls))
    game.load_state_dict(saved)
    expected = game.step()
    expected_state = game.state_dict()
    game.load_state_dict(saved)
    worker.equal(expected, game.step())
    worker.equal(expected_state, game.state_dict())
    assert worker.base_digest(backend.model) == base
    assert not torch.cuda.is_initialized()


def test_final_holdout_never_encoded():
    seen = []
    backend = SimpleNamespace(encode=lambda text: seen.append(text) or SimpleNamespace(ids=[len(seen)]))
    data = dict(train=[dict(neutral='train0', positive='train1')],
                validation=[dict(neutral='val0', positive='val1')],
                final_holdout=[dict(neutral='NEVER0', positive='NEVER1')])
    assert set(worker.evaluation_rows(backend, data)) == {'train', 'validation'}
    assert seen == ['train0', 'train1', 'val0', 'val1']


def test_dataset_rejects_leakage():
    path = ROOT / 'evidence/pg09-text-video/tiny-native-smoke-dataset.json'
    data = worker.validate_dataset(json.loads(path.read_text()))
    data['final_holdout'][0] = data['train'][0]
    with pytest.raises(ValueError, match='disjoint'):
        worker.validate_dataset(data)


def test_streamed_hash_and_base_buffer_change(tmp_path):
    path = tmp_path / 'data'
    path.write_bytes(b'tiny fixture')
    import hashlib
    assert worker.digest(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    model = torch.nn.Linear(4, 3)
    original = worker.base_digest(model)
    with torch.no_grad():
        model.bias[0] += 1
    assert worker.base_digest(model) != original


def test_resource_guard_tracks_nvme_and_rejects_tmp():
    # Metadata-only actual-host check; no tensor/model payload.
    spec = importlib.util.spec_from_file_location('bonsai_guard', ROOT / 'scripts/verify_bonsai_pg09_cached.py')
    guards = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guards)
    nvme = guards.inspect_output_filesystem('/mnt/nvme/sliders/qualification/runs/tiny-not-created/example')
    assert nvme['device'] == '/dev/nvme0n1p2' and nvme['mountpoint'] == '/mnt/nvme'
    tmp = guards.inspect_output_filesystem('/tmp/tiny-not-created/example')
    assert tmp['device'] != '/dev/nvme0n1p2'


def test_frozen_plan_preflight_never_hashes_or_loads_large_weights(monkeypatch):
    path = ROOT / 'evidence/pg09-text-video/tiny-native-cpu-smoke-proposal.json'
    plan = json.loads(path.read_text())
    for name, value in plan['environment'].items():
        monkeypatch.setenv(name, value)
    original = worker.digest

    def small_only(path):
        assert Path(path).name != 'model.safetensors'
        return original(path)
    monkeypatch.setattr(worker, 'digest', small_only)
    # Torch is already imported by pytest; preflight itself has no Torch import.
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    worker.load_plan(path)
    receipt = worker.preflight(plan)
    assert receipt['torch_imported'] is False and receipt['model_loaded'] is False
    assert receipt['final_holdout_forwarded'] is False
    assert receipt['resource_preflight']['output_filesystem']['device'] == '/dev/nvme0n1p2'


def test_plan_rejects_overrides_and_missing_required_pin(tmp_path, monkeypatch):
    path = ROOT / 'evidence/pg09-text-video/tiny-native-cpu-smoke-proposal.json'
    original = json.loads(path.read_text())
    for name, value in original['environment'].items():
        monkeypatch.setenv(name, value)
    plan = json.loads(path.read_text())
    plan['training']['stop_after'] = 5
    changed = tmp_path / 'changed.json'
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='fixed recipe'):
        worker.load_plan(changed)
    plan = json.loads(path.read_text())
    del plan['source_sha256'][str(ROOT / 'scripts/verify_tiny_pg09_native.py')]
    changed.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='omits required'):
        worker.load_plan(changed)
