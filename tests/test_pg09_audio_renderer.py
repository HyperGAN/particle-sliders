"""Native-random AR + synthetic acoustics only; no pretrained/native audio run."""
# ruff: noqa: E402 -- the official/product namespace must be bootstrapped first.
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

if sys.executable != '/ml2/sliders/.venv-yue2-retrain/bin/python':
    pytest.skip('renderer CPU qualification uses the isolated native YuE interpreter', allow_module_level=True)

from inspect_audio_pg09_assets import bootstrap_yue2_product
bootstrap_yue2_product('/mnt/nvme/sliders/audio-yue2-source-20261017')
from yue2.backend import YuE2Backend
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.pipeline import SemanticResult, SymbolicPlan
from yue2.protocol import GenerationConfig, token_prefixes
from yue2.slider import ParticleSlider
from particle_sliders import ConditionalPG09, winning_formulation
from conceptmod.textsliders.pg09_audio import AudioResponseCritic, AudioRow, YuE2ARResponseAdapter, audio_conditions
import render_yue2_pg09 as renderer

torch.set_num_threads(1)
ROW = dict(neutral='a guitar band', positive='heavy metal guitars', lyrics='a fixed lyric')


def clean_model():
    torch.manual_seed(23)
    return YuE2ForCausalLM(YuE2Config(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, intermediate_size=64,
        vocab_size=256, max_position_embeddings=256, max_latent_frames=64)).eval().requires_grad_(False)


def bind(model):
    backend = YuE2Backend(dummy=True)
    backend.model, backend.dummy = model, False
    torch.manual_seed(77)
    slider = ParticleSlider(model, winning_formulation())
    adapter = YuE2ARResponseAdapter(backend, slider, model_identity={'fixture':'renderer native-random'},
        checkpoint_modules=list(model.model.layers), **renderer.OPTIONS)
    rows = [AudioRow((11,13,17,19),(29,31,17,19), history_tokens=2),
            AudioRow((23,37,41,43),(53,59,41,43), history_tokens=2)]
    fixed = [adapter.prepare(row) for row in rows]
    heldout = [adapter.prepare(replace(row, neutral_ids=(61,)+row.neutral_ids[1:])) for row in rows]
    conditions = audio_conditions(fixed)
    critic = AudioResponseCritic(32, conditions.shape[1])
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions,
        particles=adapter.particles, response_domain='continuous_vector', preset='bispf_optimized',
        batch_size=2, total_steps=20, seed=7, **renderer.OPTIONS)
    return adapter, game, heldout


def producer_packets():
    model = clean_model()
    adapter, game, heldout = bind(model)
    for _ in range(20):
        game.step()  # Shared public game only, CPU random fixture.
    exported = game.export_state_dict()
    return exported, dict(game=game.state_dict(), export=exported,
        probes=[adapter.sample(row) for row in heldout],
        frozen_base_sha256=renderer.producer.frozen_digest(model, adapter.network))


class Tokenizer:
    def encode(self, text):
        return [29,31] if 'heavy' in text else [11,13]


class SyntheticPipe:
    backend, quantization, offload_ar, device = 'torch-eager', 'none', False, 'cpu'
    def __init__(self, mode=None):
        with torch.random.fork_rng():
            self._model = clean_model()
        self.tokenizer, self.adapter = Tokenizer(), None
        self.events, self.mode = [], mode

    def note(self, name):
        self.events.append((name, None if self.adapter is None else self.adapter.network.scale))
        random.random()
        np.random.random()
        torch.rand(3)

    def _load_model(self):
        return self._model

    def plan(self, request):
        self.note('plan')
        prefix = token_prefixes(request, self.tokenizer)
        if self.mode == 'prefix':
            prefix = prefix[:-1]
        return SymbolicPlan(request, None, [], prefix)

    def generate_semantic(self, plan, sampling, cancelled):
        self.note('semantic')
        assert plan.request.cfg_scale == 1. and plan.request.cot == 'off'
        assert sampling.max_tokens == 200 and sampling.min_tokens == 32
        cancelled()
        ids = torch.tensor([[i % 250 + 1 for i in plan.prefix]])
        # Genuine native random AR response; synthetic token mapping only.
        hidden = self._model.model(ids, position_ids=torch.arange(ids.shape[1]).unsqueeze(0), use_cache=False)
        value = hidden[0][0,-1,0]
        token = int(abs(float(value)) * 10000) % 32768
        tokens = [token] * (201 if self.mode == 'tokens' else 32)
        if self.mode == 'domain':
            tokens[0] = 32768
        return SemanticResult(plan, tokens, {}, False)

    def synthesize(self, semantic, cancelled):
        self.note('nar')
        cancelled()
        assert self.adapter is None or self.adapter.network.scale == 0
        if self.mode == 'bank' and self.adapter:
            self.adapter.particles.add_(.1)
        offset = 1 if self.mode == 'off' and self.adapter else 0
        return np.full((len(semantic.tokens),64), semantic.tokens[0]/32768 + offset, dtype=np.float32)

    def decode(self, latents):
        self.note('vae')
        assert self.adapter is None or self.adapter.network.scale == 0
        size = 400000 if self.mode == 'duration' else len(latents)*1920
        return np.full((size,2), latents[0,0], dtype=np.float32)

    def effective_config(self, request, semantic_sampling):
        value = GenerationConfig(semantic=semantic_sampling).to_dict()
        return dict(cot='off', cfg_scale=1.01 if self.mode == 'cfg' else 1.,
                    backend=self.backend, generation=value)


def run(exported, qualification, mode=None):
    pipe = SyntheticPipe(mode)
    def attach():
        assert all(scale is None for _,scale in pipe.events)
        adapter, game, heldout = bind(pipe._model)
        pipe.adapter = adapter
        return adapter, game, heldout
    result = renderer.render_case(pipe, attach, exported, qualification, ROW, 0)
    return pipe, result


@pytest.fixture(scope='module')
def packets():
    return producer_packets()


def test_public20_export_clean_zero_ar_only_scale_rng_and_native_random_host(packets):
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    pipe, (clips, receipt) = run(*packets)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])
    assert set(clips) == {'base','off','half','on','positive_caption'}
    assert [scale for stage,scale in pipe.events if stage == 'semantic'] == [None,0.,.5,1.,0.]
    assert all(scale in (None,0.) for stage,scale in pipe.events if stage in ('nar','vae'))
    assert receipt['public_export_hidden_replay_exact'] and receipt['frozen_base_unchanged']
    assert receipt['native_zero_tokens_latents_waveform_exact'] and receipt['adapter_unchanged']
    assert receipt['semantic_quality_qualified'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('mode,match', [('prefix','plan'),('tokens','token'),('domain','token'),
    ('duration','duration'),('cfg','protocol'),('off','adapter-off'),('bank','changed during inference')])
def test_invalid_native_transport_fails_without_fallback(packets, mode, match):
    with pytest.raises(ValueError, match=match):
        run(*packets, mode=mode)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault', ['steps','source','probes'])
def test_export_protocol_source_or_probe_mismatch_is_rejected(packets, fault):
    exported, qualification = deepcopy(packets)
    if fault == 'steps':
        exported['completed_steps'] = 19
    elif fault == 'source':
        exported['config']['implementation_sources']['particle_sliders.pg09'] = 'wrong'
        qualification['export'] = deepcopy(exported)
        qualification['game']['config'] = deepcopy(exported['config'])
    else:
        qualification['probes'][0] = qualification['probes'][0]+1
    with pytest.raises(ValueError):
        run(exported, qualification)


def test_explicit_root_card_required_before_cuda_or_model_construction(tmp_path):
    card = tmp_path/'not-admitted.json'
    card.write_text(json.dumps(dict(status='PROPOSED_NOT_ADMITTED')))
    result = subprocess.run([sys.executable, renderer.__file__, '--card', str(card)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True, capture_output=True, timeout=30)
    assert result.returncode and 'separate reviewed root renderer admission' in result.stderr


def test_continuous_host_floor_is_eight_gib(monkeypatch):
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path,*a,**k:
        'MemAvailable: 7340032 kB\n' if str(path)=='/proc/meminfo' else original(path,*a,**k))
    with pytest.raises(RuntimeError, match='8GiB'):
        renderer.host_floor(8)


def test_root_card_cannot_retain_old_four_gib_running_floor():
    card = dict(schema='root-yue2-pg09-render-admission-v1', status='REVIEWED_BOUNDED_AUDIO_RENDER',
        admission_id='cpu-schema-test-only', protocol=renderer.PROTOCOL, case=0,
        output_directory=str(renderer.RUNS/'yue2-matched-case-0'), limits=dict(
            host_running_available_gib_minimum=4, startup_host_available_gib_minimum=16,
            startup_gpu_free_gib_minimum=20, compute_seconds=720, maximum_wall_seconds=900,
            shutdown_grace_seconds=180))
    with pytest.raises(ValueError, match='resource/wall'):
        renderer.validate_card(card)


def test_safe_export_fresh_process_renderer_import(packets, tmp_path):
    torch.save(packets, tmp_path/'packets.pt')
    code = ('import torch; from test_pg09_audio_renderer import run; '
            'p=torch.load('+repr(str(tmp_path/'packets.pt'))+',weights_only=True,map_location="cpu"); '
            '_,(_,r)=run(*p); assert r["native_zero_tokens_latents_waveform_exact"]; '
            'assert not torch.cuda.is_initialized()')
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr


def test_native_binding_matches_unchanged_producer_on_random_host(tmp_path, monkeypatch):
    """Exercise the actual producer/renderer binding with loaders stubbed before weights."""
    from types import SimpleNamespace
    import yaml
    import yue2.backend
    import yue2.storage
    identity = {'fixture':'no pretrained tensors'}
    manifest = dict(model_identity=identity, cache_dir=str(tmp_path), records={
        'model':dict(model='m-a-p/YuE2-3B', revision='cpu-fixture', path=str(tmp_path))})
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    train_path, heldout_path = tmp_path/'train.yaml', tmp_path/'heldout.yaml'
    for path,count in ((train_path,4),(heldout_path,2)):
        path.write_text(yaml.safe_dump({'rows':[dict(neutral=f'neutral{i}',positive=f'positive{i}',
                                                    lyrics='fixture lyric') for i in range(count)]}))
    def backend():
        value = YuE2Backend(dummy=True)
        value.model, value.dummy, value.identity = clean_model(), False, identity
        value.prefix = lambda style,lyrics: ([11,13,17,19] if style.startswith('neutral') else [29,31,17,19])
        value.continuation = lambda prefix,count,seed: [71]*count
        return value
    first = backend()
    monkeypatch.setattr(yue2.backend,'YuE2Backend',lambda *a,**k:first)
    monkeypatch.setattr(yue2.storage,'resolve_model',lambda *a,**k:tmp_path)
    monkeypatch.setattr(yue2.storage,'model_identity',lambda *a,**k:identity)
    args = SimpleNamespace(family='yue2', manifest=manifest_path, model_dir=tmp_path,
        product=Path('/mnt/nvme/sliders/audio-yue2-source-20261017'), device='cpu', seed=7,
        train_prompts=train_path,heldout_prompts=heldout_path,max_seq_len=512,
        preset='bispf_optimized',total_steps=20)
    adapter,_,_,heldout,game = renderer.producer.load_native(args,renderer.OPTIONS,lambda *unused:None)
    second = backend()
    other,other_game,other_heldout = renderer.bind_native(second,manifest,train_path,heldout_path,lambda *unused:None)
    assert game.state_dict()['config'] == other_game.state_dict()['config']
    for _ in range(20):
        game.step()
    exported = game.export_state_dict()
    qualification = dict(game=game.state_dict(), export=exported,
        probes=[adapter.sample(row) for row in heldout],
        frozen_base_sha256=renderer.producer.frozen_digest(first.model,adapter.network))
    renderer.load_public_export(other_game,other,other_heldout,exported,qualification,
        renderer.producer.frozen_digest(second.model,other.network))
    assert all(torch.equal(a,b) for a,b in zip(qualification['probes'],
                                              [other.sample(row) for row in other_heldout]))
    assert not torch.cuda.is_initialized()
