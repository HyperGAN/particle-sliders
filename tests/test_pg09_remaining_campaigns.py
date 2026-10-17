"""CPU native architecture fixtures; no pretrained weights or live case placeholders."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from conceptmod.textsliders.pg09_moss_campaign import file_digest
from conceptmod.textsliders.pg09_remaining_campaigns import (
    SupraCampaignRows, build_supra_campaign, required_supra_prompts,
    export_admitted_supra_contexts, load_supra_contexts, teacher_parity,
    build_final_form_campaign,
)


class FixtureRuntime:
    def __init__(self, config):
        from conceptmod.textsliders.supra_model import SupraDiT, attach_supra_lora
        torch.manual_seed(71)
        self.device = torch.device('cpu')
        self.model = SupraDiT(latent_ch=4, d_model=32, depth=1, n_heads=4, ctx_dim=8,
                              mlp_ratio=2., num_tokens=256, patch=2)
        self.loras = attach_supra_lora(self.model, rank=2, alpha=2,
                                      targets=('ctx_proj', 'cross_attn.q', 'cross_attn.kv',
                                               'cross_attn.proj', 'self_attn.qkv', 'self_attn.proj'))
        self.cache = {p: (torch.randn(1, 3, 8), torch.ones(1, 3)) for p in required_supra_prompts(config)}
        self.stored_unconditional = True
        self.vae = SimpleNamespace(decode=lambda z: SimpleNamespace(sample=z[:, :3]))
    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]
    def encode(self, prompt):
        assert prompt in self.cache, 'fixture cannot silently encode absent frozen context'
        return self.cache[prompt]
    def noise(self, seed):
        return torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(seed))
    @contextmanager
    def scale(self, value):
        previous = [m.multiplier for m in self.loras]
        for m in self.loras:
            m.multiplier = value
        try:
            yield
        finally:
            for m, old in zip(self.loras, previous):
                m.multiplier = old
    def velocity(self, prompt, latent, time):
        with self.scale(0), torch.no_grad():
            context, mask = self.encode(prompt)
            negative, nmask = self.encode('')
            raw = self.model(torch.cat((latent, latent)), torch.tensor([time, time]),
                             torch.cat((context, negative)), torch.cat((mask, nmask)))
            positive, negative = raw.chunk(2)
            return negative + 3 * (positive - negative)


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    config = dict(rows=[dict(name='knight', neutral='plain knight', positive='boss knight')],
                  preservation=['lake'], verification=[dict(name='bridge', prompt='guardian bridge', teacher='boss bridge'),
                                                       dict(name='fruit_control', prompt='fruit')])
    runtime = FixtureRuntime(config)
    edited, holds = [], []
    for path in ('neutral', 'positive'):
        for t in (.0, .9):
            z = runtime.noise(73) + t
            edited.append(dict(prompt='plain knight', row='knight', seed=73, context=path, z=z, t=t,
                               neutral=runtime.velocity('plain knight', z, t),
                               target=runtime.velocity('boss knight', z, t), hold=False))
    for t in (.0, .9):
        z = runtime.noise(77)
        base = runtime.velocity('lake', z, t)
        holds.append(dict(prompt='lake', z=z, t=t, neutral=base, target=base.clone(), hold=True))
    path = tmp_path / 'fixture-teacher.pt'
    torch.save(dict(records=edited, holds=holds), path)
    return config, runtime, path


def rows_for(fixture, split='train'):
    config, runtime, path = fixture
    return SupraCampaignRows(path, cache_sha256=file_digest(path), config=config,
                             contexts=runtime.cache, split=split)


def test_native_rows_keep_cfg_time_target_pairing_and_useful_frozen_features(fixture):
    rows = rows_for(fixture)
    assert len(rows) == 2 * 4 + 4 * 2
    assert rows.response_shape == (1, 4, 32, 32)
    row = rows[3]
    assert row.row.timestep == pytest.approx(.9)
    assert row.row.neutral == 'plain knight' and row.row.positive == 'boss knight'
    assert row.row.negative == '' and row.row.guidance == 3 and row.row.scale == 1
    features = rows.conditions([0, 1, 2, 8])
    assert features.shape == (4, 14) and not features.requires_grad
    assert not torch.equal(features[0], features[1])  # Native time.
    assert not torch.equal(features[0], features[2])  # Frozen trajectory role.
    assert features[3, -1] == 1  # Preservation role, no row ID.
    identity = rows.identity()
    identity['policy']['edit_repetitions'] = 100
    assert rows.identity()['policy']['edit_repetitions'] == 2
    changed = rows[0]
    changed.target.add_(100)
    assert not torch.equal(changed.target, rows[0].target)
    assert rows.response_scale() > 0
    with pytest.raises(ValueError, match='training split'):
        rows_for(fixture, 'validation').response_scale()


def test_native_context_export_roundtrip_and_missing_context_is_rejected(fixture, tmp_path):
    config, runtime, _ = fixture
    export = tmp_path / 'contexts.safetensors'
    receipt = export_admitted_supra_contexts(runtime, config, export,
                                               runtime_identity=dict(model_revision='tiny CPU fixture', text_revision='tiny CPU fixture'))
    loaded = load_supra_contexts(export, receipt['sha256'], config)
    assert set(loaded) == set(required_supra_prompts(config))
    assert all(torch.equal(a, b) for p in loaded for a, b in zip(loaded[p], runtime.cache[p]))
    with pytest.raises(FileExistsError):
        export_admitted_supra_contexts(runtime, config, export,
                                               runtime_identity=dict(model_revision='tiny CPU fixture', text_revision='tiny CPU fixture'))
    runtime.cache.pop('lake')
    with pytest.raises(ValueError, match='lacks frozen context'):
        build_supra_campaign(runtime, rows_for((config, FixtureRuntime(config), fixture[2])),
                             model_identity=dict(frozen_parameter_sha256='owner-verified'))


def test_shared_gan_updates_only_owned_lora_and_critic_with_exact_resume(fixture, monkeypatch):
    from test_pg09_backends import equal
    config, runtime, _ = fixture
    rows = rows_for(fixture)
    def forbidden(*args, **kwargs):
        raise AssertionError('optimized MSE forbidden')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    frozen = {n: p.detach().clone() for n, p in runtime.model.named_parameters() if not p.requires_grad}
    before = deepcopy({n: p.detach().clone() for n, p in runtime.model.named_parameters() if p.requires_grad})
    run = build_supra_campaign(runtime, rows, model_identity=dict(frozen_parameter_sha256='tiny fixture only'),
                               total_steps=6, critic_width=8)
    assert max(r['target_relative_rms'] for r in teacher_parity(run, (0, 3), relative_rms_limit=1e-4)) < 1e-4
    critic = deepcopy(run.game.critic.state_dict())
    for _ in range(3):
        metrics = run.step()
        assert metrics['g_loss'] == pytest.approx(metrics['g_adv'])
    assert any(not torch.equal(before[n], p) for n, p in runtime.model.named_parameters() if p.requires_grad)
    assert any(not torch.equal(critic[n], p) for n, p in run.game.critic.state_dict().items())
    assert all(torch.equal(frozen[n], p) for n, p in runtime.model.named_parameters() if not p.requires_grad)
    state = run.game.state_dict()
    expected = run.step()
    run.game.load_state_dict(state)
    assert expected == run.step()
    after = run.game.state_dict()
    run.game.load_state_dict(state)
    run.step()
    equal(after, run.game.state_dict())
    assert len(list(run.render_cases())) == 16
    with pytest.raises(ValueError, match='validation'):
        build_supra_campaign(runtime, rows_for(fixture, 'validation'),
                             model_identity=dict(frozen_parameter_sha256='verified'))
    assert not torch.cuda.is_initialized()


def test_final_form_requires_full_data_and_fixed_budget_without_loading(fixture):
    class TooSmall:
        split = 'train'
        recipe = dict(concept='final-form')
        def __len__(self):
            return 48
    rows = TooSmall()
    with pytest.raises(ValueError, match='FinalForm'):
        build_final_form_campaign(None, rows, model_identity={})
    with pytest.raises(ValueError, match='FinalForm'):
        build_final_form_campaign(None, SimpleNamespace(split='validation'), model_identity={}, total_steps=3)


def test_context_and_scale_mismatch_reject_before_generator_update(fixture):
    _, runtime, _ = fixture
    rows = rows_for(fixture)
    with pytest.raises(ValueError, match='response scale'):
        build_supra_campaign(runtime, rows, model_identity=dict(frozen_parameter_sha256='fixture'), response_scale=100.)
    runtime.cache['plain knight'][0].add_(1.)
    with pytest.raises(ValueError, match='context differs'):
        build_supra_campaign(runtime, rows, model_identity=dict(frozen_parameter_sha256='fixture'))


def test_immutable_config_and_corrupt_cache_fail_closed(fixture, tmp_path):
    config, runtime, path = fixture
    rows = rows_for(fixture)
    identity = rows.identity()
    config['rows'][0]['positive'] = 'changed external config'
    assert rows.identity() == identity
    assert rows[0].row.positive == 'boss knight'
    damaged = tmp_path / 'damaged.pt'
    damaged.write_bytes(path.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='checksum'):
        SupraCampaignRows(damaged, cache_sha256=file_digest(path), config=rows.config, contexts=runtime.cache)


def test_actual_audited_proposals_keep_full_budget_heldouts_and_no_admission():
    import importlib.util
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = root / 'scripts/propose_remaining_pg09.py'
    spec = importlib.util.spec_from_file_location('remaining_campaign_proposals', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    audit_path = root / 'evidence/pg09-remaining-campaigns/audit.json'
    proposals = module.proposals(json.loads(audit_path.read_text()), audit_path,
                                 root/'conceptmod/textsliders/pg09_remaining_campaigns.py')
    final, supra = proposals['final_form'], proposals['supra_final_boss']
    assert (final['total_steps'], final['batch_size'], final['coverage']['full_balanced_cycles_per_phase']) == (3840,8,2)
    assert final['old_completed_step'] == 254 and final['old_rendered_step'] == 200
    assert (supra['total_steps'], supra['batch_size']) == (28800,4)
    assert supra['coverage']['generator_exposures'] == 192 * 600 == 28800 * 4
    assert supra['coverage']['generator_preservation_exposures'] / (28800 * 4) == .2
    assert supra['context_preparation']['max_unique_prompts'] == 19
    assert not supra['context_preparation']['placeholder_contexts']
    assert supra['heldout']['seeds'] == [42,1234]
    assert supra['validation']['seeds'] == [39001,39002]
    assert supra['preservation']['new_optimized_mse_weight'] is None
    for proposal in (final, supra):
        assert proposal['status'] == 'DRAFT_NOT_ADMITTED'
        assert proposal['objective'] == 'pg09_gan_only'
        assert proposal['probe']['use_full_total_steps_recipe']
        assert not proposal.get('quality_qualified', False)
        assert len(proposal['source_pins']) >= 6


def test_render_rejects_nonfinite_native_latent_before_vae(fixture):
    _, runtime, _ = fixture
    run = build_supra_campaign(runtime, rows_for(fixture),
                               model_identity=dict(frozen_parameter_sha256='tiny CPU fixture'), total_steps=2)
    def forbidden(*args, **kwargs):
        pytest.fail('nonfinite latent reached VAE')
    runtime.vae.decode = forbidden
    with pytest.raises(FloatingPointError, match='latent before VAE'):
        run.backend.decoder(torch.full((1,4,32,32), float('nan')))


def test_loaded_legacy_adapter_is_not_silently_used_as_gan_initialization(fixture):
    _, runtime, _ = fixture
    rows = rows_for(fixture)
    with torch.no_grad():
        runtime.loras[0].up.weight.add_(1.)
    original = runtime.loras[0].up.weight.detach().clone()
    with pytest.raises(ValueError, match='fresh zero-up'):
        build_supra_campaign(runtime, rows, model_identity=dict(frozen_parameter_sha256='tiny CPU fixture'))
    assert torch.equal(runtime.loras[0].up.weight, original)


class ContextCounterRuntime:
    stored_unconditional = True
    def __init__(self):
        self.calls = []
    def encode(self, prompt):
        self.calls.append(prompt)
        return torch.ones(1,2,8), torch.ones(1,2)


def test_context_budget_rejects_before_first_encode_directory_or_file(tmp_path):
    runtime = ContextCounterRuntime()
    config = dict(rows=[], preservation=[f'prompt{i}' for i in range(20)], verification=[])
    output = tmp_path/'not-created'/'contexts.safetensors'
    with pytest.raises(ValueError, match='19|budget'):
        export_admitted_supra_contexts(runtime, config, output,
                                       runtime_identity=dict(model_revision='CPU counter', text_revision='CPU counter'))
    assert runtime.calls == []
    assert not output.parent.exists()
    assert not torch.cuda.is_initialized()


def test_generic_fixture_export_encodes_at_most19_unique_contexts(tmp_path):
    runtime = ContextCounterRuntime()
    config = dict(rows=[], preservation=[f'fixture{i}' for i in range(18)]*2, verification=[])
    receipt = export_admitted_supra_contexts(runtime, config, tmp_path/'counter.safetensors',
                                            runtime_identity=dict(model_revision='CPU counter', text_revision='CPU counter'))
    assert len(runtime.calls) == len(set(runtime.calls)) == 19
    assert receipt['unique_contexts'] == receipt['max_encode_calls'] == 19


def test_live_wrapper_pins_exact_archived_config_and_model_t5_revisions_before_delegate(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from conceptmod.textsliders import pg09_remaining_campaigns as bridge
    audit = json.loads((Path(__file__).resolve().parents[1]/'evidence/pg09-remaining-campaigns/audit.json').read_text())
    config = audit['supra_final_boss']['config']
    identity = dict(model_revision=bridge.SUPRA_MODEL_REVISION, text_revision=bridge.SUPRA_TEXT_REVISION)
    assert bridge.validate_live_supra_context_preparation(config, identity)['unique_contexts'] == 19
    calls = []
    monkeypatch.setattr(bridge, 'export_admitted_supra_contexts', lambda *args, **kwargs: calls.append((args,kwargs)))
    runtime = ContextCounterRuntime()
    bridge.export_live_supra_contexts(runtime, config, tmp_path/'never-written', runtime_identity=identity)
    assert len(calls) == 1 and runtime.calls == []  # Only a contract check; no actual context encoding.
    for key in ('model_revision','text_revision'):
        wrong = dict(identity, **{key:'incorrect'})
        with pytest.raises(ValueError, match='revisions'):
            bridge.export_live_supra_contexts(runtime, config, tmp_path/'never-written', runtime_identity=wrong)
    altered = deepcopy(config)
    altered['rows'][0]['positive'] += ' modified'
    assert len(required_supra_prompts(altered)) == 19
    with pytest.raises(ValueError, match='exact reviewed'):
        bridge.export_live_supra_contexts(runtime, altered, tmp_path/'never-written', runtime_identity=identity)
    assert len(calls) == 1 and runtime.calls == []
    assert not (tmp_path/'never-written').exists()
