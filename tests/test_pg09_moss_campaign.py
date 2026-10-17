from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from conceptmod.textsliders.pg09_moss_campaign import (
    MossCampaignRows, canonical_digest, file_digest, build_campaign_run, coverage_budget,
)


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)


def make_cache(root):
    from moss_sliders.smoke import tiny_model
    from moss_sliders.lora import Slider
    from moss_sliders.campaign import target_bundle
    from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler
    torch.manual_seed(71)
    model = tiny_model()
    original = deepcopy(model.state_dict())
    recipe = dict(concept='bad-intent', model_revision='tiny-frozen-fixture', evaluation_seed=29001,
                  negative_caption_policy='neutral equals negative; target is base +/- (positive-neutral)',
                  training_rows=[dict(id='a', neutral='neutral a', positive='positive a', seeds=[1, 2]),
                                 dict(id='b', neutral='neutral b', positive='positive b', seeds=[1, 2])],
                  development_rows=[dict(id='heldout', neutral='heldout neutral', positive='heldout positive',
                                         seeds=[29001, 29027])],
                  settings=dict(seed=1000, rank=2, alpha=2, resolution=32, target_steps=3,
                                updates=1600, effective_batch=8,
                                evaluation_updates=[200, 400, 800, 1200, 1600]))
    root.mkdir(parents=True)
    (root / 'targets').mkdir()
    (root / 'recipe.json').write_text(json.dumps(recipe))
    prompts = list(dict.fromkeys([r[k] for r in recipe['training_rows'] + recipe['development_rows']
                                 for k in ('neutral', 'positive')] + ['']))
    encoded = {p: (torch.randn(1, 3 + (i % 2), 16), torch.ones(1, 3 + (i % 2), dtype=torch.bool))
               for i, p in enumerate(prompts)}
    tensors = {f'{kind}{i}': encoded[p][j] for i, p in enumerate(prompts)
               for j, kind in enumerate(('e', 'm'))}
    save_file(tensors, str(root / 'embeddings.safetensors'))
    (root / 'embeddings.json').write_text(json.dumps(dict(prompts_sha256=canonical_digest(prompts),
                                          sha256=file_digest(root / 'embeddings.safetensors'))))
    slider = Slider(model, rank=2)
    pipe = MOSSImagePipeline(model, MOSSImageScheduler())
    records = []
    for split, rows in [('train', recipe['training_rows']), ('dev', recipe['development_rows'])]:
        for row in rows:
            for seed in row['seeds'] if split == 'train' else [recipe['evaluation_seed']]:
                data = target_bundle(pipe, slider, encoded[row['neutral']], encoded[row['positive']],
                                     seed, resolution=32, steps=3)
                path = root / 'targets' / f"{split}-{row['id']}-{seed}.safetensors"
                save_file(data, str(path))
                ident = canonical_digest(dict(identity=canonical_digest(recipe), row=row['id'], seed=seed, split=split))
                path.with_suffix('.json').write_text(json.dumps(dict(identity=ident, sha256=file_digest(path))))
                records.append(dict(path=str(path), row=row['id'], seed=seed, split=split))
    (root / 'target-index.json').write_text(json.dumps(dict(identity=canonical_digest(recipe), records=records)))
    slider.detach()
    return original, recipe


def new_model(original):
    from moss_sliders.smoke import tiny_model
    model = tiny_model()
    model.load_state_dict(original)
    return model


def test_lazy_cache_geometry_poles_useful_condition_features_and_bounds(tmp_path):
    make_cache(tmp_path / 'cache')
    rows = MossCampaignRows(tmp_path / 'cache', bundle_cache_size=1, context_cache_size=2)
    assert rows.response_shape == (1, 4, 2, 2)
    assert len(rows) == 48
    assert rows.summary()['lru_bundles'] == 0
    plus, minus = rows[0], rows[1]
    torch.testing.assert_close(plus.target + minus.target, 2 * plus.neutral)
    assert all(torch.equal(a, b) for a, b in zip(plus.row.negative, plus.row.neutral))
    c = rows.conditions([0, 1])
    assert c.shape == (2, 21) and not c.requires_grad
    assert torch.equal(c[0, :16], c[1, :16])
    assert c[:, 17].tolist() == [1., -1.]
    assert c[:, 16].tolist() == [float(plus.row.timestep) / 1000.] * 2
    assert not torch.equal(c[0, :16], rows.conditions([24])[0, :16])
    for index in (0, 12, 24, 36):
        rows[index]
    assert len(rows._bundles) == 1 and len(rows._contexts) <= 2
    identity = rows.identity()
    identity['strength'] = 999
    assert rows.identity()['strength'] == 1.
    assert rows.condition_policy['row_id_feature'] is False
    assert not torch.cuda.is_initialized()


def test_cache_checksum_and_recipe_mismatch_fail_closed(tmp_path):
    make_cache(tmp_path / 'cache')
    rows = MossCampaignRows(tmp_path / 'cache')
    path = Path(rows.records[0]['path'])
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        rows[0]
    recipe_path = tmp_path / 'cache/recipe.json'
    recipe = json.loads(recipe_path.read_text())
    recipe['model_revision'] = 'wrong'
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='recipe identity'):
        MossCampaignRows(tmp_path / 'cache')


def test_train_only_response_scale_and_exact_row_coverage_formula(tmp_path):
    make_cache(tmp_path / 'cache')
    rows = MossCampaignRows(tmp_path / 'cache')
    expected = torch.cat([rows._bundle(i)['delta'].flatten() for i in range(len(rows.records))])
    assert rows.response_scale() == pytest.approx(float(expected.double().square().mean().sqrt()))
    budget = coverage_budget()
    assert budget['signed_rows'] == 15360
    assert budget['one_full_generator_cycle_updates'] == 1920
    assert budget['old_generator_forward_matched_updates'] == 3200
    assert budget['proposed_two_full_cycles_updates'] == 3840
    assert coverage_budget(batch_size=2)['proposed_two_full_cycles_updates'] == 15360


def test_shared_core_balanced_full_cycle_frozen_base_and_exact_resume(tmp_path, monkeypatch):
    from particle_sliders.pg09 import ConditionalPG09
    import inspect
    if 'sampling' not in inspect.signature(ConditionalPG09).parameters:
        pytest.skip('coordinated lazy/balanced core implementation pending')
    original, recipe = make_cache(tmp_path / 'cache')
    def make():
        provider = MossCampaignRows(tmp_path / 'cache', bundle_cache_size=1)
        return build_campaign_run(new_model(original), provider, response_scale=provider.response_scale(),
                                  model_identity=dict(model='CPU tiny real MOSS', revision=recipe['model_revision']),
                                  batch_size=4, total_steps=14, checkpointing=True, critic_width=8)
    run = make()
    def forbidden(*a, **kw):
        raise AssertionError('MSE must never be optimized by the campaign GAN helper')
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    g_ids, d_ids = [], []
    for _ in range(12):
        metric = run.step()
        assert metric['g_loss'] == metric['g_adv']
        g_ids += metric['g_rows']
        d_ids += metric['d_rows']
    assert sorted(g_ids) == sorted(d_ids) == list(range(len(run.rows)))
    assert g_ids != d_ids
    for name, tensor in run.backend.transformer.state_dict().items():
        assert torch.equal(tensor, original[name])
    saved = run.game.state_dict()
    expected = run.step()
    other = make()
    other.game.load_state_dict(saved)
    assert other.step() == expected
    from test_pg09_backends import equal
    equal(run.game.state_dict(), other.game.state_dict())
    cases = list(run.render_cases(None))
    assert [c['label'] for c in cases] == ['off', 'direct-positive', 'direct-positive-cfg1', 'minus', 'plus']
    assert {c['seed'] for c in cases} == {29001}
    assert cases[-1]['guidance'] == 3 and cases[-1]['scale'] == 1
    assert cases[-1]['corrected_negative_adapter'] == 0
    assert cases[-1]['corrected_negative_skip'] == []
    export = tmp_path / 'gan-adapter.safetensors'
    run.export_adapter(export)
    loaded = make()
    loaded.load_adapter(export)
    assert torch.equal(run.game.evaluate(0), loaded.game.evaluate(0))
    with pytest.raises(RuntimeError, match='inference export'):
        loaded.step()
    with pytest.raises(FileExistsError):
        run.export_adapter(export)
    assert not torch.cuda.is_initialized()


def test_heldout_cache_is_evaluation_only_and_cannot_fit_response_scale(tmp_path):
    original, recipe = make_cache(tmp_path / 'cache')
    dev = MossCampaignRows(tmp_path / 'cache', split='dev')
    assert dev[0].target.isfinite().all()  # Evaluation reads remain available.
    with pytest.raises(ValueError, match='training split'):
        dev.response_scale()
    with pytest.raises(ValueError, match='training split'):
        build_campaign_run(new_model(original), dev, response_scale=1.,
                           model_identity=dict(model='CPU fixture', revision=recipe['model_revision']))


def test_critic_uses_context_and_condition_dependent_field_gradients():
    from conceptmod.textsliders.pg09_moss_campaign import CampaignFieldCritic
    critic = CampaignFieldCritic(4, 21, width=8)
    field = torch.randn(2, 4, 2, 2, requires_grad=True)
    condition = torch.zeros(2, 21)
    condition[1, 0] = 2.
    repeated = field[0:1].expand(2, -1, -1, -1)
    score = critic(repeated, condition)
    assert score.shape == (2,) and score[0] != score[1]
    first, = torch.autograd.grad(score[0], field, retain_graph=True)
    second, = torch.autograd.grad(score[1], field)
    assert not torch.equal(first, second)
