import json

import pytest

from lumen_studio.audits import audit_report, enqueue_final, selection_report
from lumen_studio.contracts import VARIATIONS, atomic_json, canonical
from lumen_studio.dataset import compile_manifest
from lumen_studio.store import Store
from lumen_studio.sampling import selection_rows


def development_campaign(store, root, identity):
    for variation in VARIATIONS:
        rows = selection_rows(variation)
        for step in (400, 800, 1200, 1600):
            sha = variation + '-' + str(step)
            with store.connect() as db:
                db.execute('INSERT INTO checkpoints VALUES (?,?,?,?,?)',
                    (sha, variation, str(root / 'runs' / variation / f'ema-{step:06}.safetensors'),
                        canonical(dict(step=step, model_identity=identity,
                        manifest_sha256=compile_manifest('train')['sha256'])), step))
            atomic_json(root / 'runs' / variation / f'probe-{step:06}.json', dict(full_strength_raw_R=1/step))
            for row in rows:
                metadata = dict(purpose='development', energy=1, case=row['id'], width=768, height=768,
                    seed=29001, sampling_step=step, checkpoints={variation:sha})
                metadata['manifest_sha256'] = compile_manifest('dev')['sha256']
                ident = store.enqueue([metadata])['ids'][0]
                assert store.claim()['id'] == ident
                store.finish(ident, store.path.parent / 'images' / (ident + '.png'), metadata)
                store.review(ident, dict(atmosphere='win', blind=True))


def test_final_gate_selection_and_single_frozen_campaign(tmp_path):
    store = Store(tmp_path / 'studio/studio.sqlite3')
    identity = dict(model='test')
    with pytest.raises(ValueError, match='qualifying'):
        enqueue_final(store, identity, tmp_path)
    assert not (tmp_path / 'audits/final.json').exists()
    development_campaign(store, tmp_path, identity)
    selected = selection_report(store, tmp_path, 'candlelit')
    assert selected['selected']['step'] == 1600
    final = enqueue_final(store, identity, tmp_path)
    assert len(final['ids']) == 512
    assert enqueue_final(store, identity, tmp_path)['ids'] == final['ids']
    queued = [store.job(i)['payload'] for i in final['ids']]
    singles = [p for p in queued if p['purpose'] == 'final_test']
    assert len(singles) == 384
    assert len({p['character'] for p in singles}) == 8
    assert len({p['seed'] for p in singles}) == 4
    assert any(p['bare'] for p in singles)
    assert all('neutral diffuse illumination' not in p['prompt'] for p in singles if p['bare'])
    mixtures = [p for p in queued if p['purpose'] == 'mixture_audit']
    assert len(mixtures) == 128
    assert len({p['family'] for p in mixtures}) == 4
    # Audit selections remain immutable even if a later review changes preferences.
    image = store.history(1)[0]
    store.review(image['id'], dict(atmosphere='loss', major_regression=True, blind=True))
    assert enqueue_final(store, identity, tmp_path)['ids'] == final['ids']
    report = audit_report(store, tmp_path)
    assert report['completed_images'] == 0
    assert report['expected_images'] == 512
    path = tmp_path / 'audits/final.json'
    changed = json.loads(path.read_text()); changed['width'] = 512
    atomic_json(path, changed)
    with pytest.raises(ValueError, match='manifest changed'):
        enqueue_final(store, identity, tmp_path)


def test_unrated_or_nonblind_reviews_cannot_pass_selection(tmp_path):
    store = Store(tmp_path / 'studio/studio.sqlite3')
    development_campaign(store, tmp_path, dict(model='test'))
    for image in store.history(500):
        store.review(image['id'], dict(atmosphere='win', blind=False))
    report = selection_report(store, tmp_path, 'candlelit')
    assert report['selected'] is None
    assert len(report['incomplete']) == 4
    assert all(c['rated'] == 0 for c in report['incomplete'])


def test_archived_seed_with_identical_targets_cannot_supply_current_reviews(tmp_path):
    from lumen_studio.audits import active_checkpoint
    store = Store(tmp_path / 'studio/studio.sqlite3')
    development_campaign(store, tmp_path, dict(model='test'))
    assert selection_report(store, tmp_path, 'theatrical')['selected']['step'] == 1600
    with store.connect() as db:
        db.execute("UPDATE checkpoints SET path=? WHERE sha256=?", (
            '/old/host/runs/archive/theatrical-seed7/ema-001600.safetensors', 'theatrical-1600'))
    report = selection_report(store, tmp_path, 'theatrical')
    assert report['selected']['step'] == 1200
    assert {c['step'] for c in report['candidates']} == {400, 800, 1200}
    assert active_checkpoint({'path': '/new/host/runs/theatrical/ema-000200.safetensors'}, 'theatrical')
    assert not active_checkpoint({'path': '/old/host/runs/archive/theatrical-seed7/ema-000200.safetensors'}, 'theatrical')


def test_revising_one_definition_does_not_discard_unchanged_variation_reviews(tmp_path, monkeypatch):
    from dataclasses import replace
    from lumen_studio import dataset
    store = Store(tmp_path / 'studio/studio.sqlite3')
    development_campaign(store, tmp_path, dict(model='test'))
    dataset.archive_catalog(tmp_path)
    definitions = [replace(d, lighting=d.lighting + ', darker shadow edges')
                   if d.variation == 'theatrical' else d for d in dataset.load_definitions()]
    monkeypatch.setattr(dataset, 'load_definitions', lambda: definitions)
    for variation in ('candlelit', 'moonlit'):
        assert selection_report(store, tmp_path, variation)['selected']['step'] == 1600
    assert selection_report(store, tmp_path, 'theatrical')['selected'] is None


def test_blind_audit_submission_rejects_mismatched_or_incomplete_evidence(tmp_path):
    import copy
    import hashlib
    from scripts.anima_submit_audit_review import CHECKS, prepare
    groups, reviews = [], []
    for case in range(128):
        columns, ratings = {}, {}
        for letter, energy, strength in zip('ABCD', (0., .25, .5, 1.), (0., 2., 1., 3.)):
            columns[letter] = dict(id=f'{case}-{letter}', metadata=dict(energy=energy,
                prompt='one fixed character', seed=case, width=768, height=768, steps=10,
                model_identity={'sha256': 'test'}, checkpoints={'candlelit': 'fixed'},
                case=str(case), character=str(case), family='candlelit', audit='fixed-audit',
                purpose='final_test'))
            ratings[letter] = dict(atmosphere_strength=strength, notes='Synthetic validation fixture',
                **{k: k not in ('unwanted_objects', 'recurring_features', 'major_regression') for k in CHECKS})
        groups.append(dict(case=case, family='candlelit', columns=columns))
        reviews.append(dict(case=case, family='candlelit', blind=True, ratings=ratings))
    mapping = dict(audit='fixed-audit', groups=groups)
    path = tmp_path / 'ratings.json'

    def write(m, r):
        raw = json.dumps(m).encode()
        (tmp_path / 'mapping.json').write_bytes(raw)
        path.write_text(json.dumps(dict(mapping_sha256=hashlib.sha256(raw).hexdigest(), reviews=r)))

    write(mapping, reviews)
    result = prepare(path)
    assert len(result) == 512
    assert [r['atmosphere_strength'] for _, r in result[:4]] == [0., 2., 1., 3.]
    assert [r['atmosphere'] for _, r in result[:4]] == ['tie', 'win', 'win', 'win']
    # Preserve a non-monotone judgment exactly; submission cannot smooth it.
    changed = copy.deepcopy(mapping)
    changed['groups'][0]['columns']['B']['metadata']['seed'] = 999
    write(changed, reviews)
    with pytest.raises(ValueError, match='matched immutable'):
        prepare(path)
    write(mapping, reviews[:-1])
    with pytest.raises(ValueError, match='128 audit cases'):
        prepare(path)
    changed_reviews = copy.deepcopy(reviews)
    changed_reviews[0]['ratings']['C']['outfit'] = None
    write(mapping, changed_reviews)
    with pytest.raises(ValueError, match='Complete every preservation'):
        prepare(path)
    write(mapping, reviews)
    (tmp_path / 'mapping.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='different immutable'):
        prepare(path)
