import json
from dataclasses import asdict, replace

import pytest
import torch

from lumen_studio.contracts import atomic_json, digest, file_hash
from lumen_studio.dataset import compile_manifest, load_characters, load_definitions
from scripts.anima_reuse_targets import equal, main


def reuse_fixture(tmp_path, monkeypatch):
    before = compile_manifest('train')
    # Keep the changed-prompt fixture distinct after real catalogs are revised.
    definitions = [replace(d, lighting=d.lighting + ', fixture-specific directional illumination')
                   if d.id == 'theatrical-05' else d for d in load_definitions()]
    bundle = dict(definitions=[asdict(d) for d in definitions], characters=load_characters(),
                  **{split: compile_manifest(split, definitions=definitions) for split in ('train', 'dev')})
    bundle['sha256'] = digest(bundle)
    catalog, qualification = tmp_path / 'catalog.json', tmp_path / 'qualification.json'
    atomic_json(catalog, bundle)
    model = dict(model='serialized-target-test')
    atomic_json(qualification, dict(passed=True, stage='definition_reference_screen',
        candidate=bundle['sha256'], manifest_sha256=bundle['train']['sha256'], model_identity=model))
    source, destination = tmp_path / 'source', tmp_path / 'destination'
    source.mkdir()
    identity = dict(model=model, manifest_sha256=before['sha256'], split='train', variation='theatrical',
        resolution=512, steps=10, cfg=1, layout='both-trajectories/all-positions/full-velocity/shared-state-v1')
    fingerprint = digest(identity)
    rows = [r for r in before['rows'] if r['variation'] == 'theatrical']
    shards = []
    for row in (rows[0], rows[4]):
        name = row['id'] + '-1000.pt'
        records = [dict(position=t, timestep=float(10-t), trajectory=trajectory,
            latent=torch.arange(12).reshape(1, 3, 4).float() + t,
            neutral=torch.ones(1, 3, 4) * t, positive=torch.arange(12).reshape(1, 3, 4).float() - t)
            for trajectory in ('neutral', 'positive') for t in range(10)]
        torch.save(dict(fingerprint=fingerprint, row=row, seed=1000, embedding=torch.ones(1, 2, 4),
                       positive_embedding=torch.zeros(1, 2, 4), records=records), source / name)
        shards.append(dict(path=name, row=row['id'], seed=1000, count=20, sha256=file_hash(source / name)))
    atomic_json(source / 'index.json', dict(identity=identity, input_fingerprint=fingerprint,
        fingerprint=digest(dict(input=fingerprint, shards=shards)), shards=shards))
    monkeypatch.setattr('sys.argv', ['reuse', '--source', str(source), '--destination', str(destination),
        '--catalog', str(catalog), '--qualification', str(qualification), '--variation', 'theatrical',
        '--split', 'train', '--publish'])
    return source, destination, shards, fingerprint


def test_reuse_keeps_full_fields_and_recomputes_changed_prompts(tmp_path, monkeypatch):
    source, destination, shards, fingerprint = reuse_fixture(tmp_path, monkeypatch)
    main()
    original = torch.load(source / shards[0]['path'], weights_only=True)
    copied = torch.load(destination / shards[0]['path'], weights_only=True)
    assert copied['fingerprint'] != fingerprint
    copied['fingerprint'] = fingerprint
    assert equal(original, copied)
    assert not (destination / shards[1]['path']).exists()
    assert not (destination / 'index.json').exists()  # The preparer commits the complete cache.
    assert all(file_hash(source / s['path']) == s['sha256'] for s in shards)
    # A mismatched paired teacher must not be silently overwritten or admitted.
    copied['fingerprint'] = digest(json.loads((destination / 'identity.json').read_text()))
    copied['records'][0]['positive'].add_(1)
    torch.save(copied, destination / shards[0]['path'])
    with pytest.raises(ValueError, match='Existing destination tensors'):
        main()


def test_reuse_rejects_corrupt_source_before_publishing(tmp_path, monkeypatch):
    source, destination, shards, _ = reuse_fixture(tmp_path, monkeypatch)
    with (source / shards[0]['path']).open('ab') as f:
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='immutable index'):
        main()
    assert not list(destination.glob('*.pt'))
