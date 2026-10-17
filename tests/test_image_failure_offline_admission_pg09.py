"""Offline guards use metadata and tiny synthetic ZIP/tensor packets only."""
import builtins
from copy import deepcopy
import json
from pathlib import Path
import random
import zipfile

import pytest

from scripts import audit_image_checkpoint_rng_pg09 as audit
from scripts import prepare_image_failure_admission_pg09 as prepare


def test_metadata_validation_cannot_open_checkpoint_or_weight_bodies_or_import_models(monkeypatch):
    original_open, original_import = Path.open, builtins.__import__
    def guarded_open(path, *args, **kwargs):
        assert path.suffix not in prepare.BINARY, 'forbidden binary content read'
        return original_open(path, *args, **kwargs)
    def guarded_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch','diffusers','transformers'}, 'native runtime import'
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    proposal=json.loads((prepare.CHECKOUT/'evidence/pg09-image-failure-campaigns/proposals.json').read_text())
    result=prepare.inventory(proposal)
    assert len(result['verified_frozen_sources_and_inputs'])==43
    assert result['checkpoint_or_weight_bodies_opened']==0
    assert result['native_model_imports_or_forwards']==0
    assert len(result['known_packets'])==4
    assert sum(p['bytes'] for p in result['known_packets'])==2390321324
    assert all(not c['standalone_assets_found'] and c['missing'] for c in result['cases'])
    assert all(c['cached_weights'] and all(w['size_matches_manifest'] and not w['fresh_payload_checksum_verified'] for w in c['cached_weights']) for c in result['cases'])
    assert all(c['historical_replay_qualified'] is False for c in result['cases'])


def test_metadata_checksum_drift_is_rejected():
    proposal=json.loads((prepare.CHECKOUT/'evidence/pg09-image-failure-campaigns/proposals.json').read_text())
    proposal=deepcopy(proposal)
    proposal['source_bindings'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='drift'):
        prepare.inventory(proposal)
    with pytest.raises(ValueError,match='refuses'):
        prepare.source_record('/unopened/checkpoint.pt')


def test_static_pickle_inventory_never_executes_globals_or_reads_storage(tmp_path,monkeypatch):
    path=tmp_path/'synthetic.pt'
    # GLOBAL is merely reported. Even an unrecognized global must not execute.
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/data.pkl',b'cforbidden_module\nforbidden_callable\n.')
        archive.writestr('archive/data/0',b'not read')
    original=zipfile.ZipFile.read
    def guarded_read(archive,name,*args,**kwargs):
        member=name.filename if isinstance(name,zipfile.ZipInfo) else name
        assert member.endswith('/data.pkl')
        return original(archive,name,*args,**kwargs)
    monkeypatch.setattr(zipfile.ZipFile,'read',guarded_read)
    result=audit.static_inventory(path)
    assert result['global_references']==['forbidden_module forbidden_callable']
    assert not result['pickle_executed'] and not result['tensor_storage_members_read']
    assert result['opcode_counts']['GLOBAL']==1


def test_static_inventory_rejects_ambiguous_or_oversized_pickle(tmp_path):
    path=tmp_path/'ambiguous.pt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('a/data.pkl',b'N.')
        archive.writestr('b/data.pkl',b'N.')
    with pytest.raises(ValueError,match='exactly one'):
        audit.static_inventory(path)


def toy_packet(torch):
    return dict(identity='synthetic-only',step=0,trainable={'toy':torch.zeros(2)},
        optimizer={'state':{},'param_groups':[]},torch_rng=torch.get_rng_state(),
        python_rng=random.getstate(),cuda_rng=[])


def test_restricted_rng_schema_uses_cpu_weights_only_mmap_and_does_not_restore_rng(tmp_path,monkeypatch):
    import torch
    path=tmp_path/'toy.pt'
    torch.save(toy_packet(torch),path)
    original=torch.load
    calls=[]
    def guarded_load(path,**kwargs):
        calls.append(kwargs)
        assert kwargs==dict(weights_only=True,map_location='cpu',mmap=True)
        return original(path,**kwargs)
    monkeypatch.setattr(torch,'load',guarded_load)
    before=torch.get_rng_state().clone()
    result=audit.rng_schema(path,identity='synthetic-only',step=0)
    assert len(calls)==1 and torch.equal(before,torch.get_rng_state())
    assert not result['cuda_initialized'] and not result['rng_restored']
    assert result['trainable_schema']['toy']['numel']==2 and result['cuda_rng']==[]
    assert result['torch_rng']['sha256']
    assert result['python_rng']['sha256']


def test_restricted_rng_schema_rejects_wrong_identity_and_rng_type(tmp_path):
    import torch
    path=tmp_path/'toy.pt'
    packet=toy_packet(torch)
    torch.save(packet,path)
    with pytest.raises(ValueError,match='identity'):
        audit.rng_schema(path,identity='wrong',step=0)
    packet['torch_rng']=torch.zeros(4)
    torch.save(packet,path)
    with pytest.raises(ValueError,match='uint8'):
        audit.rng_schema(path,identity='synthetic-only',step=0)
