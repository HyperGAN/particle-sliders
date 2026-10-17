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


@pytest.mark.parametrize('role', ['plan','reviewed-static','source','packet','sidecar'])
def test_receipt_output_never_overwrites_any_protected_input(tmp_path,role):
    path=tmp_path/role
    path.write_bytes(b'preserved input')
    with pytest.raises((ValueError,FileExistsError),match='protected|fresh'):
        audit.FreshReceiptOutput(path,[path])
    assert path.read_bytes()==b'preserved input'


@pytest.mark.parametrize('kind', ['symlink','hardlink','dangling-symlink','parent-symlink'])
def test_receipt_output_rejects_aliases_and_symlink_directories(tmp_path,kind):
    import os
    protected=tmp_path/'input'
    protected.write_bytes(b'preserved input')
    output=tmp_path/'output'
    if kind=='symlink':
        output.symlink_to(protected)
    elif kind=='hardlink':
        os.link(protected,output)
    elif kind=='dangling-symlink':
        output.symlink_to(tmp_path/'absent')
    else:
        directory=tmp_path/'real-directory'
        directory.mkdir()
        alias=tmp_path/'alias-directory'
        alias.symlink_to(directory,target_is_directory=True)
        output=alias/'output'
    with pytest.raises((ValueError,FileExistsError),match='protected|fresh|symlink'):
        audit.FreshReceiptOutput(output,[protected])
    assert protected.read_bytes()==b'preserved input'


def test_existing_output_is_rejected_before_any_packet_or_plan_read(tmp_path,monkeypatch):
    import sys
    output=tmp_path/'known-packet.pt'
    output.write_bytes(b'preserved packet')
    def forbidden_read(*_):
        raise AssertionError('output guard ran after an audit content read')
    monkeypatch.setattr(audit,'sha_file',forbidden_read)
    monkeypatch.setattr(sys,'argv',['audit','--plan',str(tmp_path/'plan.json'),
        '--plan-sha256','unused','--mode','static','--output',str(output)])
    with pytest.raises(FileExistsError,match='fresh'):
        audit.main()
    assert output.read_bytes()==b'preserved packet'


def test_absent_output_cannot_alias_a_declared_input_before_packet_reads(tmp_path,monkeypatch):
    import sys
    output=tmp_path/'absent-source.py'
    packet=tmp_path/'never-opened.pt'
    plan=tmp_path/'plan.json'
    document=dict(audit_script_sha256=audit.sha_file(audit.__file__),
        known_packets=[dict(path=str(packet),sidecar=dict(path=str(tmp_path/'sidecar.json')))],
        source_bindings=[dict(path=str(output))])
    plan.write_text(json.dumps(document))
    argv=['audit','--plan',str(plan),'--plan-sha256',audit.sha_file(plan),
          '--mode','static','--output',str(output)]
    original=audit.sha_file
    def guarded_sha(path):
        assert Path(path)!=packet
        return original(path)
    monkeypatch.setattr(audit,'sha_file',guarded_sha)
    monkeypatch.setattr(audit.resource,'setrlimit',lambda *_:None)
    monkeypatch.setattr(audit.signal,'signal',lambda *_:None)
    monkeypatch.setattr(audit.signal,'alarm',lambda *_:None)
    for key,value in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='protected'):
        audit.main()
    assert not output.exists()


def test_receipt_publication_is_complete_and_cannot_be_repeated(tmp_path):
    output=tmp_path/'receipt.json'
    with audit.FreshReceiptOutput(output) as receipt:
        receipt.publish_json(dict(completed=True))
    original=output.read_bytes()
    assert json.loads(original)==dict(completed=True)
    with pytest.raises(FileExistsError,match='fresh'):
        audit.FreshReceiptOutput(output)
    assert output.read_bytes()==original
    assert not list(tmp_path.glob('*.pending'))


@pytest.mark.parametrize('kind',['file','symlink'])
def test_final_publication_race_preserves_competing_file_or_input(tmp_path,monkeypatch,kind):
    import os
    output=tmp_path/'receipt.json'
    protected=tmp_path/'input'
    protected.write_bytes(b'protected input')
    original_link=os.link
    def race_link(source,destination,**kwargs):
        if kind=='file':
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=kwargs['dst_dir_fd'])
            os.write(fd,b'competing receipt')
            os.close(fd)
        else:
            os.symlink(protected,destination,dir_fd=kwargs['dst_dir_fd'])
        return original_link(source,destination,**kwargs)
    monkeypatch.setattr(audit.os,'link',race_link)
    with audit.FreshReceiptOutput(output,[protected]) as receipt:
        with pytest.raises(FileExistsError):
            receipt.publish_json(dict(must_not_replace=True))
    assert protected.read_bytes()==b'protected input'
    assert output.read_bytes()==(b'competing receipt' if kind=='file' else b'protected input')
    assert not list(tmp_path.glob('*.pending'))


def test_failed_receipt_serialization_leaves_no_final_or_pending_output(tmp_path):
    output=tmp_path/'receipt.json'
    with audit.FreshReceiptOutput(output) as receipt:
        with pytest.raises(TypeError):
            receipt.publish_json({'unserializable':set()})
    assert not output.exists()
    assert not list(tmp_path.glob('*.pending'))


def test_revised_plan_cannot_reuse_old_static_receipt_before_rng_load(tmp_path,monkeypatch):
    import sys
    plan=tmp_path/'plan.json'
    static=tmp_path/'old-static.json'
    output=tmp_path/'fresh-rng.json'
    plan.write_text(json.dumps(dict(audit_script_sha256=audit.sha_file(audit.__file__),known_packets=[],source_bindings=[])))
    static.write_text(json.dumps(dict(mode='static',plan_sha256='old-v1-plan',packets=[])))
    monkeypatch.setattr(audit.resource,'setrlimit',lambda *_:None)
    monkeypatch.setattr(audit.signal,'signal',lambda *_:None)
    monkeypatch.setattr(audit.signal,'alarm',lambda *_:None)
    for key,value in dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1').items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(sys,'argv',['audit','--plan',str(plan),'--plan-sha256',audit.sha_file(plan),
        '--mode','rng','--reviewed-static-receipt',str(static),'--reviewed-static-sha256',audit.sha_file(static),
        '--output',str(output)])
    def forbidden_load(*_,**__):
        raise AssertionError('RNG packet access before strict same-plan gate')
    monkeypatch.setattr(audit,'rng_schema',forbidden_load)
    with pytest.raises(ValueError,match='static receipt plan mismatch'):
        audit.main()
    assert not output.exists()
    assert json.loads(static.read_text())['plan_sha256']=='old-v1-plan'
