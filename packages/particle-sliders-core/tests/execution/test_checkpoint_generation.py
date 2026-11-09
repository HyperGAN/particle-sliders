"""Model-independent generation fault tests; no Torch or native imports."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from particle_sliders.execution import checkpoint_generation as generation


def publish(root, step, *, build=None):
    def files(directory):
        (directory/'state.bin').write_bytes(f'state-{step}'.encode())
        (directory/'weights.bin').write_bytes(f'raw-{step}'.encode())
        (directory/'weights.json').write_text(json.dumps({'step': step}))
    return generation.publish_generation(root,f'step-{step}',identity={'step': step},
        required=['state.bin','weights.bin','weights.json'],build=build or files,
        reference_fields=lambda directory:dict(step=step,state=str(directory/'state.bin')))


def test_complete_generation_advances_one_receipt_and_preserves_older_files(tmp_path):
    legacy=tmp_path/'state.pt';legacy.write_bytes(b'historical checkpoint')
    first=publish(tmp_path,1)
    directory,manifest,receipt=generation.verify_reference(tmp_path/'latest.json',identity={'step':1})
    assert receipt==first and set(manifest['files'])=={'state.bin','weights.bin','weights.json'}
    before={name:(directory/name).read_bytes() for name in manifest['files']}
    publish(tmp_path,2)
    assert generation.verify_reference(tmp_path/'latest.json')[2]['step']==2
    assert before=={name:(directory/name).read_bytes() for name in manifest['files']}
    assert legacy.read_bytes()==b'historical checkpoint'
    with pytest.raises(FileExistsError,match='immutable'):
        publish(tmp_path,1)


def test_caller_identity_is_json_snapshotted_before_build_and_reference_callbacks(tmp_path):
    identity={'cohort':{'items':('original',)}}
    def build(directory):
        identity['cohort']['items']=('changed during build',)
        (directory/'state.bin').write_bytes(b'full state')
    def reference(directory):
        identity['cohort']['items']=('changed during reference',)
        return {'step':1}
    generation.publish_generation(tmp_path,'step-1',identity=identity,required=['state.bin'],
                                  build=build,reference_fields=reference)
    _,manifest,_=generation.verify_reference(tmp_path/'latest.json',
                                            identity={'cohort':{'items':('original',)}})
    assert manifest['identity']=={'cohort':{'items':['original']}}
    assert identity=={'cohort':{'items':('changed during reference',)}}


@pytest.mark.parametrize('phase',['build','manifest','file_sync','rename','generation_sync','pointer','replace'])
def test_failure_before_pointer_commit_keeps_previous_reference_and_evidence(tmp_path,monkeypatch,phase):
    publish(tmp_path,1)
    before=(tmp_path/'latest.json').read_bytes()
    def failed_build(directory):
        (directory/'state.bin').write_bytes(b'partial full state')
        raise OSError('injected build failure')
    if phase in ('manifest','pointer'):
        original=generation._write_json
        def write(path,value):
            if (phase=='manifest' and path.name=='generation.json') or (phase=='pointer' and path.name.startswith('.latest-')):
                raise OSError('injected '+phase)
            return original(path,value)
        monkeypatch.setattr(generation,'_write_json',write)
    elif phase in ('file_sync','generation_sync'):
        if phase=='file_sync':
            monkeypatch.setattr(generation.os,'fsync',lambda descriptor: (_ for _ in ()).throw(OSError('injected file_sync')))
        else:
            original=generation._sync_directory
            def sync(path):
                if Path(path)==tmp_path/'generations':
                    raise OSError('injected generation_sync')
                return original(path)
            monkeypatch.setattr(generation,'_sync_directory',sync)
    elif phase in ('rename','replace'):
        monkeypatch.setattr(generation.os,phase,lambda *args: (_ for _ in ()).throw(OSError('injected '+phase)))
    with pytest.raises(OSError,match='injected'):
        publish(tmp_path,2,build=failed_build if phase=='build' else None)
    assert (tmp_path/'latest.json').read_bytes()==before
    assert generation.verify_reference(tmp_path/'latest.json')[2]['step']==1
    retained=[path for path in (tmp_path/'generations').iterdir() if path.name!='step-1']
    assert len(retained)==1 and any(retained[0].iterdir())
    if retained[0].name.startswith('.'):
        with pytest.raises(ValueError,match='uncommitted'):
            generation.verify_generation(retained[0])
    else:
        assert generation.verify_generation(retained[0],identity={'step':2})


def test_final_directory_sync_failure_leaves_only_a_complete_verifiable_new_pointer(tmp_path,monkeypatch):
    publish(tmp_path,1)
    original=generation._sync_directory
    def sync(path):
        if Path(path)==tmp_path:
            raise OSError('injected terminal fsync failure')
        return original(path)
    monkeypatch.setattr(generation,'_sync_directory',sync)
    with pytest.raises(OSError,match='terminal fsync'):
        publish(tmp_path,2)
    assert generation.verify_reference(tmp_path/'latest.json',identity={'step':2})[2]['step']==2


@pytest.mark.parametrize('damage',['changed','missing','extra','symlink','manifest','missing_manifest','symlink_manifest'])
def test_verification_rejects_changed_or_incomplete_collections(tmp_path,damage):
    publish(tmp_path,1)
    directory=tmp_path/'generations/step-1'
    if damage=='changed':(directory/'weights.bin').write_bytes(b'wrong weights')
    elif damage=='missing':(directory/'state.bin').unlink()
    elif damage=='extra':(directory/'uncommitted.tmp').write_bytes(b'unexpected')
    elif damage=='symlink':
        (directory/'weights.bin').unlink()
        (directory/'weights.bin').symlink_to(directory/'state.bin')
    elif damage=='missing_manifest':(directory/'generation.json').unlink()
    elif damage=='symlink_manifest':
        (directory/'generation.json').unlink()
        (directory/'generation.json').symlink_to(directory/'state.bin')
    else:(directory/'generation.json').write_text('{}')
    with pytest.raises(ValueError):
        generation.verify_reference(tmp_path/'latest.json')


def test_missing_files_are_never_published_and_receipt_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError,match='requires exactly'):
        publish(tmp_path,1,build=lambda directory:(directory/'state.bin').write_bytes(b'partial'))
    assert not (tmp_path/'latest.json').exists()
    publish(tmp_path,2)
    path=tmp_path/'latest.json'
    receipt=json.loads(path.read_text());receipt['generation']='../outside'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='invalid generation reference'):
        generation.verify_reference(path)


def test_competing_publisher_is_refused_without_waiting_for_a_training_deadline(tmp_path):
    source=Path(__file__).resolve().parents[2]/'src'
    code="""import pathlib,sys
sys.path.insert(0,sys.argv[1])
from particle_sliders.execution import checkpoint_generation as generation
try:
 with generation._publication_lock(pathlib.Path(sys.argv[2])): pass
except RuntimeError as error: assert 'already owned' in str(error)
else: raise AssertionError('competing publisher admitted')
assert 'torch' not in sys.modules
"""
    with generation._publication_lock(tmp_path):
        child=subprocess.run([sys.executable,'-I','-S','-c',code,str(source),str(tmp_path)],
                             capture_output=True,text=True,timeout=5)
    assert child.returncode==0,child.stderr
