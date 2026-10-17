"""Pure JSON/checksum proof protocol tests; no Torch or model load required."""
import json
from types import SimpleNamespace

import pytest

from conceptmod.textsliders import pg09_remaining_worker as worker


def fixture(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoints/step-000003/state.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'Synthetic checksum marker, not a Torch packet')
    reference=tmp_path/'probe-reference.pt'
    reference.write_bytes(b'Synthetic checksum reference marker')
    recipe=SimpleNamespace(identity='a'*64)
    base=dict(combined_sha256='b'*64)
    proof=dict(schema='remaining-gan-fresh-disk-proof-v1',recipe_sha256=recipe.identity,
        reference_sha256=worker.sha(reference),checkpoint_sha256=worker.sha(checkpoint),
        frozen_base=base,probe_pid=101,resume_pid=202,replayed_from_step=2,verified_step=3,
        checkpoint_advanced=False,exact_loss=True,exact_full_state=True,exact_sample=True,
        frozen_base_unchanged=True)
    checked=[]
    monkeypatch.setattr(worker,'verify_checkpoint',lambda *args:checked.append(True))
    return recipe,base,proof,checked


def invoke(tmp_path,recipe,base,proof):
    path=tmp_path/'fresh-disk-resume.json'
    path.write_text(json.dumps(proof))
    worker.verify_disk_proof(recipe,tmp_path,dict(fresh_disk_proof_sha256=worker.sha(path)),base)


def test_complete_protocol_accepts_valid_control(tmp_path,monkeypatch):
    recipe,base,proof,checked=fixture(tmp_path,monkeypatch)
    invoke(tmp_path,recipe,base,proof)
    assert checked==[True]


@pytest.mark.parametrize('field,value',[
    ('schema','unrelated-proof'),('schema',None),
    ('replayed_from_step',0),('replayed_from_step',2.),('replayed_from_step','2'),
    ('verified_step',99),('verified_step',3.),('verified_step','3'),('verified_step',True),
    ('probe_pid',None),('probe_pid',0),('probe_pid',-1),('probe_pid',True),
    ('probe_pid',101.),('probe_pid','101'),('probe_pid',2**31),
    ('resume_pid',None),('resume_pid',0),('resume_pid',-1),('resume_pid',False),
    ('resume_pid',202.),('resume_pid','202'),('resume_pid',2**31),('resume_pid',101),
])
def test_malformed_protocol_rejects_before_checkpoint_read(tmp_path,monkeypatch,field,value):
    recipe,base,proof,checked=fixture(tmp_path,monkeypatch)
    proof[field]=value
    with pytest.raises(ValueError,match='protocol/schema/step/PID'):
        invoke(tmp_path,recipe,base,proof)
    assert checked==[]


@pytest.mark.parametrize('field',['schema','replayed_from_step','verified_step','probe_pid','resume_pid'])
def test_missing_protocol_field_rejects_before_checkpoint_read(tmp_path,monkeypatch,field):
    recipe,base,proof,checked=fixture(tmp_path,monkeypatch)
    del proof[field]
    with pytest.raises(ValueError,match='protocol/schema/step/PID'):
        invoke(tmp_path,recipe,base,proof)
    assert checked==[]
