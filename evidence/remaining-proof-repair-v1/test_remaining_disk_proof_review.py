"""Pure JSON proof gates; no model, checkpoint load, CUDA, or optimizer."""
import json
from types import SimpleNamespace
import pytest
from conceptmod.textsliders import pg09_remaining_worker as worker


def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoints/step-000003/state.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'CPU synthetic checksum marker, not a torch packet')
    reference = tmp_path / 'probe-reference.pt'
    reference.write_bytes(b'CPU synthetic reference marker')
    recipe = SimpleNamespace(identity='a' * 64)
    base = dict(combined_sha256='b' * 64)
    proof = dict(schema='remaining-gan-fresh-disk-proof-v1', recipe_sha256=recipe.identity,
                 reference_sha256=worker.sha(reference), checkpoint_sha256=worker.sha(checkpoint),
                 frozen_base=base, probe_pid=101, resume_pid=202, replayed_from_step=2,
                 verified_step=3, checkpoint_advanced=False, exact_loss=True,
                 exact_full_state=True, exact_sample=True, frozen_base_unchanged=True)
    monkeypatch.setattr(worker, 'verify_checkpoint', lambda *args: None)
    return recipe, base, proof


def invoke(tmp_path, recipe, base, proof):
    path = tmp_path / 'fresh-disk-resume.json'
    path.write_text(json.dumps(proof))
    worker.verify_disk_proof(recipe, tmp_path,
                             dict(fresh_disk_proof_sha256=worker.sha(path)), base)


def test_exact_synthetic_proof_is_accepted(tmp_path, monkeypatch):
    recipe, base, proof = fixture(tmp_path, monkeypatch)
    invoke(tmp_path, recipe, base, proof)


@pytest.mark.parametrize('field,value', [('schema','unrelated-proof'),
    ('replayed_from_step',0), ('verified_step',99)])
def test_wrong_proof_protocol_is_rejected(tmp_path, monkeypatch, field, value):
    recipe, base, proof = fixture(tmp_path, monkeypatch)
    proof[field] = value
    with pytest.raises(ValueError):
        invoke(tmp_path, recipe, base, proof)
