import pytest
from conceptmod.textsliders.reward_game import confirmation as module
from conceptmod.textsliders.reward_game.core import IntegrityError, Store, immutable, digest, write


def test_freeze_refuses_partial_development_before_loading_candidate(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_game import evaluate
    monkeypatch.setattr(module,'verify',lambda home: ({},{}))
    monkeypatch.setattr(evaluate,'inspect',lambda home,evaluation: dict(stage=16,valid_cases=15,advance=False))
    monkeypatch.setattr(module,'checkpoint',lambda *args:pytest.fail('Must not inspect or render a candidate before its development pass'))
    with pytest.raises(IntegrityError,match='strict completed'):
        module.freeze(tmp_path,'partial',tmp_path/'unused.json','batch')
    assert not (tmp_path/'confirmation').exists()


def test_frozen_protocol_tamper_is_rejected_before_assets(tmp_path,monkeypatch):
    folder=tmp_path/'confirmation/batch'
    manifest=dict(style_hashes={});protocol=dict(name='batch',manifest_sha256=digest(manifest),sources={})
    immutable(folder/'manifest.json',manifest);immutable(folder/'protocol.json',protocol)
    Store(tmp_path).event('confirmation_frozen',batch='batch',protocol_sha256=digest(protocol))
    protocol['new_clip_budget']=100
    write(folder/'protocol.json',protocol)
    monkeypatch.setattr(module,'verify',lambda home:None)
    with pytest.raises(IntegrityError,match='frozen ledger'):
        module.verify_batch(tmp_path,folder)
