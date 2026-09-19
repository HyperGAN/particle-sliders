import pytest
from conceptmod.textsliders.reward_game import complete_confirmation as module
from conceptmod.textsliders.reward_game.core import IntegrityError


def test_first_fresh_success_never_packages_failed_replication(tmp_path,monkeypatch):
    def verify(home,folder):
        first=folder.name=='first'
        return dict(replication_of=None if first else 'first',candidate=dict(path='unchanged'),seeds=[1,2] if first else [3,4],
                    cases=[dict(family='a' if first else 'b')]),{}
    monkeypatch.setattr(module,'verify_batch',verify)
    monkeypatch.setattr(module,'score',lambda home,name:dict(batch_pass=name=='first'))
    with pytest.raises(IntegrityError,match='Both fresh batches'):
        module.complete(tmp_path,'first','second','composition','winner')
    assert not (tmp_path/'confirmed').exists()


def test_replication_of_different_candidate_never_packages(tmp_path,monkeypatch):
    def verify(home,folder):
        first=folder.name=='first'
        return dict(replication_of=None if first else 'first',candidate=dict(path='a' if first else 'b')),{}
    monkeypatch.setattr(module,'verify_batch',verify)
    with pytest.raises(IntegrityError,match='same fixed adapter'):
        module.complete(tmp_path,'first','second','composition','winner')
    assert not (tmp_path/'confirmed').exists()
