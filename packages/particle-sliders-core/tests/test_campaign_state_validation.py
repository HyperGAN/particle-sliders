"""Campaign IO rejects candidates before filesystem or live adapter mutation."""
from copy import deepcopy
import json

import pytest
import torch

from particle_sliders.pg09_moss_campaign import MossCampaignRun, file_digest
from test_conditional_pg09 import equal, fixture


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def campaign():
    game = fixture()
    game.backend.model_identity = dict(model='bounded CPU campaign fixture', revision='fixture-v1')
    return MossCampaignRun(game.backend, game.rows, game)


def export_trained(tmp_path):
    source = campaign()
    source.step()
    path = tmp_path / 'adapter.safetensors'
    source.export_adapter(path)
    return source, path


def test_failed_campaign_export_creates_no_directory_or_adapter(tmp_path, monkeypatch):
    run = campaign()

    def fail_after_critic_commit(*args, **kwargs):
        raise RuntimeError('forced generator optimizer failure')

    monkeypatch.setattr(run.game.opt_g, 'step', fail_after_critic_commit)
    with pytest.raises(RuntimeError, match='forced generator'):
        run.step()
    assert run.game.completed_steps == 0
    path = tmp_path / 'not-created' / 'adapter.safetensors'
    with pytest.raises(RuntimeError, match='failed update'):
        run.export_adapter(path)
    assert not path.parent.exists()


@pytest.mark.parametrize('steps', [-1, 6, True, 1.5, None, '1', 'missing'])
def test_malformed_campaign_step_rejected_before_shared_loader(tmp_path, monkeypatch, steps):
    _, path = export_trained(tmp_path)
    receipt_path = path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    if steps == 'missing':
        del receipt['completed_steps']
    else:
        receipt['completed_steps'] = steps
    receipt_path.write_text(json.dumps(receipt))
    receiver = campaign()
    before = receiver.game.state_dict()

    def forbidden(*args, **kwargs):
        raise AssertionError('malformed steps reached adapter loading')

    monkeypatch.setattr('particle_sliders.pg09_moss_campaign.load_adapter', forbidden)
    with pytest.raises(ValueError, match='export step'):
        receiver.load_adapter(path)
    equal(before, receiver.game.state_dict())


@pytest.mark.parametrize('corruption', ['nan', 'dtype', 'shape', 'missing-key', 'metadata'])
def test_invalid_safetensors_candidate_never_mutates_live_campaign(tmp_path, corruption):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    _, path = export_trained(tmp_path)
    with safe_open(path, framework='pt', device='cpu') as file:
        metadata = file.metadata()
    state = load_file(path)
    key = next(iter(state))
    if corruption == 'nan':
        state[key].flatten()[0] = float('nan')
    elif corruption == 'dtype':
        state[key] = state[key].double()
    elif corruption == 'shape':
        state[key] = state[key].reshape(-1)[:1].clone()
    elif corruption == 'missing-key':
        del state[key]
    else:
        metadata = {'pg09_backend': '{}'}
    save_file(state, str(path), metadata=metadata)
    receipt_path = path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    receipt['sha256'] = file_digest(path)
    receipt_path.write_text(json.dumps(receipt))
    receiver = campaign()
    before = receiver.game.state_dict()
    with pytest.raises(ValueError, match='adapter export|backend protocol'):
        receiver.load_adapter(path)
    equal(before, receiver.game.state_dict())


def test_core_rejection_after_safetensors_validation_keeps_live_adapter(tmp_path, monkeypatch):
    source, path = export_trained(tmp_path)
    expected = source.game.export_state_dict()
    receiver = campaign()
    before = receiver.game.state_dict()
    assert any(not torch.equal(value, expected['network'][name])
               for name, value in before['network'].items())

    def reject_candidate(candidate):
        equal(candidate, expected)
        raise ValueError('forced core preflight rejection')

    monkeypatch.setattr(receiver.game, 'load_export_state_dict', reject_candidate)
    with pytest.raises(ValueError, match='forced core'):
        receiver.load_adapter(path)
    equal(before, receiver.game.state_dict())


def test_valid_campaign_adapter_commits_once_with_step_provenance(tmp_path, monkeypatch):
    source, path = export_trained(tmp_path)
    receiver = campaign()
    loads = []
    original_load = receiver.game.network.load_state_dict

    def record_load(state, strict=True):
        loads.append(deepcopy(state))
        return original_load(state, strict=strict)

    monkeypatch.setattr(receiver.game.network, 'load_state_dict', record_load)
    metadata = receiver.load_adapter(path)
    assert metadata['backend'] == receiver.backend.metadata()
    assert len(loads) == 1
    assert receiver.game.completed_steps == source.game.completed_steps == 1
    equal(receiver.game.export_state_dict(), source.game.export_state_dict())
    with pytest.raises(RuntimeError, match='inference export'):
        receiver.step()
    assert not torch.cuda.is_initialized()
