import importlib
import json
from pathlib import Path

import pytest

from particle_sliders.textsliders.reward_game.core import IntegrityError, read, sha, write
from particle_sliders.textsliders.reward_game.package_commit import staged_package


PREFIXES = ['', 'acoustic_', 'ff_', 'block_', 'joint_', 'robust_block_', 'transport_', 'full_flow_']


def prepare_confirmation(tmp_path, monkeypatch, module):
    source = tmp_path / 'training' / 'adapter.safetensors'
    source.parent.mkdir()
    source.write_bytes(b'fixed trained bytes')
    write(source.with_suffix('.json'), {})
    write(source.parent / 'recipe.json', {'unchanged': True})
    candidate = dict(path=str(source), multiplier=1., weights_sha256=sha(source), components=[])
    for kind in ('language_model', 'condition_encoder'):
        weights = source.parent / kind / 'adapter.safetensors'
        weights.parent.mkdir()
        weights.write_bytes(kind.encode())
        write(weights.with_suffix('.json'), {})
        write(weights.parent / 'recipe.json', {})
        candidate['components'].append(dict(path=str(weights), weights_sha256=sha(weights), structure={'kind': kind}))
    ledger = []
    for batch in ('first', 'second'):
        folder = tmp_path / 'confirmation' / batch
        for name in ('protocol.json', 'manifest.json', 'scorecard.json', 'intent/tolerances.json'):
            write(folder / name, {})
        write(folder / 'audit/off-restoration.json', {'exact': True})
        observations = {}
        for i in range(48):
            path = folder / 'observations' / f'{i}.json'
            write(path, {})
            observations[str(path)] = sha(path)
        write(folder / 'intent/preservation.json', dict(passed=True, observations_sha256=observations,
            tolerance_sha256=sha(folder / 'intent/tolerances.json')))
        ledger.append(dict(kind='confirmation_preservation_finished', batch=batch,
            result_sha256=sha(folder / 'intent/preservation.json')))
    (tmp_path / 'ledger.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in ledger))
    for name in ('manifest.json', 'scorecard.json'):
        write(tmp_path / 'composition/c' / name, {})
    write(tmp_path / 'composition/c/protocol.json', {'candidate': candidate})
    write(tmp_path / 'composition/c/off-restoration.json', {'exact': True})
    write(tmp_path / 'game.json', {'parent_home': str(tmp_path / 'parent')})
    def verify(home, folder):
        first = folder.name == 'first'
        return dict(replication_of=None if first else 'first', candidate=candidate,
            seeds=[1] if first else [2], cases=[{'family': 'a' if first else 'b'}], diagnostics={'sources': {}}), {}
    monkeypatch.setattr(module, 'verify_batch', verify)
    monkeypatch.setattr(module, 'score', lambda *args: {'batch_pass': True})
    monkeypatch.setattr(module, 'composition_score', lambda *args: {'passed': True})
    monkeypatch.setattr(module, 'checkpoint', lambda *args: None)
    monkeypatch.setattr(module, 'report', lambda *args: {})
    return source


@pytest.mark.parametrize('prefix', PREFIXES)
def test_copy_failure_leaves_no_partial_package_and_same_name_can_retry(tmp_path, monkeypatch, prefix):
    module = importlib.import_module('particle_sliders.textsliders.reward_game.' + prefix + 'complete_confirmation')
    source = prepare_confirmation(tmp_path, monkeypatch, module)
    original = module.shutil.copy2
    calls = []
    def copy(src, dest):
        calls.append(str(src))
        result = original(src, dest)
        if len(calls) == 2:
            raise OSError('copy failed after a retained prefix')
        return result
    monkeypatch.setattr(module.shutil, 'copy2', copy)
    with pytest.raises(OSError, match='copy failed'):
        module.complete(tmp_path, 'first', 'second', 'c', 'winner')
    assert not (tmp_path / 'confirmed/winner').exists()
    assert not list((tmp_path / 'confirmed').glob('.winner-*'))
    monkeypatch.setattr(module.shutil, 'copy2', original)
    result = module.complete(tmp_path, 'first', 'second', 'c', 'winner')
    destination = tmp_path / 'confirmed/winner'
    assert result['research_complete'] and (destination / 'evidence.json').is_file()
    assert (destination / 'README.md').is_file()
    assert source.read_bytes() == b'fixed trained bytes'
    if prefix == 'joint_':
        for component in read(destination / 'packaged-components.json')['components']:
            assert Path(component['weights']).is_file()
            assert '.winner-' not in component['weights']
    with pytest.raises(IntegrityError, match='already exists'):
        module.complete(tmp_path, 'first', 'second', 'c', 'winner')


def test_final_path_is_absent_until_all_files_are_ready(tmp_path):
    destination = tmp_path / 'winner'
    with staged_package(destination) as staging:
        (staging / 'a').write_bytes(b'complete')
        assert not destination.exists()
    assert (destination / 'a').read_bytes() == b'complete'
