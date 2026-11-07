"""Model-independent cache identities and immutable tensor publication, CPU only."""
from concurrent.futures import ThreadPoolExecutor
import os

import pytest
import torch

from particle_sliders.execution.cache_identity import (
    cache_key, cache_path, fingerprint_file, load_tensor, save_tensor,
)


def test_file_identity_uses_bytes_even_when_path_size_and_mtime_are_unchanged(tmp_path):
    path = tmp_path / 'weights.bin'
    path.write_bytes(b'old-weights')
    stamp = path.stat()
    original = fingerprint_file(path)
    path.write_bytes(b'new-weights')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    changed = fingerprint_file(path)
    assert original['size'] == changed['size']
    assert original['sha256'] != changed['sha256']


def test_exact_request_and_identity_changes_never_alias():
    identity, request = {'model': 'first', 'settings': {'precision': 'fp32'}}, {'duration': 1.0001, 'scale': .0001}
    original = cache_key(identity, request)
    assert cache_key(dict(reversed(list(identity.items()))), request) == original
    for changed in ({'duration': 1.0002, 'scale': .0001}, {'duration': 1.0001, 'scale': .0002}):
        assert cache_key(identity, changed) != original
    assert cache_key(dict(identity, model='second'), request) != original
    assert cache_key(dict(identity, settings={'precision': 'bf16'}), request) != original
    with pytest.raises(ValueError):
        cache_key({}, request)
    with pytest.raises(ValueError):
        cache_key(identity, {'duration': float('nan')})


def test_complete_roundtrip_retains_existing_artifact_instead_of_overwriting(tmp_path):
    identity, request = {'model': 'bytes'}, {'seed': 7}
    path = cache_path(tmp_path, identity, request)
    first = torch.arange(6, dtype=torch.float64).reshape(1, 3, 2)
    assert torch.equal(save_tensor(path, first, identity, request), first)
    captured = path.read_bytes()
    assert torch.equal(load_tensor(path, identity, request), first)
    assert torch.equal(save_tensor(path, first + 5, identity, request), first)
    assert path.read_bytes() == captured
    assert not list(tmp_path.glob('*.tmp'))


def test_wrong_identity_or_legacy_payload_is_refused_without_modification(tmp_path):
    identity, request = {'model': 'bytes'}, {'seed': 7}
    path = cache_path(tmp_path, identity, request)
    save_tensor(path, torch.ones(1), identity, request)
    captured = path.read_bytes()
    with pytest.raises(ValueError, match='identity differs'):
        load_tensor(path, {'model': 'other'}, request)
    with pytest.raises(ValueError, match='identity differs'):
        save_tensor(path, torch.ones(1), {'model': 'other'}, request)
    assert path.read_bytes() == captured
    torch.save({'condition': torch.ones(1)}, path)
    with pytest.raises(ValueError, match='schema or identity'):
        load_tensor(path, identity, request)


@pytest.mark.parametrize('tensor', [torch.empty(0), torch.tensor([float('nan')]), torch.tensor([float('inf')])])
def test_invalid_frozen_tensor_is_never_published(tmp_path, tensor):
    path = cache_path(tmp_path, {'model': 'bytes'}, {})
    with pytest.raises(ValueError, match='finite'):
        save_tensor(path, tensor, {'model': 'bytes'}, {})
    assert not path.exists()


def test_competing_writers_observe_the_same_complete_immutable_artifact(tmp_path):
    identity, request = {'model': 'bytes'}, {'seed': 7}
    path = cache_path(tmp_path, identity, request)
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(save_tensor, path, torch.full((2, 2), value), identity, request)
                   for value in (1., 2.)]
        results = [future.result(timeout=10) for future in futures]
    stored = load_tensor(path, identity, request)
    assert all(torch.equal(result, stored) for result in results)
    assert not list(tmp_path.glob('*.tmp'))
    assert not torch.cuda.is_initialized()
