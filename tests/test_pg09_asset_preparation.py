import hashlib
import json

import pytest

from scripts.prepare_text_video_pg09_verification import audit_manifest


def fixture(tmp_path):
    config = b'{"hidden_size":1024}'
    (tmp_path / 'config.json').write_bytes(config)
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    files = [dict(file='config.json', bytes=len(config), sha256=hashlib.sha256(config).hexdigest()),
             dict(file='model.safetensors', bytes=7, sha256='prior-large-file-attestation')]
    return freeze(tmp_path, files)


def freeze(root, files):
    manifest = json.dumps(dict(revision='immutable-test', files=files)).encode()
    (root / 'qualification-manifest.json').write_bytes(manifest)
    return dict(revision='immutable-test', manifest_sha256=hashlib.sha256(manifest).hexdigest())


def test_metadata_only_receipt_does_not_claim_weight_verification(tmp_path):
    expected = fixture(tmp_path)
    # Treat the small synthetic weight as large using the explicit threshold.
    receipt = audit_manifest(tmp_path, expected, metadata_limit=3)
    assert receipt['model_loaded'] is False
    assert receipt['fresh_large_weight_hash_verification'] is False
    assert all(x['hash_evidence'] == 'prior manifest attestation; stat only' for x in receipt['files'])


def test_missing_weight_stops_preparation(tmp_path):
    expected = fixture(tmp_path)
    (tmp_path / 'model.safetensors').unlink()
    with pytest.raises(ValueError, match='missing or wrong-sized'):
        audit_manifest(tmp_path, expected)


def test_changed_authorized_manifest_stops_preparation(tmp_path):
    expected = fixture(tmp_path)
    (tmp_path / 'qualification-manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='authorized asset pin'):
        audit_manifest(tmp_path, expected)


def test_changed_metadata_stops_preparation(tmp_path):
    expected = fixture(tmp_path)
    (tmp_path / 'config.json').write_bytes(b'{"hidden_size":2048}')
    with pytest.raises(ValueError, match='metadata hash differs'):
        audit_manifest(tmp_path, expected)


@pytest.mark.parametrize('relative', ['../other', '/tmp/other'])
def test_escaping_manifest_path_rejected(tmp_path, relative):
    expected = freeze(tmp_path, [dict(file=relative, bytes=1, sha256='x')])
    with pytest.raises(ValueError, match='inside its authorized'):
        audit_manifest(tmp_path, expected)


def test_escaping_symlink_rejected(tmp_path):
    root = tmp_path / 'assets'
    root.mkdir()
    outside = tmp_path / 'elsewhere'
    outside.write_bytes(b'x')
    (root / 'model').symlink_to(outside)
    expected = freeze(root, [dict(file='model', bytes=1, sha256=hashlib.sha256(b'x').hexdigest())])
    with pytest.raises(ValueError, match='symlink escapes'):
        audit_manifest(root, expected)


def test_exact_linked_metadata_cache_allowed_but_large_weights_not(tmp_path):
    root, cache = tmp_path / 'assets', tmp_path / 'metadata-cache'
    root.mkdir()
    cache.mkdir()
    (cache / 'config').write_bytes(b'{}')
    (root / 'config').symlink_to(cache / 'config')
    expected = freeze(root, [dict(file='config', bytes=2, sha256=hashlib.sha256(b'{}').hexdigest())])
    expected['linked_metadata_roots'] = [str(cache)]
    assert audit_manifest(root, expected)['files'][0]['hash_evidence'] == 'fresh metadata hash'
    with pytest.raises(ValueError, match='symlink escapes'):
        audit_manifest(root, expected, metadata_limit=1)
