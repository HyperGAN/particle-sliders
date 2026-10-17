"""Hash existing staged audio bytes; never deserialize pretrained tensors."""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


started = time.monotonic()
music_root = Path('/mnt/nvme/sliders/qualification/models/music3-bd348f9')
music_manifest = music_root / 'qualification-manifest.json'
yue_manifest = Path('/mnt/nvme/sliders/qualification/models/yue2-cached/qualification-manifest.json')
receipts = {}
for family in ('yue2', 'music3'):
    manifest_path = yue_manifest if family == 'yue2' else music_manifest
    manifest = json.loads(manifest_path.read_text())
    files = []
    if family == 'music3':
        candidates = [(music_root / entry['filename'], entry['sha256'], entry['bytes'])
                      for entry in manifest['files']]
    else:
        candidates = []
        for record in manifest['records'].values():
            root = Path(record['path'])
            for name, entry in record['identity']['files'].items():
                candidates.append((root / name, entry['sha256'], entry['bytes']))
            candidates.append((root / 'config.json', record['identity']['config_sha256'], None))
        tokenizer = Path(manifest['records']['model']['path']) / 'qwen.tiktoken'
        candidates.append((tokenizer, 'b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186', None))
    for path, expected, expected_bytes in candidates:
        actual_bytes = path.stat().st_size
        actual = sha(path)
        if actual != expected or (expected_bytes is not None and actual_bytes != expected_bytes):
            raise ValueError('staged audio file identity differs: ' + str(path))
        files.append(dict(path=str(path), bytes=actual_bytes, sha256=actual, verified=True))
    receipts[family] = dict(manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
                            files=files, total_bytes=sum(f['bytes'] for f in files))
output = Path('/ml2/codex-storage/future-workspaces/2026-10-17/task-4/audio-next-assets.json')
output.write_text(json.dumps(dict(status='actual_existing_bytes_verified', families=receipts,
    elapsed_seconds=time.monotonic() - started, model_tensors_loaded=False,
    cuda_used=False, downloads=False), indent=2) + '\n')
print(json.dumps(dict(output=str(output), elapsed_seconds=time.monotonic() - started,
    total_verified_bytes=sum(r['total_bytes'] for r in receipts.values()))))
