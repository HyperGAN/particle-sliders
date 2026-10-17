#!/usr/bin/env python3
"""Compose a review-only video wheel from exact core and video Git objects.

This does not modify the shared checkout or publish a package. The base wheel
lineage and all four video source hashes are recorded beside the candidate.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

PACKAGE = 'packages/particle-sliders-core'
MODULES = ('ltx25_backend.py', 'ltx25_uni.py', 'ltx25_cli.py', 'pg09_ltx_embed.py')


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-repo', required=True)
    p.add_argument('--core-head', required=True)
    p.add_argument('--video-repo', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--video-head', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).absolute()
    out.mkdir(parents=True, exist_ok=True)
    core = git(args.core_repo, 'rev-parse', args.core_head + '^{commit}').decode().strip()
    video = git(args.video_repo, 'rev-parse', args.video_head + '^{commit}').decode().strip()
    sources = {}
    with tempfile.TemporaryDirectory(prefix='video-wheel-build-') as directory:
        root = Path(directory)
        archive = git(args.core_repo, 'archive', '--format=tar', core, PACKAGE)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter='data')
        for name in MODULES:
            path = PACKAGE + '/src/particle_sliders/' + name
            data = git(args.video_repo, 'show', video + ':' + path)
            (root / path).write_bytes(data)
            sources[path] = hashlib.sha256(data).hexdigest()
        result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                                 '--no-cache-dir', '--wheel-dir', str(out), str(root / PACKAGE)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / 'build.log').write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout)
    wheels = list(out.glob('particle_sliders_core-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('candidate output must contain exactly one core wheel')
    receipt = dict(schema='video-wheel-candidate-composition-v1', core_head=core, video_head=video,
                   video_sources=sources, wheel=wheels[0].name,
                   wheel_sha256=hashlib.sha256(wheels[0].read_bytes()).hexdigest(),
                   review_candidate_only=True, runtime_admission=False)
    (out / 'composition.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
