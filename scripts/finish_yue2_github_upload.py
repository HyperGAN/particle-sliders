#!/usr/bin/env python3
"""Finish the verified GitHub mirror after its large uploads complete."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PAGE=ROOT/'eval/listen/yue2-lora-distill-20260917'
ANALYSIS=ROOT/'analysis/yue2_lora_distill_20260917'
REPO='mikkel/yue2-concept-sliders'
TAG='distilled-rank8-20260917'
API=f'repos/{REPO}/releases/390932810'
COMMIT='75c55aaf278157ce96a68649fc841290fa5f01fc'


def gh(*args):
    return subprocess.check_output(['/home/mikkel/.local/bin/gh',*args],text=True)


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def status(stage,**values):
    value=dict(stage=stage,time=datetime.now(timezone.utc).isoformat(),
        url=f'https://github.com/{REPO}/releases/tag/{TAG}',**values)
    for path in (PAGE/'github-publication.json',ANALYSIS/'github-publication.json'):
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(path)
    print(json.dumps(value),flush=True)


def process_identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None


def missing(expected):
    release=json.loads(gh('api',API))
    assert release['tag_name']==TAG and release['target_commitish']==COMMIT
    found={a['name']:a for a in release['assets']}
    wanted=[]
    for entry in expected:
        actual=found.get(entry['name'],{})
        if (actual.get('state')!='uploaded' or actual.get('size')!=entry['bytes']
                or actual.get('digest')!='sha256:'+entry['sha256']):
            wanted.append(entry['name'])
    return wanted,release


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wait-pid',type=int,default=836765)
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    manifest=PAGE/'artifact-manifest.json'
    expected=json.loads(manifest.read_text())['assets']
    expected.append(dict(name=manifest.name,bytes=manifest.stat().st_size,sha256=digest(manifest)))
    for entry in expected:
        path=PAGE/entry['name']
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
    wanted,release=missing(expected)
    if args.check:
        print(json.dumps(dict(local_checksums_valid=True,pending=wanted,draft=release['draft'])))
        return
    identity=process_identity(args.wait_pid)
    while identity and process_identity(args.wait_pid)==identity:
        status('uploading',pending=missing(expected)[0])
        time.sleep(30)
    for attempt in range(3):
        wanted,release=missing(expected)
        if not wanted:break
        status('retrying',pending=wanted,attempt=attempt+1)
        for name in wanted:
            try:gh('release','upload',TAG,str(PAGE/name),'--repo',REPO,'--clobber')
            except subprocess.CalledProcessError:
                if attempt==2:raise
                time.sleep(30)
    wanted,release=missing(expected)
    assert not wanted, f'Incomplete or invalid uploaded assets: {wanted}'
    if release['draft']:
        gh('release','edit',TAG,'--repo',REPO,'--draft=false','--prerelease')
    wanted,release=missing(expected)
    assert not wanted and not release['draft'] and release['prerelease']
    status('published',assets=expected,commit=COMMIT)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc))
        raise
