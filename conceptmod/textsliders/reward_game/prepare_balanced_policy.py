"""Verify and copy exact frozen parent targets, without recovering or rendering."""
import argparse
from pathlib import Path
import shutil
from .core import read,write,immutable,sha,digest,IntegrityError,locked


def prepare(source,folder):
    source=Path(source).resolve();folder=Path(folder).resolve()
    with locked(folder/'prepare.lock'):
        episodes=read(source/'episodes.json');recovery=read(source/'recovery-recipe.json')
        if len(episodes)!=8 or read(source/'prepare-status.json')['verified_targets']!=8:raise IntegrityError('Need eight exact parent targets')
        if recovery['manifest_sha256']!=sha(source/'manifest.json') or recovery['episodes_sha256']!=sha(source/'episodes.json'):
            raise IntegrityError('Parent target metadata changed')
        for path,h in recovery['source_hashes'].items():
            if sha(path)!=h:raise IntegrityError('Parent target recovery source changed')
        files=['manifest.json','episodes.json','policy-parent.json','recovery-recipe.json','prepare-status.json']
        for e in episodes:
            name=f"tokens/{e['id']}.pt";audit=read((source/name).with_suffix('.json'))
            if not audit['passed'] or audit['sha256']!=sha(source/name):raise IntegrityError('Parent target failed integrity')
            files.append(name)
        plan=dict(source=str(source),files={name:sha(source/name) for name in files},new_audio_generated=0,new_replays=0,reused_exact_targets=8,
                  source_hash=sha(__file__),reference_cache='not copied; regenerated for the new optimizer recipe')
        immutable(folder/'target-reuse.json',plan)
        for name,h in plan['files'].items():
            target=folder/name;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(source/name,target)
            if sha(target)!=h:raise IntegrityError('Copied target changed')
            if name.endswith('.pt'):
                old=(source/name).with_suffix('.json');audit=read(old)
                audit.update(reused_exact_target=True,reused_from=str(source/name),source_audit_sha256=sha(old),seconds=0.)
                immutable(target.with_suffix('.json'),audit)
        return plan


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--folder',required=True)
    a=p.parse_args();print(__import__('json').dumps(prepare(a.source,a.folder),indent=2))
