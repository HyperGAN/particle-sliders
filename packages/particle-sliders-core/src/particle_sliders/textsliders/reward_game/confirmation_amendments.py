"""Explicit source amendments preserve the original frozen protocol and code."""
import json
from pathlib import Path
from .core import read, sha, digest, IntegrityError


def effective_sources(home, folder, protocol):
    sources=dict(protocol['sources'])
    files=sorted((Path(folder)/'amendments').glob('*/amendment.json'))
    if not files:
        return sources
    events=[json.loads(line) for line in (Path(home)/'ledger.jsonl').read_text().splitlines() if line]
    for file in files:
        amendment=read(file)
        entries=[e for e in events if e.get('kind')=='confirmation_amendment_frozen'
                 and e.get('batch')==protocol['name'] and e.get('name')==amendment['name']]
        if len(entries)!=1 or entries[0]['amendment_sha256']!=digest(amendment):
            raise IntegrityError('Confirmation amendment differs from its ledger entry')
        if amendment['protocol_sha256']!=digest(protocol):
            raise IntegrityError('Confirmation amendment belongs to a different protocol')
        if amendment['rendering_changed'] or amendment['ce_scoring_changed']:
            raise IntegrityError('This amendment mechanism cannot change rendering or CE')
        for path,h in amendment.get('preserved_observations_and_audio',{}).items():
            if sha(path)!=h:
                raise IntegrityError('A completed pre-amendment observation or recording changed')
        for change in amendment['changes']:
            path=change['path']
            if sources.get(path)!=change['previous_sha256'] or sha(change['archived_source'])!=change['previous_sha256']:
                raise IntegrityError('Amended original source was not preserved exactly')
            sources[path]=change['current_sha256']
        for path,h in amendment['added_sources'].items():
            if path in sources and sources[path]!=h:
                raise IntegrityError('Amendment conflicts with an existing source')
            sources[path]=h
    return sources
