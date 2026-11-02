"""Package a fixed ordinary acoustic adapter only after every declared confirmation check."""
import argparse
from pathlib import Path
import shutil
import json
from .core import read,write,immutable,sha,IntegrityError,Store
from .ff_artifact import checkpoint
from .ff_confirmation import verify_batch,score
from .ff_composition import score as composition_score
from .costs import report


def complete(home,first,replication,composition,name):
    home=Path(home).resolve();p1,_=verify_batch(home,home/'confirmation'/first);p2,_=verify_batch(home,home/'confirmation'/replication)
    if p1['replication_of'] is not None or p2['replication_of']!=first or p1['candidate']!=p2['candidate']:
        raise IntegrityError('Need a first batch and independent replication of the same fixed adapter')
    if set(p1['seeds'])&set(p2['seeds']) or {c['family'] for c in p1['cases']}&{c['family'] for c in p2['cases']}:
        raise IntegrityError('Replication families and seeds must be new')
    cards={first:score(home,first),replication:score(home,replication)}
    if not all(c['batch_pass'] for c in cards.values()):raise IntegrityError('Both fresh batches must independently pass practical and uncertainty gates')
    for batch,p in ((first,p1),(replication,p2)):
        folder=home/'confirmation'/batch;intent=read(folder/'intent/preservation.json')
        if not intent['passed']:raise IntegrityError('Independent preservation review failed')
        events=[json.loads(l) for l in (home/'ledger.jsonl').read_text().splitlines()]
        finished=[r for r in events if r['kind']=='confirmation_preservation_finished' and r['batch']==batch]
        if not finished or any(r['result_sha256']!=sha(folder/'intent/preservation.json') for r in finished):raise IntegrityError('Preservation report changed from the completed ledger entry')
        if len(intent['observations_sha256'])!=48 or sha(folder/'intent/tolerances.json')!=intent['tolerance_sha256']:raise IntegrityError('Preservation evidence incomplete or changed')
        for file,h in intent['observations_sha256'].items():
            if sha(file)!=h:raise IntegrityError('Preservation observation changed')
        if not read(folder/'audit/off-restoration.json')['exact']:raise IntegrityError('Fresh renderer Off restoration failed')
        # Result metadata is tied to all declared diagnostic source files.
        for file,h in p['diagnostics']['sources'].items():
            if sha(file)!=h:raise IntegrityError('Diagnostic implementation changed after confirmation')
    comp=composition_score(home,composition);cp=read(home/'composition'/composition/'protocol.json')
    if not comp['passed'] or cp['candidate']!=p1['candidate'] or not read(home/'composition'/composition/'off-restoration.json')['exact']:
        raise IntegrityError('Matching fixed-energy composition evidence is incomplete or failed')
    c=p1['candidate'];checkpoint(c['path'],c['multiplier'])
    folder=home/'confirmed'/name
    if folder.exists():raise IntegrityError('Completed package name already exists')
    evidence=[home/'confirmation'/batch/f for batch in (first,replication) for f in ('protocol.json','manifest.json','scorecard.json','intent/preservation.json','intent/tolerances.json','audit/off-restoration.json')]
    evidence += [home/'composition'/composition/f for f in ('protocol.json','manifest.json','scorecard.json','off-restoration.json')]
    source=Path(c['path']);metadata=read(source.with_suffix('.json'));recipe=source.parent/'recipe.json'
    if not recipe.exists():raise IntegrityError('Exact training recipe is missing')
    folder.mkdir(parents=True)
    for file,target in ((source,folder/'reward-ce-confirmed.safetensors'),(source.with_suffix('.json'),folder/'reward-ce-confirmed.json'),(recipe,folder/'training-recipe.json')):
        shutil.copy2(file,target)
        if sha(file)!=sha(target):raise IntegrityError('Final adapter package copy changed')
    for label,path in (('first-confirmation',home/'confirmation'/first),('independent-replication',home/'confirmation'/replication),('composition',home/'composition'/composition),('training-evidence',source.parent)):
        (folder/label).symlink_to(path,target_is_directory=True)
    result=dict(research_complete=True,scope='Frozen first-20-second CE with independently measured preservation and diversity; natural full-song completion remains unestablished',
        adapter_sha256=c['weights_sha256'],multiplier=c['multiplier'],training_recipe_sha256=sha(recipe),first=first,replication=replication,composition=composition,
        evidence_sha256={str(p):sha(p) for p in evidence},cost=report(Path(read(home/'game.json')['parent_home'])),production_deployment=False)
    immutable(folder/'evidence.json',result)
    (folder/'README.md').write_text('Confirmed ordinary reward adapter at multiplier '+str(c['multiplier'])+'.\n\nBoth fresh batches independently passed the fixed practical CE gates and whole-family uncertainty intervals, plus the declared preservation diagnostics. The separate fixed per-host-energy composition comparison passed. Exact evidence, raw audio, losses and listening players are linked in this directory.\n\nThe supported scope is the frozen first 20 seconds of generated music. Natural full-song completion remains unestablished. This research package has not been added to the production registry.\n')
    Store(home).event('research_confirmed',package=str(folder),adapter_sha256=c['weights_sha256'],evidence_sha256=sha(folder/'evidence.json'))
    Store(home).status('confirmed',research_complete=True,package=str(folder),adapter_sha256=c['weights_sha256'])
    parent=Path(read(home/'game.json')['parent_home'])
    Store(parent).event('research_confirmed',package=str(folder),adapter_sha256=c['weights_sha256'],evidence_sha256=sha(folder/'evidence.json'))
    Store(parent).status('confirmed',research_complete=True,package=str(folder),adapter_sha256=c['weights_sha256'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--first',required=True);p.add_argument('--replication',required=True)
    p.add_argument('--composition',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    print(__import__('json').dumps(complete(a.home,a.first,a.replication,a.composition,a.name),indent=2))
