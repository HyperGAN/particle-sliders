"""Registered bounded attempts; rejection advances the backlog, errors do not."""
import os
from pathlib import Path
import subprocess
import shutil
import signal
from contextlib import nullcontext
import time

from .core import Store, read, write, immutable, digest, sha, locked, IntegrityError
from .block_evaluate import evaluate

REQUIRED=('hypothesis','failure_mechanism','method','parent_checkpoint','changed_variables','budget','sources')


def _command(recipe,env,log):
    process=subprocess.Popen(recipe['argv'],cwd=recipe.get('cwd'),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        process.wait(timeout=recipe['budget']['max_seconds'])
    except BaseException:
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
        raise
    return process


def register(home,recipe_path):
    store=Store(home);recipe=read(recipe_path)
    if any(k not in recipe for k in REQUIRED):raise ValueError('Recipe needs '+', '.join(REQUIRED))
    if recipe['method'] not in ('checkpoint','command'):raise ValueError('Registered methods: checkpoint, command')
    if recipe['method']=='command' and (not recipe.get('argv') or recipe['budget'].get('max_seconds',0)<=0):
        raise ValueError('Command methods require argv and an elapsed-time budget')
    for path,h in recipe['sources'].items():
        if sha(path)!=h:raise IntegrityError('Recipe source changed')
    ident='attempt-'+digest(recipe)[:16];folder=store.home/'attempts'/ident
    immutable(folder/'recipe.json',recipe)
    for path,h in recipe['sources'].items():
        target=folder/'sources'/h/Path(path).name
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(path,target)
        if sha(target)!=h:raise IntegrityError('Archived recipe source differs')
    if not (folder/'status.json').exists():
        write(folder/'status.json',dict(state='registered',created_unix=time.time()))
        store.event('attempt_registered',attempt_id=ident,recipe_sha256=digest(recipe),method=recipe['method'])
    return dict(attempt_id=ident,recipe_sha256=digest(recipe))


def execute(home,folder,recipe,evaluator=evaluate):
    store=Store(home)
    for path,h in recipe['sources'].items():
        if sha(path)!=h:raise IntegrityError('Training source changed; register a new recipe')
    if recipe['method']=='command' and not (folder/'training-result.json').exists():
        write(folder/'status.json',dict(state='training',started_unix=time.time()))
        env=dict(os.environ,**recipe.get('env',{}),REWARD_GAME_ATTEMPT=str(folder))
        began=time.monotonic()
        from .resources import gpu_lease
        lease=gpu_lease(home,recipe['training_gpu']) if recipe.get('training_gpu') is not None else nullcontext()
        with lease, (folder/'training.log').open('a') as log:
            result=_command(recipe,env,log)
        if result.returncode:raise RuntimeError(f'Training exited {result.returncode}; retained state in {folder}')
        status=read(recipe['training_status']) if recipe.get('training_status') else {}
        no_candidate=status.get('state')=='no_candidate_selected'
        if no_candidate and not recipe.get('allow_no_candidate',False):raise IntegrityError('Undeclared selection exit')
        if not no_candidate and not Path(recipe['checkpoint']).exists():raise IntegrityError('Method exited without its declared adapter')
        actual_updates=None
        if recipe.get('expected_updates') is not None:
            actual_updates=status['actual_updates']
            if actual_updates!=recipe['expected_updates']:raise IntegrityError('Actual optimizer step count differs from declared checkpoint')
        local_updates=actual_updates-recipe.get('initial_updates',0) if actual_updates is not None else None
        if local_updates is not None and (local_updates<0 or local_updates>recipe['budget'].get('optimizer_updates',local_updates)):
            raise IntegrityError('Actual new optimizer updates exceed the attempt budget')
        state=recipe.get('training_state')
        selection=recipe.get('selection_result')
        if selection and (status.get('selection_sha256')!=sha(selection) or read(selection)['selected_off']!=no_candidate):
            raise IntegrityError('Selection result does not match method status')
        immutable(folder/'training-result.json',dict(elapsed_seconds=time.monotonic()-began,
                   checkpoint_sha256=None if no_candidate else sha(recipe['checkpoint']),no_candidate_selected=no_candidate,
                   actual_updates=actual_updates,local_optimizer_updates=local_updates,training_state=state,training_state_sha256=sha(state) if state else None,
                   selection_result=selection,selection_sha256=sha(selection) if selection else None))
    if (folder/'training-result.json').exists() and read(folder/'training-result.json').get('no_candidate_selected'):
        result=read(folder/'training-result.json');selection=read(result['selection_result'])
        if sha(result['selection_result'])!=result['selection_sha256']:raise IntegrityError('Selection changed after completion')
        immutable(folder/'training-selection.json',selection)
        decision=dict(advance=False,decision='rejected_training_selection',run_id=None,
                      next_action='try_next_registered_attempt',reason='Off won the predeclared training objective')
        immutable(folder/'decision.json',decision);write(folder/'status.json',dict(state='decided',**decision))
        from .block_artifacts import archive_selection
        archive_selection(home,folder.name)
        store.event('attempt_decided',attempt_id=folder.name,**decision)
        return 'rejected'
    if recipe.get('checkpoint') and Path(recipe['checkpoint']).exists():
        for path in (Path(recipe['checkpoint']),Path(recipe['checkpoint']).with_suffix('.json')):
            target=folder/('adapter'+path.suffix)
            if not target.exists():shutil.copy2(path,target)
            if sha(target)!=sha(path):raise IntegrityError('Attempt adapter archive differs from evaluated weights')
    kwargs={'max_new_clips':recipe['budget'].get('new_clips',16)} if evaluator is evaluate else {}
    card=evaluator(home,recipe['checkpoint'],recipe.get('multiplier',1.),'auto',folder.name,**kwargs)
    if card['decision'] in ('engineering_failure','incomplete'):
        write(folder/'status.json',dict(state='engineering_failure',run_id=card['run_id']))
        return 'engineering_failure'
    immutable(folder/'scorecard.json',card)
    decision=dict(advance=card['advance'],decision=card['decision'],run_id=card['run_id'],
                  next_action='freeze_fresh_confirmation_protocol' if card['advance'] and card['stage']==16 else 'try_next_registered_attempt')
    immutable(folder/'decision.json',decision);write(folder/'status.json',dict(state='decided',**decision))
    from .block_artifacts import archive
    archive(home,folder.name)
    store.event('attempt_decided',attempt_id=folder.name,**decision)
    return 'confirmation_required' if card['advance'] and card['stage']==16 else 'budget_exhausted' if card['decision']=='attempt_budget_exhausted' else 'rejected'


def search(home,resume=True,evaluator=evaluate):
    store=Store(home)
    with locked(store.home/'search.lock'):
        completed=[]
        while True:
            if (store.home/'STOP').exists():
                store.status('user_stopped',research_complete=False);return dict(state='user_stopped',completed=completed)
            folders=sorted((store.home/'attempts').glob('attempt-*'),key=lambda p:read(p/'status.json')['created_unix'] if 'created_unix' in read(p/'status.json') else p.stat().st_ctime)
            pending=[p for p in folders if not (p/'decision.json').exists()]
            if not pending:
                store.event('method_selection_required',completed_attempts=len(folders))
                store.status('method_selection_required',research_complete=False,completed_attempts=len(folders),
                             next_action='Supervising agent reviews failures and registers a materially different method')
                return dict(state='method_selection_required',research_complete=False,completed=completed)
            folder=pending[0];recipe=read(folder/'recipe.json')
            store.status('research_attempt',attempt_id=folder.name,research_complete=False)
            try:state=execute(home,folder,recipe,evaluator)
            except Exception as exc:
                write(folder/'status.json',dict(state='engineering_failure',error=repr(exc)))
                store.event('attempt_engineering_failure',attempt_id=folder.name,error=repr(exc))
                store.status('engineering_failure',attempt_id=folder.name,error=repr(exc),research_complete=False)
                return dict(state='engineering_failure',error=repr(exc),completed=completed)
            completed.append(dict(attempt_id=folder.name,state=state))
            if state not in ('rejected','budget_exhausted'):
                store.status(state,attempt_id=folder.name,research_complete=False)
                return dict(state=state,completed=completed)
