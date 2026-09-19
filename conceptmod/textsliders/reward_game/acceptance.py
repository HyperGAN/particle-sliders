"""Live acceptance: one Off PCM audit, cached controls, and a failed adapter."""
import subprocess
import time
import signal

from .core import ROOT,DEFAULT_HOME,read,write,sha,Store,immutable
from .evaluate import evaluate,engineering_audit,compare
from .controller import register,search


def main():
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    home=DEFAULT_HOME;store=Store(home)
    unit='music-reward-game-merged-audit-v1-20260908.service'
    while subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:
        time.sleep(5)
    game=read(read(home/'game.json')['spec']);path=game['original_reference']['checkpoint']
    results=[]
    for checkpoint,multiplier in ((None,0.),(path,1.),(path,.5)):
        card=evaluate(home,checkpoint,multiplier,'16','reference-acceptance')
        repeated=evaluate(home,checkpoint,multiplier,'16','repeat-reference-acceptance')
        if card!=repeated or card['new_clips']!=0 or card['cache_hits']!=16:
            raise RuntimeError('Reference/cache acceptance failed')
        results.append(card)
    immutable(home/'audit/reference-interface.json',dict(passed=True,run_ids=[r['run_id'] for r in results],
              identical_repeated_scorecards=True,new_clips=0))
    write(home/'audit/reference-comparison.json',compare(home,results[2]['run_id'],results[1]['run_id']))
    engineering_audit(home)
    failed=ROOT/'analysis/reward_preference_20260908/students/gentle/reward-ce-preference-gentle_step60.safetensors'
    for index,checkpoint,multiplier,hypothesis,mechanism in (
        (0,str(failed),1.,'Previously rejected preference continuation is evaluated through the frozen game interface.',
         'Acceptance diagnostic: failed preference checkpoints must receive a scientific decision.'),
        (1,path,.5,'The fixed half-strength reference follows a rejection through the same controller.',
         'Acceptance diagnostic: rejecting one checkpoint must advance to another registered attempt.')):
        recipe=dict(hypothesis=hypothesis,failure_mechanism=mechanism,method='checkpoint',parent_checkpoint=path,
                    changed_variables={'checkpoint':index,'fixed_multiplier':multiplier},budget={'new_clips':16,'optimizer_updates':0},
                    sources={str(checkpoint):sha(checkpoint)},checkpoint=checkpoint,multiplier=multiplier,
                    acceptance_only=True)
        target=home/'recipes'/f'acceptance-{index}.json';immutable(target,recipe);register(home,target)
    result=search(home);write(home/'audit/controller-live.json',result)
    store.event('acceptance_campaign_finished',result=result,research_complete=False)


if __name__=='__main__':main()
