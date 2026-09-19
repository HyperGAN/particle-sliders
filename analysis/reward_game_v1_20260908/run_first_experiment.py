"""Execute the predeclared pilot only after the live game acceptance checks."""
from pathlib import Path
import signal
import subprocess
import time
from conceptmod.textsliders.reward_game.core import DEFAULT_HOME, Store, read, write
from conceptmod.textsliders.reward_game.controller import register, search


def main():
    home=DEFAULT_HOME;store=Store(home)
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    unit='music-reward-game-acceptance-v3-20260908.service'
    while subprocess.run(['systemctl','--user','is-active','--quiet',unit]).returncode==0:
        if (home/'STOP').exists():return
        time.sleep(5)
    if (home/'STOP').exists():return
    checks=[home/'audit/initialization.json',home/'audit/reference-interface.json',home/'audit/off-pcm-v1/result.json',home/'training/merged-imitation-v1/live-forward-audit.json']
    for path in checks:
        if not path.exists() or not read(path)['passed']:
            store.status('engineering_failure',research_complete=False,reason='Pilot awaits a passed acceptance check',check=str(path))
            return
    result=read(home/'audit/controller-live.json')
    if result['state']!='method_selection_required':
        store.status(result['state'],research_complete=False,reason='Resolve the live controller result before changing the method')
        return
    recipe=home/'recipes/merged-imitation-10.json'
    registration=register(home,recipe)
    store.event('new_learning_method_started',**registration,recipe=str(recipe))
    result=search(home)
    write(home/'first-learning-attempt-result.json',result)


if __name__=='__main__':main()
