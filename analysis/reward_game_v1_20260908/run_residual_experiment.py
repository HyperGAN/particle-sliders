"""Start the registered richer-loss pilot only after target recovery passes."""
import signal
import subprocess
import time
from conceptmod.textsliders.reward_game.core import DEFAULT_HOME,Store,read,write
from conceptmod.textsliders.reward_game.controller import search


def main():
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    home=DEFAULT_HOME;store=Store(home);folder=home/'training/residual-imitation-v1'
    while subprocess.run(['systemctl','--user','is-active','--quiet','music-reward-game-residual-recovery-v1-20260908.service']).returncode==0:
        if (home/'STOP').exists():return
        time.sleep(5)
    if (home/'STOP').exists():return
    status=read(folder/'status.json')
    if status['state']!='targets_prepared' or status['verified_targets']!=24 or not read(folder/'off-restoration-limit24.json')['exact']:
        store.status('engineering_failure',research_complete=False,reason='Resolve exact residual target recovery before training')
        return
    result=search(home);write(home/'residual-learning-result.json',result)


if __name__=='__main__':main()
