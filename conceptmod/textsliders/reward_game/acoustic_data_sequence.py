"""Collect exact training states, then verify the acoustic game Off bridge."""
import argparse
from pathlib import Path
import subprocess
import os
import signal
import sys
import time
from .core import read,immutable,sha,Store


def run(home,folder,game_home):
    home=Path(home).resolve();folder=Path(folder).resolve();game_home=Path(game_home).resolve()
    immutable(folder/'data-sequence-protocol.json',dict(source_sha256=sha(__file__),collection_recipe_sha256=sha(folder/'collection-recipe.json'),
        acoustic_benchmark_sha256=read(game_home/'game.json')['benchmark_sha256'],new_engineering_clip_budget=8,optimizer_updates=0,max_seconds=7200))
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel);began=time.monotonic()
    commands=[['conceptmod.textsliders.reward_game.collect_acoustic','--home',str(home),'--folder',str(folder),'run'],
              ['conceptmod.textsliders.reward_game.acoustic','--home',str(game_home),'audit-off']]
    for index,command in enumerate(commands):
        Store(home).status('acoustic_training_collection' if index==0 else 'acoustic_game_off_audit',research_complete=False,folder=str(folder),game_home=str(game_home))
        process=subprocess.Popen([sys.executable,'-u','-m',*command],start_new_session=True)
        try:code=process.wait(timeout=max(1.,7200-(time.monotonic()-began)))
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=180)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        if code:raise RuntimeError(f'Acoustic data prerequisite exited {code}; outputs retained')
    Store(home).status('acoustic_data_ready',research_complete=False,folder=str(folder),game_home=str(game_home))
    Store(home).event('acoustic_data_ready',folder=str(folder),new_engineering_clip_budget=8)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True);p.add_argument('--game-home',required=True);a=p.parse_args();run(a.home,a.folder,a.game_home)
