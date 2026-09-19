"""Audit the separate Off bridge, then evaluate the fixed block composition."""
import argparse,os,signal,subprocess,sys
from pathlib import Path
from .core import read,Store


def run(home):
    home=Path(home).resolve();root=Path(read(home/'game.json')['parent_home'])
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    for stage,argv in (
        ('acoustic_block_off_audit',[sys.executable,'-u','-m','conceptmod.textsliders.reward_game.block','--home',str(home),'audit-off']),
        ('acoustic_block_development',[sys.executable,'-u','-m','conceptmod.textsliders.reward_game.block_search','--home',str(home),'search','--resume'])):
        Store(root).status(stage,research_complete=False,game_home=str(home))
        process=subprocess.Popen(argv,start_new_session=True)
        try:code=process.wait(timeout=7200)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait()
            raise
        if code:raise RuntimeError(f'{stage} exited {code}; preserve its original records and logs')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);a=p.parse_args();run(a.home)
