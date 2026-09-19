"""Bounded dependency, target recovery and training for the matched-policy attempt."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from .core import read
from .resources import gpu_lease


def run(home,folder,total,collection_unit):
    folder=Path(folder);deadline=time.monotonic()+3600
    while True:
        state=subprocess.run(['systemctl','--user','show',collection_unit,'-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()
        if state not in ('active','activating','deactivating'):
            status=read(folder/'collection-status.json') if (folder/'collection-status.json').exists() else {}
            if status.get('state')!='pairs_frozen':raise RuntimeError('Collection stopped without all frozen pairs; retained for review')
            break
        if time.monotonic()>deadline:raise TimeoutError('Matched collection dependency exceeded one hour')
        time.sleep(5)
    subprocess.run([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.prepare_matched','--home',str(home),'--folder',str(folder)],check=True)
    with gpu_lease(home,1):
        subprocess.run([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.train_matched_imitation','--folder',str(folder),'--gpu','1','--total',str(total),'--lr','1e-6'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--total',type=int,default=10);p.add_argument('--collection-unit',required=True)
    a=p.parse_args();run(a.home,a.folder,a.total,a.collection_unit)
