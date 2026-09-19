"""Separate GPU leases for exact parent target preparation and short training."""
import argparse
import subprocess
import sys

from .resources import gpu_lease


def run(home,folder,source,total):
    subprocess.run([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.prepare_parent_policy',
        '--home',home,'--folder',folder,'--source',source],check=True)
    with gpu_lease(home,1):
        subprocess.run([sys.executable,'-u','-m','conceptmod.textsliders.reward_game.train_parent_policy',
            '--folder',folder,'--gpu','1','--total',str(total),'--lr','1e-6'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--folder',required=True)
    p.add_argument('--source',required=True);p.add_argument('--total',type=int,default=10)
    a=p.parse_args();run(a.home,a.folder,a.source,a.total)
