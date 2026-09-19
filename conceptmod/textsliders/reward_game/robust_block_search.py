"""Attempt ledger for the declared acoustic host, linked to the parent campaign."""
import argparse
from pathlib import Path
from .core import read,Store
from . import robust_block_controller


def main():
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('register');q.add_argument('--recipe',required=True)
    q=sub.add_parser('search');q.add_argument('--resume',action='store_true')
    a=p.parse_args();home=Path(a.home).resolve();parent=Path(read(home/'game.json')['parent_home'])
    if a.action=='register':result=robust_block_controller.register(home,a.recipe)
    else:
        Store(parent).status('acoustic_candidate_search',research_complete=False,game_home=str(home))
        result=robust_block_controller.search(home)
        Store(parent).status('block_search_review_required',research_complete=False,game_home=str(home),result=result)
    print(__import__('json').dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
