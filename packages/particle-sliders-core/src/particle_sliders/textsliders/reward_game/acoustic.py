"""Explicit CLI for the separate acoustic adapter format and game identity."""
import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('init');q.add_argument('--parent-home',required=True);q.add_argument('--audit-folder',required=True)
    sub.add_parser('audit-off');sub.add_parser('leaderboard')
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--multiplier',type=float,default=1.);q.add_argument('--stage',default='auto',choices=('auto','4','8','16'));q.add_argument('--tag',default='acoustic-manual');q.add_argument('--max-new-clips',type=int,default=16)
    q=sub.add_parser('inspect');q.add_argument('--run',required=True)
    a=p.parse_args()
    if a.action=='init':
        from .acoustic_setup import init
        result=init(a.home,a.parent_home,a.audit_folder)
    else:
        from . import acoustic_evaluate as e
        if a.action=='evaluate':result=e.evaluate(a.home,a.checkpoint,a.multiplier,a.stage,a.tag,max_new_clips=a.max_new_clips)
        elif a.action=='inspect':result=e.inspect(a.home,a.run)
        elif a.action=='audit-off':result=e.engineering_audit(a.home)
        else:result=e.leaderboard(a.home)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
