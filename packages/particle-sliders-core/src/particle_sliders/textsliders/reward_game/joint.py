"""CLI for fixed ordinary LM plus acoustic candidates."""
import argparse,json


def main():
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('init');q.add_argument('--parent-home',required=True);q.add_argument('--acoustic-home',required=True);q.add_argument('--bundle',required=True)
    sub.add_parser('audit-off');sub.add_parser('leaderboard')
    q=sub.add_parser('evaluate');q.add_argument('--checkpoint',required=True);q.add_argument('--stage',default='auto',choices=('auto','4','8','16'));q.add_argument('--tag',default='joint-manual')
    q=sub.add_parser('inspect');q.add_argument('--run',required=True)
    a=p.parse_args()
    if a.action=='init':
        from .joint_setup import init
        result=init(a.home,a.parent_home,a.acoustic_home,a.bundle)
    else:
        from . import joint_evaluate as e
        if a.action=='audit-off':result=e.engineering_audit(a.home)
        elif a.action=='leaderboard':result=e.leaderboard(a.home)
        elif a.action=='inspect':result=e.inspect(a.home,a.run)
        else:result=e.evaluate(a.home,a.checkpoint,1.,a.stage,a.tag)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
