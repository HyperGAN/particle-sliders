"""Reusable reward-slider game. JSON on stdout; readable scorecards on stderr."""
import argparse
import json
import sys
import signal

from .core import DEFAULT_HOME, DEFAULT_SPEC, markdown


def main():
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--home',default=str(DEFAULT_HOME))
    commands=p.add_subparsers(dest='command',required=True)
    q=commands.add_parser('init');q.add_argument('--spec',default=str(DEFAULT_SPEC))
    q=commands.add_parser('evaluate');q.add_argument('--checkpoint',required=True)
    q.add_argument('--multiplier',type=float,default=1.);q.add_argument('--stage',choices=['auto','4','8','16'],default='auto');q.add_argument('--tag',default='manual')
    commands.add_parser('leaderboard')
    q=commands.add_parser('inspect');q.add_argument('--run',required=True)
    q=commands.add_parser('compare');q.add_argument('--left',required=True);q.add_argument('--right',required=True)
    q=commands.add_parser('register');q.add_argument('--recipe',required=True)
    q=commands.add_parser('search');q.add_argument('--resume',action='store_true')
    commands.add_parser('audit-off')
    a=p.parse_args()
    from . import setup,evaluate,controller
    try:
        if a.command=='init':result=setup.init(a.home,a.spec)
        elif a.command=='evaluate':result=evaluate.evaluate(a.home,None if a.checkpoint=='off' else a.checkpoint,a.multiplier,a.stage,a.tag)
        elif a.command=='leaderboard':result=evaluate.leaderboard(a.home)
        elif a.command=='inspect':result=evaluate.inspect(a.home,a.run)
        elif a.command=='compare':result=evaluate.compare(a.home,a.left,a.right)
        elif a.command=='register':result=controller.register(a.home,a.recipe)
        elif a.command=='search':result=controller.search(a.home,a.resume)
        else:result=evaluate.engineering_audit(a.home)
        print(json.dumps(result,indent=2,allow_nan=False))
        if 'comparisons' in result:print(markdown(result),file=sys.stderr)
        if result.get('decision') in ('engineering_failure','incomplete') or result.get('state')=='engineering_failure':return 2
        return 0
    except Exception as exc:
        print(json.dumps(dict(state='engineering_failure',type=type(exc).__name__,error=str(exc))),file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
