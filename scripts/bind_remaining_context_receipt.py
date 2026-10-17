#!/usr/bin/env python3
"""Offline only: bind a completed native preparation into a NEW unadmitted draft."""
import argparse
import json
from pathlib import Path

from conceptmod.textsliders.pg09_remaining_bind import bind_prepared_contexts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft',type=Path,required=True)
    parser.add_argument('--preparation-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(bind_prepared_contexts(args.draft,args.preparation_output,args.output),indent=2))


if __name__=='__main__':
    main()
