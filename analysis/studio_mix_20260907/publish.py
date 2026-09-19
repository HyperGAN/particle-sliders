"""Expose completed local study files through the existing studio static server."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

WORK = Path(__file__).resolve().parent
APP_ROOT = WORK.parents[2]


def main():
    state = json.loads((WORK/'renders.json').read_text())
    output = Path(state['output'])
    public = APP_ROOT/'app/static/studio-mix-20260907'
    public.mkdir(exist_ok=True)
    for rec in state['records'].values():
        if rec['status'] != 'complete':
            continue
        for field in ('audio', 'excerpt'):
            source = Path(rec[field])
            dest = public/source.name
            if not dest.exists():
                os.link(source, dest)
            elif not os.path.samefile(source, dest):
                raise ValueError(f'Unexpected existing listening file: {dest}')
    temporary = public/'index.html.tmp'
    shutil.copyfile(output/'index.html', temporary)
    temporary.replace(public/'index.html')
    print(f'{sum(r["status"] == "complete" for r in state["records"].values())}/25 clips at /studio-mix-20260907/')
    return state['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        status = main()
        if not args.watch or status != 'running':
            break
        time.sleep(15)
