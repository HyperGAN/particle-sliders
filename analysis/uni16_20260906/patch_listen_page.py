#!/usr/bin/env python3
"""Keep uni16 listen index usable while an older train.py still rewrites it.

Opens sections that have audio and removes the 30s hard reload.
Safe to leave running; exits once the live page already uses the soft template
from a restarted train.py for a while, or after --hours.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

PAGE = Path('/ml2/music/sliders-conceptmod/eval/listen/uni16-gan-v1/index.html')
SOFT = '''<script>
(()=>{
  const y=sessionStorage.getItem('uni16y'); if(y) scrollTo(0,+y);
  document.querySelectorAll('details').forEach(d=>{ if(d.querySelector('audio')) d.open=true; });
  setInterval(()=>{
    fetch('status.json',{cache:'no-store'}).then(r=>r.json()).then(s=>{
      const nodes=[...document.querySelectorAll('body > p')];
      if(nodes[1] && s.phase) nodes[1].textContent=s.phase;
    }).catch(()=>{});
  },10000);
})();
</script>'''


def fix(text: str) -> str:
    def open_if_audio(match: re.Match[str]) -> str:
        block = match.group(0)
        if '<audio' in block and not block.startswith('<details open'):
            return '<details open' + block[len('<details'):]
        return block

    text = re.sub(r'<details(?: open)?>(?:(?!</details>).)*</details>', open_if_audio, text, flags=re.S)
    text = text.replace('<script>setTimeout(()=>location.reload(),30000)</script>', SOFT)
    if 'setTimeout(()=>location.reload()' in text:
        text = re.sub(r'<script>setTimeout\(\(\)=>location\.reload\(\),\s*\d+\)</script>', SOFT, text)
    if 'sessionStorage.getItem(\'uni16y\')' not in text and 'audioSig' not in text:
        text = text.rstrip() + SOFT
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hours', type=float, default=12.0)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    deadline = time.time() + args.hours * 3600
    soft_streak = 0
    while time.time() < deadline:
        raw = PAGE.read_text() if PAGE.exists() else ''
        if 'audioSig' in raw and 'setTimeout(()=>location.reload()' not in raw:
            soft_streak += 1
            if soft_streak >= 30:
                print(time.strftime('%H:%M:%S'), 'train.py soft template stable; exiting', flush=True)
                return 0
        else:
            soft_streak = 0
            fixed = fix(raw)
            if fixed != raw:
                PAGE.write_text(fixed)
                print(time.strftime('%H:%M:%S'), 'patched index.html', flush=True)
        if args.once:
            return 0
        time.sleep(1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
