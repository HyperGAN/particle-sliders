"""Listening-page presentation, separate from the frozen training protocol.

Receives copies of queue data and writes only the listening directory. Changes
here do not invalidate model checkpoints or training-source fingerprints.
"""
import html
import json
from pathlib import Path


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)


def publish(catalog, status, PAGE):
    PAGE.mkdir(parents=True,exist_ok=True)
    cards=[]
    for item in catalog['sliders']:
        info=status['sliders'].get(item['id'],{})
        text=info.get('status','queued')
        for stage in ('bounded','warmup'):
            entry=info.get(stage,{})
            if info.get('status')=='training' and entry.get('directory') and entry.get('status')=='running':
                run=Path(entry['directory']); logs=list(run.glob('*train*.jsonl'))
                if logs:
                    try:
                        with logs[-1].open('rb') as f:
                            f.seek(max(0,logs[-1].stat().st_size-65536))
                            lines=f.read().decode().splitlines()
                        step=json.loads(lines[-1])['step'];text=f'{stage}: update {step}'
                    except (ValueError,IndexError):pass
                break
        audio=[]
        for folder in sorted((PAGE/item['id']).glob('row-*')):
            for wav in sorted(folder.glob('*-s*/0[123]_*.wav')):
                rel=wav.relative_to(PAGE)
                audio.append(f'<p>{html.escape(str(rel))}<audio controls preload="none" src="{html.escape(str(rel))}"></audio></p>')
        opened=' open' if audio else ''
        cards.append(f'<section><h2>{html.escape(item["label"])}</h2><p>{html.escape(item["description"])}</p>'
            f'<p class="slider-status" data-slider="{html.escape(item["id"])}">{html.escape(text)}</p>'
            f'<details{opened}><summary>Matched audio</summary>{"".join(audio)}</details></section>')
    audio_sig='|'.join(
        str(wav.relative_to(PAGE))
        for item in catalog['sliders']
        for folder in sorted((PAGE/item['id']).glob('row-*'))
        for wav in sorted(folder.glob('*-s*/0[123]_*.wav')))
    status['audio_sig']=audio_sig
    document='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>16 sounds · GAN v1</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;background:#131a20;color:#e6edf1;font:16px system-ui}section{padding:20px 0;border-bottom:1px solid #3b454d}audio{display:block;width:100%;margin:8px 0}a{color:#9bd1ff}p{line-height:1.5}summary{cursor:pointer}</style>
<h1>16 sounds · GAN v1</h1><p>Two vocal controls and fourteen genre and sound controls. Same lyrics and seed in each comparison. Off, trained slider and target-caption reference are retained.</p>'''
    document+=f'<p id="phase">{html.escape(status["phase"])}</p><p><a href="status.json">Progress data</a> · <a href="combos/index.html">Combination comparisons</a></p>'
    # Soft refresh: update progress from status.json; full reload only when
    # new audio appears and nothing is playing. Sections with clips start open.
    document+=''.join(cards)+f'''<script>
const audioSig={json.dumps(audio_sig)};
const y=sessionStorage.getItem('uni16y'); if(y) scrollTo(0,+y);
setInterval(()=>{{
  fetch('status.json',{{cache:'no-store'}}).then(r=>r.json()).then(s=>{{
const phase=document.getElementById('phase');
if(phase&&s.phase) phase.textContent=s.phase;
for(const el of document.querySelectorAll('.slider-status[data-slider]')){{
  const info=(s.sliders||{{}})[el.dataset.slider];
  if(!info) continue;
  let text=info.status||'queued';
  for(const stage of ['bounded','warmup']){{
    const entry=(info[stage]||{{}});
    if(info.status==='training'&&entry.status==='running'){{ text=stage+': running'; break; }}
  }}
  if(info.status==='complete') text='complete';
  el.textContent=text;
}}
if(s.audio_sig&&s.audio_sig!==audioSig&&[...document.querySelectorAll('audio')].every(a=>a.paused)){{
  sessionStorage.setItem('uni16y',String(scrollY));
  location.reload();
}}
  }}).catch(()=>{{}});
}},10000);
</script>'''
    (PAGE/'index.html').write_text(document);write(PAGE/'status.json',status)
