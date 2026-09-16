#!/usr/bin/env python3
"""Matched held-out YuE2 Arm B renders at -1, 0, 0.5, 1 plus caption references."""
from pathlib import Path
import argparse
import html
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.yue2_arm_b import load_prompts,RECIPE
from conceptmod.textsliders.yue2_backend import YuE2Slider,file_digest
from conceptmod.textsliders.infer_yue2 import render
from conceptmod.textsliders.train_lora_yue2_fresh import write_json

TAKES=[('clean',-1.,'neutral'),('off',0.,'neutral'),('half',.5,'neutral'),
       ('metal',1.,'neutral'),('metal-caption',0.,'positive'),('clean-caption',0.,'negative')]


def page(output,rows,seeds):
    cards=[]
    for i,row in enumerate(rows):
        for seed in seeds:
            cells=[]
            for name,scale,field in TAKES:
                relative=Path(f'row-{i}-seed-{seed}')/name
                meta=output/relative/'evaluation.json'
                if meta.exists():
                    d=json.loads(meta.read_text())
                    cells.append(f'<div><b>{html.escape(name)}</b> · {d["duration"]:.1f}s'
                        f'<audio controls preload="none" src="{relative}/audio.flac"></audio></div>')
                else:cells.append(f'<div><b>{html.escape(name)}</b> · pending</div>')
            cards.append(f'<section><h2>Prompt {i+1}, seed {seed}</h2><p>{html.escape(row["neutral"])}</p>'
                +''.join(cells)+'</section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>YuE2 metal · Arm B</title>'
        '<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}'
        'section{padding:20px;border:1px solid #46505b;margin:20px 0}audio{display:block;width:100%;margin:10px 0}'
        'section>div{display:inline-block;vertical-align:top;width:46%;margin:1%}a{color:#a9d7ff}</style>'
        '<h1>YuE2 metal · Music Arm B</h1><p id="status">Matched prompts and seeds. Experimental checkpoints.</p>'
        '<p><a href="metal-yue2.safetensors">Download trained slider</a> · <a href="metal-yue2.json">Training details</a></p>'
        +''.join(cards)+
        '<script>async function poll(){try{const r=await fetch("status.json",{cache:"no-store"});'
        'if(r.ok){const s=await r.json();document.getElementById("status").textContent=s.stage||'
        '`${s.status}: ${s.completed}/${s.total} updates`;}}catch(e){}}poll();setInterval(poll,15000);</script>')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1709,2903])
    p.add_argument('--max_tokens',type=int,default=500)
    args=p.parse_args(argv)
    if args.max_tokens<1 or not args.seeds or any(not 0<=s<2**63 for s in args.seeds):p.error('Invalid sampling budget or seeds')
    rows,meta=load_prompts(args.prompts_file)
    args.output_dir.mkdir(parents=True,exist_ok=True);page(args.output_dir,rows,args.seeds)
    from yue2 import YuE2Pipeline
    from yue2.storage import verify_result
    digest=file_digest(args.weights)
    with YuE2Pipeline.from_pretrained('m-a-p/YuE2-3B',vae='m-a-p/YuE2-Vae',local_files_only=True,
            device='cuda:0',backend='torch-eager',memory_budget_gib=18,quantization='none',offload_ar=False) as pipe:
        network,record=YuE2Slider.load(pipe._load_model(),args.weights)
        if record.get('recipe')!=RECIPE['name']:raise ValueError('Expected a native Arm B checkpoint')
        if record['model_identity']!=pipe.weights['mot']:raise ValueError('Base model differs from training')
        if {r['lyrics'] for r in record['rows']}&{r['lyrics'] for r in rows}:raise ValueError('Evaluation lyrics overlap training')
        for i,row in enumerate(rows):
            for seed in args.seeds:
                for name,scale,field in TAKES:
                    dest=args.output_dir/f'row-{i}-seed-{seed}'/name
                    request=dict(weights_sha256=digest,scale=scale,style=row[field],lyrics=row['lyrics'],seed=seed,
                        max_tokens=args.max_tokens,backend='torch-eager',cot='off')
                    if (dest/'evaluation.json').exists():
                        prior=json.loads((dest/'evaluation.json').read_text())
                        if prior['request']!=request:raise ValueError('Existing render request differs')
                        verify_result(dest)
                        if file_digest(dest/'audio.flac')!=prior['audio_sha256']:raise ValueError('Audio digest differs')
                        continue
                    if dest.exists():raise FileExistsError(f'Incomplete output: {dest}')
                    staging=dest.with_name(dest.name+f'.pending-{time.time_ns()}')
                    result=render(pipe,network,style=row[field],lyrics=row['lyrics'],scale=scale,seed=seed,
                        adapter_identity=digest,semantic_sampling={'min_tokens':min(200,args.max_tokens),'max_tokens':args.max_tokens})
                    audio=result.audio
                    if audio.ndim!=2 or audio.shape[1]!=2 or not len(audio) or not np.isfinite(audio).all():
                        raise ValueError('Invalid native audio')
                    result.save_artifacts(staging);verify_result(staging)
                    stats=dict(request=request,duration=len(audio)/48000,rms=float(np.sqrt(np.mean(audio.astype(float)**2))),
                        clipped_fraction=float(np.mean(np.abs(audio)>=.999)),truncated=result.truncated,
                        audio_sha256=file_digest(staging/'audio.flac'))
                    write_json(staging/'evaluation.json',stats);staging.rename(dest)
                    page(args.output_dir,rows,args.seeds)
                    print(json.dumps(dict(row=i,seed=seed,take=name,duration=stats['duration'],rms=stats['rms'])),flush=True)
    import shutil
    shutil.copy2(args.weights,args.output_dir/'metal-yue2.safetensors')
    shutil.copy2(args.weights.with_suffix('.json'),args.output_dir/'metal-yue2.json')
    write_json(args.output_dir/'status.json',dict(stage='Training and matched rendering complete'))
    page(args.output_dir,rows,args.seeds)

if __name__=='__main__':main()
