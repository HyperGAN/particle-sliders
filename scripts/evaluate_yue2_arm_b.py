#!/usr/bin/env python3
"""Matched held-out unipolar YuE2 renders at 0, 0.5, 1 plus the metal caption."""
from pathlib import Path
import argparse
import html
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders.yue2_arm_b import load_prompts,RECIPE
from conceptmod.textsliders.yue2_gan_plus_neu import RECIPE as PLUS_NEU_RECIPE
from conceptmod.textsliders.yue2_backend import YuE2Backend,YuE2Slider,file_digest
from conceptmod.textsliders.infer_yue2 import render
from conceptmod.textsliders.train_lora_yue2_fresh import write_json
from scripts.yue2_training_dashboard import dashboard_html

TAKES=[('off',0.,'neutral'),('half',.5,'neutral'),
       ('metal',1.,'neutral'),('metal-caption',0.,'positive')]


def takes(include_canary=False, only=None):
    items=TAKES + ([('minus-canary',-1.,'neutral')] if include_canary else [])
    if not only:return items
    alias={'on':'metal'}
    names=[alias.get(name,name) for name in only]
    by_name={item[0]:item for item in items}
    missing=[name for name in names if name not in by_name]
    if missing:raise SystemExit(f'Unknown takes: {missing}. Choose from {", ".join(by_name)}')
    return [by_name[name] for name in names]


@torch.no_grad()
def hidden_diagnostics(model,tokenizer,network,rows,include_canary=False):
    """Held-out hidden geometry only; these are not toy continuation gates."""
    backend=SimpleNamespace(model=model,tokenizer=tokenizer)
    records=[]
    for i,row in enumerate(rows):
        ids=YuE2Backend.prefix(backend,row['neutral'],row['lyrics'])
        pos_ids=YuE2Backend.prefix(backend,row['positive'],row['lyrics'])
        with network.scaled(0.):
            neutral=YuE2Backend.hidden(backend,ids)[:,-1].float()
            positive=YuE2Backend.hidden(backend,pos_ids)[:,-1].float()
        target=positive-neutral
        norm=target.norm()
        if not torch.isfinite(norm) or norm<=0:raise ValueError('Invalid held-out caption delta')
        for scale in ([0.,.5,1.,-1.] if include_canary else [0.,.5,1.]):
            with network.scaled(scale):delta=YuE2Backend.hidden(backend,ids)[:,-1].float()-neutral
            projection=(delta*target).sum()/norm.square()
            orthogonal=(delta-projection*target).norm()/norm
            records.append(dict(row=i,scale=scale,canary=scale<0,
                caption_projection=float(projection),orthogonal_ratio=float(orthogonal),
                target_relative_error=float((delta-target).norm()/norm),
                neutral_delta_norm=float(delta.norm()),zero_exact=bool(torch.equal(delta,torch.zeros_like(delta))) if scale==0 else None))
    return dict(kind='unscored_hidden_geometry',records=records,
        definitions=dict(caption_projection='dot(student-neutral, positive-neutral) / ||positive-neutral||^2',
            orthogonal_ratio='norm of residual orthogonal to caption delta / norm of caption delta',
            target_relative_error='norm(student-positive) / norm(positive-neutral)',
            zero_exact='bitwise equality with the base prompt state at scale zero'),
        limitation='Not calibrated audio cover/leak or lyric-preservation gates; -1 is unscored.')


def page(output,rows,seeds,recipe='unipolar_gan',include_canary=False,label='Metal',only=None):
    label=html.escape(label)
    title = 'YuE2 metal · GAN + neutral' if recipe == 'gan_plus_neu' else 'YuE2 metal · Unipolar GAN'
    detail = 'Trained with the +/0 conditional GAN.' if recipe == 'gan_plus_neu' else 'Trained only at +1.'
    if recipe == 'particle_bridge':
        title = 'YuE2 metal · Routed particle bridge'
        detail = 'Paired-error GAN with routed particles and particle VIC; EMA weights. Trained at +1.'
    title=title.replace('YuE2 metal',f'YuE2 {label}')
    display={'off':'Off','half':'Half','metal':label,'metal-caption':f'{label} caption','minus-canary':'−1 canary'}
    cards=[]
    for i,row in enumerate(rows):
        for seed in seeds:
            cells=[]
            for name,scale,field in takes(include_canary, only):
                relative=Path(f'row-{i}-seed-{seed}')/name
                meta=output/relative/'evaluation.json'
                if meta.exists():
                    d=json.loads(meta.read_text())
                    cells.append(f'<div><b>{display[name]}</b> · {d["duration"]:.1f}s'
                        f'<audio controls preload="none" src="{relative}/audio.flac"></audio></div>')
                else:cells.append(f'<div><b>{display[name]}</b> · pending</div>')
            cards.append(f'<section><h2>Prompt {i+1}, seed {seed}</h2><p>{html.escape(row["neutral"])}</p>'
                +''.join(cells)+'</section>')
    document=(f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>'
        '<style>body{max-width:1100px;margin:40px auto;padding:0 20px;background:#16191d;color:#eee;font:16px system-ui}'
        'section{padding:20px;border:1px solid #46505b;margin:20px 0}audio{display:block;width:100%;margin:10px 0}'
        'section>div{display:inline-block;vertical-align:top;width:46%;margin:1%}a{color:#a9d7ff}</style>'
        f'<h1>{title}</h1><p>0 = Off · 1 = {label}. {detail}</p>'
        + ('<p>−1 is an untrained, unscored canary.</p>' if include_canary else '') +
        '<p id="status">Matched prompts and seeds. Experimental checkpoints.</p>'
        '<p><a href="metal-yue2.safetensors">Download trained slider</a> · <a href="metal-yue2.json">Training details</a></p>'
        +dashboard_html(recipe)+'<h2>Held-out listening comparisons</h2>'+''.join(cards))
    temporary=output/'index.html.tmp';temporary.write_text(document);temporary.replace(output/'index.html')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',choices=['unipolar_gan','gan_plus_neu','particle_bridge'],default='unipolar_gan')
    p.add_argument('--include_canary',action='store_true',help='Render -1 as an unscored diagnostic')
    p.add_argument('--hidden_diagnostics',action='store_true',help='Record unscored held-out caption geometry')
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--prompts_file',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-yue2-metal-arm-b-eval.yaml')
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--takes',nargs='+',default=None,help='Subset to render, e.g. off metal. Default is every take. on is an alias for the scale-1 take.')
    p.add_argument('--seeds',type=int,nargs='+',default=[1709,2903])
    p.add_argument('--max_tokens',type=int,default=500)
    args=p.parse_args(argv)
    if args.max_tokens<1 or not args.seeds or any(not 0<=s<2**63 for s in args.seeds):p.error('Invalid sampling budget or seeds')
    rows,meta=load_prompts(args.prompts_file)
    args.output_dir.mkdir(parents=True,exist_ok=True);page(args.output_dir,rows,args.seeds,args.recipe,args.include_canary,meta.get('plus_label','Metal'),args.takes)
    from yue2 import YuE2Pipeline
    from yue2.storage import verify_result
    digest=file_digest(args.weights)
    with YuE2Pipeline.from_pretrained('m-a-p/YuE2-3B',vae='m-a-p/YuE2-Vae',local_files_only=True,
            device='cuda:0',backend='torch-eager',memory_budget_gib=18,quantization='none',offload_ar=False) as pipe:
        network,record=YuE2Slider.load(pipe._load_model(),args.weights)
        expected = PLUS_NEU_RECIPE if args.recipe == 'gan_plus_neu' else RECIPE
        if args.recipe == 'particle_bridge':
            from conceptmod.textsliders.yue2_particle_bridge import RECIPE as expected
        if record.get('recipe')!=expected['name'] or record.get('trained_scales')!=expected['trained_scales']:
            raise ValueError('Expected a unipolar GAN checkpoint; bipolar checkpoints are not accepted')
        if record['model_identity']!=pipe.weights['mot']:raise ValueError('Base model differs from training')
        if {r['lyrics'] for r in record['rows']}&{r['lyrics'] for r in rows}:raise ValueError('Evaluation lyrics overlap training')
        if args.hidden_diagnostics:
            diagnostics=hidden_diagnostics(pipe._load_model(),pipe.tokenizer,network,rows,args.include_canary)
            write_json(args.output_dir/'hidden-diagnostics.json',dict(diagnostics,weights_sha256=digest))
        for i,row in enumerate(rows):
            for seed in args.seeds:
                for name,scale,field in takes(args.include_canary, args.takes):
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
                    page(args.output_dir,rows,args.seeds,args.recipe,args.include_canary,meta.get('plus_label','Metal'),args.takes)
                    print(json.dumps(dict(row=i,seed=seed,take=name,duration=stats['duration'],rms=stats['rms'])),flush=True)
    import shutil
    shutil.copy2(args.weights,args.output_dir/'metal-yue2.safetensors')
    shutil.copy2(args.weights.with_suffix('.json'),args.output_dir/'metal-yue2.json')
    write_json(args.output_dir/'status.json',dict(stage='Training and matched rendering complete'))
    page(args.output_dir,rows,args.seeds,args.recipe,args.include_canary,meta.get('plus_label','Metal'),args.takes)

if __name__=='__main__':main()
