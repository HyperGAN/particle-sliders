"""Stage the exact final audit selections, their original samples and CLI conversions."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
CAMPAIGN = ROOT / 'analysis/uni16_fresh3400_20260912'
AUDIT = ROOT / 'analysis/uni16_release_audit_20260914'
AUDIO = ROOT / 'eval/listen/uni16-fresh3400-20260912'
PACKAGE = WORK / 'package'
CONVERTER = Path('/tmp/music3-hf-release-20260916/conceptmod')
VERSION = 'uni16-fresh-selected-v2'
REPO = 'ntc-ai/minimax-music3-concept-sliders'
sys.path.insert(0, str(ROOT.parent))
from app.rewriter import _artist_name_hit

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def names(value, location):
    if isinstance(value,dict):
        for k,v in value.items(): names(v,f'{location}.{k}')
    elif isinstance(value,list):
        for i,v in enumerate(value): names(v,f'{location}[{i}]')
    elif isinstance(value,str) and _artist_name_hit('',value):
        raise ValueError(f'Prohibited name in {location}; release stopped')
def copy(source, dest, expected=None):
    source=Path(source);digest=sha(source)
    if expected and digest != expected: raise ValueError(f'Source changed: {source.name}')
    target=PACKAGE/dest;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or sha(target)!=digest:shutil.copyfile(source,target)
    return dest

def build():
    torch.set_num_threads(4)
    summary=read(WORK/'audit-snapshot.json')
    assert summary['complete']==16 and summary['candidates_scored']==80
    policy=summary['selection_policy']
    assert policy['version']=='quality-later-v2' and policy['tolerance']==.2
    assert sha(AUDIT/'selection.py')==policy['source_sha256']
    assert sha(AUDIT/'selection-policy.json')==policy['policy_sha256']
    manifest=read(CAMPAIGN/'manifest.json')
    for name,expected in manifest['files'].items():
        assert sha(CAMPAIGN/name)==expected, name
    for name,expected in summary['measurement_protocol']['sources'].items():
        assert sha(name)==expected, Path(name).name
    assert sha(CAMPAIGN/'manifest.json')==summary['measurement_protocol']['training_manifest_sha256']
    # Re-run the decision layer against frozen worker measurements, then compare
    # both selected labels and hashes with the captured live page.
    sys.path.insert(0,str(AUDIT))
    from selection import select
    catalog={r['id']:r for r in read(CAMPAIGN/'catalog.json')['sliders']}
    concepts=[];pairs=[];audits=[];conversion_inputs=[];selection=[]
    for slider in summary['sliders']:
        sid=slider['id'];item=catalog[sid];q=slider['recommendation'];step=q['step']
        assert slider['all_budget_samples_scored'] and slider['training_status']=='complete'
        decision=select(read(AUDIT/'sliders'/f'{sid}.json')['candidates'],summary['measurement_protocol'],policy,
                        slider['waveform_audit']['flags'])
        assert decision['recommendation']['label']==q['label']
        assert decision['recommendation']['weights_sha256']==q['weights_sha256']
        selected=next(r for r in decision['ranking'] if r['label']==q['label'])
        assert selected['risk']==0 and selected['within_quality_tolerance']
        assert all(v<=.2+1e-12 for v in q['quality_shortfall'].values())
        source=Path(q['weights']);native=f'weights/{VERSION}/{sid}/{sid}_step{step}.safetensors'
        meta=read(source.with_suffix('.json'));names(meta,f'{sid} native sidecar')
        prompts={}
        for split in ('train','eval'):
            p=Path(item[f'{split}_prompts']);data=yaml.safe_load(p.read_text());names(data,f'{sid} {split} prompts')
            dest=f'prompts/{VERSION}/{p.name}';copy(p,dest);prompts[split]=dest
        with safe_open(source,framework='pt',device='cpu') as f:names(f.metadata() or {},f'{sid} tensor metadata')
        tensors=load_file(str(source));assert len(tensors)==432
        assert all(torch.isfinite(t).all() for t in tensors.values())
        assert {float(t) for k,t in tensors.items() if k.endswith('.alpha')}=={8.}
        state_path=source.parent/f'state-step{step}.pt';state_sha=sha(state_path)
        assert state_sha==selected['integrity']['full_state_sha256']
        state=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)
        assert state['completed']==step and state['signature']['slider']==sid
        assert tensors.keys()==state['network'].keys()
        assert all(torch.equal(t,state['network'][k]) for k,t in tensors.items())
        assert len(state['history'])==step-600
        copy(source,native,q['weights_sha256'])
        meta['prompts_file']=prompts['train'];meta['weights_sha256']=q['weights_sha256']
        meta['release']=VERSION;meta['selection_policy']='quality-later-v2'
        write((PACKAGE/native).with_suffix('.json'),meta)
        comfy=f'comfyui/{VERSION}/{sid}_step{step}_comfyui.safetensors'
        concepts.append(dict(id=sid,label=item['label'],description=item['description'],weights=native,
            sha256=q['weights_sha256'],comfyui_weights=comfy,kind='language_model',unit_scale=1,
            training_steps=step,training_budget=slider['target'],train_prompts=prompts['train'],
            eval_prompts=prompts['eval'],comparison_status='complete',comparison_pairs=4,
            enjoyment=selected['mean']['enjoyment'],production=selected['mean']['production'],
            selection_policy='quality-later-v2',quality_shortfall=q['quality_shortfall'],
            tolerance_sensitive=q['tolerance_sensitive']))
        conversion_inputs.append(str(PACKAGE/native))
        audits.append(dict(id=sid,step=step,weights_sha256=q['weights_sha256'],finite=True,
            tensors=len(tensors),export_matches_full_state=True,full_state_sha256=state_sha,
            fresh_updates=step-600,unique_seeds=len({r['history']['seed'] for r in state['history']}),
            row_counts=read(source.parent/f'audit-step{step}.json')['row_counts']))
        assert audits[-1]['fresh_updates']==audits[-1]['unique_seeds']
        fixture=yaml.safe_load(Path(item['eval_prompts']).read_text())['rows']
        for clip in sorted(selected['clips'],key=lambda c:(c['row'],c['seed'])):
            row,seed=clip['row'],clip['seed'];base=AUDIO/sid/f'row-{row}-seed-{seed}'
            pair=dict(id=sid,label=item['label'],checkpoint=q['label'],row=row,seed=seed,
                weights_sha256=q['weights_sha256'],caption=fixture[row]['neutral'],lyrics=fixture[row]['lyrics'],
                featured=row==2 and seed==1709)
            for arm,key,label in [('off','baseline','off'),('on','candidate',q['label']),('reference','positive_reference','caption')]:
                wav=base/f'{label}.wav';info=read(wav.with_suffix('.json'));assert sha(wav)==info['sha256']==clip[key]['sha256']
                spec=info['spec'];assert spec['seed']==seed and spec['campaign_sha256']==sha(CAMPAIGN/'manifest.json')
                assert spec['weights_sha256']==(q['weights_sha256'] if arm=='on' else None)
                assert spec['scale']==(1. if arm=='on' else 0.)
                expected_prompt=fixture[row]['positive' if arm=='reference' else 'neutral']
                digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
                assert spec['prompt_sha256']==digest(expected_prompt) and spec['lyrics_sha256']==digest(fixture[row]['lyrics'])
                sound,rate=sf.read(wav,dtype='float32',always_2d=True)
                assert len(sound)>rate and np.isfinite(sound).all()
                assert np.sqrt(np.mean(sound.astype(np.float64)**2))>1e-5
                prefix=f'samples/{VERSION}/{sid}/row{row}-seed{seed}-{arm}'
                dest=copy(wav,prefix+'.wav',info['sha256']);mp3=copy(wav.with_suffix('.mp3'),prefix+'.mp3')
                duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(PACKAGE/mp3)],text=True))
                assert abs(duration-len(sound)/rate)<.2
                pair[arm]=dict(wav=dest,mp3=mp3,sha256=info['sha256'],duration_s=len(sound)/rate,
                    measurements={k:clip[key][k] for k in ('concept','enjoyment','production','lyrics','rms','clipped_fraction','hf14k_fraction')})
            pairs.append(pair)
        selection.append(dict(id=sid,selected_checkpoint=q['label'],selected_sha256=q['weights_sha256'],
            quality_anchors=slider['quality_anchors'],quality_shortfall=q['quality_shortfall'],
            sensitivity=slider['sensitivity'],candidates=[{k:r.get(k) for k in
              ('label','step','weights_sha256','risk','flags','mean','mean_style_gain','quality_shortfall','within_quality_tolerance','worst_quality','prompt_quality')}
                for r in decision['ranking']]))
        print(f'Validated {sid} step {step}: state equality and four matched comparisons',flush=True)
        del state,tensors
    # Execute the upstream CLI on the staged native files. No hand-written
    # converter and no edits to source training exports.
    command=[sys.executable,str(CONVERTER/'scripts/convert_lora_comfyui.py'),*conversion_inputs,'--force']
    env=dict(os.environ,PYTHONPATH=str(CONVERTER),CUDA_VISIBLE_DEVICES='')
    with (WORK/'comfyui-conversion.log').open('w') as log:
        subprocess.run(command,cwd=CONVERTER,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    conversion=[]
    for item in concepts:
        native=PACKAGE/item['weights'];generated=native.with_name(native.stem+'_comfyui.safetensors')
        assert generated.exists()
        original=load_file(str(native));converted=load_file(str(generated));assert len(original)==len(converted)==432
        matched=0
        for k,t in original.items():
            match=re.fullmatch(r'lora_te-model-layers-(\d+)-self_attn-(q_proj|k_proj|v_proj|o_proj)\.(alpha|lora_down.weight|lora_up.weight)',k)
            assert match,k
            suffix={'alpha':'alpha','lora_down.weight':'lora_A.weight','lora_up.weight':'lora_B.weight'}[match[3]]
            key=f'text_encoders.model.layers.{match[1]}.self_attn.{match[2]}.{suffix}'
            expected=t if suffix=='alpha' else t.to(torch.bfloat16)
            assert torch.equal(converted[key],expected) and torch.isfinite(converted[key]).all(),key
            matched+=1
        destination=PACKAGE/item['comfyui_weights'];destination.parent.mkdir(parents=True,exist_ok=True)
        generated.replace(destination)
        side=read(Path(str(generated)+'.json'));Path(str(generated)+'.json').unlink()
        assert side['host']=='lm' and side['fused_qkv'] is False and side['n_keys']==432
        item['comfyui_sha256']=sha(destination)
        side.update(source_weights=item['weights'],source_weights_sha256=item['sha256'],
            weights_sha256=item['comfyui_sha256'],rank=8,alpha=8,steps=item['training_steps'],
            recommended_range=[0.,1.],converter_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CONVERTER,text=True).strip(),
            conversion_precision='LoRA factors cast from FP32 to BF16; alpha scalars preserved in FP32.',
            clip_strength=1.,model_strength=0.,requires='Music 3 text encoder with separate q_proj/k_proj/v_proj/o_proj projections')
        write(Path(str(destination)+'.json'),side)
        conversion.append(dict(id=item['id'],native_sha256=item['sha256'],comfyui_sha256=item['comfyui_sha256'],
            verified_tensors=matched,projections=144,all_factors_equal_to_native_bf16_cast=True))
    copy(CAMPAIGN/'runtime/conceptmod/textsliders/lora.py','source/lora.py')
    write(PACKAGE/'catalog.json',dict(version=VERSION,release_date='2026-09-16',sliders=concepts,
        selection_policy=f'evidence/{VERSION}/selection-policy.json',reward_slider_separate='reward/refined-block-step2/README.md'))
    write(PACKAGE/f'samples/{VERSION}/pairs.json',dict(selection='All four fixed cases per selected checkpoint; first row/seed featured consistently, no per-clip score selection.',pairs=pairs))
    write(PACKAGE/f'evidence/{VERSION}/integrity.json',dict(checkpoints=audits))
    # Retain only public method fields; machine paths and conversation text stay local.
    public_policy={k:policy[k] for k in ('version','quality_metrics','tolerance','sensitivity_tolerances','rule','interpretation','style_policy','lyrics_policy','technical_policy','coverage','legacy_policy','source_sha256','policy_sha256')}
    write(PACKAGE/f'evidence/{VERSION}/selection-policy.json',public_policy)
    write(PACKAGE/f'evidence/{VERSION}/selection.json',dict(candidates_scored=80,sliders=selection))
    training={k:manifest[k] for k in ('version','initialization','warmup','continuation','save','evaluation','health_stop')}
    training['approved_budgets']=read(CAMPAIGN/'budget.json')['targets']
    training['training_manifest_sha256']=sha(CAMPAIGN/'manifest.json')
    training['release_selection']='Selection performed after training using quality-later-v2. See selection-policy.json.'
    write(PACKAGE/f'evidence/{VERSION}/training.json',training)
    write(PACKAGE/f'evidence/{VERSION}/conversion.json',dict(
        repository='https://github.com/mikkel/conceptmod',revision=side['converter_revision'],
        command='python scripts/convert_lora_comfyui.py <16 selected native exports> --force',
        sources={p:sha(CONVERTER/p) for p in ('scripts/convert_lora_comfyui.py','conceptmod/convert.py','conceptmod/convert_klein.py')},
        precision='BF16 adapter factors, FP32 alpha scalars',checkpoints=conversion,
        scope='Every mapped factor checked against the source cast; see comfyui/README.md for loader compatibility.'))
    write(WORK/'asset-validation.json',dict(sliders=16,native_checkpoints=16,comfyui_checkpoints=16,
        pairs=len(pairs),original_wavs=len(pairs)*3,mp3_previews=len(pairs)*3,
        native_state_equality=True,comfyui_tensors_verified=16*432,selection_reproduced=True))
    assert len(pairs)==64
    print('Staged 16 selected native checkpoints, 16 CLI-converted ComfyUI checkpoints and 64 three-arm comparisons.',flush=True)

if __name__=='__main__':build()
