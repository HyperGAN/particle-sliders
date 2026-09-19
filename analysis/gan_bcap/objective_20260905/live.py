"""Research continuation with one conditional full-state energy objective.

This deliberately changes the objective and critic. It is not an exact resume
of the old game. Source LoRA and Adam moments are imported explicitly, all
targets are the positive-caption teacher, and rejected G proposals roll back.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from conceptmod.textsliders import train_lm_slider_music3 as legacy
from conceptmod.textsliders.lora import LoRANetwork
from conceptmod.textsliders.gan_v2.data import ROOT, prepare_rows, sha, gather, validate_prompts, check_disjoint
from conceptmod.textsliders.gan_v2.state import cpu, code_fingerprints
from .conditional import ConditionalMetric, conditional_energy, backtrack
from .distribution import paired_mmd


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def state_sequence(lm,row,device):
    prompt = row['prompt_embeds'].to(device)
    frames = row['frame_embeds'].to(device) if row['frame_embeds'] is not None else None
    embeds = prompt if frames is None else torch.cat([prompt,frames],1)
    h = lm.model(inputs_embeds=embeds,attention_mask=torch.ones(embeds.shape[:2],
        dtype=torch.long,device=device),use_cache=False).last_hidden_state.float()
    span = gather(h[:,:prompt.shape[1]],row['span_mask'])
    # The appended audio-start token is counted once, as the first continuation
    # state. Every later cached history position is included, not only EOS logits.
    return torch.cat([span[:,:-1],h[:,prompt.shape[1]-1:]],1)


def prepare(lm,tokenizer,rows,args,device):
    prepared = prepare_rows(lm,tokenizer,rows,model_dir=legacy.DEFAULT_MODEL,
        cache_dir=ROOT/'cache/endreg',device=device,frames=args.frames,seeds=(7,),policy_stride=1)
    with torch.no_grad():
        for index,row in enumerate(prepared):
            if args.cfg_branches:
                from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CFG_TOKEN_ID
                source=rows[index]
                nt,nm=legacy._tokenize(tokenizer,legacy._assemble(source.get('neutral') or source['target'],source['lyrics']),device)
                pt,pm=legacy._tokenize(tokenizer,legacy._assemble(source['positive'],source['lyrics']),device)
                ns,ps=legacy._assert_lyric_span(nt,nm,pt,pm,tokenizer,source['lyrics'],where=f'energy CFG row {index}')
                def pair(tokens):
                    unconditional=tokens.clone();unconditional[:,1:-2]=_AUDIO_CFG_TOKEN_ID
                    return lm.model.embed_tokens(torch.cat([tokens,unconditional],0))
                row['prompt_embeds']=pair(nt).cpu()
                row['span_mask']=ns.repeat(2,1).cpu()
                if row['frame_embeds'] is not None:row['frame_embeds']=row['frame_embeds'].repeat(2,1,1)
                teacher=dict(row,prompt_embeds=pair(pt).cpu(),span_mask=ps.repeat(2,1).cpu())
                positive=state_sequence(lm,teacher,device).cpu()
                neutral=state_sequence(lm,row,device).cpu()
                row['energy_neutral']=neutral
                row['energy_target']=positive-neutral
                continue
            neutral = state_sequence(lm,row,device).cpu()
            positive_span = row['real']+row['neutral_span']
            positive = torch.cat([positive_span[:,:-1],row['continuation_teacher']],1)
            if positive.shape!=neutral.shape: raise ValueError('Teacher history alignment failed')
            row['energy_neutral'] = neutral
            row['energy_target'] = positive-neutral
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-state',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--updates',type=int,default=120)
    p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--frames',type=int,default=250)
    p.add_argument('--save-every',type=int,default=30)
    p.add_argument('--prompts',type=Path,default=ROOT/'conceptmod/textsliders/data/prompts-gender-uni-v2.yaml')
    p.add_argument('--evaluation-prompts',type=Path,default=ROOT/'analysis/gan_bcap/v2_20260905/fixtures/evaluation.yaml')
    p.add_argument('--evaluation-rows',type=int,default=2)
    p.add_argument('--fresh-optimizer',action='store_true',help='Reset G moments for the changed objective; record explicitly')
    p.add_argument('--cfg-branches',action='store_true',help='Match both actual conditional/unconditional inference branches')
    p.add_argument('--fixed-kernel',action='store_true',help='Use one fixed multiscale RBF MMD; no discriminator')
    args = p.parse_args()
    if args.out.exists(): raise ValueError('Use a fresh research directory')
    if min(args.updates,args.save_every,args.evaluation_rows)<1:raise ValueError('Invalid run budget')
    args.out.mkdir(parents=True)
    # Capture before model I/O. A source edit during a long model load must
    # never be mislabeled as the code already imported by this process.
    sources = code_fingerprints()
    sources.update({str(path.resolve()):sha(path) for path in Path(__file__).parent.glob('*.py')})
    for filename in sources:
        path=Path(filename)
        dest=args.out/'provenance'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        if sha(dest)!=sources[filename]:raise RuntimeError('Source changed during capture')
    torch.set_num_threads(4); torch.manual_seed(7)
    device = torch.device('cuda:0')
    rows,metadata = legacy._load_rows(args.prompts)
    eval_rows,_ = legacy._load_rows(args.evaluation_prompts)
    eval_rows = eval_rows[:args.evaluation_rows]
    validate_prompts(rows);validate_prompts(eval_rows);check_disjoint(rows,eval_rows)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(str(legacy.DEFAULT_MODEL/'tokenizer'),local_files_only=True)
    lm = AutoModelForCausalLM.from_pretrained(str(legacy.DEFAULT_MODEL/'language_model'),
        torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache=False
    train = prepare(lm,tokenizer,rows,args,device)
    heldout = prepare(lm,tokenizer,eval_rows,args,device)
    scale = math.sqrt(sum(float(row['energy_target'].double().square().sum()) for row in train)
                      /sum(row['energy_target'].numel() for row in train))
    if scale<=0 or not math.isfinite(scale):raise ValueError('Degenerate target calibration')
    network = LoRANetwork(lm,multiplier=1.,rank=8,alpha=8,delimiter='-',
        target_replace=['Qwen3Attention'],prefix='lora_te',train_method='full').to(device)
    network.requires_grad_(True)
    source = torch.load(args.source_state,map_location='cpu',weights_only=True)
    network.load_state_dict(source['modules']['lora'],strict=True)
    parameters = list(network.parameters())
    optimizer = torch.optim.AdamW(parameters,lr=args.lr,betas=(0.,.999),weight_decay=0.)
    if not args.fresh_optimizer: optimizer.load_state_dict(source['optimizers']['lora'])
    for group in optimizer.param_groups:
        group.update(lr=args.lr,betas=(0.,.999),weight_decay=0.)
    metric = None if args.fixed_kernel else ConditionalMetric(train[0]['energy_target'].shape[-1]).to(device)
    dop = None if metric is None else torch.optim.Adam(metric.parameters(),lr=1.5*args.lr,betas=(0.,.999))
    origin = source['completed_updates']
    del source
    manifest = dict(arguments={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()},
        objective=('conditional multiscale RBF MMD in fixed RMS coordinates' if args.fixed_kernel else
            'conditional energy distance in the injective, spectrally constrained metric [raw RMS coordinates, scalar learned feature]'),
        source_sha256=sha(args.source_state),prompts_sha256=sha(args.prompts),
        evaluation_prompts_sha256=sha(args.evaluation_prompts),scale=scale,smoothing=None if args.fixed_kernel else .1,
        kernel_bandwidths=[.03,.1,.3,1.,3.] if args.fixed_kernel else None,
        source_step=origin,sources=sources,teacher_history='cached neutral-caption histories',
        branches=['conditional','unconditional'] if args.cfg_branches else ['conditional'],
        generator_optimizer=('fresh moments' if args.fresh_optimizer else 'imported source moments')+
            '; beta1 zero; no weight decay; fixed LR; objective-decrease line search',
        discriminator='none; fixed characteristic kernel' if args.fixed_kernel else 'new conditional MLP, exact spectral projection <=1 per linear layer',
        limitations=['fixed histories, not sampled student trajectories','two reserved prompts are diagnostic only',
                     'loss descent applies to the current full training batch and fixed critic','not catalog weights'])
    write_json(args.out/'manifest.json',manifest)
    def forward(row):
        legacy._set_scale(network,1.)
        return state_sequence(lm,row,device)-row['energy_neutral'].to(device)
    def loss_for(fake,row):
        if args.fixed_kernel:return paired_mmd(fake,row['energy_target'].to(device),scale=scale)
        return conditional_energy(metric,fake,row['energy_target'].to(device),
            row['energy_neutral'].to(device),scale=scale)
    @torch.no_grad()
    def batch_value():
        return sum(float(loss_for(forward(row),row)) for row in train)/len(train)
    @torch.no_grad()
    def evaluate(step):
        results = {}
        for label,group in [('train',train),('heldout',heldout)]:
            values=[]
            for row in group:
                fake=forward(row);real=row['energy_target'].to(device)
                error=(fake-real).double()
                span_length=row['real'].shape[1]
                values.append(dict(loss=float(loss_for(fake,row)),
                    relative_error=float(error.norm()/real.double().norm().clamp_min(1e-10)),
                    continuation_rmse=float(error[:,span_length-1:].square().mean().sqrt()),
                    magnitude=float(fake.double().norm()/real.double().norm().clamp_min(1e-10)),
                    branch_continuation_rmse=[float(e.square().mean().sqrt()) for e in error[:,span_length-1:]]))
            results[label]=values
        results['step']=step
        write_json(args.out/f'evaluation-{step}.json',results)
        print('EVALUATION',json.dumps(results),flush=True)
    def save(step,history):
        path=args.out/f'{args.out.name}_step{step}.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()},str(path))
        write_json(path.with_suffix('.json'),dict(steps=step,rank=8,alpha=8,kind='language_model',
            target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',unit_scale=1.,
            plus_label=metadata.get('plus_label','On'),minus_label=metadata.get('minus_label','Off'),
            prompts_file=str(args.prompts.resolve()),recommended_range=[0.,1.],
            research_objective=manifest['objective'],quality_status='unvalidated_research_candidate'))
        torch.save(dict(schema='conditional-energy-research-1',manifest=manifest,step=step,
            network=cpu(network.state_dict()),metric=None if metric is None else cpu(metric.state_dict()),
            optimizer=cpu(optimizer.state_dict()),d_optimizer=None if dop is None else cpu(dop.state_dict()),history=history,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),args.out/'state.pt')
    # Exact zero-scale identity in the same geometry, including continuations.
    legacy._set_scale(network,0.)
    with torch.no_grad():
        for row in train+heldout:
            if not torch.equal(state_sequence(lm,row,device).cpu(),row['energy_neutral']):
                raise RuntimeError('Zero-scale identity failed')
    evaluate(origin)
    save(origin,[])
    history=[]
    start=time.monotonic()
    with (args.out/'train.jsonl').open('w') as log:
        for update in range(1,args.updates+1):
            if metric is not None:
                metric.requires_grad_(True)
                dop.zero_grad(set_to_none=True)
                for row in train:
                    with torch.no_grad():fake=forward(row)
                    (-loss_for(fake,row)/len(train)).backward()
                if not all(torch.isfinite(q.grad).all() for q in metric.parameters() if q.grad is not None):
                    raise FloatingPointError('Nonfinite metric gradient')
                dop.step();metric.project();metric.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            initial_loss=0.
            for row in train:
                loss=loss_for(forward(row),row)/len(train)
                initial_loss+=float(loss.detach());loss.backward()
            gradient_norm=math.sqrt(sum(float(q.grad.double().square().sum()) for q in parameters if q.grad is not None))
            if not math.isfinite(gradient_norm):raise FloatingPointError('Nonfinite generator gradient')
            before=[q.detach().clone() for q in parameters]
            optimizer_before=copy.deepcopy(optimizer.state_dict())
            optimizer.step()
            proposed=[q.detach().clone() for q in parameters]
            result=backtrack(parameters,before,proposed,batch_value,initial_loss)
            if not result['accepted']:optimizer.load_state_dict(optimizer_before)
            parameter_step=math.sqrt(sum(float((q.detach().double()-old.double()).square().sum()) for q,old in zip(parameters,before)))
            record=dict(step=origin+update,loss_before=initial_loss,gradient_norm=gradient_norm,
                parameter_step=parameter_step,line_search=result,seconds=time.monotonic()-start)
            history.append(record);log.write(json.dumps(record,allow_nan=False)+'\n');log.flush()
            if update%10==0:print(json.dumps(record),flush=True)
            if update%args.save_every==0 or update==args.updates:
                save(origin+update,history);evaluate(origin+update)
    write_json(args.out/'completion.json',dict(status='budget_complete',step=origin+args.updates,
        accepted=sum(r['line_search']['accepted'] for r in history),seconds=time.monotonic()-start))


if __name__=='__main__':main()
