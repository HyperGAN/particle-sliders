"""Fresh repaired-smoke GAN training with the measured parameter-step bound.

Research wrapper: the existing trainer and original continuation drivers stay
unchanged. Full states reject a different wrapper or bound on continuation.
"""
from contextlib import contextmanager, nullcontext
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.optim.adamw import AdamW

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from conceptmod.textsliders import train_lm_slider_music3 as trainer
from conceptmod.textsliders import lm_gan_state as game
from conceptmod.textsliders import lora as lora_module
from analysis.gan_bcap.parameter_step_limit import bound_step
from analysis.gan_bcap.parameter_step_limit import step_limited_hooks
from analysis.gan_bcap.lyric_preservation_experiment import lyric_hooks


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def bounded_training(maximum, telemetry_path):
    original_network, original_step=lora_module.LoRANetwork, AdamW.step
    original_signature, original_restore=game.signature, game.restore
    network=None; completed=0
    telemetry=Path(telemetry_path)
    if telemetry.exists():raise ValueError('Use a fresh run directory')
    telemetry.parent.mkdir(parents=True,exist_ok=True)
    config=dict(maximum=maximum, implementation_sha256=sha(__file__),
                bound_function_sha256=sha(ROOT/'analysis/gan_bcap/parameter_step_limit.py'),
                scope='Global actual LoRA parameter displacement after AdamW; moments retained; float rounding allowed.')

    def make_network(*args,**kwargs):
        nonlocal network
        network=original_network(*args,**kwargs)
        return network

    def signature(*args,**kwargs):
        result=original_signature(*args,**kwargs)
        result['catalog_step_limit']=config
        return result

    def restore(*args,**kwargs):
        nonlocal completed
        history=original_restore(*args,**kwargs)
        completed=len(history)
        return history

    def step(optimizer,*args,**kwargs):
        nonlocal completed
        params=[p for g in optimizer.param_groups for p in g['params']]
        target=[] if network is None else [p for p in network.parameters() if p.requires_grad]
        if not target or {id(p) for p in target}!={id(p) for p in params}:
            return original_step(optimizer,*args,**kwargs)
        before=[p.detach().clone() for p in params]
        result=original_step(optimizer,*args,**kwargs)
        record=bound_step(params,before,maximum)
        completed+=1
        with telemetry.open('a') as h:
            h.write(json.dumps(dict(step=completed,**record),allow_nan=False)+'\n')
        return result

    lora_module.LoRANetwork,AdamW.step=make_network,step
    game.signature,game.restore=signature,restore
    try:yield config
    finally:
        lora_module.LoRANetwork,AdamW.step=original_network,original_step
        game.signature,game.restore=original_signature,original_restore


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--steps',type=int,required=True)
    p.add_argument('--maximum',type=float,default=2.)
    p.add_argument('--lyric-hold-weight',type=float,default=0.)
    p.add_argument('--save-every',type=int,default=150)
    p.add_argument('--resume-state',type=Path)
    p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    if args.steps<1 or args.maximum<0 or args.lyric_hold_weight<0: p.error('Invalid budget or bound')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.9,0)
    run=ROOT/'models/gan-autonomous-20260905'/args.name
    run.mkdir(parents=True,exist_ok=True)
    if (run/f'{args.name}_train.jsonl').exists():raise ValueError('Use a fresh run name')
    train=trainer.parse_args([
        '--name',args.name,'--prompts_file',str(args.prompts.resolve()),'--save_dir',str(run),
        '--lm_target','faithful_plus_neu_lyric','--pole_mode','hidden',
        '--rank','8','--alpha','8','--lr','5e-4','--steps',str(args.steps),'--seed',str(args.seed),
        '--no-early_stop','--endreg_weight','1','--save_every',str(args.save_every),'--device','0',
        '--adv_arch','tx','--adv_in','scaled','--adv_readout','mean_last','--adv_condition','none',
        '--adv_weight','1','--fm_weight','1','--fm_mode','batch','--pole_weight','0','--lyrichold_weight','0',
        '--adv_reg_coeff','1','--adv_reg_kappa','1','--adv_batch','4','--gan_beta1','0',
        '--gan_lr_schedule','constant','--grad_account','--parts','0','--save_training_state',
    ])
    train.lyrichold_weight=args.lyric_hold_weight
    if args.resume_state:
        train.resume_state=str(args.resume_state.resolve())
        stored=torch.load(args.resume_state,map_location='cpu',weights_only=True)
        baseline=stored['signature'].get('stability_experiment',{}).get('baseline_signature',stored['signature'])
        # Preserve the warm-up card. All intended changes are explicit below.
        vars(train).update(baseline['settings'])
        train.lyrichold_weight=args.lyric_hold_weight
        kwargs=dict(baseline_signature=baseline,weight=args.lyric_hold_weight,
                    telemetry_path=run/f'{args.name}_telemetry.jsonl')
        hooks=(step_limited_hooks(maximum=args.maximum,**kwargs) if args.maximum else lyric_hooks(**kwargs))
        config=hooks.config
        context=hooks.installed()
    elif args.maximum:
        context=bounded_training(args.maximum,run/f'{args.name}_step_limit.jsonl')
        config=None
    else:
        config=dict(maximum=None,enabled=False)
        context=nullcontext()
    with context as fresh_config:
        if config is None:config=fresh_config
        (run/'experiment.json').write_text(json.dumps(dict(status='training',arguments=vars(args),
            step_limit=config,physical_gpu=1),default=str,indent=2)+'\n')
        weights=trainer.train(train)
    meta=json.loads(weights.with_suffix('.json').read_text())
    meta['catalog_step_limit']=config
    weights.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    (run/'experiment.json').write_text(json.dumps(dict(status='complete',arguments=vars(args),
        step_limit=config,physical_gpu=1,weights=str(weights)),default=str,indent=2)+'\n')


if __name__=='__main__':main()
