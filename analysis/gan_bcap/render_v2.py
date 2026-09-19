"""Matched live/EMA rendering across LoRA ranks, sharing identical controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.gan_bcap.lm_evaluate import checkpoint_metadata,topology
from conceptmod.textsliders.gan_v2.data import sha,validate_prompts


def detach_network(network):
    """Restore exactly the Linear methods replaced by this LoRA network."""
    for wrapper in network.unet_loras:
        module=wrapper.org_forward.__self__
        if module.forward!=wrapper.forward:raise RuntimeError('Another wrapper replaced an attached LoRA')
        module.forward=wrapper.org_forward


def render(weights,out,prompts,*,row=0,seeds=(7,23,101,303),duration=20.):
    import torch
    from safetensors.torch import load_file
    from conceptmod.textsliders import generate_listen as G
    from conceptmod.textsliders.lora import LoRANetwork
    torch.set_num_threads(4)
    weights=[Path(p).resolve() for p in weights];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if len({p.stem for p in weights})!=len(weights):raise ValueError('Checkpoint stems must be distinct')
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seeds')
    metadata=[checkpoint_metadata(p) for p in weights]
    example=G._load_prompt_row(prompts,row);validate_prompts([example])
    spec=dict(prompts=str(Path(prompts).resolve()),prompts_sha256=sha(prompts),row=row,seeds=list(seeds),duration=duration,
        scales=[0.,1.],seed_retries=0,renderer_sha256=sha(__file__),
        checkpoints=[dict(path=str(p),sha256=sha(p),steps=m[2]) for p,m in zip(weights,metadata)])
    manifest=out/'render_spec.json'
    if manifest.exists() and json.loads(manifest.read_text())!=spec:raise ValueError('Locked render specification changed')
    manifest.write_text(json.dumps(spec,indent=2)+'\n')
    pipe=G._load_pipeline(G.DEFAULT_MODEL_DIR,'cuda:0')
    network=None;shape=None;shared={}
    try:
        for path,(meta,meta_path,steps) in zip(weights,metadata):
            requested=topology(meta)
            if requested!=shape:
                if network is not None:
                    detach_network(network);del network
                network=LoRANetwork(pipe.language_model,multiplier=0.,**requested).to('cuda:0').eval()
                if not network.unet_loras:raise RuntimeError('No LoRA modules')
                shape=requested
            network.load_state_dict(load_file(str(path)),strict=True)
            for seed in seeds:
                folder=out/f'{path.stem}-s{seed}';folder.mkdir(exist_ok=True)
                opts=SimpleNamespace(name=path.stem,out_dir=str(folder),scales='0,1',
                    plus_label=meta['plus_label'],minus_label=meta['minus_label'],seed=seed,duration=duration,kind='lm')
                stats={}
                for dest,prompt,scale in G._jobs(opts,example):
                    scale=float(scale or 0.)
                    identity=(prompt,example['lyrics'],seed,scale,str(path) if scale else 'base')
                    if dest.exists():stats[dest.name]=G._inspect_wav(dest)
                    elif identity in shared:
                        shutil.copyfile(shared[identity],dest);stats[dest.name]=G._inspect_wav(dest)
                    else:
                        network.set_lora_slider(scale)
                        print(f'Rendering {path.stem}, seed {seed}, scale {scale}',flush=True)
                        with torch.inference_mode(),network:
                            audio=pipe(prompt=prompt,lyrics=example['lyrics'],audio_duration=duration,
                                generator=torch.Generator('cuda:0').manual_seed(seed),output='audios')[0]
                        stats[dest.name]=G._write_wav(dest,audio,int(pipe.sampling_rate),duration,
                            accept_short=True,accept_silent=True)
                    shared[identity]=dest
                G._write_readme(opts,path,example,[0.,1.],stats,meta['rank'],meta['alpha'],unit_scale=1.)
                (folder/'checkpoint.json').write_text(json.dumps(dict(weights=str(path),sha256=sha(path),
                    steps=steps,metadata_source=str(meta_path),seed=seed),indent=2)+'\n')
    finally:
        if network is not None:detach_network(network)
    return spec


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',type=Path,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prompts',type=Path,required=True)
    parser.add_argument('--row',type=int,default=0)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,23,101,303])
    parser.add_argument('--duration',type=float,default=20.)
    args=parser.parse_args()
    render(args.weights,args.out,args.prompts,row=args.row,seeds=args.seeds,duration=args.duration)
