"""Compare attached training arithmetic with the actual ordinary CPU merger."""
import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from .core import ROOT, read, write, sha, immutable


def audit(home, gpu=1):
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import load_file
    from ..reward_sliders.specs import MODEL
    from ..reward_sliders.data import generation_hidden
    from ..reward_preference.objective import guided_log_probs, reference_kl
    from ..reward_sliders.evaluate import component
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET, _SEMANTIC_VOCAB_SIZE, _AUDIO_END_TOKEN_ID
    sys.path.insert(0, str(ROOT.parent))
    from app import generator
    from app.lora_runtime import LoRANetwork
    from ..reward_sliders.train import audit_adapter
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise ValueError('Wrong GPU')
    folder = Path(home) / 'audit' / 'forward-v1'; folder.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'analysis/reward_preference_20260908'; m = read(source / 'manifest.json')
    pairs = read(source / 'preferences.json')
    # One saved take per voice, with at least one style-bearing family.
    families = {f['family']: f for f in m['families']}; selected = []; voices = set()
    for pair in pairs:
        voice = families[pair['family']]['voice']
        if voice not in voices:
            selected.append(pair); voices.add(voice)
    initial = m['preference']['initial']; path = initial['checkpoint']
    signature = dict(checkpoint_sha256=sha(path), multiplier=.5, physical_gpu=gpu,
                     selected=[p['chosen'] for p in selected], frames=500,
                     source_sha256=sha(__file__), tokens={p['chosen']: sha(source/'tokens'/f"{p['chosen']}.pt") for p in selected})
    immutable(folder / 'recipe.json', signature)
    torch.set_num_threads(4); torch.cuda.set_device(0); started = time.monotonic()
    lm = AutoModelForCausalLM.from_pretrained(str(MODEL/'language_model'), torch_dtype=torch.bfloat16,
                                            local_files_only=True).to('cuda:0').eval().requires_grad_(False)
    lm.config.use_cache=False; pipe=SimpleNamespace(language_model=lm)
    load_seconds=time.monotonic()-started; results=[]
    def policy(hidden, boundary):
        return guided_log_probs(hidden[:,boundary:],lm.lm_head.weight,offset=_AUDIO_CODE_OFFSET,
                                vocabulary=_SEMANTIC_VOCAB_SIZE,eos=_AUDIO_END_TOKEN_ID)
    try:
        for pair in selected:
            began=time.monotonic(); family=pair['family']; row=torch.load(source/'tokens'/f"{pair['chosen']}.pt",map_location='cpu',weights_only=True)
            embeds=torch.cat((row['prompt_embeds'],row['frame_embeds']),1).to('cuda:0'); boundary=row['prompt_embeds'].shape[1]
            styles=m['style_components'][family]
            generator._merge_sliders(pipe,'cuda:0',styles)
            with torch.no_grad(): off=generation_hidden(lm,embeds); offp=policy(off,boundary)
            network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=.5,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full').to('cuda:0')
            network.load_state_dict(load_file(path),strict=True); audit_adapter(network,lm)
            with torch.no_grad(): attached=generation_hidden(lm,embeds); ap=policy(attached,boundary)
            network.detach(); del network
            generator._merge_sliders(pipe,'cuda:0',styles+[component(path,.5)])
            with torch.no_grad(): merged=generation_hidden(lm,embeds); mp=policy(merged,boundary)
            a=attached[:,boundary:].float(); b=merged[:,boundary:].float(); base=off[:,boundary:].float()
            mismatch=float((a-b).square().mean().sqrt()); effect=float((b-base).square().mean().sqrt())
            result=dict(family=family,voice=families[family]['voice'],styles=len(styles),hidden_rmse=mismatch,
                        merged_effect_rmse=effect,mismatch_to_effect=mismatch/max(effect,1e-12),
                        kl_merged_to_attached=float(reference_kl(ap,mp)),kl_off_to_merged=float(reference_kl(mp,offp)),
                        top1_mismatch=float((ap.argmax(-1)!=mp.argmax(-1)).float().mean()),
                        top1_effect=float((offp.argmax(-1)!=mp.argmax(-1)).float().mean()),seconds=time.monotonic()-began)
            results.append(result);write(folder/'progress.json',dict(rows=results,load_seconds=load_seconds))
            print(result,flush=True)
        result=dict(rows=results,load_seconds=load_seconds,elapsed_seconds=time.monotonic()-started,new_clips=0,
                    interpretation='Numerical mismatch relative to the adapter effect; does not establish an audio failure cause.')
        write(folder/'result.json',result)
    finally:
        generator._merge_sliders(pipe,'cuda:0',[])
        exact=all(torch.equal(module.weight.detach().cpu(),base) for module,base in generator._merge_state('cuda:0').pristine.items())
        write(folder/'off-restoration.json',dict(exact=exact))
        if not exact: raise RuntimeError('Off restoration failed')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--home',required=True);p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args();audit(a.home,a.gpu)
