"""Zero-update audit of reference and actual checkpointed training numerics."""
import argparse
import os
from pathlib import Path
import signal
import time

from .core import DEFAULT_HOME,ROOT,read,write,immutable,sha,digest,locked
from .merged_forward import MergedForward


def audit(home,gpu):
    import torch
    from diffusers import ModularPipeline
    from safetensors.torch import load_file
    from ..reward_sliders.specs import MODEL
    from ..reward_sliders.data import generation_hidden
    from ..reward_preference.objective import guided_log_probs,reference_kl
    from diffusers.modular_pipelines.minimax_music3.encoders import _AUDIO_CODE_OFFSET,_AUDIO_END_TOKEN_ID,_SEMANTIC_VOCAB_SIZE
    from app import generator
    from app.lora_runtime import LoRANetwork
    from .resources import gpu_lease
    from .setup import verify
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong GPU')
    home=Path(home);folder=home/'audit/reference-mode-v2';source=ROOT/'analysis/reward_preference_20260908'
    training=home/'training/residual-imitation-v1';recipe=read(training/'recipe.json');manifest=read(source/'manifest.json')
    pairs=[next(p for p in read(source/'preferences.json') if p['family']==f) for f in ['search-train-03','search-train-01']]
    signature=dict(physical_gpu=gpu,optimizer_updates=0,initial=recipe['initial'],selected=[p['chosen'] for p in pairs],
        cached_reference_sha256=sha(training/'reference.pt'),training_recipe_sha256=digest(recipe),source_sha256=sha(__file__),
        question='Compare the exact advanced-index selection used by the trainer across modes, with a separate strided-view control.')
    immutable(folder/'recipe.json',signature)
    with locked(folder/'audit.lock'),gpu_lease(home,gpu):
        verify(home);torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(71);started=time.monotonic()
        pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
        pipe.load_components(names='language_model',pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
        pipe.to('cuda:0');lm=pipe.language_model;lm.eval().requires_grad_(False);lm.config.use_cache=False
        network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',attach=False).to('cuda:0').requires_grad_(True)
        weights=load_file(recipe['initial']['checkpoint']);weights={k:v*.5 if k.endswith('.lora_up.weight') else v for k,v in weights.items()}
        network.load_state_dict(weights,strict=True);wrapper=MergedForward(network);wrapper.attach();load_seconds=time.monotonic()-started
        reference=torch.load(training/'reference.pt',map_location='cpu',weights_only=True)['rows'];results=[]
        try:
            for pair in pairs:
                row=torch.load(training/'tokens'/f"{pair['chosen']}.pt",map_location='cpu',weights_only=True)
                nets=[(generator._slider_network(pipe,'cuda:0',c,attach=False),c['multiplier']) for c in manifest['style_components'][pair['family']]]
                wrapper.styles(nets);embeds=torch.cat((row['prompt_embeds'],row['frame_embeds']),1).to('cuda:0');boundary=row['prompt_embeds'].shape[1]
                def forward(layout_control=False):
                    hidden=generation_hidden(lm,embeds)
                    positions=torch.arange(0,500,16,device=hidden.device)
                    selected=hidden[:,boundary+positions]
                    policy=lambda h:guided_log_probs(h,lm.lm_head.weight,offset=_AUDIO_CODE_OFFSET,vocabulary=_SEMANTIC_VOCAB_SIZE,eos=_AUDIO_END_TOKEN_ID)
                    output=policy(selected)
                    if layout_control:
                        view=hidden[:,boundary:][:,::16]
                        return output,policy(view),dict(values_equal=torch.equal(selected,view),advanced_stride=list(selected.stride()),view_stride=list(view.stride()))
                    return output
                lm.gradient_checkpointing_disable();lm.eval()
                with torch.no_grad():
                    evaluation,view_control,layout=forward(True);evaluation=evaluation.cpu();view_control=view_control.cpu()
                lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});lm.train()
                with torch.no_grad():training_no_grad=forward().cpu()
                with torch.enable_grad():output=forward();training_grad=output.detach().cpu();del output
                cached=reference[pair['chosen']]['logp']
                result=dict(family=pair['family'],zero_updates=True,layout=layout,strided_view_control_kl=float(reference_kl(view_control,evaluation)),
                    cached_vs_recomputed_exact=torch.equal(cached,evaluation),cached_vs_recomputed_kl=float(reference_kl(evaluation,cached)),
                    train_mode_without_grad_kl=float(reference_kl(training_no_grad,evaluation)),
                    checkpointed_training_with_grad_kl=float(reference_kl(training_grad,evaluation)),
                    checkpointed_training_vs_cached_kl=float(reference_kl(training_grad,cached)),
                    grad_vs_no_grad_max_logprob_difference=float((training_grad-training_no_grad).abs().max()))
                results.append(result);write(folder/'progress.json',dict(rows=results));print(result,flush=True)
            unchanged=all(torch.equal(v.cpu(),weights[k]) for k,v in network.state_dict().items())
            write(folder/'result.json',dict(rows=results,adapter_unchanged=unchanged,load_seconds=load_seconds,
                elapsed_seconds=time.monotonic()-started,new_clips=0,optimizer_updates=0,
                interpretation='Numerical diagnostic only; no audio claim. Preserve both evaluated candidates regardless of the finding.'))
        finally:
            wrapper.detach();generator._merge_sliders(pipe,'cuda:0',[])
            exact=wrapper.base_unchanged();write(folder/'off-restoration.json',dict(exact=exact,physical_gpu=gpu))
            if not exact:raise RuntimeError('Base changed')


if __name__=='__main__':
    def cancel(signum,frame):raise KeyboardInterrupt(f'Cancellation signal {signum}')
    signal.signal(signal.SIGTERM,cancel)
    p=argparse.ArgumentParser();p.add_argument('--home',default=str(DEFAULT_HOME));p.add_argument('--gpu',type=int,default=1)
    a=p.parse_args();audit(a.home,a.gpu)
