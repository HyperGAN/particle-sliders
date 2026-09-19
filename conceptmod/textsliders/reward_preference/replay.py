"""Recover semantic tokens from exact saved feedback and sampler state.

No waveform is inverted or regenerated. Every recovered frame must reconstruct
the saved feedback embedding bit for bit, using the original frozen decoder.
"""
import argparse
import os
from pathlib import Path
from types import SimpleNamespace
import torch

from ..reward_search.renderer import SearchRenderer
from ..reward_sliders.specs import MODEL,read_json,write_json,save_tensor,sha,digest
from ..reward_sliders.experiment import verify


@torch.no_grad()
def recover(pipe, trajectory, frames=500):
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET, _AUDIO_END_TOKEN_ID, _SEMANTIC_VOCAB_SIZE,
        _AR_CFG_SCALE, _AR_CFG_TOP_K, _sample_top_k, _generate_depth_codes, _embed_audio_frame)
    lm = pipe.language_model
    device = next(lm.parameters()).device
    feedback = trajectory['frame_embeds']
    if feedback.shape[0]!=2 or feedback.shape[1]<frames+1:
        raise ValueError('Need both CFG branches and a verified feedback frame for every target')
    generator = torch.Generator(device)
    generator.set_state(trajectory['rng_before']['generator'])
    prompt = trajectory['prompt_embeds'].to(device)
    output = lm.model(inputs_embeds=prompt,use_cache=True)
    cache = output.past_key_values
    hidden = output.last_hidden_state[:,-1]
    mask = torch.ones(lm.config.vocab_size,dtype=torch.bool,device=device)
    mask[_AUDIO_CODE_OFFSET:_AUDIO_CODE_OFFSET+_SEMANTIC_VOCAB_SIZE] = False
    mask[_AUDIO_END_TOKEN_ID] = False
    tokens=[]
    for index in range(frames+1):
        logits = lm.lm_head(hidden).float().masked_fill(mask,-float('inf'))
        conditional,unconditional = logits[:1],logits[1:2]
        guided = unconditional+(conditional-unconditional)*_AR_CFG_SCALE
        threshold = torch.topk(conditional,_AR_CFG_TOP_K,dim=-1).values[...,-1,None]
        guided = guided.masked_fill(conditional<threshold,-float('inf')).masked_fill(mask[None],-float('inf'))
        sampled = _sample_top_k(guided,generator)
        if int(sampled.item())==_AUDIO_END_TOKEN_ID:
            raise ValueError('Replay ended before the saved feedback history')
        code = sampled-_AUDIO_CODE_OFFSET
        codes,_ = _generate_depth_codes(pipe,hidden,code.repeat(2),generator)
        rebuilt = _embed_audio_frame(pipe,codes)
        if not torch.equal(rebuilt.cpu(),feedback[:,index:index+1]):
            raise ValueError(f'Saved feedback does not exactly reproduce at frame {index}; no inferred tokens accepted')
        tokens.append(code.cpu())
        if index<frames:
            output = lm.model(inputs_embeds=rebuilt,past_key_values=cache,use_cache=True)
            cache = output.past_key_values
            hidden = output.last_hidden_state[:,-1]
    # Prefill predicts the discarded warmup (token zero). Feedback k predicts k+1.
    return dict(tokens=torch.cat(tokens)[1:],prompt_embeds=trajectory['prompt_embeds'],
                frame_embeds=feedback[:,:frames].clone(),verified_feedback_frames=frames+1,
                cfg_branches=2,warmup_excluded=True,exact_feedback_reconstruction=True)


def run(folder,gpu):
    folder=Path(folder);m=read_json(folder/'manifest.json');verify(m)
    from app import generator
    from diffusers import ModularPipeline
    from ..reward_search.resources import studio
    if gpu not in (0,1) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Wrong physical GPU')
    if gpu==0 and studio()['loaded']:raise ValueError('Studio still owns GPU 0')
    torch.set_num_threads(4);torch.cuda.set_device(0)
    # Exact replay needs only the composer and residual-code decoder. Loading
    # nine GB of acoustic-transformer files does not contribute to these targets.
    pipe=ModularPipeline.from_pretrained(str(MODEL),local_files_only=True)
    for name in ('language_model','rvq_depth_decoder'):
        pipe.load_components(names=name,pretrained_model_name_or_path=str(MODEL),local_files_only=True,dtype=torch.bfloat16)
    pipe.to('cuda:0')
    if len(pipe.language_model.model.layers)!=36 or pipe.language_model.config.hidden_size!=4096:raise ValueError('Wrong composer')
    renderer=SimpleNamespace(pipe=pipe,host=generator,device='cuda:0')
    cases=read_json(folder/f'replay-gpu{gpu}.json')
    try:
        for case in cases:
            source=case['observation'];path=folder/'tokens'/f"{source['id']}.pt"
            signature=dict(trajectory_sha256=source['trajectory_sha256'],audio_sha256=source['audio_sha256'],
                           manifest_sha256=digest(m),frames=m['preference']['frames'])
            if path.exists():
                assert torch.load(path,map_location='cpu',weights_only=True)['signature']==signature
                continue
            assert sha(source['trajectory'])==source['trajectory_sha256']
            assert sha(source['audio'])==source['audio_sha256']
            renderer.host._merge_sliders(renderer.pipe,renderer.device,m['style_components'][source['family']])
            trajectory=torch.load(source['trajectory'],map_location='cpu',weights_only=True)
            recovered=recover(renderer.pipe,trajectory,m['preference']['frames'])
            save_tensor(path,dict(signature=signature,**recovered))
            write_json(path.with_suffix('.json'),dict(passed=True,source=source['id'],physical_gpu=gpu,
                token_file_sha256=sha(path),frames=len(recovered['tokens']),verified_feedback_frames=recovered['verified_feedback_frames'],
                exact_feedback_reconstruction=True,audio_regenerated=False))
            print('RECOVERED '+source['id']+'; 501 exact feedback frames; no new audio',flush=True)
    finally:
        renderer.host._merge_sliders(renderer.pipe,renderer.device,[])
        exact=all(torch.equal(module.weight.detach().cpu(),base)
                  for module,base in renderer.host._merge_state(renderer.device).pristine.items())
        write_json(folder/'audit'/f'replay-gpu{gpu}-off.json',dict(exact=exact))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int,required=True)
    args=p.parse_args();run(args.run_dir,args.gpu)
