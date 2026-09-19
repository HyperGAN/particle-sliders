"""Exact saved-history replay retaining all eight codebooks, including checks.

This extends the frozen semantic replay without modifying that implementation.
No waveform is inverted or regenerated. Merged weights must match the capture.
"""
import torch


@torch.no_grad()
def recover(pipe,trajectory,frames=500):
    from diffusers.modular_pipelines.minimax_music3.encoders import (
        _AUDIO_CODE_OFFSET,_AUDIO_END_TOKEN_ID,_SEMANTIC_VOCAB_SIZE,
        _AR_CFG_SCALE,_AR_CFG_TOP_K,_sample_top_k,_generate_depth_codes,_embed_audio_frame)
    lm=pipe.language_model;device=next(lm.parameters()).device;feedback=trajectory['frame_embeds']
    if feedback.shape[0]!=2 or feedback.shape[1]<frames+1:
        raise ValueError('Need both CFG branches and feedback after every target')
    generator=torch.Generator(device);generator.set_state(trajectory['rng_before']['generator'])
    output=lm.model(inputs_embeds=trajectory['prompt_embeds'].to(device),use_cache=True)
    cache=output.past_key_values;hidden=output.last_hidden_state[:,-1]
    mask=torch.ones(lm.config.vocab_size,dtype=torch.bool,device=device)
    mask[_AUDIO_CODE_OFFSET:_AUDIO_CODE_OFFSET+_SEMANTIC_VOCAB_SIZE]=False;mask[_AUDIO_END_TOKEN_ID]=False
    retained=[]
    for index in range(frames+1):
        logits=lm.lm_head(hidden).float().masked_fill(mask,-float('inf'))
        conditional,unconditional=logits[:1],logits[1:2]
        guided=unconditional+(conditional-unconditional)*_AR_CFG_SCALE
        threshold=torch.topk(conditional,_AR_CFG_TOP_K,dim=-1).values[...,-1,None]
        guided=guided.masked_fill(conditional<threshold,-float('inf')).masked_fill(mask[None],-float('inf'))
        sampled=_sample_top_k(guided,generator)
        if int(sampled.item())==_AUDIO_END_TOKEN_ID:raise ValueError('Replay ended before the saved history')
        codes,_=_generate_depth_codes(pipe,hidden,(sampled-_AUDIO_CODE_OFFSET).repeat(2),generator)
        if codes.shape!=(2,8) or not torch.equal(codes[0],codes[1]):raise ValueError('Unexpected eight-code CFG alignment')
        rebuilt=_embed_audio_frame(pipe,codes)
        if not torch.equal(rebuilt.cpu(),feedback[:,index:index+1]):
            raise ValueError(f'Exact feedback mismatch at frame {index}; inferred targets rejected')
        retained.append(codes[0].cpu())
        if index<frames:
            output=lm.model(inputs_embeds=rebuilt,past_key_values=cache,use_cache=True)
            cache=output.past_key_values;hidden=output.last_hidden_state[:,-1]
    # Prefill's first sampled frame is warmup. Feedback k predicts code frame k+1.
    codes=torch.stack(retained)[1:]
    return dict(codes=codes,tokens=codes[:,0].clone(),prompt_embeds=trajectory['prompt_embeds'],
                frame_embeds=feedback[:,:frames].clone(),verified_feedback_frames=frames+1,
                cfg_branches=2,codebooks=8,warmup_excluded=True,exact_feedback_reconstruction=True)
