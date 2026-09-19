from types import SimpleNamespace
import torch


def test_tail_capture_replays_actual_flow_blocks_and_short_overlap_exactly():
    from diffusers import MiniMaxMusic3Transformer1DModel,FlowMatchEulerDiscreteScheduler
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines.minimax_music3.denoise import MiniMaxMusic3ChunkDenoiseInner,MiniMaxMusic3ChunkUpdateStep,MiniMaxMusic3ChunkSetTimestepsStep
    from conceptmod.textsliders.reward_game.acoustic_tail import TailCapture,replay_latents
    torch.set_num_threads(2);torch.manual_seed(15)
    tf=MiniMaxMusic3Transformer1DModel(in_channels=4,condition_dim=8,num_layers=2,num_attention_heads=2,attention_head_dim=8,ff_inner_dim=32,rotary_dim=8).eval().requires_grad_(False)
    pipe=SimpleNamespace(transformer=tf,condition_encoder=torch.nn.Identity(),vocoder=torch.nn.Identity(),
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1,invert_sigmas=True),guider=ClassifierFreeGuidance(guidance_scale=1.7),
        _execution_device='cpu',latent_hop_length=512,sampling_rate=44100)
    state=SimpleNamespace(previous_latent=None,previous_condition=None,latent_chunks=[],num_inference_steps=4,progress_bar=SimpleNamespace(update=lambda:None))
    with TailCapture(pipe,steps=4,retained=2) as capture:
        for chunk in range(2):
            state.condition=pipe.condition_encoder(torch.randn(1,200,8));state.overlap=0
            if state.previous_latent is not None:
                state.overlap=min(state.previous_latent.shape[-1],200)
                state.condition[:,:state.overlap]=state.previous_condition[:,:state.overlap]
            state.latents=torch.randn(1,4,200)
            state.noise_prompt=state.latents[...,:state.overlap].clone() if state.overlap else None
            MiniMaxMusic3ChunkSetTimestepsStep()(pipe,state,chunk)
            MiniMaxMusic3ChunkDenoiseInner()(pipe,state,chunk)
            MiniMaxMusic3ChunkUpdateStep()(pipe,state,chunk)
        for latent in state.latent_chunks:pipe.vocoder(latent)
    result=capture.result()
    # 200 latent frames produce a 28-frame carry, so hardcoding 172 fails.
    assert state.previous_latent.shape[-1]==28
    replayed,checks=replay_latents(tf,result,device='cpu',record_checks=True)
    assert all(c['exact'] for c in checks)
    assert all(torch.equal(a,b) for a,b in zip(replayed,state.latent_chunks))
    one,_=replay_latents(tf,result,device='cpu',chunk_indices=(1,))
    assert len(one)==1 and torch.equal(one[0],state.latent_chunks[1])
