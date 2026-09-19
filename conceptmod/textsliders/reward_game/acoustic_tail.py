"""Research-only capture/replay of the final two flow steps in each audio chunk.

Earlier flow states and inter-chunk overlap are fixed reference values. This is
truncated differentiation, never claimed to be a full generation gradient.
No production generator or pipeline block is changed.
"""
from contextlib import AbstractContextManager
import torch


class TailCapture(AbstractContextManager):
    def __init__(self,pipe,steps=30,retained=2):
        self.pipe=pipe;self.steps=steps;self.retained=retained;self.handles=[];self.chunks=[];self.vocoded=0
        self.pending=None

    def condition(self,module,args):
        self.chunks.append(dict(calls=0,steps={},latent=None))

    def before(self,module,args,kwargs):
        chunk=self.chunks[-1];index=chunk['calls']//2;branch=chunk['calls']%2
        if index>=self.steps:raise ValueError('Unexpected number of flow/CFG calls')
        if index>=self.steps-self.retained:
            latent=kwargs['hidden_states'];condition=kwargs['encoder_hidden_states'];timestep=kwargs['timestep']
            saved=chunk['steps'].setdefault(index,dict(branches={}))
            saved['branches'][branch]=dict(latent=latent.detach().cpu().clone(),condition=condition.detach().cpu().clone(),
                                           timestep=timestep.detach().cpu().clone())
            self.pending=(chunk,index,branch)
        else:self.pending=None
        chunk['calls']+=1

    def after(self,module,args,kwargs,output):
        if self.pending:
            chunk,index,branch=self.pending
            chunk['steps'][index]['branches'][branch]['velocity']=output[0].detach().cpu().clone()
        self.pending=None

    def vocoder(self,module,args):
        self.chunks[self.vocoded]['latent']=args[0].detach().cpu().clone();self.vocoded+=1

    def __enter__(self):
        self.handles=[self.pipe.condition_encoder.register_forward_pre_hook(self.condition),
            self.pipe.transformer.register_forward_pre_hook(self.before,with_kwargs=True),
            self.pipe.transformer.register_forward_hook(self.after,with_kwargs=True),
            self.pipe.vocoder.register_forward_pre_hook(self.vocoder)]
        return self

    def __exit__(self,*exc):
        for handle in reversed(self.handles):handle.remove()
        self.handles=[]

    def result(self):
        if not self.chunks or self.vocoded!=len(self.chunks):raise ValueError('Incomplete acoustic capture')
        for chunk in self.chunks:
            if chunk['calls']!=self.steps*2 or chunk['latent'] is None:raise ValueError('Incomplete flow calls')
            for step in chunk['steps'].values():
                branches=step['branches']
                if set(branches)!={0,1} or not torch.equal(branches[0]['latent'],branches[1]['latent']):raise ValueError('Unexpected CFG layout')
                if not torch.equal(branches[0]['timestep'],branches[1]['timestep']):raise ValueError('CFG timestep mismatch')
                zero=[b for b,v in branches.items() if not bool(v['condition'].count_nonzero())]
                if len(zero)!=1:raise ValueError('Need one zero unconditional branch')
                step['unconditional']=zero[0];step['conditional']=1-zero[0]
        return dict(version=1,steps=self.steps,retained_steps=self.retained,chunks=self.chunks,
                    scheduler_config=dict(self.pipe.scheduler.config),guidance_config=dict(self.pipe.guider.config),
                    latent_hop_length=self.pipe.latent_hop_length,sampling_rate=self.pipe.sampling_rate,
                    gradient_scope='last two flow steps; reference earlier flow and chunk-overlap states detached')


def replay_latents(transformer,capture,*,device='cuda:0',record_checks=False,chunk_indices=None):
    """Ordinary scheduler and CFG, preserving cached overlap before each step."""
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines.minimax_music3.denoise import _OVERLAP_LATENT_LENGTH
    scheduler=FlowMatchEulerDiscreteScheduler.from_config(capture['scheduler_config'])
    guider=ClassifierFreeGuidance.from_config(capture['guidance_config'])
    chunks=[];checks=[]
    for chunk_index,chunk in enumerate(capture['chunks']):
        if chunk_indices is not None and chunk_index not in chunk_indices:continue
        scheduler.set_timesteps(sigmas=np.linspace(1.,1./capture['steps'],capture['steps']),device=device)
        first=capture['steps']-capture['retained_steps'];scheduler.set_begin_index(first)
        latent=chunk['steps'][first]['branches'][0]['latent'].to(device)
        # The actual carry is 172 frames, limited for a short final chunk.
        if chunk_index==0:overlap=0
        else:
            previous_length=capture['chunks'][chunk_index-1]['latent'].shape[-1]
            carry_start=max(0,previous_length-2*_OVERLAP_LATENT_LENGTH)
            carry_end=max(carry_start,previous_length-_OVERLAP_LATENT_LENGTH)
            overlap=min(carry_end-carry_start,latent.shape[-1])
        for index in range(first,capture['steps']):
            reference=chunk['steps'][index];cond=reference['branches'][reference['conditional']]['condition'].to(device)
            cached=reference['branches'][0]['latent'].to(device)
            if overlap:latent=torch.cat((cached[...,:overlap],latent[...,overlap:]),-1)
            t=scheduler.timesteps[index];timestep=t.expand(latent.shape[0]).to(latent.dtype)
            guider.set_state(step=index,num_inference_steps=capture['steps'],timestep=t)
            states=guider.prepare_inputs({'encoder_hidden_states':(cond,torch.zeros_like(cond))})
            for state in states:
                guider.prepare_models(transformer)
                state.noise_pred=transformer(hidden_states=latent,timestep=timestep,encoder_hidden_states=state.encoder_hidden_states,return_dict=False)[0]
                guider.cleanup_models(transformer)
            velocity=guider(states)[0]
            latent=scheduler.step(velocity,t,latent,return_dict=False)[0]
        if overlap:latent=torch.cat((chunk['latent'][...,:overlap].to(device),latent[...,overlap:]),-1)
        chunks.append(latent)
        if record_checks:
            original=chunk['latent'].to(device);difference=(latent.float()-original.float()).abs()
            checks.append(dict(chunk=chunk_index,exact=torch.equal(latent,original),max_absolute_difference=float(difference.max()),
                               relative_l2=float(difference.norm()/original.float().norm().clamp_min(1e-12))))
    return chunks,checks


def decode(vocoder,chunks,hop_length):
    return torch.cat([decode_piece(vocoder,latent,index,len(chunks),hop_length) for index,latent in enumerate(chunks)],-1)


def decode_piece(vocoder,latent,index,num_chunks,hop_length):
    from diffusers.modular_pipelines.minimax_music3.decoders import _CROP_LEFT_LATENT,_CROP_RIGHT_LATENT
    wave=vocoder(latent.to(vocoder.dtype));left=0 if index==0 else _CROP_LEFT_LATENT*hop_length
    right=0 if index==num_chunks-1 else _CROP_RIGHT_LATENT*hop_length
    return wave[...,left:wave.shape[-1]-right].float().clamp(-1.,1.)
