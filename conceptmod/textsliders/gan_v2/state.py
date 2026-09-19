"""Complete, versioned research checkpoints with fixed schedule semantics."""
from __future__ import annotations

from dataclasses import asdict
import importlib.metadata
from pathlib import Path
import random

import torch

from .data import sha, digest


def cpu(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {key:cpu(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)): return type(value)(cpu(item) for item in value)
    return value


def code_fingerprints():
    folder=Path(__file__).parent
    paths=list(folder.glob('*.py')) + [folder.parent/name for name in (
        'train_lm_slider_music3.py','lm_adv.py','lm_gan.py','lora.py','lm_gan_state.py')]
    paths.append(folder.parents[2]/'analysis/gan_bcap/parameter_step_limit.py')
    return {str(path.resolve()):sha(path) for path in sorted(paths)}


def signature(recipe,critic_config,fixtures,*,rank,alpha,ema_config,model_identity,source=None):
    return dict(version='music-gan-v2-1',recipe=asdict(recipe),critic=critic_config,
        fixtures=fixtures,rank=rank,alpha=alpha,ema=ema_config,model_identity=model_identity,
        source=source,sources=code_fingerprints(),
        packages={name:importlib.metadata.version(name) for name in ('torch','transformers','diffusers')})


class RowSampler:
    def __init__(self,count,batch,seed=7):
        if not 0<batch<=count: raise ValueError('Invalid minibatch size')
        self.count,self.batch=count,batch
        self.generator=torch.Generator().manual_seed(seed)
        self.order=[];self.cursor=0

    def next(self):
        if self.count==self.batch:return list(range(self.count))
        if self.cursor>=len(self.order):
            self.order=torch.randperm(self.count,generator=self.generator).tolist();self.cursor=0
        end=min(self.cursor+self.batch,len(self.order))
        result=self.order[self.cursor:end];self.cursor=end
        return result

    def state_dict(self):
        return dict(count=self.count,batch=self.batch,order=self.order,cursor=self.cursor,rng=self.generator.get_state())

    def load_state_dict(self,state):
        if (self.count,self.batch)!=(state['count'],state['batch']):raise ValueError('Sampler geometry changed')
        if not 0<=state['cursor']<=len(state['order']):raise ValueError('Invalid sampler cursor')
        if state['order'] and sorted(state['order'])!=list(range(self.count)):raise ValueError('Invalid sampler permutation')
        self.order,self.cursor=state['order'],state['cursor'];self.generator.set_state(state['rng'])


def save(path,engine,sampler,ema,run_signature,history):
    path=Path(path)
    blob=dict(schema='music-gan-v2-1',signature=run_signature,completed=engine.completed,
        network=cpu(engine.network.state_dict()),critic=cpu(engine.critic.state_dict()),
        g_optimizer=cpu(engine.g_optimizer.state_dict()),d_optimizer=cpu(engine.d_optimizer.state_dict()),
        sampler=sampler.state_dict(),ema=None if ema is None else ema.state_dict(),history=history,
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(blob,temporary);temporary.replace(path)


def restore(path,engine,sampler,ema,run_signature):
    blob=torch.load(path,map_location='cpu',weights_only=True)
    if blob.get('schema')!='music-gan-v2-1' or blob['signature']!=run_signature:
        raise ValueError('Resume configuration, fixtures, model, or source code changed')
    if bool(blob['ema'] is not None)!=bool(ema is not None):raise ValueError('EMA topology changed')
    if blob['history'] and blob['history'][-1]['step']!=blob['completed']:raise ValueError('History/state mismatch')
    engine.network.load_state_dict(blob['network'],strict=True)
    engine.critic.load_state_dict(blob['critic'],strict=True)
    engine.g_optimizer.load_state_dict(blob['g_optimizer'])
    engine.d_optimizer.load_state_dict(blob['d_optimizer'])
    engine.completed=blob['completed'];sampler.load_state_dict(blob['sampler'])
    if ema is not None:ema.load_state_dict(blob['ema'])
    random.setstate(blob['python_rng']);torch.set_rng_state(blob['torch_rng'])
    if blob['cuda_rng']:
        if len(blob['cuda_rng'])!=torch.cuda.device_count():raise ValueError('Visible GPU topology changed')
        torch.cuda.set_rng_state_all(blob['cuda_rng'])
    return blob['history']


def initialize_from_legacy(path,engine,*,reuse_critic):
    """Explicit migration, never masquerading as an unchanged old experiment."""
    blob=torch.load(path,map_location='cpu',weights_only=True)
    if blob.get('schema')!=1:raise ValueError('Not a legacy full GAN state')
    for name,expected in blob['signature']['sources'].items():
        if sha(Path(__file__).parent.parent/name)!=expected:raise ValueError(f'Legacy source changed: {name}')
    engine.network.load_state_dict(blob['modules']['lora'],strict=True)
    engine.g_optimizer.load_state_dict(blob['optimizers']['lora'])
    if reuse_critic:
        engine.critic.load_state_dict(blob['modules']['critic'],strict=True)
        engine.d_optimizer.load_state_dict(blob['optimizers']['critic'])
    engine.completed=blob['completed_updates']
    # Restore source RNG only after all newly initialized components exist.
    random.setstate(blob['python_rng']);torch.set_rng_state(blob['torch_rng'])
    if blob['cuda_rng'] and torch.cuda.is_available():
        if len(blob['cuda_rng'])!=torch.cuda.device_count():raise ValueError('Visible GPU topology changed')
        torch.cuda.set_rng_state_all(blob['cuda_rng'])
    return dict(path=str(Path(path).resolve()),sha256=sha(path),completed=engine.completed,
                critic_and_optimizer='restored' if reuse_critic else 'new topology, initialized explicitly',
                generator_optimizer='restored',legacy_signature_digest=digest(blob['signature']))
