"""Real Turbo parity, rendering and 24GB training-memory gate; no qualification claims."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
import torch
from lumen_studio.contracts import atomic_json
from lumen_studio.dataset import compile_manifest
from lumen_studio.game import Game
from lumen_studio.particles import ParticleAdapter
from lumen_studio.backends.anima import TurboRuntime


def main():
    torch.set_num_threads(4)
    out=ROOT/'artifacts/anima/smoke'
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    r=TurboRuntime(ROOT/'artifacts/anima/model','cuda:0',checkpointing=True)
    report=dict(device=torch.cuda.get_device_name(),model=r.identity,load_seconds=time.monotonic()-start)
    row=compile_manifest('dev')['rows'][0]
    prompt=row['neutral']
    z=r.noise(29001,512,512)
    from diffusers import ClassifierFreeGuidance
    r.pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=1.))
    captured=[]
    def capture(module,args,kwargs,output):
        captured.append({k:v.detach().cpu() for k,v in kwargs.items() if torch.is_tensor(v)} | {'velocity':output[0].detach().float().cpu()})
    hook=r.transformer.register_forward_hook(capture,with_kwargs=True)
    with torch.no_grad():
        official=r.pipe(prompt=prompt,latents=z.clone(),height=512,width=512,num_inference_steps=10,max_sequence_length=512,output='latents')
    hook.remove()
    assert len(captured)==10, len(captured)
    errors=[]
    embedding=r.encode(prompt)
    schedule=r.scheduler(10)
    latent=z.clone()
    with torch.no_grad():
        for i,t in enumerate(schedule.timesteps):
            prediction=r.predict(latent,t,embedding)
            error=(prediction.cpu()-captured[i]['velocity']).abs().max().item()
            errors.append(error)
            if not torch.equal(embedding.cpu(),captured[i]['encoder_hidden_states']):
                raise AssertionError('Text-conditioning mismatch')
            latent=r.step(schedule,prediction,t,latent)
    if not torch.is_tensor(official):
        print('Official output type:',type(official),flush=True)
        official=official.get('latents') if hasattr(official,'get') else official.latents
    report.update(velocity_max_abs_by_timestep=errors,latent_max_abs=float((latent-official).abs().max()))
    assert max(errors)==0, errors
    assert torch.equal(latent,official)
    r.decode(latent).save(out/'neutral-512-10.png')
    report['renders']=[]
    for label,p,steps,size in [('positive',row['positive'],10,512),('neutral8',prompt,8,512),
        ('neutral12',prompt,12,512),('neutral768',prompt,10,768),
        ('bare','an adult man with curly black hair, wearing a gray coat in a stone hall',10,512)]:
        torch.cuda.reset_peak_memory_stats()
        start=time.monotonic()
        image=r.render(p,29001,width=size,height=size,steps=steps)
        image.save(out/f'{label}-{size}-{steps}.png')
        torch.cuda.synchronize()
        report['renders'].append(dict(label=label,steps=steps,size=size,seconds=time.monotonic()-start,
            peak_gib=torch.cuda.max_memory_allocated()/2**30))
        atomic_json(out/'report.json',report)
    # Full-dimensional gmix + all native branches + exact active lazy cap.
    # This is a memory/correctness fixture, never a training or quality pilot.
    emb_neu=r.encode(row['neutral']);emb_pos=r.encode(row['positive'])
    latent=r.noise(29001,512,512)
    t=r.scheduler(10).timesteps[4]
    with torch.no_grad(): target=r.predict(latent,t,emb_pos)
    adapter=ParticleAdapter(r.transformer).to(r.device)
    r.mixer.add('candlelit',adapter)
    game=Game(adapter,dict(scale=torch.ones(10,target.numel()),edit_rms=torch.ones(10)),count=16,seed=7)
    report['training_smoke']=[]
    def predict(indices):
        return torch.cat([(r.predict(latent,t,emb_neu)-target).flatten(1) for _ in indices])
    for step in range(1,5):
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        with r.mixer.scales({'candlelit':1.}): result=game.update(predict,[4]*16,step)
        torch.cuda.synchronize()
        result.update(seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        report['training_smoke'].append(result)
        atomic_json(out/'report.json',report)
    assert report['training_smoke'][-1]['particle_gan_grad_norm']>0
    assert all(not p.requires_grad and p.grad is None for p in r.transformer.parameters())
    report['passed']=True
    atomic_json(out/'report.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
