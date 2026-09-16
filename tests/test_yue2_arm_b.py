"""Numeric Arm B parity, native polarity gradients, and exact recovery."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
import torch
from torch import nn
import torch.nn.functional as F
import yaml

pytest.importorskip('yue2')
from conceptmod.textsliders import yue2_arm_b as arm
from conceptmod.textsliders.lm_adv import LMDiscriminator
from conceptmod.textsliders.slider_targets import lm_slider_loss, lm_faithful_guard_e
from conceptmod.textsliders.train_lm_slider_music3 import ARM_B
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args,train
from conceptmod.textsliders.yue2_backend import YuE2Backend,YuE2Slider
from conceptmod.textsliders.train_lora_yue2_fresh import fresh_history

@pytest.fixture(autouse=True)
def single_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

class Network(nn.Module):
    def __init__(self):
        super().__init__();self.w=nn.Parameter(torch.randn(3,3)*.1);self.scale=0.
    @contextmanager
    def scaled(self,s):
        prev=self.scale;self.scale=s
        try:yield
        finally:self.scale=prev

class Backend:
    def __init__(self,network):
        self.network=network;self.model=None
    def hidden(self,ids,checkpointing=False):
        x=torch.tensor(ids,dtype=torch.float32).reshape(1,-1,3)
        return x + self.network.scale*(x@self.network.w)

def optimizers(network,critic):
    return (torch.optim.AdamW(network.parameters(),lr=.0005,betas=(0.,.999),weight_decay=1e-6),
            torch.optim.Adam(critic.parameters(),lr=.00075,betas=(0.,.999)))

def reference_update(backend,network,critic,g,d,rows):
    """Independent equations, full batch backward, explicit calibrated b_cap."""
    def outputs(row,scale):
        with network.scaled(scale):
            h=backend.hidden(row['ids']);i=row['prefix_len']-1
            return h[:,i],h[:,i:].mean(-1)
    real=[];fake=[]
    with torch.no_grad():
        for row in rows:
            for i,s in enumerate([1.,-1.]):
                real.append(row['targets'][i:i+1]-row['neutral'])
                fake.append(outputs(row,s)[0]-row['neutral'])
    real=torch.cat(real);fake=torch.cat(fake)
    critic.requires_grad_(True);d.zero_grad(set_to_none=True)
    penalties=[]
    for x in [real,fake]:
        z=(x/critic.input_scale).detach().clone().requires_grad_(True)
        grad=torch.autograd.grad(critic.net(z).sum(),z,create_graph=True)[0]
        penalties.append(F.relu(torch.sqrt(grad.square().sum(-1)+1e-12)-1).square().mean())
    loss_d=F.softplus(critic(fake)-critic(real)).mean()+.5*sum(penalties)
    loss_d.backward();d.step();d.zero_grad(set_to_none=True);critic.requires_grad_(False)
    g.zero_grad(set_to_none=True);terms=[]
    for row in rows:
        pos,ep=outputs(row,1.);neg,en=outputs(row,-1.)
        rp=critic(row['targets'][0:1]-row['neutral']).detach()
        rn=critic(row['targets'][1:2]-row['neutral']).detach()
        adv=.5*(F.softplus(rp-critic(pos-row['neutral'])).mean()+F.softplus(rn-critic(neg-row['neutral'])).mean())
        pole=lm_slider_loss(pos,neg,row['targets'][0:1],row['targets'][1:2])
        end=.5*(F.mse_loss(ep,row['end_teacher'])+F.mse_loss(en,row['end_teacher']))
        terms.append(adv+pole+end)
    total=sum(terms)/len(rows);total.backward()
    grads=[p.grad.detach().clone() for p in network.parameters()]
    torch.nn.utils.clip_grad_value_(network.parameters(),1.);g.step()
    return float(total.detach()),float(loss_d.detach()),grads

@pytest.mark.parametrize('critic_gain',[.1,2.])
def test_complete_update_matches_independent_equations(monkeypatch,critic_gain):
    torch.manual_seed(101)
    a=Network();b=deepcopy(a)
    ca=LMDiscriminator(3,hidden_dim=8,in_mode='scaled',input_scale=2.7)
    with torch.no_grad():
        for p in ca.parameters():p.fill_(critic_gain)
    cb=deepcopy(ca);ga,da=optimizers(a,ca);gb,db=optimizers(b,cb)
    rows=[]
    for i in range(4):
        ids=[float(i+1),.5,1.,.4,.2,.3]
        base=torch.tensor(ids[:3]).reshape(1,3)
        rows.append(dict(ids=ids,prefix_len=1,neutral=base,
            targets=torch.cat([base+torch.tensor([[.2,.4,-.1]]),base+torch.tensor([[-.3,-.1,.2]])]),
            end_teacher=torch.tensor([[.2,.1]])))
    monkeypatch.setattr(arm,'end_margins',lambda model,h:h.mean(-1))
    for step in [1,2]:
        expected,expected_d,grads=reference_update(Backend(b),b,cb,gb,db,rows)
        actual=arm.update(Backend(a),a,ca,ga,da,rows,step=step,checkpointing=False)
        assert actual['loss']==pytest.approx(expected,rel=2e-6,abs=1e-6)
        assert actual['d_loss']==pytest.approx(expected_d,rel=2e-6,abs=1e-6)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
        for x,y in zip(ca.parameters(),cb.parameters()):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
        for x,y in zip(a.parameters(),grads):torch.testing.assert_close(x.grad,y.clamp(-1,1),rtol=1e-5,atol=1e-6)
        assert all(p.grad is None for p in ca.parameters())
        assert a.scale==0


def rows():
    return [dict(neutral=f'Guitar band {i}',positive=f'Heavy metal guitar band {i}',
                 negative=f'Clean picked guitar band {i}',lyrics=f'[verse]\nWe carry crate {i}') for i in range(4)]


def test_locked_recipe_and_exact_regularizer():
    for k,v in ARM_B.items():assert arm.RECIPE[k]==v
    reg=arm.make_regularizer()
    assert (reg.arm,reg.coeff,reg.kappa,reg.norm,reg.method,reg.lazy_k,reg.target_anneal)==('b_cap',1.,1.,'l2','autograd',1,'none')
    assert arm.RECIPE['adv_batch']==4
    assert arm.RECIPE['phase_changes']==[]
    assert arm.RECIPE['cover_mapping']=='pole_loss_once'


def test_native_targets_are_shared_guard_targets():
    torch.manual_seed(18);backend=YuE2Backend(dummy=True)
    data=rows();meta=dict(slider_positive='Metal band',slider_negative='Clean band',
        leak_positive='Male singer',leak_negative='Female singer')
    fixed=arm.prepare(backend,data,meta,4,512)
    def enc(style,lyrics):return backend.hidden(backend.prefix(style,lyrics))[:,-1].float()
    with torch.no_grad():
        axis=enc(meta['slider_positive'],data[0]['lyrics'])-enc(meta['slider_negative'],data[0]['lyrics'])
        leak=enc(meta['leak_positive'],data[0]['lyrics'])-enc(meta['leak_negative'],data[0]['lyrics'])
        for row,item in zip(data,fixed):
            p=enc(row['positive'],row['lyrics']);n=enc(row['negative'],row['lyrics']);z=enc(row['neutral'],row['lyrics'])
            a,b=lm_faithful_guard_e(p,n,z,leak,slider_dir=axis)
            assert torch.equal(item['targets'],torch.cat([a,b]))
            assert torch.equal(item['neutral'],z)


def test_native_checkpointed_polarities_match_eager():
    torch.manual_seed(19);a=YuE2Backend(dummy=True);fixed=arm.prepare(a,rows(),{},4,512)
    b=YuE2Backend(dummy=True);b.model.load_state_dict(a.model.state_dict())
    na=YuE2Slider(a.model,rank=8,alpha=8);nb=YuE2Slider(b.model,rank=8,alpha=8);nb.load_state_dict(na.state_dict())
    ca,ga,da=arm.build_game(a,na,fixed);cb,gb,db=arm.build_game(b,nb,fixed);cb.load_state_dict(ca.state_dict())
    current=[fresh_history(a,na,row,item,4,100+i,False)[0] for i,(row,item) in enumerate(zip(rows(),fixed))]
    before={k:v.clone() for k,v in a.model.state_dict().items()}
    eager=arm.update(a,na,ca,ga,da,current,step=1,checkpointing=False)
    checkpointed=arm.update(b,nb,cb,gb,db,current,step=1,checkpointing=True)
    for k,v in na.state_dict().items():torch.testing.assert_close(v,nb.state_dict()[k],rtol=1e-5,atol=1e-7)
    assert eager['loss']==pytest.approx(checkpointed['loss'],rel=1e-6)
    assert eager['grad_norm']>0 and eager['end']==0
    assert all(torch.equal(before[k],v) for k,v in a.model.state_dict().items())
    assert all(p.grad is None for p in a.model.parameters())
    with na.scaled(0):
        assert torch.equal(a.hidden(fixed[0]['prefix']),b.hidden(fixed[0]['prefix']))


def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def test_resume_exact_fresh_batches_and_reject_recipe_change(tmp_path):
    prompts=tmp_path/'prompts.yaml';prompts.write_text(yaml.safe_dump(dict(rows=rows())))
    common=['--dummy','--prompts_file',str(prompts),'--steps','2','--train_tokens','4','--history_backend','eager']
    full=tmp_path/'full';split=tmp_path/'split'
    train(parse_args(common+['--save_dir',str(full)]))
    train(parse_args(common+['--save_dir',str(split),'--until','1']))
    train(parse_args(common+['--save_dir',str(split)]))
    a=torch.load(full/'state.pt',weights_only=True);b=torch.load(split/'state.pt',weights_only=True)
    for key in ['network','critic','g_optimizer','d_optimizer','sampler','rng']:assert same(a[key],b[key]),key
    assert [x['seed'] for h in b['history'] for x in h['histories']]==list(range(4000001,4000009))
    assert all(sorted(h['rows'])==[0,1,2,3] for h in b['history'])
    assert len({x['tokens_sha256'] for h in b['history'] for x in h['histories']})==8
    # Reopening a completed state does not overwrite pinned milestones.
    before=(split/'state-step2.pt').stat().st_mtime_ns
    train(parse_args(common+['--save_dir',str(split)]))
    assert (split/'state-step2.pt').stat().st_mtime_ns==before
    with pytest.raises(ValueError,match='Resume'):
        train(parse_args(common+['--save_dir',str(split),'--seed','8']))


def test_prompt_loader_does_not_silently_drop_negative_or_axes(tmp_path):
    p=tmp_path/'p.yaml';bad=rows();bad[0].pop('negative');p.write_text(yaml.safe_dump(dict(rows=bad)))
    with pytest.raises(ValueError):arm.load_prompts(p)
    p.write_text(yaml.safe_dump(dict(rows=rows(),leak_positive='Male singer')))
    with pytest.raises(ValueError):arm.load_prompts(p)
