from copy import deepcopy
from dataclasses import asdict

import pytest
import torch
from particlegan import get_recipe

from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun, load_export

ROWS=[dict(neutral='a quiet song',positive='an intense metal song'),
      dict(neutral='a calm melody',positive='an aggressive metal melody')]


def make(dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    backend=TinyLLMBackend(dummy=True)
    backend.model.to(dtype=dtype)
    return PG09TinyRun(backend,ROWS,total_steps=6,batch_size=4)


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):
            equal(x,y)
    else:
        assert a==b


def test_actual_neutral_callbacks_recipe_and_all_parameter_roles_update():
    run=make()
    expected=get_recipe('bispf_optimized',num_particles=128,z_dim=4,batch_size=4,total_steps=6)
    assert run.signature['recipe']==asdict(expected)
    base=deepcopy(run.backend.model.state_dict())
    before=deepcopy(run.network.state_dict())
    owned=[p for g in run.game.opt_g.param_groups for p in g['params']]
    assert len(owned)==len({id(p) for p in owned})
    assert sum(p is run.network.particles for p in owned)==1
    assert not ({id(p) for p in owned}&{id(p) for p in run.backend.model.parameters()})
    for _ in range(3):
        metrics=run.step()
        assert all(torch.isfinite(torch.tensor(metrics[k])) for k in ('d_loss','g_loss','d_pen'))
        assert {r['phase'] for r in run.last_forward_rows}=={'g','d'}
        for record in run.last_forward_rows:
            fixed=run.fixed[record['row']]
            assert record['input_ids']==fixed['neutral_ids']
            assert record['input_ids']!=fixed['positive_ids']
    changes={k:float((v-before[k]).square().sum()) for k,v in run.network.state_dict().items()}
    assert changes['particles']>0
    for role in ('lora_up','lora_down','bridge.router','bridge.net'):
        assert sum(delta for key,delta in changes.items() if role in key)>0,role
    for name,tensor in run.backend.model.state_dict().items():
        assert torch.equal(tensor,base[name])
    for i,row in enumerate(ROWS):
        assert torch.equal(run.sample(row['neutral'],0),run.fixed[i]['neutral'])
        assert not torch.equal(run.sample(row['neutral'],1),run.fixed[i]['neutral'])


def test_resume_all_state_and_sampling_are_exact_and_legacy_rejected():
    live=make()
    live.step()
    saved=live.state_dict()
    baseline=live.sample(ROWS[0]['neutral'],1)
    # Sampling does not consume the private game streams.
    equal(live.state_dict(),saved)
    expected=live.step()
    restored=make()
    restored.load_state_dict(saved)
    assert torch.equal(restored.sample(ROWS[0]['neutral'],1),baseline)
    observed=restored.step()
    assert observed==expected
    equal(restored.state_dict(),live.state_dict())
    before=restored.state_dict()
    with pytest.raises(ValueError,match='protocol'):
        restored.load_state_dict({'network':live.network.state_dict(),'g_optimizer':{}})
    equal(restored.state_dict(),before)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_export_reload_preserves_fp32_adapter_on_frozen_base(tmp_path,dtype):
    trained=make(dtype)
    trained.step()
    trained.step()
    path=tmp_path/'tiny-pg09.safetensors'
    trained.export(path)
    torch.manual_seed(91)
    backend=TinyLLMBackend(dummy=True)
    backend.model.to(dtype=dtype)
    loaded,metadata=load_export(backend,path)
    assert metadata['student_conditioning']=='neutral'
    assert all(p.dtype==torch.float32 for p in loaded.parameters())
    assert all(p.dtype==dtype for p in backend.model.parameters())
    ids=backend.encode(ROWS[0]['neutral']).ids
    for scale in (0.,.5,1.):
        with loaded.scaled(scale):
            actual=backend.hidden(ids)[:,-1].float().detach().cpu()
        assert torch.equal(actual,trained.sample(ROWS[0]['neutral'],scale))
    assert not torch.cuda.is_initialized()
