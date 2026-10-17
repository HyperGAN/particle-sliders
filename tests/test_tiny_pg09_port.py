from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from particlegan import get_recipe

from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun, load_export
from particle_sliders.pg09 import ConditionalPG09

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


def test_actual_neutral_callbacks_recipe_and_all_parameter_roles_update(monkeypatch):
    run=make()
    assert type(run.game) is ConditionalPG09
    assert run.game.backend is run.adapter
    assert run.game.prior.z is run.network.particles
    calls = []
    original = run.backend.hidden
    def hidden(ids):
        calls.append(tuple(ids))
        return original(ids)
    monkeypatch.setattr(run.backend, 'hidden', hidden)
    def forbidden(*a, **kw):
        raise AssertionError('MSE and local PairedPG09 are forbidden')
    from conceptmod.textsliders.pg09_game import PairedPG09
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    monkeypatch.setattr(PairedPG09, 'step', forbidden)
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
        assert set(calls) == {fixed.row.neutral_ids for fixed in run.fixed}
        assert not (set(calls) & {fixed.row.positive_ids for fixed in run.fixed})
        assert metrics['g_loss']==pytest.approx(metrics['g_adv']+metrics['prior_regularization'])
        assert metrics['d_loss']==pytest.approx(metrics['d_adv']+metrics['d_pen'])
        assert metrics['g_row_exposures']==metrics['step']*4
        assert metrics['d_row_exposures']==metrics['step']*4
    changes={k:float((v-before[k]).square().sum()) for k,v in run.network.state_dict().items()}
    assert changes['particles']>0
    for role in ('lora_up','lora_down','bridge.router','bridge.net'):
        assert sum(delta for key,delta in changes.items() if role in key)>0,role
    for name,tensor in run.backend.model.state_dict().items():
        assert torch.equal(tensor,base[name])
    for i,row in enumerate(ROWS):
        assert torch.equal(run.sample(row['neutral'],0),run.fixed[i].neutral)
        assert not torch.equal(run.sample(row['neutral'],1),run.fixed[i].neutral)


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


def test_resume_retains_inference_vocabulary_and_rejects_corruption_transactionally():
    live=make()
    live.step()
    live.sample('new held out words',1)
    saved=live.state_dict()
    expected=live.sample('second held out phrase',1)
    other=make()
    other.load_state_dict(saved)
    assert torch.equal(expected,other.sample('second held out phrase',1))
    bad=deepcopy(saved)
    bad['game']['opt_d']['param_groups'][0]['params'].append(999999)
    before=other.state_dict()
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='parameter count'):
        other.load_state_dict(bad)
    equal(before,other.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
    bad=deepcopy(saved)
    bad['tokenizer_vocab']['corrupt']=10000
    with pytest.raises(ValueError,match='tokenizer'):
        other.load_state_dict(bad)
    equal(before,other.state_dict())


def test_previous_local_protocol_rejected_before_state_mutation():
    run=make()
    before=run.state_dict()
    old=deepcopy(before)
    old['protocol']='tiny-qwen-pg090-neutral-v1'
    with pytest.raises(ValueError,match='protocol'):
        run.load_state_dict(old)
    equal(before,run.state_dict())


def test_dummy_context_uses_its_actual_config_before_forward():
    backend=TinyLLMBackend(dummy=True)
    limit=backend.model.config['max_position_embeddings']
    assert len(backend.encode(' '.join(['word']*limit)).ids)==limit
    with pytest.raises(ValueError,match='prompt exceeds context'):
        backend.encode(' '.join(['word']*(limit+1)))


def test_failed_shared_update_cannot_export(tmp_path,monkeypatch):
    run=make()
    monkeypatch.setattr(run.adapter,'generated',lambda row: row.neutral)
    with pytest.raises(ValueError,match='detached generator'):
        run.step()
    path=tmp_path/'failed.safetensors'
    with pytest.raises(RuntimeError,match='failed update'):
        run.export(path)
    assert not path.exists()


@pytest.mark.parametrize('corruption', ['targets', 'shape', 'vocab', 'conditioning', 'source', 'inference_source', 'recipe'])
def test_export_rejection_does_not_attach_or_change_tokenizer(tmp_path, corruption):
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    trained=make()
    path=tmp_path/'tiny.safetensors'
    trained.export(path)
    state=load_file(str(path))
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        metadata=json.loads(handle.metadata()['conceptmod'])
    if corruption=='targets':
        metadata['targets']=[]
    elif corruption=='shape':
        state['particles']=state['particles'][:1]
    elif corruption=='vocab':
        metadata['dummy_tokenizer_vocab']['new']=10000
    elif corruption=='conditioning':
        metadata['student_conditioning']='positive'
    elif corruption=='source':
        metadata['source_sha256']='wrong'
    elif corruption=='inference_source':
        metadata['inference_sources']={}
    else:
        metadata['resolved_recipe']['lr']*=10
    save_file(state,str(path),metadata={'conceptmod':json.dumps(metadata)})
    torch.manual_seed(91)
    backend=TinyLLMBackend(dummy=True)
    base=deepcopy(backend.model.state_dict())
    before_vocab=deepcopy(backend.tokenizer._vocab)
    bindings=[m.forward for m in backend.model.modules()]
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        load_export(backend,path)
    equal(base,backend.model.state_dict())
    assert before_vocab==backend.tokenizer._vocab
    assert bindings==[m.forward for m in backend.model.modules()]
    assert torch.equal(rng,torch.get_rng_state())


def test_cpu_cli_selects_shared_core_and_resumes_in_fresh_process(tmp_path):
    import particle_sliders.pg09 as shared_pg09
    core=Path(shared_pg09.__file__).resolve().parents[4]
    script=Path(__file__).resolve().parents[1]/'scripts/qualify_tiny_pg09_cpu.py'
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
             OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
             PYTHONDONTWRITEBYTECODE='1')
    for folder, until in [('resumed',2),('resumed',4),('uninterrupted',4)]:
        result=subprocess.run([sys.executable,str(script),'--core',str(core),
            '--output',str(tmp_path/folder),'--until',str(until)],env=env,
            text=True,capture_output=True,timeout=40)
        assert result.returncode==0,result.stdout+result.stderr
    resumed=torch.load(tmp_path/'resumed/training.pt',weights_only=True)
    expected=torch.load(tmp_path/'uninterrupted/training.pt',weights_only=True)
    equal(resumed,expected)
    receipt=json.loads((tmp_path/'resumed/receipt.json').read_text())
    assert receipt['game_class']=='particle_sliders.pg09.ConditionalPG09'
    assert receipt['core_module']==shared_pg09.__file__
    assert receipt['completed_steps']==4
    assert receipt['cuda_initialized'] is False
    assert receipt['quality_qualified'] is False


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
