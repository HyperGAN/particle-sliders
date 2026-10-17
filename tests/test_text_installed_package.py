"""Installed-wheel execution and native-width text response software gates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from importlib.metadata import distribution, version

from particle_sliders.pg09 import ConditionalPG09, PG_VERSION
from particle_sliders.tiny_llm_backend import TinyLLMBackend, DummyTinyCausalLM
from conceptmod.textsliders.tiny_llm_pg09 import PG09TinyRun, load_export
from conceptmod.textsliders.text_package import require_text_package

ROWS=[dict(neutral='a quiet song',positive='an intense metal song'),
      dict(neutral='a calm melody',positive='an aggressive metal melody')]


def test_imports_match_the_locked_installed_wheel():
    identity=require_text_package()
    assert identity['versions']=={'particlegan':'0.9.1','particle-sliders-core':'0.6.0'}
    assert identity['review_status'] in ('pending_ada','ada_approved')
    assert identity['critical_modules']['particle_sliders.tiny_native']['path'].endswith('particle_sliders/tiny_native.py')
    assert not torch.cuda.is_initialized()


def test_source_checkout_shadow_rejected_before_product_execution(tmp_path):
    installed=Path(distribution('particle-sliders-core').locate_file('')).resolve()
    shadow=tmp_path/'shadow/particle_sliders'
    shadow.mkdir(parents=True)
    (shadow/'__init__.py').write_text('# source checkout shadow\n')
    root=Path(__file__).resolve().parents[1]
    program='''
import sys
sys.path[:0]=sys.argv[1:4]
from conceptmod.textsliders.text_package import require_text_package
try:
    require_text_package()
except RuntimeError as exc:
    assert 'shadows its installed distribution' in str(exc), str(exc)
else:
    raise AssertionError('source checkout shadow accepted')
'''
    result=subprocess.run([sys.executable,'-I','-c',program,str(shadow.parent),str(installed),str(root)],
                          cwd=tmp_path,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_bounded_cli_validates_pin_before_backend_construction(tmp_path,monkeypatch):
    import importlib.util
    from conceptmod.textsliders import text_package
    import particle_sliders.tiny_llm_backend as native
    script=Path(__file__).resolve().parents[1]/'scripts/qualify_tiny_pg09_cpu.py'
    spec=importlib.util.spec_from_file_location('tiny_bounded_cli',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    def reject():
        calls.append('pin')
        raise RuntimeError('pin rejected before backend')
    def forbidden(*a,**kw):
        calls.append('backend')
        raise AssertionError('backend constructed before package pin validation')
    monkeypatch.setattr(text_package,'require_text_package',reject)
    monkeypatch.setattr(native,'TinyLLMBackend',forbidden)
    monkeypatch.setattr(sys,'argv',[str(script),'--output',str(tmp_path/'out'),'--until','1'])
    with pytest.raises(RuntimeError,match='pin rejected'):
        module.main()
    assert calls==['pin']
    assert not (tmp_path/'out').exists()


def make_native_width():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    backend=TinyLLMBackend(dummy=True)
    backend.model=DummyTinyCausalLM(hidden_size=1024,num_hidden_layers=1,
                                  intermediate_size=64,vocab_size=64).eval().requires_grad_(False)
    return backend


def test_source_free_installed_wheel_executes_native_game(tmp_path):
    installed=Path(distribution('particle-sliders-core').locate_file('')).resolve()
    program=r'''
import importlib.abc,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class NoRepository(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'conceptmod','analysis','transformers','diffusers','llama_cpp','huggingface_hub'}:
            raise AssertionError('unexpected repository or native runtime import: '+fullname)
sys.meta_path.insert(0,NoRepository())
import torch
from particle_sliders.pg09 import ConditionalPG09,PG_VERSION
from particle_sliders.tiny_llm_backend import TinyLLMBackend
from particle_sliders.tiny_native import TinyParticleSlider
from particle_sliders.pg09_text_video import TinyQwenResponseAdapter,TextResponseRow
from particle_sliders.pg09_readout_critic import ConditionalReadoutCritic
torch.set_num_threads(1);torch.manual_seed(91)
native=TinyLLMBackend(dummy=True)
base={k:v.clone() for k,v in native.model.state_dict().items()}
slider=TinyParticleSlider(native.model).float()
adapter=TinyQwenResponseAdapter(native,slider)
rows=[adapter.prepare(TextResponseRow((1,2),(3,4))),adapter.prepare(TextResponseRow((5,6),(7,8)))]
conditions=torch.cat([r.neutral for r in rows])
game=ConditionalPG09(adapter,rows,ConditionalReadoutCritic(32,32),conditions=conditions,
                     particles=slider.particles,response_domain='continuous_vector',total_steps=2)
before=slider.particles.detach().clone()
for _ in range(2):
    metrics=game.step()
    assert abs(metrics['g_loss']-metrics['g_adv']-metrics['prior_regularization'])<1e-6
assert not torch.equal(before,slider.particles)
assert all(torch.equal(v,native.model.state_dict()[k]) for k,v in base.items())
sources=game.state_dict()['config']['implementation_sources']
assert not any(k.startswith(('conceptmod','analysis')) for k in sources)
assert 'particle_sliders.tiny_native' in sources and 'particle_sliders.reference' in sources
assert not torch.cuda.is_initialized()
print(json.dumps(dict(pg_version=PG_VERSION,game=type(game).__module__,sources=sources,cuda_initialized=False)))
'''
    result=subprocess.run([sys.executable,'-I','-c',program,str(installed)],cwd=tmp_path,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
        text=True,capture_output=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['pg_version']==PG_VERSION=='0.9.1'
    assert receipt['game']=='particle_sliders.pg09'


def test_native_1024_release_guard_precedes_attachment_and_tokenization():
    backend=make_native_width()
    from particle_sliders.tiny_native import _attention_targets
    bindings=[module.forward for module in _attention_targets(backend.model).values()]
    with pytest.raises(ValueError,match='width 1024'):
        PG09TinyRun(backend,ROWS,total_steps=4,batch_size=2)
    assert not backend.tokenizer._vocab
    assert bindings==[module.forward for module in _attention_targets(backend.model).values()]


def test_explicit_native_width_shared_run_replay_and_inference(tmp_path,monkeypatch):
    backend=make_native_width()
    base=deepcopy(backend.model.state_dict())
    run=PG09TinyRun(backend,ROWS,total_steps=4,batch_size=2,
                   pg_source_profile='alz-continuous-v1',penalty_input_domain='continuous')
    assert type(run.game) is ConditionalPG09
    assert type(run.network).__module__=='particle_sliders.tiny_native'
    assert type(run.adapter).__module__=='particle_sliders.pg09_text_video'
    assert version('particle-sliders-core')=='0.6.0'
    def forbidden(*a,**kw):
        raise AssertionError('MSE forbidden')
    monkeypatch.setattr(torch.nn.functional,'mse_loss',forbidden)
    bank=run.network.particles.detach().clone()
    run.step()
    state=run.state_dict()
    expected=run.step()
    assert expected['d_pen']>0
    assert run.game.penalty.last_stats['resolved_domain']=='continuous'
    assert not torch.equal(bank,run.network.particles)
    assert all(torch.equal(value,backend.model.state_dict()[key]) for key,value in base.items())
    other=PG09TinyRun(make_native_width(),ROWS,total_steps=4,batch_size=2,
                     pg_source_profile='alz-continuous-v1',penalty_input_domain='continuous')
    other.load_state_dict(state)
    assert other.step()==expected
    for key,value in run.network.state_dict().items():
        assert torch.equal(value,other.network.state_dict()[key])
    export=tmp_path/'native1024.safetensors'
    run.export(export)
    loaded_backend=make_native_width()
    loaded,metadata=load_export(loaded_backend,export,pg_source_profile='alz-continuous-v1')
    for scale in (-1.,0.,.5,1.):
        ids=loaded_backend.encode(ROWS[0]['neutral']).ids
        with loaded.scaled(scale),torch.no_grad():
            value=loaded_backend.hidden(ids)[:,-1].float()
        assert torch.equal(value,run.sample(ROWS[0]['neutral'],scale))
    assert metadata['pg_version']=='0.9.1'
    assert not torch.cuda.is_initialized()
