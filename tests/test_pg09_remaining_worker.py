"""CPU toy-core fixtures only: no pretrained/native forward, T5 or GPU calls."""
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conceptmod.textsliders.pg09_backends import PreparedResponse,ResponseRow
from conceptmod.textsliders.pg09_remaining_admission import RemainingRecipe,digest,sha
from conceptmod.textsliders import pg09_remaining_worker as worker
from conceptmod.textsliders.pg09_remaining_offline import safe_header

ROOT=Path(__file__).resolve().parents[1]
DRAFTS=ROOT/'evidence/remaining-proof-repair-v1/admission-final'


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain=nn.Parameter(torch.zeros(()))
        self.scale=1.
    @contextmanager
    def scaled(self,value):
        previous,self.scale=self.scale,value
        try:
            yield
        finally:
            self.scale=previous


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset=nn.Parameter(torch.tensor(.3),requires_grad=False)
        self.register_buffer('fixed',torch.tensor(2.))


class ToyBackend:
    def __init__(self):
        self.network=ToyNetwork()
        self.transformer=ToyBase()
        self.model_identity=dict(qualification='CPU toy fixture only')
    def prepare(self,row):
        base=torch.zeros_like(row.latent)+self.transformer.offset
        return PreparedResponse(row,base+.2*row.latent,base)
    def generated(self,prepared):
        return prepared.neutral+self.network.gain*self.network.scale*prepared.row.latent
    def metadata(self):
        return dict(family='CPU toy fixture',native_forward=False,model_identity=self.model_identity)


class ToyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.score=nn.Linear(6,1)
    def forward(self,response,condition):
        return self.score(torch.cat((response.flatten(1),condition),1)).squeeze(-1)


def toy_run():
    from particle_sliders.pg09 import ConditionalPG09
    torch.set_num_threads(1)
    torch.manual_seed(7)
    backend=ToyBackend()
    row=ResponseRow('CPU neutral','CPU positive','',torch.ones(1,1,2,2),timestep=torch.tensor(.5))
    rows=[backend.prepare(row),backend.prepare(replace(row,latent=row.latent*.5))]
    game=ConditionalPG09(backend,rows,ToyCritic(),conditions=torch.tensor([[0.,1.],[1.,1.]]),
                         batch_size=8,total_steps=3840,seed=1000,preset='bispf_optimized')
    return SimpleNamespace(backend=backend,game=game,rows=rows,step=game.step)


def toy_recipe():
    return SimpleNamespace(identity=digest(dict(case='CPU toy only',total_steps=3840)),
                           data=dict(total_steps=3840,teacher_indices=[0,1],checkpoint_every=80))


def toy_base(run):
    return worker.frozen_identity(run.backend.transformer,lambda:None)


def initial_gate(run,*,emit,indices):
    # Reuse the existing pure strict grad/no-grad/cache checker; no native runtime.
    from moss_original_runtime import initial_gradient_parity
    return initial_gradient_parity(run,emit=emit,indices=indices)


def test_default_cli_and_card_recipe_fail_closed_before_cuda_or_load(tmp_path,monkeypatch):
    data=json.loads((DRAFTS/'final-form.draft.json').read_text())
    contract=RemainingRecipe.validate(data)
    assert 'no owner-reviewed' in contract.phase_reason('probe',{})
    def forbidden(*args,**kwargs):
        pytest.fail('DRAFT CLI reached live model boundary')
    monkeypatch.setattr(worker,'cuda_setup',forbidden)
    monkeypatch.setattr(worker,'load_native',forbidden)
    # Source check is covered independently; the unadmitted contract rejects first.
    monkeypatch.setattr(RemainingRecipe,'verify_sources',lambda self:None)
    with pytest.raises(ValueError,match='no owner-reviewed'):
        worker.execute(contract,data['output_directory'],'probe',[])
    assert not torch.cuda.is_initialized()
    for key,value in [('physical_gpu',2),('total_steps',3),('batch_size',2),('hf_home','/wrong/cache')]:
        changed=dict(data,**{key:value})
        with pytest.raises(ValueError):
            RemainingRecipe.validate(changed)
    reviewed=RemainingRecipe.validate(dict(data,admission='REVIEWED_REMAINING_GAN_RECIPE'))
    assert 'priority' in reviewed.phase_reason('probe',{})
    with pytest.raises(ValueError,match='both exact'):
        RemainingRecipe.validate(dict(data,lease_paths=['one']))


def test_missing19_context_asset_and_phase_proof_do_not_become_capacity_only():
    data=json.loads((DRAFTS/'supra-final-boss.draft.json').read_text())
    data.update(admission='REVIEWED_REMAINING_GAN_RECIPE',owner_priority_gate=dict(path='/future/owner.json',sha256='a'*64))
    contract=RemainingRecipe.validate(data)
    assert 'missing reviewed19' in contract.phase_reason('probe',{})
    data['contexts']['sha256']='b'*64
    data['contexts']['receipt_sha256']='c'*64
    data['contexts']['preparation_phase_sha256']='d'*64
    data['contexts']['case_budget_sha256']='e'*64
    data['expected_base_parameter_sha256']='f'*64
    contract=RemainingRecipe.validate(data)
    assert contract.phase_reason('probe',{}) is None
    assert 'previous phase' in contract.phase_reason('train',{})
    prior=dict(recipe_sha256=contract.identity,status='probe_passed',completed_steps=3,
               frozen_base_unchanged=True,generator_changed=True,critic_changed=True,
               exact_in_memory_replay=True,exact_export_reload=True,file_reference_written=True)
    assert 'separate-process' in contract.phase_reason('train',prior)
    assert contract.phase_reason('resume-check',prior) is None
    assert contract.phase_reason('train',dict(prior,fresh_disk_resume_exact=True)) is None
    assert 'full GAN budget' in contract.phase_reason('evaluate',prior)


def test_immutable_file_reference_and_cpu_rng_packet_restore(tmp_path):
    packet=dict(rng=torch.Generator().get_state(),tensor=torch.ones(2))
    path=tmp_path/'packet.pt'
    first=worker.immutable_packet(path,packet)
    assert worker.immutable_packet(path,packet)==first
    with pytest.raises(ValueError,match='immutable'):
        worker.immutable_packet(path,dict(packet,tensor=torch.zeros(2)))
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    assert loaded['rng'].device.type=='cpu' and loaded['rng'].dtype==torch.uint8
    assert sha(path)==first


def test_full_recipe_probe_shared_core_and_actual_separate_cpu_process_disk_replay(tmp_path):
    run=toy_run()
    recipe=toy_recipe()
    base=toy_base(run)
    observed=[]
    result=worker.qualify_probe(run,recipe,tmp_path,base,
        emit=lambda event,**fields:observed.append((event,fields)),guard=lambda:None,cap_check=lambda:None,
        base_check=lambda:toy_base(run)==base,initial_gate=initial_gate)
    assert result['status']=='probe_passed' and run.game.completed_steps==3
    assert all(result[k] for k in ('generator_changed','critic_changed','exact_export_reload','exact_in_memory_replay'))
    before=sha(tmp_path/'checkpoints/step-000003/state.pt')
    with pytest.raises(ValueError,match='separate process'):
        worker.resume_disk_reference(toy_run(),recipe,tmp_path,base,result,base_check=lambda:True)
    (tmp_path/'previous.json').write_text(json.dumps(result))
    code='''
import json
from pathlib import Path
from test_pg09_remaining_worker import toy_run,toy_recipe,toy_base
from conceptmod.textsliders.pg09_remaining_worker import resume_disk_reference
root=Path(__import__('sys').argv[1]);run=toy_run();base=toy_base(run)
proof=resume_disk_reference(run,toy_recipe(),root,base,json.loads((root/'previous.json').read_text()),base_check=lambda:toy_base(run)==base)
print(json.dumps(proof))
'''
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=str(ROOT/'tests')+os.pathsep+os.environ['PYTHONPATH'])
    process=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=environment,capture_output=True,text=True,timeout=45)
    assert process.returncode==0,process.stderr
    proof=json.loads(process.stdout)
    assert proof['fresh_disk_resume_exact']
    worker.verify_disk_proof(recipe,tmp_path,dict(result,**proof),base)
    assert sha(tmp_path/'checkpoints/step-000003/state.pt')==before
    saved=json.loads((tmp_path/'fresh-disk-resume.json').read_text())
    assert saved['probe_pid']!=saved['resume_pid'] and saved['checkpoint_advanced'] is False
    assert not torch.cuda.is_initialized()


def test_mixed_loaded_module_origin_and_phase_caps_reject(tmp_path):
    data=json.loads((DRAFTS/'final-form.draft.json').read_text())
    recipe=RemainingRecipe.validate(data)
    unrelated=tmp_path/'module.py'
    unrelated.write_text('x=1\n')
    with pytest.raises(ValueError,match='actual imported'):
        recipe.verify_origin(SimpleNamespace(__file__=str(unrelated)))
    def measurement(_):
        return dict(rss_bytes=0,cuda_reserved_bytes=0,cuda_peak_reserved_bytes=0,output_bytes=0)
    worker.check_caps(recipe,tmp_path,0,'probe',clock=lambda:1,measure=measurement)
    with pytest.raises(RuntimeError,match='cap exceeded'):
        worker.check_caps(recipe,tmp_path,0,'probe',clock=lambda:1801,measure=measurement)
    with pytest.raises(RuntimeError,match='cap exceeded'):
        worker.check_caps(recipe,tmp_path,0,'probe',clock=lambda:1,measure=measurement,cohort_deadline=0)
    with pytest.raises(RuntimeError,match='cap exceeded'):
        worker.check_caps(recipe,tmp_path,0,'probe',clock=lambda:1,
                          measure=lambda _:dict(rss_bytes=0,cuda_reserved_bytes=25*1024**3,output_bytes=0))


def synthetic_factory(path,name='supra_particle_model'):
    def create():
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return SimpleNamespace(MODEL_SOURCE=path,model_module=create)


def test_dynamic_native_namespace_stable_source_and_collision_failure_atomicity(tmp_path):
    source=tmp_path/'native_definition.py'
    source.write_text('class Model:\n    def forward(self,x):\n        return x\n')
    recipe=SimpleNamespace(data=dict(source_pins={str(source):sha(source)}))
    old=sys.modules.pop('supra_particle_model',None)
    try:
        assert worker.register_native_supra_source(synthetic_factory(source),recipe)==str(source)
        registered=sys.modules['supra_particle_model']
        assert registered.Model.__module__=='supra_particle_model'
        assert registered.Model.forward.__code__.co_filename==str(source)
        other=tmp_path/'other.py'
        other.write_text(source.read_text())
        with pytest.raises(ValueError,match='another origin'):
            worker.register_native_supra_source(synthetic_factory(other),SimpleNamespace(data=dict(source_pins={str(other):sha(other)})))
        assert sys.modules['supra_particle_model'] is registered
        def broken_import():
            sys.modules['supra_particle_model']=SimpleNamespace(__file__='/invalid/source.py')
            raise RuntimeError('synthetic import failure')
        with pytest.raises(RuntimeError,match='synthetic import failure'):
            worker.register_native_supra_source(SimpleNamespace(MODEL_SOURCE=source,model_module=broken_import),recipe)
        assert sys.modules['supra_particle_model'] is registered
        with pytest.raises(ValueError,match='deterministic'):
            worker.register_native_supra_source(synthetic_factory(source,name='pid_based_99'),recipe)
        assert 'pid_based_99' not in sys.modules
    finally:
        sys.modules.pop('supra_particle_model',None)
        if old is not None:
            sys.modules['supra_particle_model']=old


def test_dynamic_namespace_reconstructs_same_class_source_in_two_fresh_cpu_processes(tmp_path):
    source=tmp_path/'synthetic_native.py'
    source.write_text('class Model:\n    def forward(self,x):\n        return x\n')
    script=tmp_path/'source_only_check.py'
    script.write_text('''
import inspect,json
from pathlib import Path
from types import SimpleNamespace
from test_pg09_remaining_worker import synthetic_factory,ToyBackend,ToyCritic
from conceptmod.textsliders.pg09_remaining_worker import register_native_supra_source
from conceptmod.textsliders.pg09_remaining_admission import sha
source=Path(__import__('sys').argv[1]);module=synthetic_factory(source)
register_native_supra_source(module,SimpleNamespace(data=dict(source_pins={str(source):sha(source)})))
import supra_particle_model as native
from particle_sliders.pg09 import _implementation_sources
backend=ToyBackend();backend.transformer=native.Model()
sources=_implementation_sources(backend,ToyCritic())
print(json.dumps(dict(namespace=native.Model.__module__,class_source=inspect.getsourcefile(native.Model),forward_source=native.Model.forward.__code__.co_filename,sha256=sha(source),core_implementation_sources=sources)))
''')
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                     PYTHONPATH=str(ROOT/'tests')+os.pathsep+os.environ['PYTHONPATH'])
    outputs=[]
    for _ in range(2):
        process=subprocess.run([sys.executable,str(script),str(source)],env=environment,capture_output=True,text=True,timeout=30)
        assert process.returncode==0,process.stderr
        outputs.append(json.loads(process.stdout))
    assert outputs[0]==outputs[1]
    assert outputs[0]['namespace']=='supra_particle_model'
    assert outputs[0]['core_implementation_sources']['supra_particle_model']==sha(source)


def test_safetensors_header_bounds_reject_without_tensor_or_model_load(tmp_path):
    path=tmp_path/'bad.safetensors'
    path.write_bytes(struct.pack('<Q',128*1024*1024))
    with pytest.raises(ValueError,match='unbounded'):
        safe_header(path)
    header=json.dumps(dict(x=dict(dtype='F32',shape=[1],data_offsets=[0,8]))).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+b'12345678')
    with pytest.raises(ValueError,match='byte range'):
        safe_header(path)


def test_disk_reference_rejects_corrupt_checkpoint_before_any_replay(tmp_path):
    recipe=toy_recipe()
    base=dict(combined_sha256='a'*64)
    class NoReplay:
        completed_steps=3
        device=torch.device('cpu')
        def state_dict(self):
            return dict(completed_steps=3,value=torch.ones(1))
        def load_state_dict(self,value):
            pytest.fail('corrupt immutable checkpoint reached model restore')
    run=SimpleNamespace(game=NoReplay())
    worker.save_checkpoint(tmp_path,recipe,run,base['combined_sha256'])
    packet=dict(schema='remaining-gan-file-reference-v1',recipe_sha256=recipe.identity,frozen_base=base,
                probe_pid=os.getpid()-1,before={},expected=run.game.state_dict(),metrics={},sample=torch.ones(1))
    reference=tmp_path/'probe-reference.pt'
    previous=dict(reference_sha256=worker.immutable_packet(reference,packet))
    cp=tmp_path/'checkpoints/step-000003/state.pt'
    cp.write_bytes(cp.read_bytes()+b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        worker.resume_disk_reference(run,recipe,tmp_path,base,previous,base_check=lambda:True)


def test_probe_interrupt_checkpoints_completed_step_without_release_proof(tmp_path):
    run=toy_run()
    recipe=toy_recipe()
    base=toy_base(run)
    stopped=False
    def emit(event,**fields):
        nonlocal stopped
        if event=='probe_step' and fields['completed_steps']==1:
            stopped=True
    result=worker.qualify_probe(run,recipe,tmp_path,base,emit=emit,guard=lambda:None,cap_check=lambda:None,
                               base_check=lambda:toy_base(run)==base,initial_gate=initial_gate,stop_requested=lambda:stopped)
    assert result['status']=='stopped' and result['completed_steps']==1
    assert result['file_reference_written'] is False
    assert (tmp_path/'checkpoints/step-000001/state.pt').is_file()
    assert not (tmp_path/'probe-reference.pt').exists()


def test_streamed_frozen_parameters_and_buffers_exclude_owned_adapter():
    base=ToyBase()
    base.adapter=nn.Parameter(torch.tensor(.1))
    identity=worker.frozen_identity(base,lambda:None)
    with torch.no_grad():
        base.adapter.add_(1)
    assert worker.frozen_identity(base,lambda:None)==identity
    with torch.no_grad():
        base.fixed.add_(1)
    assert worker.frozen_identity(base,lambda:None)['parameters_sha256']==identity['parameters_sha256']
    assert worker.frozen_identity(base,lambda:None)['buffers_sha256']!=identity['buffers_sha256']
    with torch.no_grad():
        base.offset.add_(1)
    assert worker.frozen_identity(base,lambda:None)['parameters_sha256']!=identity['parameters_sha256']


def test_offline_context_binding_requires_actual_native_phase_and_same_sources():
    from copy import deepcopy
    from conceptmod.textsliders.pg09_remaining_bind import validate_preparation
    data=json.loads((DRAFTS/'supra-final-boss.draft.json').read_text())
    draft=RemainingRecipe.validate(data)
    admitted=dict(deepcopy(data),admission='REVIEWED_REMAINING_GAN_RECIPE')
    preparation=RemainingRecipe.validate(admitted)
    runtime_source=next(p for p in data['source_pins'] if p.endswith('/supra/runtime.py'))
    base=dict(parameters_sha256='a'*64,buffers_sha256='b'*64,parameters=104094736)
    base['combined_sha256']=digest(base)
    phase=dict(status='contexts_prepared',mode='contexts',recipe_sha256=preparation.identity,
               training_updates=0,completed_steps=0,native_encode_calls_at_most=19,frozen_base=base)
    receipt=dict(schema='supra-frozen-contexts-v1',unique_contexts=19,max_encode_calls=19,
                 config_sha256=data['config_canonical_sha256'],stored_unconditional=True,
                 runtime_identity=dict(model_revision=data['model_revision'],text_revision=data['text_revision'],
                                       preparation_recipe_sha256=preparation.identity),
                 torch_version=data['versions']['torch'],runtime_source=runtime_source,
                 runtime_source_sha256=data['source_pins'][runtime_source])
    budget=dict(schema='remaining-cohort-wall-v1',cohort_id=data['cohort_id'],case=data['case'],
                wall_seconds=data['caps']['wall_seconds'],started_unix=0,deadline_unix=data['caps']['wall_seconds'])
    validate_preparation(draft,preparation,phase,receipt,budget)
    for key,value in [('status','failed_no_retry'),('training_updates',1),('native_encode_calls_at_most',20)]:
        with pytest.raises(ValueError,match='zero-update'):
            validate_preparation(draft,preparation,dict(phase,**{key:value}),receipt,budget)
    with pytest.raises(ValueError,match='exact native'):
        validate_preparation(draft,preparation,phase,dict(receipt,stored_unconditional=False),budget)
    with pytest.raises(ValueError,match='cumulative'):
        validate_preparation(draft,preparation,phase,receipt,dict(budget,deadline_unix=99999))
    changed=deepcopy(admitted)
    changed['source_pins'][runtime_source]='f'*64
    with pytest.raises(ValueError,match='source or cohort'):
        validate_preparation(draft,RemainingRecipe.validate(changed),phase,receipt,budget)


def test_supra_noise_and_cfg_rounding_measurements_fail_closed_before_update():
    noise=[dict(seed=7063,exact=True),dict(seed=7064,exact=True)]
    measured=[dict(index=0,relative_rms=0.),dict(index=239,relative_rms=.0005)]
    result=worker.check_supra_protocol_measurements(noise,measured)
    assert result['relative_rms_limit']==.001 and result['optimized_loss'] is False
    with pytest.raises(ValueError,match='seeded noise'):
        worker.check_supra_protocol_measurements([noise[0],dict(seed=7064,exact=False)],measured)
    with pytest.raises(ValueError,match='rounding parity') as error:
        worker.check_supra_protocol_measurements(noise,[dict(index=239,relative_rms=.0011)])
    assert error.value.diagnostics['velocity'][0]['relative_rms']==.0011
