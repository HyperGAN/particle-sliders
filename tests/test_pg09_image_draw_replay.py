"""Private RNG and lazy assets use synthetic CPU states/files only, never archives."""
import hashlib
import json
import random

import pytest
import torch

from conceptmod.textsliders.pg09_image_draw_replay import (
    cpu_draw_reconstruction,FileDescriptorAssets,json_sha,tensor_sha,python_rng_sha,
)
from scripts.verify_image_draw_replay_pg09 import load_private_inputs


def private_inputs(case,updates=500):
    generator=torch.Generator('cpu').manual_seed(12345)
    initial=generator.get_state().clone()
    expected=[]
    if case=='sd3':
        for _ in range(updates):
            expected.append((int(torch.randint(0,4,(1,),generator=generator).item()),
                             int(torch.randint(1,50,(1,),generator=generator).item())))
    python=random.Random(123).getstate()
    return initial,generator.get_state().clone(),python,expected


@pytest.mark.parametrize('case',['sana','sd3'])
def test_private_source_draws_match_endpoint_and_never_touch_global_rng(case,monkeypatch):
    initial,final,python,expected=private_inputs(case)
    before_cpu,before_python=torch.get_rng_state().clone(),random.getstate()
    def forbidden(*_,**__):
        raise AssertionError('global RNG seeding/restoration forbidden')
    for name in ('set_rng_state','manual_seed'):
        monkeypatch.setattr(torch,name,forbidden)
    for name in ('setstate','seed'):
        monkeypatch.setattr(random,name,forbidden)
    result=cpu_draw_reconstruction(case,initial,final,python,python)
    assert result['cpu_end_state_exact'] and result['python_end_state_exact']
    assert result['schedule_candidate_usable'] and len(result['draws'])==500
    if case=='sana':
        assert [r['prompt_index'] for r in result['draws']]==[i%2 for i in range(500)]
        assert all(r['fixed_training_timestep']==500. for r in result['draws'])
    else:
        assert [(r['prompt_index'],r['selected_prefix_index']) for r in result['draws']]==expected
    assert not result['historical_draw_replay_qualified'] and not result['cuda_noise_generated']
    assert torch.equal(before_cpu,torch.get_rng_state()) and before_python==random.getstate()
    assert torch.equal(initial,private_inputs(case)[0])
    assert not torch.cuda.is_initialized()


def test_hidden_cpu_or_python_consumers_report_mismatch_without_repair():
    initial,final,python,_=private_inputs('sd3')
    g=torch.Generator('cpu')
    g.set_state(final)
    torch.rand(1,generator=g)
    result=cpu_draw_reconstruction('sd3',initial,g.get_state(),python,python)
    assert not result['cpu_end_state_exact'] and not result['schedule_candidate_usable']
    p=random.Random(0)
    p.setstate(python)
    p.random()
    result=cpu_draw_reconstruction('sd3',initial,final,python,p.getstate())
    assert result['cpu_end_state_exact'] and not result['python_end_state_exact']
    assert not result['schedule_candidate_usable']
    assert not result['historical_draw_replay_qualified']


def test_invalid_reconstruction_policy_fails_before_any_global_rng_change():
    initial,final,python,_=private_inputs('sana')
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='bounded known'):
        cpu_draw_reconstruction('unknown',initial,final,python,python)
    with pytest.raises(ValueError,match='bounded known'):
        cpu_draw_reconstruction('sana',initial,final,python,python,updates=501)
    with pytest.raises(ValueError,match='uint8'):
        cpu_draw_reconstruction('sana',initial.float(),final,python,python)
    assert torch.equal(before,torch.get_rng_state())


def fixture_assets(tmp_path,case):
    draws=[]
    records=[]
    shape=(1,32,16,16) if case=='sana' else (1,16,64,64)
    def condition():
        if case=='sana':
            return dict(embeds=torch.ones(1,3,4),mask=torch.tensor([[1,1,0]]))
        return dict(embeds=torch.ones(1,3,4,dtype=torch.bfloat16),pooled=torch.ones(1,2,dtype=torch.bfloat16))
    conditions={role:condition() for role in ('neutral','positive','negative','unconditional')}
    text=tmp_path/'conditions.pt'
    torch.save(conditions,text)
    def asset(path):
        return dict(path=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    for i in range(3):
        draw=(dict(update=i+1,prompt_index=i%2,fixed_training_timestep=500.) if case=='sana' else
              dict(update=i+1,prompt_index=i,selected_prefix_index=i+1,native_training_step_count=50))
        draws.append(draw)
        path=tmp_path/f'noise-{i}.pt'
        torch.save(torch.full(shape,float(i)),path)
        records.append(dict(draw=draw,noise=asset(path),conditions=asset(text)))
    bindings=dict(draw_schedule_sha256=json_sha(draws),conditions_policy_sha256='a'*64,
                  native_source_sha256='b'*64,complete_condition_gate_sha256='c'*64)
    index=tmp_path/'index.json'
    index.write_text(json.dumps(dict(schema='image-descriptor-assets-cpu-prototype-v1',case=case,bindings=bindings,rows=records)))
    sha=hashlib.sha256(index.read_bytes()).hexdigest()
    return index,sha,bindings


@pytest.mark.parametrize('case',['sana','sd3'])
def test_file_reader_loads_only_selected_assets_cpu_and_binds_identity(tmp_path,case,monkeypatch):
    index,sha,bindings=fixture_assets(tmp_path,case)
    original=torch.load
    calls=[]
    def guarded(*args,**kwargs):
        calls.append(kwargs)
        assert kwargs==dict(weights_only=True,map_location='cpu')
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',guarded)
    reader=FileDescriptorAssets(index,expected_sha256=sha,expected_bindings=bindings)
    assert len(reader)==3 and calls==[]
    before_cpu,before_python=torch.get_rng_state().clone(),random.getstate()
    row=reader[1]
    assert len(calls)==2 and torch.all(row['initial_noise_fp32']==1.)
    assert row['draw']['update']==2 and not row['initial_noise_fp32'].requires_grad
    row['draw']['prompt_index']=999
    assert reader[1]['draw']['prompt_index']==1
    assert torch.equal(before_cpu,torch.get_rng_state()) and random.getstate()==before_python
    assert reader.identity()['historical_draw_replay_qualified'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('kind',['noise','index','private-index'])
def test_file_reader_rejects_source_or_asset_mutation(tmp_path,kind):
    index,sha,bindings=fixture_assets(tmp_path,'sana')
    reader=FileDescriptorAssets(index,expected_sha256=sha,expected_bindings=bindings)
    if kind=='noise':
        with (tmp_path/'noise-0.pt').open('ab') as stream:
            stream.write(b'changed')
    elif kind=='index':
        index.write_text(index.read_text()+' ')
    else:
        reader._manifest=b'{}'
    with pytest.raises(ValueError,match='changed'):
        reader[0]


@pytest.mark.parametrize('kind',['escape','symlink'])
def test_file_reader_rejects_escaping_or_symlink_assets(tmp_path,kind):
    index,_,bindings=fixture_assets(tmp_path,'sana')
    doc=json.loads(index.read_text())
    if kind=='escape':
        doc['rows'][0]['noise']['path']='../unrelated.pt'
    else:
        (tmp_path/'alias.pt').symlink_to(tmp_path/'noise-0.pt')
        doc['rows'][0]['noise']['path']='alias.pt'
    index.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='inside|symlink'):
        FileDescriptorAssets(index,expected_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),expected_bindings=bindings)


def test_unadmitted_packet_reader_contract_uses_only_cpu_restricted_load_on_toy(tmp_path,monkeypatch):
    from scripts.audit_image_checkpoint_rng_pg09 import expected_case_schema
    initial,_,python,_=private_inputs('sana')
    path=tmp_path/'toy-packet.pt'
    torch.save(dict(identity='synthetic',step=0,trainable={},optimizer={},torch_rng=initial,python_rng=python,cuda_rng=[]),path)
    expected=dict(torch_rng=dict(shape=list(initial.shape),dtype=str(initial.dtype),sha256=tensor_sha(initial)),
        cuda_rng=[],python_rng=dict(version=python[0],integer_count=len(python[1]),sha256=python_rng_sha(python)))
    original=torch.load
    def guarded(*args,**kwargs):
        assert kwargs==dict(weights_only=True,map_location='cpu',mmap=True)
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',guarded)
    packet=dict(path=str(path),case='sana',identity='synthetic',step=0,expected_schema=expected_case_schema('sana'))
    before=torch.get_rng_state().clone()
    cpu,p=load_private_inputs(packet,expected,torch)
    assert torch.equal(cpu,initial) and p==python and torch.equal(before,torch.get_rng_state())
    expected['torch_rng']['sha256']='0'*64
    with pytest.raises(ValueError,match='recovered CPU'):
        load_private_inputs(packet,expected,torch)


def test_metadata_plan_creation_cannot_reload_packets_or_run_reconstruction(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from scripts import prepare_image_draw_replay_pg09 as proposal
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.suffix not in {'.pt','.safetensors','.bin','.ckpt','.npy','.npz','.model'}
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    monkeypatch.setattr(sys,'argv',['prepare','--output-dir',str(tmp_path/'proposal')])
    proposal.main()
    plan=json.loads((tmp_path/'proposal/private-draw-plan.json').read_text())
    assert plan['execution_admitted'] is False and not plan['actual_packet_reload_executed']
    assert plan['metadata_observations']['sana']['cpu_hash_unchanged']
    assert not plan['metadata_observations']['sd3']['cpu_hash_unchanged']
    assert all(v['python_hash_unchanged'] and not v['cuda_hash_unchanged'] for v in plan['metadata_observations'].values())


@pytest.mark.parametrize('kind',['noise-shape','mask-type','nonbinary-mask'])
def test_asset_reader_rejects_checksums_for_wrong_native_numeric_schema(tmp_path,kind):
    index,_,bindings=fixture_assets(tmp_path,'sana')
    doc=json.loads(index.read_text())
    if kind=='noise-shape':
        path=tmp_path/'noise-0.pt'
        torch.save(torch.zeros(1,32),path)
        record=doc['rows'][0]['noise']
    else:
        path=tmp_path/'conditions.pt'
        value=torch.load(path,weights_only=True,map_location='cpu')
        value['neutral']['mask']=(torch.ones(1,3) if kind=='mask-type' else torch.full((1,3),2,dtype=torch.int64))
        torch.save(value,path)
        record=doc['rows'][0]['conditions']
    record.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    index.write_text(json.dumps(doc))
    reader=FileDescriptorAssets(index,expected_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),expected_bindings=bindings)
    with pytest.raises(ValueError,match='schema|binary'):
        reader[0]


def assert_exact(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:
            assert_exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            assert_exact(x,y)
    else:
        assert a==b


class ReaderBackendFixture:
    def __init__(self,reader):
        self.reader=reader
        self.network=torch.nn.Linear(1,1,bias=False)
        self.network.weight.data.fill_(.2)

    def metadata(self):
        return dict(qualification='CPU fixtures only; no native model',reader=self.reader.identity())

    def generated(self,item):
        return (item.row.latent+1)*self.network.weight.reshape(1,1,1,1)


def reader_builder_fixture(backend,descriptor,step):
    from conceptmod.textsliders.pg09_backends import PreparedResponse
    from conceptmod.textsliders.pg09_image_failure_campaigns import CampaignRow
    item=backend.reader[descriptor['asset_row']]
    c=item['conditions']
    row=CampaignRow(c['neutral'],c['positive'],c['negative'],item['initial_noise_fp32'],torch.tensor(500.),
        unconditional=c['unconditional'],provenance=dict(logical_step=step,asset_identity=item['source_identity']))
    return PreparedResponse(row,row.latent+.1,row.latent)


class ReaderCriticFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.SiLU(),torch.nn.Linear(4,1))

    def forward(self,response,conditions):
        return self.net(torch.cat([response.flatten(1).mean(1,keepdim=True),conditions],dim=1)).squeeze(-1)


@pytest.mark.parametrize('preset',['bispf','bispf_optimized'])
def test_file_reader_online_core_fixture_resume_rebuilds_exact_selected_union(tmp_path,preset):
    from copy import deepcopy
    from particle_sliders.pg09 import ConditionalPG09
    from conceptmod.textsliders.pg09_image_failure_campaigns import OnlineSnapshotProvider,online_campaign_step
    index,sha,bindings=fixture_assets(tmp_path,'sana')
    def make():
        reader=FileDescriptorAssets(index,expected_sha256=sha,expected_bindings=bindings)
        backend=ReaderBackendFixture(reader)
        provider=OnlineSnapshotProvider(backend,[dict(asset_row=i) for i in range(3)],
            torch.tensor([[i/10.,500.] for i in range(3)]),(1,32,16,16),reader_builder_fixture,
            protocol_identity=dict(qualification='CPU file-reader online fixture',assets=reader.identity()))
        game=ConditionalPG09(backend,provider,ReaderCriticFixture(),conditions=None,particles=None,
            response_domain='continuous_field',preset=preset,batch_size=2,total_steps=5,
            pg_source_profile='release',penalty_input_domain='auto')
        return game,provider
    game,provider=make()
    assert provider.audit==[]
    for _ in range(2):
        online_campaign_step(game,provider)
    checkpoint=deepcopy(game.state_dict())
    expected=[online_campaign_step(game,provider) for _ in range(3)]
    state=deepcopy(game.state_dict())
    audit=deepcopy(provider.audit)
    other,other_provider=make()
    other.load_state_dict(checkpoint)
    observed=[online_campaign_step(other,other_provider) for _ in range(3)]
    assert_exact(expected,observed)
    assert_exact(state,other.state_dict())
    assert_exact(audit,other_provider.audit)
    assert 1<=len(audit)<=3 and not torch.cuda.is_initialized()
