"""Real Diffusers native API with tiny random CPU weights; no pretrained qualification."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from diffusers import ZImageTransformer2DModel
from torch import nn

from embedded_zimage_shared import backend as product
from embedded_zimage_shared.graphs import ZImageLoRA, targets
from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.field_critic import FieldCritic
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig
from particle_sliders.pg09 import _hash, validate_pg09_domain
from particle_sliders.response import open_response

ROOT = Path(__file__).resolve().parents[3]
torch.set_num_threads(1)


def equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


def model():
    return ZImageTransformer2DModel(all_patch_size=(2,), all_f_patch_size=(1,), in_channels=4,
        dim=32,n_layers=1,n_refiner_layers=1,n_heads=4,n_kv_heads=2,cap_feat_dim=16,
        axes_dims=[4,2,2],axes_lens=[128,128,128])


@pytest.fixture
def staged(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(23)
        native=model()
        native.save_pretrained(tmp_path/'model')
        identity=product.bundle_identity(tmp_path/'model')
        rows=[dict(neutral={'embeds':torch.randn(3,16)},positive={'embeds':torch.randn(4,16)},
            negative={'embeds':torch.randn(2,16)},latent=torch.randn(1,4,4,4),timestep=torch.tensor(300.+i*100.)) for i in range(3)]
        torch.save(dict(schema=product.PROTOCOL,model_identity=identity,rows=rows),tmp_path/'rows.pt')
    config=SessionConfig(product.NAME,product.COHORT,model_dir=str(tmp_path/'model'),model_identity=identity,
        rows=str(tmp_path/'rows.pt'),rows_sha256=product.file_sha256(tmp_path/'rows.pt'),response_shape=[1,4,4,4],steps=4,batch=2,seed=7,ema_decay=.8)
    return native,rows,config


def test_real_hypergan_native_attention_outputs_input_and_all_adapter_vjps(staged):
    native,rows,config=staged
    old_spec=importlib.util.spec_from_file_location('historical_zimage_lora',ROOT/'conceptmod/textsliders/lora.py')
    historical=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(historical)
    old_model=deepcopy(native).eval().requires_grad_(False)
    old=historical.LoRANetwork(old_model,rank=16,alpha=16,target_replace=['Attention'],delimiter='-')
    new_model=deepcopy(native).eval().requires_grad_(False)
    new=ZImageLoRA(new_model)
    assert len(new.names)==12
    assert [layer.lora_name for layer in old.unet_loras]==list(new.names)
    with torch.no_grad():
        for layer in old.unet_loras:
            nn.init.normal_(layer.lora_up.weight,std=.04)
            graph=new.projections[layer.lora_name].graph
            assert type(graph).__module__=='hndl.torch'
            graph['down'].weight.copy_(layer.lora_down.weight)
            graph['up'].weight.copy_(layer.lora_up.weight)
    z=rows[0]['latent'].clone().requires_grad_(); z2=z.detach().clone().requires_grad_()
    c=rows[0]['neutral']['embeds'].clone().requires_grad_(); c2=c.detach().clone().requires_grad_()
    def predict(m,z,c): return -torch.stack([v.squeeze(1) for v in m(list(z.unsqueeze(2).unbind(0)),torch.tensor([.7]),[c],return_dict=False)[0]])
    old.set_lora_slider(.6)
    with old, new.scaled(.6):
        expected,actual=predict(old_model,z,c),predict(new_model,z2,c2)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        g0=torch.autograd.grad(expected.square().sum(),(z,c,*old.parameters()))
        g1=torch.autograd.grad(actual.square().sum(),(z2,c2,*new.parameters()))
        for a,b in zip(g1,g0): torch.testing.assert_close(a,b,atol=4e-6,rtol=3e-4)
    with new.scaled(0):
        actual=predict(new_model,z,c)
    new.close()
    assert torch.equal(actual,predict(new_model,z,c))
    assert not any(m._forward_hooks for m in new_model.modules())


def test_shared_hndl_critic_outputs_vjps_and_penalty_second_derivatives():
    torch.default_generator.manual_seed(44)
    old=FieldCritic(4,106);new=HNDLFieldCritic(4,106,(4,4))
    for name,layer in [('field0',old.field[0]),('field2',old.field[2]),('condition',old.condition),('score0',old.score[0]),('score2',old.score[2])]:
        new.graph[name].load_state_dict(layer.state_dict())
    assert type(new.graph).__module__=='hndl.torch'
    x=torch.randn(2,4,4,4,requires_grad=True); y=x.detach().clone().requires_grad_()
    c=torch.randn(2,106)
    a,b=old(x,c),new(y,c)
    torch.testing.assert_close(a,b)
    g0=torch.autograd.grad(a.sum(),x,create_graph=True)[0]
    g1=torch.autograd.grad(b.sum(),y,create_graph=True)[0]
    torch.testing.assert_close(g0,g1)
    second0=torch.autograd.grad(g0.square().sum(),tuple(old.parameters()),allow_unused=True)
    second1=torch.autograd.grad(g1.square().sum(),tuple(new.parameters()),allow_unused=True)
    for a,b in zip(second0,second1):
        if a is None: assert b is None
        else: torch.testing.assert_close(a,b,atol=2e-7,rtol=3e-5)


@pytest.mark.parametrize('preset',['bispf','bispf_optimized'])
def test_actual_shared_updates_exact_resume_ema_strength_and_native_only_inference(staged,tmp_path,preset):
    native,rows,config=staged
    options=config.options; options['preset']=preset
    config=SessionConfig(product.NAME,product.COHORT,**options)
    with open_response(config) as full:
        initial=deepcopy(full.game.network.state_dict());critic=deepcopy(full.game.critic.state_dict())
        full.train()
        expected=full.checkpoint()
        assert any(not torch.equal(v,full.game.network.state_dict()[k]) for k,v in initial.items())
        assert any(not torch.equal(v,full.game.critic.state_dict()[k]) for k,v in critic.items())
        assert full.game.prior is None and full.capabilities.generator_arm=='adapter_gan'
        assert set(id(p) for group in full.game.opt_g.param_groups for p in group['params'])==set(map(id,full.game.network.parameters()))
        outputs={strength:full.evaluate(strength=strength) for strength in (0,.5,1)}
        assert outputs[0].shape==(1,4,4,4)
        assert not torch.equal(outputs[.5],outputs[1])
        assert not torch.equal(outputs[0],outputs[.5])
        raw=full.export_inference(selection='raw');ema=full.export_inference(selection='ema')
    with open_response(config) as split:
        split.train(stop_after=2);split.checkpoint(tmp_path/'split.pt')
    with open_response(config,resume=tmp_path/'split.pt') as resumed:
        resumed.train();equal(expected,resumed.checkpoint())
    (tmp_path/'rows.pt').unlink()
    request=dict(condition=rows[0]['neutral'],latent=rows[0]['latent'],timestep=rows[0]['timestep'])
    for packet in (raw,ema):
        assert set(packet)=={'schema','specification','origin','selection','network','sha256'}
        assert 'rows' not in packet['specification']['configuration']
        with open_inference(packet,options={'model_dir':str(tmp_path/'model')},operation='response') as inference:
            assert not hasattr(inference,'game')
            for strength in (0,.5,1):
                value=inference.response(**request,strength=strength)
                if packet is raw: assert torch.equal(value,outputs[strength])
                if strength==0: assert torch.equal(value,outputs[0])
            with pytest.raises(Exception,match='cannot resume'): inference.step()
    assert any(not torch.equal(raw['network'][k],ema['network'][k]) for k in raw['network'])


@pytest.mark.parametrize('changed',['model','shape','profile','source','rows','hooks'])
def test_mismatch_rejects_before_native_mutation(staged,monkeypatch,changed):
    native,rows,config=staged
    options=config.options
    flags=[p.requires_grad for p in native.parameters()];modes=[m.training for m in native.modules()]
    before=deepcopy(native.state_dict());rng=torch.get_rng_state().clone()
    if changed=='model': options['model_identity']['state_sha256']='0'*64
    elif changed=='shape': options['response_shape']=[1,256]
    elif changed=='profile': options['penalty_input_domain']='continuous'
    elif changed=='source': monkeypatch.setitem(product.CORE_HASHES,'pg09.py','0'*64)
    elif changed=='rows':
        rows[0]['positive']['embeds']=torch.randn(3,15)
        torch.save(dict(schema=product.PROTOCOL,model_identity=options['model_identity'],rows=rows),options['rows'])
        rng=torch.get_rng_state().clone()
    else:
        handle=native.register_forward_hook(lambda *args: None)
    config=SessionConfig(product.NAME,product.COHORT,**options)
    with pytest.raises((ValueError,RuntimeError)):
        product.BACKEND.open(config,model=native)
    if changed=='hooks':handle.remove()
    assert not any(m._forward_hooks for m in native.modules())
    assert [p.requires_grad for p in native.parameters()]==flags
    assert [m.training for m in native.modules()]==modes
    equal(before,native.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_builder_tamper_and_wide_vector_are_refused(staged,monkeypatch):
    import particle_sliders.fibo_hndl as binding
    native,rows,config=staged
    original=binding._sha
    monkeypatch.setattr(binding,'_sha',lambda p:'0'*64 if str(p).endswith('hndl_networks.py') else original(p))
    with pytest.raises(ValueError,match='builder closure'):product.BACKEND.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    with pytest.raises(ValueError,match='width >=170'):validate_pg09_domain((1,2048),response_domain='continuous_vector')


def test_borrowed_modes_gradients_checkpoint_flags_and_nested_scales_restore(staged):
    native,rows,config=staged
    native.train();native.layers[0].eval();next(native.parameters()).requires_grad_(False)
    native.enable_gradient_checkpointing()
    flags=[p.requires_grad for p in native.parameters()];modes=[m.training for m in native.modules()]
    checkpoint=native._gradient_checkpointing_func
    gradients=[]
    for p in native.parameters():
        p.grad=torch.ones_like(p);gradients.append(p.grad.clone())
    with product.BACKEND.open(config,model=native) as session:
        network=session.game.network
        assert all(not p.requires_grad for p in native.parameters())
        with network.scaled(.3):
            with network.scaled(0): assert network.scale==0
            assert network.scale==.3
        assert network.scale==0
        session.step()
    assert native._gradient_checkpointing_func is checkpoint
    assert [p.requires_grad for p in native.parameters()]==flags
    assert [m.training for m in native.modules()]==modes
    for p,g in zip(native.parameters(),gradients):assert torch.equal(p.grad,g)
    assert not any(m._forward_hooks for m in native.modules())


def test_prepare_and_update_failures_remove_every_hook_and_restore_borrowed_state(staged,monkeypatch):
    native,rows,config=staged
    modes=[m.training for m in native.modules()];flags=[p.requires_grad for p in native.parameters()]
    marker=RuntimeError('teacher failure')
    original=product.ImageResponseAdapter.prepare
    monkeypatch.setattr(product.ImageResponseAdapter,'prepare',lambda *args: (_ for _ in ()).throw(marker))
    with pytest.raises(RuntimeError) as caught:product.BACKEND.open(config,model=native)
    assert caught.value is marker
    assert not any(m._forward_hooks for m in native.modules())
    assert [m.training for m in native.modules()]==modes
    assert [p.requires_grad for p in native.parameters()]==flags
    monkeypatch.setattr(product.ImageResponseAdapter,'prepare',original)
    session=product.BACKEND.open(config,model=native)
    session.game.backend.generated=lambda *args: (_ for _ in ()).throw(marker)
    with pytest.raises(RuntimeError) as caught:session.step()
    assert caught.value is marker and session.closed
    assert not any(m._forward_hooks for m in native.modules())
    assert [m.training for m in native.modules()]==modes
    assert [p.requires_grad for p in native.parameters()]==flags


def test_live_restore_failure_rolls_back_and_packet_rejection_precedes_load_hooks(staged):
    _,_,config=staged
    with open_response(config) as source,open_response(config) as target:
        source.train(stop_after=2);target.train(stop_after=1)
        before=target.checkpoint();packet=source.checkpoint();calls=[]
        marker=RuntimeError('post-copy load hook')
        def fail(module,incompatible):
            calls.append(1)
            if len(calls)==1:
                module.eval().requires_grad_(False)
                raise marker
        hook=target.game.network.register_load_state_dict_post_hook(fail)
        try:
            with pytest.raises(RuntimeError) as caught:target.restore(packet)
            assert caught.value is marker and len(calls)==2
            equal(before,target.checkpoint())
            invalid=deepcopy(packet);invalid['manifest']['config']['cohort']='historical-uni'
            invalid['sha256']=_hash({k:v for k,v in invalid.items() if k!='sha256'})
            with pytest.raises(CheckpointError):target.restore(invalid)
            assert len(calls)==2
        finally:hook.remove()


def test_every_hook_removal_attempted_if_cleanup_fails(staged,monkeypatch):
    native,rows,config=staged
    session=product.BACKEND.open(config,model=native)
    handle=session.game.network._handles[0];original=handle.remove
    def fail():
        original();raise RuntimeError('remove failed after removal')
    monkeypatch.setattr(handle,'remove',fail)
    with pytest.raises(RuntimeError,match='remove failed'):session.close()
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_executable_cli_native_training_and_inference_after_data_removal(staged,tmp_path):
    native,rows,config=staged
    args=[sys.executable,'-m','embedded_zimage_shared.cli','train','--model-dir',config.options['model_dir'],
        '--rows',config.options['rows'],'--shape','1','4','4','4','--steps','2','--checkpoint',str(tmp_path/'cli.pt'),
        '--export',str(tmp_path/'infer.pt')]
    result=subprocess.run(args,cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['metrics']['step']==2
    (tmp_path/'rows.pt').unlink()
    torch.save(dict(condition=rows[0]['neutral'],latent=rows[0]['latent'],timestep=rows[0]['timestep']),tmp_path/'request.pt')
    result=subprocess.run([sys.executable,'-m','embedded_zimage_shared.cli','response','--model-dir',config.options['model_dir'],
        '--artifact',str(tmp_path/'infer.pt'),'--request',str(tmp_path/'request.pt'),'--strength','0.5',
        '--output',str(tmp_path/'value.pt')],cwd=tmp_path,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert torch.load(tmp_path/'value.pt',weights_only=True).shape==(1,4,4,4)


def test_partial_native_hook_attachment_failure_removes_earlier_hooks(staged,monkeypatch):
    native,rows,config=staged
    original=nn.Module.register_forward_hook
    calls=[]
    def register(module,*args,**kwargs):
        calls.append(module)
        if len(calls)==3: raise RuntimeError('third native hook rejected')
        return original(module,*args,**kwargs)
    monkeypatch.setattr(nn.Module,'register_forward_hook',register)
    with pytest.raises(RuntimeError,match='third native hook'):
        product.BACKEND.open(config,model=native)
    assert len(calls)==3
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_foreign_native_attention_processor_is_refused_before_attachment(staged):
    native,rows,config=staged
    native.layers[0].attention.processor._attention_backend='math'
    with pytest.raises(ValueError,match='default native CPU attention'):
        product.BACKEND.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_inference_rejects_changed_model_and_graph_source_before_attachment(staged,monkeypatch):
    _,_,config=staged
    with open_response(config) as session:
        session.step();packet=session.export_inference()
    calls=[]
    original=product.ZImageLoRA
    def build(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(product,'ZImageLoRA',build)
    bad=deepcopy(packet)
    bad['specification']['configuration']['model_identity']['state_sha256']='0'*64
    bad['sha256']=_hash({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError,match='identity differs'):
        open_inference(bad,options={'model_dir':config.options['model_dir']},operation='response')
    assert calls==[]
    bad=deepcopy(packet)
    bad['specification']['factory_sources']['prepare']='0'*64
    bad['sha256']=_hash({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(CheckpointError,match='factory source'):
        open_inference(bad,options={'model_dir':config.options['model_dir']},operation='response')
    assert calls==[]


def test_actual_historical_uni_safetensors_refuses_before_native_factory(staged,tmp_path,monkeypatch):
    _,_,config=staged
    result=subprocess.run([sys.executable,str(ROOT/'conceptmod/textsliders/train_lora_zimage.py'),
        '--dummy','--steps','2','--save_dir',str(tmp_path/'legacy')],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    weights=next((tmp_path/'legacy').glob('*.safetensors'))
    from safetensors import safe_open
    with safe_open(str(weights),framework='pt') as f:
        assert any('lora_down' in name for name in f.keys())
    calls=[]
    monkeypatch.setattr(product.ZImageLoRA,'__init__',lambda *args:calls.append(1))
    with pytest.raises(CheckpointError):open_response(config,resume=weights)
    assert calls==[]


@pytest.mark.parametrize('mutation',['last_row','trailing_bytes'])
def test_complete_rows_bytes_reject_before_native_attachment(staged,mutation):
    native,rows,config=staged
    path=Path(config.options['rows'])
    if mutation=='last_row':
        rows[-1]['positive']['embeds'][0,0]+=1
        torch.save(dict(schema=product.PROTOCOL,model_identity=config.options['model_identity'],rows=rows),path)
    else:
        with path.open('ab') as stream:stream.write(b'ignored-by-torch-load-but-bound-by-full-hash')
    with pytest.raises(ValueError,match='complete frozen rows file hash'):
        product.BACKEND.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_rows_changed_after_preflight_reject_before_prepare_attaches(staged):
    from dataclasses import replace
    native,rows,config=staged
    def mutate(options,request,staging):
        product.preflight(options,request,staging)
        with Path(options['rows']).open('ab') as stream:stream.write(b'changed after preflight')
    backend=replace(product.BACKEND,preflight=mutate)
    with pytest.raises(ValueError,match='complete frozen rows file hash'):
        backend.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_bundle_identity_streams_bytes_without_loading_weight_tensors(staged,monkeypatch):
    native,rows,config=staged
    import safetensors.torch
    def forbidden(*args,**kwargs):raise AssertionError('identity must not deserialize weight tensors')
    monkeypatch.setattr(safetensors.torch,'load_file',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(ZImageTransformer2DModel,'from_pretrained',forbidden)
    original=Path.open;requests=[]
    class BoundedReads:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def read(self,n=-1):
            assert 0<=n<=product.HASH_CHUNK_BYTES
            requests.append(n);return self.stream.read(n)
        def __getattr__(self,name):return getattr(self.stream,name)
    def opened(path,*args,**kwargs):
        stream=original(path,*args,**kwargs)
        return BoundedReads(stream) if path.suffix=='.safetensors' else stream
    monkeypatch.setattr(Path,'open',opened)
    identity=product.bundle_identity(config.options['model_dir'])
    assert identity==config.options['model_identity']
    assert identity['state_sha256']==product.digest_state(native.state_dict())
    assert requests and max(requests)<=product.HASH_CHUNK_BYTES


def test_streamed_safetensors_digest_covers_multiple_chunks(tmp_path):
    from safetensors.torch import save_file
    state={'large':torch.arange(700000,dtype=torch.float32),'scalar':torch.tensor(2.)}
    path=tmp_path/'large.safetensors';save_file(state,str(path))
    assert path.stat().st_size>2*product.HASH_CHUNK_BYTES
    assert product._safetensors_state_digest(path)==product.digest_state(state)


def test_recursive_core_closure_rejects_nested_unexpected_source(staged,tmp_path,monkeypatch):
    import particle_sliders
    import shutil
    root=Path(particle_sliders.__file__).parent;copy=tmp_path/'core-copy'
    for relative in product.source_closure(root):
        target=copy/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,target)
    assert product.source_closure(copy)==product.source_closure(root)
    nested=copy/'unexpected'/'nested.py';nested.parent.mkdir();nested.write_text('injected = True\n')
    assert 'unexpected/nested.py' in product.source_closure(copy)
    monkeypatch.setattr(particle_sliders,'__file__',str(copy/'__init__.py'))
    native,rows,config=staged
    with pytest.raises(ValueError,match='core lifecycle/source'):
        product.BACKEND.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_training_paths_are_fixed_but_native_only_export_relocates(staged,tmp_path,monkeypatch):
    import shutil
    _,rows,config=staged
    with open_response(config) as session:
        session.step();session.checkpoint(tmp_path/'resume.pt')
        raw=session.export_inference(selection='raw');ema=session.export_inference(selection='ema')
        expected=session.evaluate(strength=.5)
    moved_model=tmp_path/'relocated-model';moved_rows=tmp_path/'relocated-rows.pt'
    shutil.copytree(config.options['model_dir'],moved_model);shutil.copyfile(config.options['rows'],moved_rows)
    options=config.options;options.update(model_dir=str(moved_model),rows=str(moved_rows))
    moved_config=SessionConfig(product.NAME,product.COHORT,**options)
    with monkeypatch.context() as patch:
        patch.setattr(product,'_native',lambda *args,**kwargs:pytest.fail('changed training paths must reject before native prepare'))
        with pytest.raises(CheckpointError,match='config/cohort differs'):
            open_response(moved_config,resume=tmp_path/'resume.pt')
    shutil.rmtree(config.options['model_dir']);Path(config.options['rows']).unlink();moved_rows.unlink()
    request=dict(condition=rows[0]['neutral'],latent=rows[0]['latent'],timestep=rows[0]['timestep'])
    for artifact in (raw,ema):
        assert set(artifact['specification']['configuration'])=={'model_identity','response_shape','initialization'}
        with open_inference(artifact,options={'model_dir':str(moved_model)},operation='response') as inference:
            value=inference.response(**request,strength=.5)
            if artifact is raw:assert torch.equal(value,expected)
            assert value.shape==(1,4,4,4)


def test_equal_weights_reserialized_file_is_a_different_model_identity(staged):
    from safetensors.torch import load_file,save_file
    native,rows,config=staged
    path=Path(config.options['model_dir'])/'diffusion_pytorch_model.safetensors'
    state=load_file(str(path));save_file(state,str(path),metadata={'changed':'same tensor values, different exact file'})
    actual=product.bundle_identity(config.options['model_dir'])
    assert actual['state_sha256']==config.options['model_identity']['state_sha256']
    assert actual['files']!=config.options['model_identity']['files']
    with pytest.raises(ValueError,match='identity differs before attachment'):
        product.BACKEND.open(config,model=native)
    assert not any(m._forward_hooks for m in native.modules())
    assert all(p.requires_grad for p in native.parameters())


def test_native_base_loader_deserializes_once_per_session(staged,monkeypatch):
    native,rows,config=staged
    original=ZImageTransformer2DModel.from_pretrained;calls=[]
    def load(*args,**kwargs):
        calls.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(ZImageTransformer2DModel,'from_pretrained',load)
    with open_response(config) as session:
        assert session.game.response_shape==(1,4,4,4)
    assert calls==[config.options['model_dir']]


def test_seeded_initialization_is_independent_of_caller_rng_and_borrowing(staged,monkeypatch):
    native,rows,config=staged
    def no_cuda(*args,**kwargs):pytest.fail('CPU qualification must not initialize/seed/read a CUDA RNG')
    for name in ('init','_lazy_init','device_count','manual_seed','manual_seed_all','get_rng_state','set_rng_state'):
        monkeypatch.setattr(torch.cuda,name,no_cuda)
    packets=[]
    for caller_seed,borrow in ((101,False),(902,False),(305,True)):
        torch.default_generator.manual_seed(caller_seed);torch.rand(11)
        before=torch.get_rng_state().clone()
        staging={'model':deepcopy(native)} if borrow else {}
        with product.BACKEND.open(config,**staging) as session:
            packets.append(session.checkpoint())
            assert torch.equal(before,torch.get_rng_state())
            assert session.binding.inference.configuration['initialization']==product.initialization_identity(config.options['seed'])
        assert torch.equal(before,torch.get_rng_state())
    equal(packets[0],packets[1]);equal(packets[0],packets[2])
    assert _hash(packets[0])==_hash(packets[1])==_hash(packets[2])
    options=config.options;options['seed']+=1
    other_config=SessionConfig(product.NAME,product.COHORT,**options)
    with open_response(other_config) as other:
        other_packet=other.checkpoint()
    for part in ('network','critic'):
        expected=packets[0]['payload']['game'][part];different=other_packet['payload']['game'][part]
        assert any(not torch.equal(v,different[k]) for k,v in expected.items())


@pytest.mark.parametrize('component',['adapter','critic','native_loader'])
def test_direct_factory_initialization_failure_restores_caller_rng(staged,monkeypatch,component):
    native,rows,config=staged
    def no_cuda(*args,**kwargs):pytest.fail('CPU failure qualification must not touch CUDA RNG')
    for name in ('init','_lazy_init','device_count','manual_seed','manual_seed_all','get_rng_state','set_rng_state'):
        monkeypatch.setattr(torch.cuda,name,no_cuda)
    marker=RuntimeError('initialization failure at '+component)
    def fail(*args,**kwargs):torch.rand(31);raise marker
    if component=='adapter':monkeypatch.setattr(product,'ZImageLoRA',fail)
    elif component=='critic':monkeypatch.setattr(product,'HNDLFieldCritic',fail)
    else:monkeypatch.setattr(ZImageTransformer2DModel,'from_pretrained',fail)
    torch.default_generator.manual_seed(611);torch.rand(7)
    before=torch.get_rng_state().clone();modes=[m.training for m in native.modules()]
    flags=[p.requires_grad for p in native.parameters()]
    with pytest.raises(RuntimeError) as caught:
        product.prepare(config.options,**({} if component=='native_loader' else {'model':native}))
    assert caught.value is marker
    assert torch.equal(before,torch.get_rng_state())
    assert not any(m._forward_hooks for m in native.modules())
    assert [m.training for m in native.modules()]==modes
    assert [p.requires_grad for p in native.parameters()]==flags


def test_direct_factory_success_restores_rng_and_binds_policy(staged):
    native,rows,config=staged
    torch.default_generator.manual_seed(981);before=torch.get_rng_state().clone()
    binding=product.prepare(config.options,model=native)
    try:
        assert torch.equal(before,torch.get_rng_state())
        assert binding.inference.configuration['initialization']['policy']==product.INITIALIZATION_POLICY
        assert product.COHORT.endswith('-v3')
        assert product.PROTOCOL=='embedded-zimage-native-4d-raw-velocity-v2'
        assert all(item.row.guidance==1. for item in binding.game.rows)
        row=binding.game.rows[0].row
        with torch.no_grad():
            raw=binding.game.backend.velocity(row.neutral,row.latent,row.timestep,scale=0)
            actual=binding.game.backend.cfg(row,row.latent,row.timestep,scale=0)
        assert torch.equal(raw,actual)  # Native g=0 maps to shared guidance=1.
    finally:binding.close()
    assert torch.equal(before,torch.get_rng_state())


def test_changed_initialization_policy_and_previous_cohort_refuse_without_relabel(staged,tmp_path,monkeypatch):
    _,_,config=staged
    with open_response(config) as session:
        session.step();artifact=session.export_inference();checkpoint=session.checkpoint()
    bad=deepcopy(artifact);bad['specification']['configuration']['initialization']['policy']='previous-unseeded-policy'
    bad['sha256']=_hash({k:v for k,v in bad.items() if k!='sha256'})
    with monkeypatch.context() as patch:
        patch.setattr(product,'_native',lambda *args,**kwargs:pytest.fail('policy mismatch must precede attachment'))
        with pytest.raises(ValueError,match='initialization policy'):
            open_inference(bad,options={'model_dir':config.options['model_dir']},operation='response')
        checkpoint['manifest']['config']['cohort']='embedded-zimage-adapter-gan-pg094-hg-hndl010-v2'
        checkpoint['sha256']=_hash({k:v for k,v in checkpoint.items() if k!='sha256'})
        torch.save(checkpoint,tmp_path/'previous-policy.pt')
        with pytest.raises(CheckpointError,match='config/cohort differs'):
            open_response(config,resume=tmp_path/'previous-policy.pt')



def test_direct_inference_factory_preserves_rng_on_success_and_initialization_failure(staged,monkeypatch):
    from particle_sliders.inference import InferenceSpec
    _,rows,config=staged
    with open_response(config) as session:
        session.step();artifact=session.export_inference()
    spec=InferenceSpec(artifact['specification']);options={'model_dir':config.options['model_dir']}
    torch.default_generator.manual_seed(314);before=torch.get_rng_state().clone()
    binding=product.prepare_inference(options,spec)
    try:assert torch.equal(before,torch.get_rng_state())
    finally:binding.close()
    marker=RuntimeError('inference adapter initialization failure')
    def fail(*args,**kwargs):torch.rand(17);raise marker
    monkeypatch.setattr(product,'ZImageLoRA',fail)
    with pytest.raises(RuntimeError) as caught:product.prepare_inference(options,spec)
    assert caught.value is marker
    assert torch.equal(before,torch.get_rng_state())
