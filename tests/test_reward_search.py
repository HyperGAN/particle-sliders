import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from conceptmod.textsliders.reward_search import queue
from conceptmod.textsliders.reward_search.data import prepare_window_rows,WindowStudentForward,slices
from conceptmod.textsliders.reward_sliders.specs import sha,save_tensor,write_json,read_json
from conceptmod.textsliders.reward_sliders.directions import RewardTeacher
from conceptmod.textsliders.reward_sliders.data import generation_hidden
from conceptmod.textsliders.reward_sliders.capture import ResidualCapture


def test_atomic_bundle_claims_and_immutable_reenqueue(tmp_path):
    jobs=[dict(family={'family':str(i)},seed=i,arms=[{'name':'off'},{'name':'on'}]) for i in range(12)]
    queue.add(tmp_path,'screen',jobs)
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed=list(pool.map(lambda gpu:queue.claim(tmp_path,gpu),[0,1]*6))
    assert len({j['id'] for j in claimed})==12
    assert all(len(j['payload']['arms'])==2 for j in claimed)
    assert queue.claim(tmp_path,0) is None
    for job in claimed:queue.finish(tmp_path,job['id'])
    queue.add(tmp_path,'screen',jobs)
    assert queue.counts(tmp_path)=={'complete':12}
    changed=[dict(jobs[0],arms=[{'name':'different'}])]
    with pytest.raises(ValueError,match='changed'):queue.add(tmp_path,'screen',changed)


@pytest.fixture
def window_case(tmp_path,monkeypatch):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    import conceptmod.textsliders.reward_search.data as data
    torch.manual_seed(13);torch.set_num_threads(2)
    lm=Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=64)).eval().requires_grad_(False)
    trajectory=dict(prompt_embeds=torch.randn(2,3,16),frame_embeds=torch.randn(2,6,16))
    path=tmp_path/'trajectory.pt';save_tensor(path,trajectory)
    observation=dict(id='train-example',family='train',split='train',status='complete',
                     seed=7,trajectory=str(path),trajectory_sha256=sha(path))
    teacher=RewardTeacher(layer=1,coefficient=.03,residual_norm=1.,direction=torch.ones(16)/4,
                          center=torch.zeros(16),fitting_families=['train'])
    # A small-vocabulary numerical test uses one head column for the EOS diagnostic.
    monkeypatch.setattr(data,'end_margins',lambda lm,h: h.float()@lm.lm_head.weight[0].float())
    rows=prepare_window_rows(lm,[observation],teacher,lambda _:None,device='cpu',windows=[(0,4),(2,6)],stride=2)
    return lm,trajectory,observation,teacher,rows


def test_later_windows_keep_prefix_and_align_both_cfg_branches(window_case):
    lm,trajectory,_,teacher,rows=window_case
    assert len(rows)==4
    for row in rows:
        branch=row['branch'];stop=row['window_stop']
        expected=torch.cat([trajectory['prompt_embeds'][branch:branch+1],
                            trajectory['frame_embeds'][branch:branch+1,:stop]],1)
        assert torch.equal(expected,row['embeds'])
        span,end=slices(row)
        with torch.no_grad():
            baseline=generation_hidden(lm,expected)
            with ResidualCapture(lm,capture=False,prompt_length=3,**teacher.hook_kwargs()):
                positive=generation_hidden(lm,expected)
        assert torch.equal(baseline[:,span],row['neutral_span'])
        assert torch.equal(positive[:,span]-baseline[:,span],row['real'])
        assert torch.equal(positive[:,:3],baseline[:,:3])
        assert row['neutral_span'].shape==(1,2,16)
        assert row['end_teacher'].shape==(1,2)
    late=rows[-1]
    assert late['embeds'].shape[1]==9 and late['span_start']==5
    altered=late['embeds'].clone();altered[:,3:5]*=-1
    with torch.no_grad():changed=generation_hidden(lm,altered)[:,slices(late)[0]]
    assert not torch.equal(changed,late['neutral_span'])


def test_window_student_zero_and_real_gan_update_leave_base_frozen(window_case):
    from app.lora_runtime import LoRANetwork
    from conceptmod.textsliders.gan_v2.critic import SpanCritic,pad_sequences
    from conceptmod.textsliders.gan_v2.engine import GANEngine
    from conceptmod.textsliders.gan_v2.train import arm_settings
    lm,_,_,_,rows=window_case
    pristine={name:p.detach().clone() for name,p in lm.named_parameters()}
    network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=0.,target_replace=['Qwen3Attention'],
                       prefix='lora_te',delimiter='-',train_method='full').requires_grad_(True)
    forward=WindowStudentForward(lm,network,lambda _:None,'cpu')
    for row in rows:assert torch.count_nonzero(forward(row)['fake'])==0
    recipe,_=arm_settings('baseline',origin=0,horizon=2,diagnostics_every=0)
    critic=SpanCritic(16,width=16,layers=1,heads=2,conditioned=False,ordered=False,normalized_features=False)
    real,mask=pad_sequences([r['real'] for r in rows]);critic.calibrate_input_scale(real,mask)
    engine=GANEngine(network,critic,forward,rows,replace(recipe,g_lr=.0001,d_lr=.0001))
    engine.update([0,1,2,3])
    assert engine.completed==1
    assert any(torch.count_nonzero(m.lora_up.weight) for m in network.unet_loras)
    assert all(torch.equal(pristine[name],p) for name,p in lm.named_parameters())
    assert all(torch.isfinite(p).all() for p in network.parameters())


def test_short_trajectory_rejects_late_window(window_case):
    lm,_,observation,teacher,_=window_case
    with pytest.raises(ValueError,match='geometry'):
        prepare_window_rows(lm,[observation],teacher,lambda _:None,device='cpu',windows=[(4,8)],stride=2)


def test_render_worker_passes_one_component_to_legacy_host(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_search import renderer
    checkpoint=tmp_path/'weights.safetensors';checkpoint.write_bytes(b'identity-only fixture')
    metadata=dict(kind='language_model',rank=8,alpha=8,target_replace=['Qwen3Attention'],
                  prefix='lora_te',delimiter='-',train_method='full',unit_scale=1.,weights_sha256=sha(checkpoint))
    write_json(checkpoint.with_suffix('.json'),metadata)
    write_json(tmp_path/'manifest.json',dict(families=[{'family':'x','split':'train'}],reward_spec={}))
    treatment=dict(name='on',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),coefficient=.75)
    queue.add(tmp_path,'screen',[dict(family={'family':'x'},seed=7,arms=[treatment])])
    seen=[]
    class FakeRenderer:
        def __init__(self,*args):
            self.host=SimpleNamespace(_merge_sliders=lambda *args:None,
                                      _merge_state=lambda *args:SimpleNamespace(pristine={}))
            self.pipe=None;self.device='cpu'
        def parity(self,*args):pass
        def observe(self,family,seed,arm,scorer,**kwargs):
            extra=kwargs['extra'];assert isinstance(extra,dict)
            assert extra['multiplier']==.75 and extra['weights']==str(checkpoint)
            seen.append(extra);(tmp_path/'stop-gpu1').touch()
            return dict(status='complete',id='test',error=None)
    monkeypatch.setattr(renderer,'SearchRenderer',FakeRenderer)
    monkeypatch.setattr(renderer,'verify',lambda _:None)
    renderer.worker(tmp_path,1)
    assert len(seen)==1 and queue.counts(tmp_path)=={'complete':1}


@pytest.mark.parametrize('disappears',[False,True])
def test_restart_transient_worker_without_discarding_its_definition(tmp_path,monkeypatch,disappears):
    import subprocess
    from conceptmod.textsliders.reward_search import resources
    calls=[]
    monkeypatch.setattr(resources,'active',lambda _:False)
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *args,**kwargs:'loaded\n')
    def run(command,**kwargs):
        calls.append(command)
        if command[:3]==['systemctl','--user','start'] and disappears:
            return subprocess.CompletedProcess(command,5,'','Unit not found.')
        return subprocess.CompletedProcess(command,0,'','')
    monkeypatch.setattr(resources.subprocess,'run',run)
    resources.start_worker(tmp_path,1)
    assert not any('reset-failed' in command for command in calls)
    recreated=[command for command in calls if command[0]=='systemd-run']
    assert bool(recreated)==disappears
    if recreated:assert '--setenv=CUDA_VISIBLE_DEVICES=1' in recreated[0]


def test_studio_handoff_uses_restart_and_restores_settings(tmp_path,monkeypatch):
    from conceptmod.textsliders.reward_search import resources
    override=tmp_path/'config/override.conf';calls=[];restored=[];phase={'borrowed':False}
    settings=dict(enabled=False,prompt='A close dry drum kit',duration=120.,sliders=[],
                  energy={'language_model':2.},randomize_sliders=False)
    def studio():
        return dict(active=[],queued=[],keep=dict(settings),
                    workers=0 if phase['borrowed'] else 1,loaded=[] if phase['borrowed'] else ['cuda:0'])
    def command(args,**kwargs):
        calls.append(args)
        if args==['systemctl','--user','restart','music-studio.service']:
            phase['borrowed']=override.exists()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(resources,'OVERRIDE',override)
    monkeypatch.setattr(resources,'studio',studio);monkeypatch.setattr(resources,'wait_studio',studio)
    monkeypatch.setattr(resources,'restore_keep',lambda value:restored.append(value))
    monkeypatch.setattr(resources.subprocess,'run',command)
    monkeypatch.setattr(resources,'start_worker',lambda run,gpu:(Path(run)/'experiment-done').touch())
    monkeypatch.setattr(resources,'stop_worker',lambda *args:None)
    monkeypatch.setattr(resources.time,'sleep',lambda _:None)
    resources.monitor(tmp_path)
    assert sum(args==['systemctl','--user','restart','music-studio.service'] for args in calls)==2
    assert not any(args[:3] in (['systemctl','--user','start'],['systemctl','--user','stop']) for args in calls)
    assert restored==[settings,settings] and not override.exists()
    assert read_json(tmp_path/'audit/studio-restored.json')['workers']==1


@pytest.mark.parametrize('busy',['queued','keep'])
def test_studio_is_not_taken_while_work_remains(tmp_path,monkeypatch,busy):
    from conceptmod.textsliders.reward_search import resources
    class StopObservation(Exception):pass
    monkeypatch.setattr(resources,'OVERRIDE',tmp_path/'override.conf')
    monkeypatch.setattr(resources,'studio',lambda:dict(active=[],queued=['job'] if busy=='queued' else [],
                                                      keep={'enabled':busy=='keep'}))
    monkeypatch.setattr(resources.subprocess,'run',lambda *args,**kwargs:pytest.fail('Studio was changed before it drained'))
    def stop(_):raise StopObservation()
    monkeypatch.setattr(resources.time,'sleep',stop)
    with pytest.raises(StopObservation):resources.monitor(tmp_path)
    assert not (tmp_path/'override.conf').exists()
