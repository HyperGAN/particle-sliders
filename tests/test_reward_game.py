"""Scientific accounting, cache and recovery tests without audio generation."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from conceptmod.textsliders.reward_game.core import (
    Store, BusyError, IntegrityError, read, write, digest, sha, scorecard,
    generation_identity, reference_summary, DEFAULT_SPEC, candidate_identity)
from conceptmod.textsliders.reward_game import evaluate as evaluator
from conceptmod.textsliders.reward_game import controller


@pytest.fixture
def game():return read(DEFAULT_SPEC)


def candidate(value='weights',multiplier=1.):
    return dict(path=None,weights_sha256=value,structure={'rank':8,'alpha':8},multiplier=multiplier)


def observations(game,deltas):
    return {c['id']:dict(status='complete',reward=dict(valid=True,scalar=c['controls']['off']['ce']+d)) for c,d in zip(game['cases'],deltas)}


def test_prepared_reference_cards_reproduce_exactly(game):
    saved=read(DEFAULT_SPEC.parent/'reference-scorecards.json')
    assert digest(game)==saved['benchmark_sha256']
    for n,ids in game['stages'].items():
        cases=[next(c for c in game['cases'] if c['id']==ident) for ident in ids]
        for label in game['controls']:assert reference_summary(cases,label)==saved['stages'][n][label]


def test_stage_gates_ties_losses_and_invalid_results(game):
    assert scorecard(game,observations(game,[.2,.2,.2,-.1]),4)['advance']
    assert not scorecard(game,observations(game,[1.,1.,1.,-.51]),4)['advance']
    assert not scorecard(game,observations(game,[1.,1.,0.,0.]),4)['advance']
    rows=observations(game,[.3]*16);rows[game['cases'][0]['id']]['status']='invalid_output'
    card=scorecard(game,rows,16)
    assert not card['advance'] and card['valid_cases']==15 and len(card['invalid'])==1
    assert not scorecard(game,observations(game,[.3]*4),8)['advance']


def test_family_weight_is_not_window_or_case_weight(game):
    g=deepcopy(game)
    g['cases'][1]['family']['family']=g['cases'][0]['family']['family']
    c=scorecard(g,observations(g,[.1,.3,.6,.8]),4)
    assert c['comparisons']['off']['equal_family_gain']==pytest.approx((.2+.6+.8)/3)


def test_render_identity_changes_only_on_content_and_actual_settings(game):
    m=read(game['source_manifest']);case=game['cases'][0];a=candidate()
    key=digest(generation_identity(game,m,case,a))
    assert digest(generation_identity(game,m,case,dict(a,path='/elsewhere',tag='different')))==key
    for changed in (dict(a,weights_sha256='different'),dict(a,multiplier=.5),dict(a,structure={'rank':16})):
        assert digest(generation_identity(game,m,case,changed))!=key
    assert digest(generation_identity(game,m,dict(case,physical_gpu=0),a))!=key
    assert digest(generation_identity(dict(game,duration_seconds=21),m,case,a))!=key
    styled=next(c for c in game['cases'] if len(c['style_components'])==2)
    with pytest.raises(IntegrityError,match='host energy'):
        generation_identity(game,m,styled,candidate(multiplier=3.))


def test_atomic_double_claim_and_original_failure_retained(tmp_path):
    store=Store(tmp_path);identity={'test':1};key=digest(identity);entered=threading.Event();release=threading.Event()
    def owner():
        with store.claim(key) as folder:
            entered.set();release.wait(5)
            write(folder/'observation.json',dict(key=key,identity=identity,status='interrupted'))
    with ThreadPoolExecutor(1) as pool:
        f=pool.submit(owner);entered.wait(5)
        try:
            with pytest.raises(BusyError):
                with store.claim(key):pass
        finally:release.set()
        f.result()
    with pytest.raises(BusyError):
        with store.claim(key):pass
    assert store.observation(key)['status']=='interrupted'


def test_raw_audio_corruption_and_independent_scoring_cache(tmp_path):
    store=Store(tmp_path);audio=tmp_path/'sample.wav';audio.write_bytes(b'original PCM')
    identity={'render':1};key=digest(identity);row=dict(key=key,identity=identity,status='complete',audio=str(audio),audio_sha256=sha(audio))
    write(tmp_path/'renders'/key/'observation.json',row)
    store.save_score(dict(valid=True,scalar=6.,audio_sha256=sha(audio)),'reward-v1')
    assert store.score(row,'reward-v1')['scalar']==6
    assert store.score(row,'reward-v2') is None
    audio.write_bytes(b'changed PCM')
    with pytest.raises(IntegrityError,match='Corrupted'):store.observation(key)


@pytest.fixture
def evaluation(tmp_path,game,monkeypatch):
    m=read(game['source_manifest']);current=candidate();counts=[]
    monkeypatch.setattr(evaluator,'verify',lambda home:(game,m))
    monkeypatch.setattr(evaluator,'checkpoint',lambda p,s:dict(current,multiplier=s))
    monkeypatch.setattr(evaluator,'leaderboard',lambda home:None)
    def runner(home,jobs):
        store=Store(home)
        for job in jobs:
            counts.append(job['case']['id']);identity=generation_identity(game,m,job['case'],job['candidate'])
            with store.claim(job['key']) as folder:
                path=folder/'audio.wav';path.write_bytes(b'simulated-audio')
                row=dict(key=job['key'],identity=identity,status='complete',audio=str(path),audio_sha256=sha(path),
                         reward=dict(valid=True,scalar=job['case']['controls']['off']['ce']-.1))
                write(folder/'observation.json',row)
    return tmp_path,game,m,current,counts,runner


def test_repeating_evaluation_is_identical_and_does_not_render(evaluation):
    home,game,m,candidate,counts,runner=evaluation
    a=evaluator.evaluate(home,None,1.,runner=runner)
    b=evaluator.evaluate(home,None,1.,tag='new name',runner=runner)
    assert a==b and len(counts)==4 and a['early_rejected'] and a['new_clips']==4
    c=evaluator.evaluate(home,None,.5,runner=runner)
    assert c['run_id']!=a['run_id'] and len(counts)==8
    d=evaluator.evaluate(home,None,1.,stage='16',runner=runner)
    assert d['stage']==16 and d['forced_stage'] and d['cache_hits']==4 and len(counts)==20


def test_partial_resume_runs_only_unattempted_cases(evaluation):
    home,game,m,candidate,counts,runner=evaluation
    def interrupted(home,jobs):
        runner(home,jobs[:2]);raise RuntimeError('worker stopped before case 3')
    a=evaluator.evaluate(home,None,1.,runner=interrupted)
    assert a['decision']=='engineering_failure' and len(counts)==2
    b=evaluator.evaluate(home,None,1.,runner=runner)
    assert b['decision']=='rejected' and len(counts)==4 and len(set(counts))==4


def test_inspect_running_evaluation_reports_progress_without_rejection(evaluation):
    home,game,m,candidate,counts,runner=evaluation
    def inspect_then_render(home,jobs):
        run_id=next((home/'evaluations').glob('eval-*')).name
        result=evaluator.inspect(home,run_id)
        assert result['decision']=='in_progress' and result['valid_cases']==0 and result['stage']==4
        runner(home,jobs)
    result=evaluator.evaluate(home,None,1.,runner=inspect_then_render)
    assert result['decision']=='rejected' and len(counts)==4


def test_recorded_engineering_failure_does_not_reroll(evaluation):
    home,game,m,candidate,counts,runner=evaluation
    def broken(home,jobs):
        runner(home,jobs)
        path=home/'renders'/jobs[0]['key']/'observation.json';row=read(path);row['status']='engineering_error';write(path,row)
    a=evaluator.evaluate(home,None,1.,runner=broken)
    b=evaluator.evaluate(home,None,1.,runner=runner)
    assert a['decision']==b['decision']=='engineering_failure' and len(counts)==4


def test_rejected_candidate_advances_controller_to_next_attempt(tmp_path):
    for index in range(2):
        recipe=dict(hypothesis=f'bounded mechanism {index}',failure_mechanism=f'mechanism {index}',method='checkpoint',
                    parent_checkpoint=None,changed_variables={'mechanism':index},budget={'new_clips':4},sources={},checkpoint=f'candidate-{index}')
        path=tmp_path/f'recipe-{index}.json';write(path,recipe);controller.register(tmp_path,path)
    calls=[]
    def reject(home,path,*args):
        calls.append(path)
        return dict(decision='rejected',advance=False,stage=4,run_id=path,render_keys=[],new_clips=0)
    result=controller.search(tmp_path,evaluator=reject)
    assert len(calls)==2 and result['state']=='method_selection_required' and not result['research_complete']
    controller.search(tmp_path,evaluator=reject)
    assert len(calls)==2


def test_merged_forward_matches_deployment_and_updates_only_adapter():
    import sys
    import torch
    from pathlib import Path
    from transformers import Qwen3Config,Qwen3ForCausalLM
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from app.lora_runtime import LoRANetwork
    from conceptmod.textsliders.reward_game.merged_forward import MergedForward
    from conceptmod.textsliders.reward_sliders.data import generation_hidden
    torch.manual_seed(71);torch.set_num_threads(2)
    lm=Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=32)).to(torch.bfloat16).eval().requires_grad_(False)
    def network():return LoRANetwork(lm,rank=8,alpha=8.,multiplier=1.,target_replace=['Qwen3Attention'],prefix='lora_te',delimiter='-',train_method='full',attach=False)
    reward=network();style=network()
    with torch.no_grad():
        for n in (reward,style):
            for module in n.unet_loras:module.lora_up.weight.normal_(0,.03)
    wrapper=MergedForward(reward);wrapper.styles([(style,.7)]);wrapper.attach()
    embeds=torch.randn(2,8,16,dtype=torch.bfloat16)
    with torch.no_grad():wrapped=generation_hidden(lm,embeds)
    wrapper.detach()

    with torch.no_grad():
        for r,s in zip(reward.unet_loras,style.unet_loras):
            host=reward.hosts[r.lora_name]
            style_delta=(s.lora_up.weight.float()@s.lora_down.weight.float())*.7
            reward_delta=r.lora_up.weight.float()@r.lora_down.weight.float()
            host.weight.copy_((wrapper.pristine[r.lora_name].float()+(style_delta+reward_delta)).to(host.weight.dtype))
        actual=generation_hidden(lm,embeds)
        assert torch.equal(wrapped,actual)
        for name,base in wrapper.pristine.items():reward.hosts[name].weight.copy_(base)
    wrapper.attach();optimizer=torch.optim.AdamW(reward.parameters(),lr=.001)
    before={k:v.clone() for k,v in reward.state_dict().items()}
    loss=generation_hidden(lm,embeds).float().square().mean();loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in reward.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in lm.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k],v) for k,v in reward.state_dict().items()) and wrapper.base_unchanged()
    wrapper.detach()


def test_gpu_lease_restores_settings_on_worker_error(tmp_path,monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from conceptmod.textsliders.reward_game import resources
    from conceptmod.textsliders.reward_game.core import locked as real_lock
    monkeypatch.setattr(resources,'locked',lambda path:real_lock(tmp_path/'gpu.lock'))
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    state=dict(active=[],queued=[],loaded=[],workers=1,keep=dict(enabled=False,prompt='dry drums',duration=60,
                                                               sliders=[],energy=None,randomize_sliders=False))
    calls=[];settings=[]
    monkeypatch.setattr(resources,'studio',lambda:state)
    monkeypatch.setattr(resources,'wait_studio',lambda:state)
    monkeypatch.setattr(resources,'systemctl',lambda *args:(calls.append(args) or SimpleNamespace(stdout='original unit')))
    monkeypatch.setattr(resources,'restore_keep',lambda value:settings.append(value))
    with pytest.raises(RuntimeError,match='worker'):
        with resources.gpu_lease(tmp_path/'ledger',0):raise RuntimeError('worker stopped')
    assert calls.count(('restart','music-studio.service'))==2
    assert len(settings)==2 and settings[0]['prompt']==settings[1]['prompt']=='dry drums'
    assert not (tmp_path/'.config/systemd/user/music-studio.service.d/30-reward-game-lease.conf').exists()


def test_gpu_lease_refuses_other_owner(tmp_path,monkeypatch):
    from pathlib import Path
    from conceptmod.textsliders.reward_game import resources
    from conceptmod.textsliders.reward_game.core import locked as real_lock
    monkeypatch.setattr(resources,'locked',lambda path:real_lock(tmp_path/'gpu.lock'))
    monkeypatch.setattr(Path,'home',classmethod(lambda cls:tmp_path))
    path=tmp_path/'.config/systemd/user/music-studio.service.d/30-reward-game-lease.conf'
    path.parent.mkdir(parents=True);path.write_text('another owner')
    with pytest.raises(BusyError):
        with resources.gpu_lease(tmp_path/'ledger',0):pass
    assert path.read_text()=='another owner'


def test_depth_gain_controls_and_negative_delta_penalty():
    import torch
    from conceptmod.textsliders.reward_game.gains import fold,training_objective
    weights={f'lora_te-model-layers-{layer}-self_attn-q_proj.lora_up.weight':torch.ones(4,8) for layer in (0,12,24)}
    weights['unchanged.alpha']=torch.tensor(8.)
    assert all(torch.equal(v,weights[k]) for k,v in fold(weights,[1.,1.,1.]).items())
    zero=fold(weights,[0.,0.,0.]);assert all(torch.count_nonzero(v)==0 for k,v in zero.items() if 'lora_up' in k)
    assert torch.equal(zero['unchanged.alpha'],weights['unchanged.alpha'])
    values=fold(weights,[.35,.5,.65]);assert float(values[next(iter(weights))][0,0])==pytest.approx(.35)
    assert training_objective([.1,.2,-.2,0.])==pytest.approx(-.025)
    with pytest.raises(ValueError):fold(weights,[1.1,0.,0.])


def test_direct_selection_keeps_off_eligible_and_ignores_invalid_proposals():
    from conceptmod.textsliders.reward_game.direct_search import choose
    negative=dict(index=0,valid=True,objective=-.01,gains=[.35,.5,.65])
    invalid=dict(index=1,valid=False,objective=10.,gains=[.65,.5,.35])
    assert choose([negative,invalid]) is None
    tie=dict(negative,objective=0.)
    assert choose([tie]) is None
    positive=dict(negative,objective=.05)
    assert choose([positive,invalid])==positive


def test_training_selection_rejection_skips_development_and_advances(tmp_path,monkeypatch):
    from types import SimpleNamespace
    method=tmp_path/'method';method.mkdir()
    selection=method/'selection.json';write(selection,dict(selected_off=True,actual_updates=0,new_clips=8))
    write(method/'audio-observations.json',[]);write(method/'generation-jobs.json',[])
    status=method/'status.json';write(status,dict(state='no_candidate_selected',actual_updates=0,selection_sha256=sha(selection)))
    recipe=dict(hypothesis='direct score pilot',failure_mechanism='surrogate mismatch',method='command',parent_checkpoint=None,
                changed_variables={'objective':'direct'},budget={'new_clips':16,'max_seconds':5},sources={},
                checkpoint=str(method/'absent.safetensors'),argv=['declared-command'],allow_no_candidate=True,
                training_status=str(status),selection_result=str(selection),expected_updates=0)
    path=tmp_path/'recipe.json';write(path,recipe);registration=controller.register(tmp_path,path)
    monkeypatch.setattr(controller,'_command',lambda *args,**kwargs:SimpleNamespace(returncode=0))
    def never(*args,**kwargs):raise AssertionError('Off selection must not schedule development')
    result=controller.search(tmp_path,evaluator=never)
    assert result['state']=='method_selection_required'
    folder=tmp_path/'attempts'/registration['attempt_id']
    assert read(folder/'decision.json')['decision']=='rejected_training_selection'
    assert read(folder/'training-state.json')['local_optimizer_updates']==0
    assert not (folder/'scorecard.json').exists()
    assert controller.search(tmp_path,evaluator=never)['completed']==[]


def test_command_timeout_terminates_the_owned_process_group(monkeypatch):
    import subprocess
    import signal
    calls=[]
    class Process:
        pid=12345
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if len(calls)==1:raise subprocess.TimeoutExpired(['experiment'],timeout)
            return 0
    monkeypatch.setattr(controller.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(controller.os,'killpg',lambda pid,sig:calls.append(('signal',pid,sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        controller._command(dict(argv=['experiment'],budget={'max_seconds':1}),{},None)
    assert ('signal',12345,signal.SIGTERM) in calls
    assert calls[-1]==('wait',120)
