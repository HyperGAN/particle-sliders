import json

import pytest
import torch

from analysis.gan_bcap.autonomous_audio import score_components, summarize, coverage_weights
from analysis.gan_bcap.catalog_train import bounded_training, lora_module
from analysis.gan_bcap.autonomous_campaign import search_progress


def test_score_is_relative_to_control_and_monotone_in_each_component():
    baseline=dict(concept=.2,enjoyment=7.,production=8.,lyrics=.8)
    assert score_components(baseline,baseline)[0]==0
    for key in baseline:
        changed={**baseline,key:baseline[key]+.1}
        assert score_components(changed,baseline)[0]>0


def test_excess_high_frequency_noise_penalty_allows_reference_brightness():
    base=dict(concept=.2,enjoyment=7.,production=8.,lyrics=.8,hf14k_fraction=.001)
    noisy={**base,'hf14k_fraction':.021}
    assert score_components(noisy,base)[0]<0
    bright_reference={**base,'hf14k_fraction':.022}
    assert score_components(noisy,base,bright_reference)[0]==pytest.approx(0)


def test_overlapping_tail_windows_do_not_double_count_the_ending():
    weights=coverage_weights([dict(start_s=0,end_s=10),dict(start_s=10,end_s=20),dict(start_s=10.02,end_s=20.02)])
    assert sum(weights)==pytest.approx(1)
    assert weights[0]==pytest.approx(10/20.02)
    assert sum(weights[1:])==pytest.approx(10.02/20.02)


def test_ranking_rejects_missing_or_duplicate_fixtures():
    def row(path,fixture):
        return dict(checkpoint=dict(path=path,steps=600),fixture=fixture,
                    heuristic_score=1.,eligible=True,
                    components=dict(concept=0,enjoyment=0,production=0,lyrics=0))
    with pytest.raises(ValueError,match='same prompt/seed'):
        summarize([row('a','x'),row('a','y'),row('b','x')])
    with pytest.raises(ValueError,match='twice'):
        summarize([row('a','x'),row('a','x')])


def test_fresh_bound_marks_lora_only_and_restores_hooks(tmp_path,monkeypatch):
    monkeypatch.setattr(lora_module,'LoRANetwork',lambda:torch.nn.Linear(2,1,bias=False))
    factory=lora_module.LoRANetwork
    other=torch.nn.Parameter(torch.zeros(2))
    other_opt=torch.optim.AdamW([other],lr=.5)
    original_step=torch.optim.AdamW.step
    with bounded_training(.01,tmp_path/'steps.jsonl'):
        network=lora_module.LoRANetwork()
        opt=torch.optim.AdamW(network.parameters(),lr=.5)
        before=[p.detach().clone() for p in network.parameters()]
        for p in network.parameters():p.grad=torch.ones_like(p)
        opt.step()
        other.grad=torch.ones_like(other);other_opt.step()
        distance=sum((p-b).double().square().sum() for p,b in zip(network.parameters(),before)).sqrt()
        assert float(distance.detach())==pytest.approx(.01,abs=3e-8)
        assert float(other.detach().norm())>.5
    assert lora_module.LoRANetwork is factory
    assert torch.optim.AdamW.step is original_step
    records=[json.loads(l) for l in (tmp_path/'steps.jsonl').read_text().splitlines()]
    assert len(records)==1 and records[0]['step']==1
    assert records[0]['factor']<1


def test_catalog_bound_exact_resume_and_changed_bound_rejection(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from analysis.gan_bcap.catalog_train import game
    monkeypatch.setattr(lora_module,'LoRANetwork',lambda:torch.nn.Linear(2,1,bias=False))
    args=SimpleNamespace(lr=.2)

    def run(label,count,resume=None,maximum=.02):
        torch.manual_seed(7)
        with bounded_training(maximum,tmp_path/f'{label}.jsonl'):
            net=lora_module.LoRANetwork();opt=torch.optim.AdamW(net.parameters(),lr=.2)
            signature=game.signature(args,[],{})
            history=[] if resume is None else game.restore(resume,run_signature=signature,
                modules={'lora':net},optimizers={'lora':opt})
            for step in range(len(history),count):
                opt.zero_grad();net(torch.tensor([[1.,2.]])).square().sum().backward();opt.step()
                history.append({'step':step+1})
            game.save(tmp_path/f'{label}.pt',run_signature=signature,modules={'lora':net},
                optimizers={'lora':opt},history=history)
            return {k:v.detach().clone() for k,v in net.state_dict().items()}

    full=run('full',3)
    run('prefix',1)
    resumed=run('resumed',3,tmp_path/'prefix.pt')
    assert all(torch.equal(v,resumed[k]) for k,v in full.items())
    records=[json.loads(l) for l in (tmp_path/'resumed.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==[2,3]
    with pytest.raises(ValueError,match='different training settings'):
        run('wrong',3,tmp_path/'prefix.pt',maximum=.03)


def test_plateau_reconstruction_keeps_improvement_and_rejects_silent_high_score():
    initial=[{'score':v} for v in [.1,.2,.3]]
    history=initial+[{'score':.4},{'score':.39},{'score':3.,'eligible':False}]
    assert search_progress(history)==pytest.approx((.4,2))
    assert search_progress(history+[{'score':.5}])==pytest.approx((.5,0))


def test_interrupted_stage_resumes_saved_state_in_new_directory(tmp_path):
    from analysis.gan_bcap.autonomous_campaign import Campaign
    campaign=Campaign.__new__(Campaign)
    campaign.state={};campaign.save=lambda:None;campaign.event=lambda message:None
    partial=tmp_path/'trial';partial.mkdir()
    (partial/'trial_train.jsonl').write_text('{"step": 600}\n')
    source=partial/'trial_state.pt';torch.save({'completed_updates':600},source)
    calls=[]
    def run(command,log):
        name,resume=command;calls.append((name,resume))
        folder=tmp_path/name;folder.mkdir()
        (folder/f'{name}_last.safetensors').write_bytes(b'fixture')
        (folder/f'{name}_last.json').write_text('{"steps": 900}')
        torch.save({'completed_updates':900},folder/f'{name}_state.pt')
    campaign.run=run
    weights,state=campaign.train_stage('job','trial',tmp_path,900,lambda name,resume:[name,str(resume)])
    assert calls==[('trial-resume1',str(source))]
    assert weights.parent.name=='trial-resume1'
    assert source.exists() and state.exists()
    again=campaign.train_stage('job','trial',tmp_path,900,lambda name,resume:[name,str(resume)])
    assert again==(weights,state) and len(calls)==1
