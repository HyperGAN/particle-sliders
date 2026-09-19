"""Check recovery and composition before committing the GPU to the catalog."""
from contextlib import ExitStack
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from safetensors.torch import save_file

from analysis.uni16_20260906 import train as campaign
from conceptmod.textsliders.lora import LoRANetwork


def test_catalog_labels_work_with_native_renderer_and_flat_score_discovery():
    from types import SimpleNamespace
    import yaml
    from conceptmod.textsliders.generate_listen import _jobs
    catalog=campaign.read(campaign.WORK/'catalog.json')
    folder=Path('/tmp/uni16-label-check')
    for item in catalog['sliders']:
        data=yaml.safe_load(Path(item['train_prompts']).read_text())
        options=SimpleNamespace(out_dir=folder,scales='0,1',plus_label=data['plus_label'],minus_label='Off')
        jobs=_jobs(options,data['rows'][0])
        assert all(path.parent==folder for path,_,_ in jobs), item['id']
        assert len({path for path,_,_ in jobs})==len(jobs)
        assert jobs[1][0].match('02_slider_*_plus1.wav')


def test_recover_complete_warmup_without_retraining(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign,'verify_frozen',lambda:None)
    run=tmp_path/'female-warmup600-a01';run.mkdir()
    weights=run/f'{run.name}_last.safetensors';weights.touch()
    state=run/f'{run.name}_state.pt';torch.save({'completed_updates':600},state)
    driver=object.__new__(campaign.Campaign)
    driver.status={'sliders':{'female':{'warmup':{'status':'running','attempt':1,'directory':str(run)}}}}
    monkeypatch.setattr(driver,'save',lambda *args:None)
    monkeypatch.setattr(driver,'command',lambda *args:pytest.fail('Recovered stage must not launch training'))
    monkeypatch.setattr(campaign,'audit_checkpoint',lambda *args:{'weights_sha256':'checked'})
    assert driver.stage({'id':'female'},'warmup')==(weights,state)
    assert driver.status['sliders']['female']['warmup']['status']=='complete'


def test_partial_warmup_resumes_own_state_in_fresh_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(campaign,'verify_frozen',lambda:None)
    run=tmp_path/'female-warmup600-a01';run.mkdir()
    state=run/f'{run.name}_state.pt';torch.save({'completed_updates':450},state)
    driver=object.__new__(campaign.Campaign)
    driver.status={'sliders':{'female':{'warmup':{'status':'running','attempt':1,'directory':str(run)}}}}
    monkeypatch.setattr(campaign,'MODELS',tmp_path)
    monkeypatch.setattr(driver,'save',lambda *args:None)
    commands=[]
    monkeypatch.setattr(driver,'command',lambda args,*rest:commands.append(args))
    monkeypatch.setattr(campaign,'audit_checkpoint',lambda *args:{'weights_sha256':'checked'})
    weights,_=driver.stage({'id':'female','label':'Female','train_prompts':'new.yaml'},'warmup')
    command=commands[0]
    assert command[command.index('--resume_state')+1]==str(state)
    assert weights.parent!=run and weights.parent.name.endswith('a02')


@pytest.mark.parametrize('bad', ['different','nonfinite'])
def test_reject_inference_state_mismatch_and_nonfinite_weights(tmp_path,bad):
    state=tmp_path/'state.pt';weights=tmp_path/'weights.safetensors'
    value=torch.ones(2)
    exported=torch.full((2,),2. if bad=='different' else float('nan'))
    torch.save(dict(completed_updates=600,modules={'lora':{'weight':value}}),state)
    save_file({'weight':exported},str(weights))
    weights.with_suffix('.json').write_text(json.dumps({'steps':600}))
    with pytest.raises(ValueError,match='Nonfinite or mismatched'):
        campaign.audit_checkpoint(weights,state,600,'warmup')


def test_two_native_adapters_add_and_detach_without_changing_base():
    class Qwen3Attention(nn.Module):
        def __init__(self):
            super().__init__();self.q_proj=nn.Linear(12,12,bias=False)
        def forward(self,x):return self.q_proj(x)
    model=nn.Sequential(Qwen3Attention())
    torch.manual_seed(712)
    x=torch.randn(2,3,12)
    original=model[0].q_proj.forward
    base=model(x).detach()
    networks=[LoRANetwork(model,multiplier=0.,rank=8,alpha=8.,delimiter='-',
        target_replace=['Qwen3Attention'],prefix='lora_te',train_method='full') for _ in range(2)]
    for net in networks:
        assert len(net.unet_loras)==1
        with torch.no_grad():net.unet_loras[0].lora_up.weight.normal_(std=.1)
    def sample(a,b):
        for net,scale in zip(networks,(a,b)):net.set_lora_slider(scale)
        with torch.no_grad(),ExitStack() as stack:
            for net in networks:stack.enter_context(net)
            return model(x).clone()
    torch.testing.assert_close(sample(0,0),base,rtol=0,atol=0)
    a,b=sample(.65,0),sample(0,.35)
    torch.testing.assert_close(sample(.65,.35),a+b-base,rtol=1e-5,atol=1e-6)
    model[0].q_proj.forward=original
    torch.testing.assert_close(model(x),base,rtol=0,atol=0)


def test_presentation_edits_are_excluded_but_training_edits_still_block(tmp_path,monkeypatch):
    from conceptmod.textsliders.gan_v2 import state
    monkeypatch.setattr(campaign,'WORK',tmp_path)
    monkeypatch.setattr(state,'code_fingerprints',lambda:{})
    core=tmp_path/'train.py';core.write_text('training = 1\n')
    page=tmp_path/'report.py';page.write_text('page = 1\n')
    (tmp_path/'patch_listen_page.py').write_text('page = 2\n')
    sources=campaign.signed_files()
    assert str(core) in sources
    assert str(page) not in sources
    assert str(tmp_path/'patch_listen_page.py') not in sources
    campaign.write(tmp_path/'manifest.json',dict(source_files=sources))
    page.write_text('page = 3\n')
    campaign.verify_frozen()
    core.write_text('training = 2\n')
    with pytest.raises(campaign.CampaignBlocked,match='train.py'):
        campaign.verify_frozen()


def test_page_failure_and_mutation_do_not_corrupt_queue_state(tmp_path,monkeypatch):
    import report
    monkeypatch.setattr(campaign.importlib,'reload',lambda module:module)
    monkeypatch.setattr(campaign,'WORK',tmp_path)
    monkeypatch.setattr(campaign,'PAGE',tmp_path/'page')
    driver=object.__new__(campaign.Campaign)
    driver.catalog={'sliders':[]}
    driver.status={'status':'running','sliders':{}}
    def broken_page(catalog,status,page):
        status['status']='corrupted';catalog['sliders'].append('corrupted')
        raise ValueError('Bad page template')
    monkeypatch.setattr(report,'publish',broken_page)
    driver.save('A checkpoint was saved')
    assert driver.status['status']=='running'
    assert campaign.read(tmp_path/'status.json')['status']=='running'
    assert driver.catalog=={'sliders':[]}


def test_global_block_preserves_later_jobs_and_completed_warmup(monkeypatch):
    monkeypatch.setattr(campaign,'verify_frozen',lambda:None)
    driver=object.__new__(campaign.Campaign)
    driver.catalog={'sliders':[{'id':'a','label':'A'},{'id':'b','label':'B'}]}
    driver.status={'status':'running','sliders':{'a':{'warmup':{'status':'complete'}}}}
    monkeypatch.setattr(driver,'save',lambda *args:None)
    visited=[]
    def stage(item,phase,source=None):
        visited.append((item['id'],phase))
        if phase=='bounded':raise campaign.CampaignBlocked('Source changed')
        return Path('weights'),Path('state')
    monkeypatch.setattr(driver,'stage',stage)
    assert driver.run()==1
    assert driver.status['status']=='blocked'
    assert driver.status['sliders']['a']['status']=='blocked'
    assert driver.status['sliders']['a']['warmup']['status']=='complete'
    assert 'b' not in driver.status['sliders']
    assert visited==[('a','warmup'),('a','bounded')]


def test_initial_global_block_records_reason_without_failing_sliders(monkeypatch):
    driver=object.__new__(campaign.Campaign)
    driver.status={'status':'running','sliders':{'done':{'status':'complete'}}}
    monkeypatch.setattr(driver,'save',lambda *args:None)
    def blocked():raise campaign.CampaignBlocked('Prompt changed')
    monkeypatch.setattr(campaign,'verify_frozen',blocked)
    assert driver.run()==1
    assert driver.status['status']=='blocked'
    assert driver.status['sliders']=={'done':{'status':'complete'}}
