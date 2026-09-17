"""Queue budgets, device isolation, interrupted campaigns and neutral display."""
import json
from pathlib import Path

import pytest

from scripts import queue_yue2_particle_catalog as queue
from scripts.evaluate_yue2_arm_b import page


def manifest(tmp_path):
    source=tmp_path/'source.py';source.write_text('frozen training source\n')
    m=dict(source_root=str(queue.ROOT),python='python',steps=1200,
        files={str(source):queue.sha(source)},jobs=[])
    for i in range(16):
        job=dict(id=f'slider{i}',name=f'slider{i}',gpu=i%2,save_dir=str(tmp_path/f'run{i}'),
            output_dir=str(tmp_path/f'page{i}'),train_prompts=str(tmp_path/f'train{i}.yaml'),
            eval_prompts=str(tmp_path/f'eval{i}.yaml'))
        job['argv']=queue.campaign_argv(m,job);m['jobs'].append(job)
    return m


def test_queue_pins_source_budget_and_command(tmp_path):
    m=manifest(tmp_path);queue.verify(m)
    m['steps']=1201
    with pytest.raises(ValueError):queue.verify(m)
    m['steps']=1200;m['jobs'][0]['argv'].append('--propose_only_lr_scale')
    with pytest.raises(ValueError):queue.verify(m)
    m=manifest(tmp_path);Path(next(iter(m['files']))).write_text('changed\n')
    with pytest.raises(ValueError,match='Frozen'):queue.verify(m)


def test_both_gpus_get_exact_particle_recipe_and_no_rate_override(tmp_path):
    m=manifest(tmp_path)
    for j in m['jobs']:
        a=j['argv']
        assert a[a.index('--gpu')+1]==str(j['gpu'])
        assert a[a.index('--steps')+1]=='1200'
        assert a[a.index('--recipe')+1]=='particle_bridge'
        assert a[a.index('--seed')+1]=='7'
        assert '--include_canary' in a and '--hidden_diagnostics' in a
        assert not any('lr_scale' in v or 'c9_g4x' in v for v in a)
    assert {j['gpu'] for j in m['jobs']}=={0,1}


def test_zero_exit_or_old_budget_is_not_campaign_completion(tmp_path):
    m=manifest(tmp_path);job=m['jobs'][0]
    train=Path(job['save_dir'])/'status.json';out=Path(job['output_dir'])/'status.json'
    queue.write(train,dict(completed=600))
    queue.write(out,dict(stage='Training and held-out comparisons complete'))
    assert not queue.completed(m,job)
    queue.write(train,dict(completed=1200));queue.write(out,dict(stage='Stopped'))
    assert not queue.completed(m,job)
    queue.write(out,dict(stage='Training and held-out comparisons complete'))
    assert queue.completed(m,job)


def test_wait_tracks_process_identity_and_named_pages_escape_labels(tmp_path):
    import os
    identity=queue.process_identity(os.getpid())
    assert identity and identity==queue.process_identity(os.getpid())
    assert queue.process_identity(999999999) is None
    rows=[dict(neutral='Plain sung song',positive='One adult female lead',lyrics='[verse]\nWe fold the map')]
    page(tmp_path,rows,[1709],'particle_bridge',True,label='Female')
    s=(tmp_path/'index.html').read_text()
    assert '1 = Female.' in s and '<b>Female caption</b>' in s
    assert '1 = Metal.' not in s and '<b>Metal' not in s
    page(tmp_path,rows,[1709],'particle_bridge',True,label='A <test>')
    assert 'A &lt;test&gt;' in (tmp_path/'index.html').read_text()


def test_workers_isolate_devices_continue_after_failed_job_and_skip_finished(tmp_path,monkeypatch):
    from types import SimpleNamespace
    m=manifest(tmp_path)
    m.update(wait_for={'0':[],'1':[]},gpu_lock_root=str(tmp_path),min_free_mib=10000)
    launched=[]
    monkeypatch.setattr(queue.signal,'signal',lambda *a:None)
    monkeypatch.setattr(queue.subprocess,'check_output',lambda *a,**k:'20000\n')
    class Child:
        def __init__(self,argv,**kwargs):
            gpu=argv[argv.index('--gpu')+1]
            assert kwargs['env']['CUDA_VISIBLE_DEVICES']==gpu
            assert len(kwargs['pass_fds'])==2
            name=argv[argv.index('--name')+1];launched.append((name,gpu))
            self.returncode=1 if name=='slider2' else 0
            queue.write(Path(argv[argv.index('--save_dir')+1])/'status.json',dict(completed=1200))
            queue.write(Path(argv[argv.index('--output_dir')+1])/'status.json',
                dict(stage='Training and held-out comparisons complete'))
        def poll(self):return self.returncode
    monkeypatch.setattr(queue.subprocess,'Popen',Child)
    for gpu in (0,1):queue.worker(SimpleNamespace(queue_dir=tmp_path,gpu=gpu),m)
    assert len(launched)==16 and len({name for name,gpu in launched})==16
    assert queue.read(tmp_path/'jobs/slider2.json')['status']=='failed'
    assert queue.read(tmp_path/'jobs/slider14.json')['status']=='complete'
    for gpu in (0,1):queue.worker(SimpleNamespace(queue_dir=tmp_path,gpu=gpu),m)
    assert len(launched)==16  # Finished work and failures both retain their evidence.
