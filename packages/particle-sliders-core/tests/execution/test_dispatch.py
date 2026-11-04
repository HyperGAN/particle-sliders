"""Public controller contracts; no model imports or GPU allocation."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from particle_sliders.execution import resolve_request, execute_resolved_request
from particle_sliders.execution.request import PROFILE_SCHEMA, REQUEST_SCHEMA, validate_resolved, materialize_training
from particle_sliders.execution.local import check_request
from particle_sliders.execution.cli import main, should_dispatch


@pytest.fixture
def product(tmp_path):
    (tmp_path / 'worker.py').write_text('''import argparse,json,pathlib,sys
p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
p.add_argument('--checkpoint-every',type=int);p.add_argument('--resume');p.add_argument('--check',action='store_true')
a=p.parse_args();c=json.loads(pathlib.Path(a.config).read_text())
if a.check:
 print(json.dumps({'checked':True}));sys.exit(0)
o=pathlib.Path(a.output);o.mkdir()
for step in range(1,c['steps']+1):
 if step%a.checkpoint_every==0 or step==c['steps']:
  cp=o/'checkpoints'/('step-%06d'%step);cp.mkdir(parents=True)
  (cp/'training.pt').write_bytes(b'full-fixture-state')
  (cp/'checkpoint.json').write_text(json.dumps({'step':step}))
 (o/'progress.json').write_text(json.dumps({'status':'training','completed_steps':step}))
(o/'adapter.bin').write_bytes(b'fixture-adapter')
(o/'result.json').write_text(json.dumps({'status':'completed','completed_steps':c['steps'],
 'checkpoint':str(cp),'adapter':str(o/'adapter.bin')}))
''')
    spec = dict(schema=PROFILE_SCHEMA, name='fixture-worker', product='fixture-product',
                config_kind='native', status='fixture', command=['{python}', 'worker.py'],
                runtime={'python': sys.executable}, capabilities={'resume': True, 'check': True, 'periodic_checkpoint': True})
    path = tmp_path / 'training_profile.json'
    path.write_text(json.dumps(spec))
    return path, spec


def request_for(product, tmp_path, **kwargs):
    return resolve_request({'steps': 5}, base=tmp_path, profile=str(product[0]), output=tmp_path/'run', **kwargs)


def test_host_is_stdlib_only(product, tmp_path):
    source = Path(__file__).resolve().parents[2] / 'src'
    code = '''import sys
from particle_sliders.session_cli import main
assert main(['plan','--profile',sys.argv[1],'--config',sys.argv[2]])==0
assert 'torch' not in sys.modules
assert 'particle_sliders.lifecycle' not in sys.modules
'''
    config = tmp_path/'recipe.json';config.write_text('{"steps":5}')
    result = subprocess.run([sys.executable, '-S', '-c', code, str(product[0]), str(config)],
        env=dict(os.environ, PYTHONPATH=str(source)), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_request_transport_does_not_change_semantics(product, tmp_path):
    r = request_for(product, tmp_path)
    other = deepcopy(r)
    other['execution'].update(runner='modal', device='cuda:0')
    other['output']['root'] = '/different/volume'
    other['resolved']['profile']['runtime']['python'] = '/remote/env/bin/python'
    other['resolved']['profile_dir'] = '/remote/source'
    validate_resolved(other)
    other['training']['steps'] += 1
    with pytest.raises(ValueError, match='semantics changed'):
        validate_resolved(other)


def test_source_only_is_descriptive_not_gate(product, tmp_path):
    spec = deepcopy(product[1]);spec['status']='source_only'
    r = resolve_request({'steps': 5}, base=tmp_path, profile=spec, output=tmp_path/'run')
    assert check_request(r)['qualification'] == 'source_only'


def test_blocked_before_output_or_worker(product, tmp_path):
    spec = deepcopy(product[1]);spec.update(status='blocked', reason='native model factory is unimplemented')
    r = resolve_request({'steps':5}, base=tmp_path, profile=spec, output=tmp_path/'run')
    with pytest.raises(ValueError, match='unimplemented'):
        execute_resolved_request(r)
    assert not (tmp_path/'run').exists()


def test_native_roundtrip_and_checkpoints(product, tmp_path):
    r = request_for(product,tmp_path,checkpoint_every=2)
    checkpoints=[]
    result=execute_resolved_request(r,on_checkpoint=checkpoints.append)
    assert result['status']=='completed' and result['completed_steps']==5
    assert [p.name for p in sorted((tmp_path/'run/artifacts/checkpoints').iterdir())]==['step-000002','step-000004','step-000005']
    assert len(set(checkpoints))==3
    assert Path(result['checkpoint']).is_dir()
    saved=json.loads((tmp_path/'run/request.json').read_text())
    assert saved['training']=={'steps':5}
    with pytest.raises(FileExistsError): execute_resolved_request(r)


def test_check_is_no_training_output(product,tmp_path):
    r=request_for(product,tmp_path)
    assert check_request(r)['status']=='checked'
    assert not (tmp_path/'run').exists()


def test_input_hash_reference_and_changed_input(product,tmp_path):
    file=tmp_path/'labels.txt';file.write_text('labels')
    value=dict(schema=REQUEST_SCHEMA,profile=str(product[0]),training={'steps':5,'labels':'input://labels'},
               input_files=[{'name':'labels','local_path':'labels.txt'}])
    r=resolve_request(value,base=tmp_path)
    assert materialize_training(r)['labels']==str(file)
    file.write_text('changed')
    with pytest.raises(ValueError,match='changed'): execute_resolved_request(r)
    assert not Path(r['output']['root']).exists()


def test_session_config_is_not_polluted_by_runner(tmp_path):
    value=dict(backend='example',cohort='new-cohort',options={'steps':3,'dummy':True})
    r=resolve_request(value,base=tmp_path,runner='modal')
    assert r['training']==value
    assert r['resolved']['profile']['interface']=='image'
    assert should_dispatch(['train','--runner','modal'])
    assert not should_dispatch(['train','--config','missing.json'])


def test_failure_record_and_missing_result(product,tmp_path):
    (tmp_path/'worker.py').write_text('raise SystemExit(9)')
    r=request_for(product,tmp_path)
    with pytest.raises(RuntimeError,match='code 9'):execute_resolved_request(r)
    result=json.loads((tmp_path/'run/result.json').read_text())
    assert result['status']=='failed' and result['worker_reaped'] is True


def test_resume_uri_only_remote(product,tmp_path):
    r=request_for(product,tmp_path,runner='modal',resume='modal://abc/latest')
    assert r['execution']['resume']=='modal://abc/latest'
    with pytest.raises(ValueError,match='local checkpoint'):request_for(product,tmp_path,resume='modal://abc/latest')


def test_unknown_profile_and_invalid_budget(product,tmp_path):
    with pytest.raises(ValueError,match='expected one'):resolve_request({'steps':2},profile='missing-profile',base=tmp_path)
    for steps in (0, True, 1.5):
        with pytest.raises(ValueError,match='budget'):resolve_request({'steps':steps},profile=str(product[0]),base=tmp_path)


def test_config_only_uses_sibling_profile(product, tmp_path):
    config = tmp_path / 'recipe.json'; config.write_text('{"steps":5}')
    assert should_dispatch(['train', '--config', str(config)])
    assert main(['train', '--config', str(config), '--check']) == 0
    legacy = tmp_path / 'legacy.json'
    legacy.write_text(json.dumps(dict(backend='test', cohort='cohort', options={'steps':5})))
    assert not should_dispatch(['train', '--config', str(legacy)])


def test_cancel_reaps_owned_worker_and_updates_progress(product, tmp_path):
    (tmp_path/'worker.py').write_text("import time;time.sleep(60)")
    r = request_for(product,tmp_path)
    import particle_sliders.execution.local as local
    original = local.subprocess.Popen
    owned = []
    def start(*args, **kwargs):
        child = original(*args, **kwargs)
        owned.append(child)
        return child
    # Interrupt at first queue poll, after launch, just as foreground Ctrl-C.
    from unittest.mock import patch
    with patch.object(local.subprocess, 'Popen', start), patch.object(local.queue.Queue, 'get', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt): execute_resolved_request(r)
    assert owned and owned[0].poll() is not None
    assert json.loads((tmp_path/'run/progress.json').read_text())['status']=='cancelled'
    assert json.loads((tmp_path/'run/result.json').read_text())['worker_reaped'] is True


def test_resolved_budget_rejects_transport_tamper(product, tmp_path):
    r = request_for(product,tmp_path)
    r['resolved']['target_steps'] = 999
    with pytest.raises(ValueError,match='target budget'): validate_resolved(r)


def test_remote_does_not_inherit_local_device(product, tmp_path):
    spec = deepcopy(product[1]); spec['runtime']['device']='cuda:1'
    r = resolve_request({'steps':2},base=tmp_path,profile=spec,runner='modal')
    assert 'device' not in r['execution']


def test_numeric_environment_is_semantic(product, tmp_path):
    r = request_for(product,tmp_path)
    r['resolved']['profile']['runtime']['env']={'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    with pytest.raises(ValueError,match='semantics changed'):validate_resolved(r)


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux parent-death contract')
def test_controller_death_stops_own_worker(product, tmp_path):
    import signal
    import time
    (tmp_path/'worker.py').write_text("import os,pathlib,time;pathlib.Path('worker.pid').write_text(str(os.getpid()));time.sleep(60)")
    r=request_for(product,tmp_path)
    request_file=tmp_path/'request.json';request_file.write_text(json.dumps(r))
    source=Path(__file__).resolve().parents[2]/'src'
    code="import json,sys;from particle_sliders.execution import execute_resolved_request;execute_resolved_request(json.load(open(sys.argv[1])))"
    controller=subprocess.Popen([sys.executable,'-c',code,str(request_file)],
        env=dict(os.environ,PYTHONPATH=str(source)),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    worker_pid=None
    try:
        until=time.monotonic()+5
        while not (tmp_path/'worker.pid').exists() and time.monotonic()<until:
            time.sleep(.02)
        assert (tmp_path/'worker.pid').is_file()
        worker_pid=int((tmp_path/'worker.pid').read_text())
        controller.kill();controller.wait(timeout=5)
        until=time.monotonic()+5
        while time.monotonic()<until:
            stat=Path(f'/proc/{worker_pid}/stat')
            if not stat.exists() or stat.read_text().split(') ',1)[1].startswith('Z'):
                break
            time.sleep(.02)
        else:
            pytest.fail('worker survived controller death')
    finally:
        if controller.poll() is None:
            controller.kill();controller.wait()
        if worker_pid:
            try:os.kill(worker_pid,signal.SIGKILL)
            except ProcessLookupError:pass


def test_native_artifact_defaults_are_product_owned(product,tmp_path):
    r=request_for(product,tmp_path)
    assert 'selection' not in r['output'] and 'export_format' not in r['output']
    for options in ({'selection':'raw'}, {'selection':'ema'}, {'export_format':'v1'}, {'export_format':'v2'}):
        with pytest.raises(ValueError,match='override is unsupported'):
            request_for(product,tmp_path,**options)


def test_declared_native_artifact_flags_reach_worker(product,tmp_path):
    from particle_sliders.execution.local import command_for
    spec=deepcopy(product[1]);spec['capabilities'].update(selection=True,export_format=True)
    r=resolve_request({'steps':5},base=tmp_path,profile=spec,selection='ema',export_format='v2')
    command,_,_=command_for(r,tmp_path/'config.json',tmp_path/'out')
    assert command[command.index('--selection')+1]=='ema'
    assert command[command.index('--export-format')+1]=='v2'


def test_native_device_fixed_or_explicitly_supported(product,tmp_path):
    from particle_sliders.execution.local import command_for
    with pytest.raises(ValueError,match='device override'):
        request_for(product,tmp_path,device='cuda:1')
    fixed=deepcopy(product[1]);fixed['runtime']['device']='cuda:1'
    r=resolve_request({'steps':5},base=tmp_path,profile=fixed,device='cuda:1')
    assert r['execution']['device']=='cuda:1'
    with pytest.raises(ValueError,match='fixed runtime'):
        resolve_request({'steps':5},base=tmp_path,profile=fixed,device='cuda:0')
    flexible=deepcopy(product[1]);flexible['capabilities']['device']=True
    r=resolve_request({'steps':5},base=tmp_path,profile=flexible,device='cuda:0')
    command,_,_=command_for(r,tmp_path/'config.json',tmp_path/'out')
    assert command[command.index('--device')+1]=='cuda:0'


def test_session_device_override_is_effective(tmp_path):
    config=dict(backend='example',cohort='new',options={'steps':5,'device':'cuda:0'})
    r=resolve_request(config,base=tmp_path,device='cuda:1')
    assert r['training']['options']['device']=='cuda:1'
    assert config['options']['device']=='cuda:0'
    r['execution']['device']='cuda:2'
    with pytest.raises(ValueError,match='effective session'):validate_resolved(r)


def test_native_selection_must_be_confirmed_by_worker(product,tmp_path):
    spec=deepcopy(product[1]);spec['capabilities']['selection']=True
    worker=tmp_path/'worker.py'
    worker.write_text(worker.read_text().replace("p.add_argument('--checkpoint-every',type=int);", "p.add_argument('--selection');p.add_argument('--checkpoint-every',type=int);"))
    r=resolve_request({'steps':2},base=tmp_path,profile=spec,selection='ema',output=tmp_path/'unconfirmed')
    with pytest.raises(RuntimeError,match='confirm requested selection'):
        execute_resolved_request(r)
    worker.write_text(worker.read_text().replace("'checkpoint':str(cp),", "'selection':a.selection,'checkpoint':str(cp),"))
    r=resolve_request({'steps':2},base=tmp_path,profile=spec,selection='ema',output=tmp_path/'confirmed')
    assert execute_resolved_request(r)['native_result']['selection']=='ema'
