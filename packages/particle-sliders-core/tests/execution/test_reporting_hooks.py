"""Controller/observer wiring; actual HyperGAN reporting is tested separately."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from test_dispatch import product, request_for
from particle_sliders.execution import resolve_request, execute_resolved_request
from particle_sliders.execution.request import validate_resolved
import particle_sliders.execution.local as local


@pytest.fixture
def reporting(tmp_path, monkeypatch):
    script=tmp_path/'fake_reporter.py'
    script.write_text('''import argparse,json,pathlib,sys,time,os
p=argparse.ArgumentParser();p.add_argument('--run');p.add_argument('--runtime-path');p.add_argument('--runtime-commit');p.add_argument('--runtime-sha256');p.add_argument('--check',action='store_true');p.add_argument('--follow',action='store_true');a=p.parse_args()
if a.runtime_path.endswith('missing'):
 print('reporting runtime missing',file=sys.stderr);sys.exit(7)
if a.check:sys.exit(0)
r=pathlib.Path(a.run);o=r/'reporting/hypergan';o.mkdir(parents=True)
(o/'pid').write_text(str(os.getpid()));(o/'reporter-state.json').write_text(json.dumps({'status':'ready'}))
if a.runtime_path.endswith('dies'):time.sleep(.2);sys.exit(9)
while not (r/'result.json').is_file():time.sleep(.01)
(o/'observed-result.json').write_text((r/'result.json').read_text())
(o/'reporter-state.json').write_text(json.dumps({'status':'complete'}))
''')
    def command(request, output, *, check=False):
        config=request['reporting']
        return [sys.executable,str(script),'--run',str(output),'--runtime-path',config['runtime_path'],
                '--runtime-commit',config['runtime_commit'],'--runtime-sha256',config['runtime_sha256'],
                '--check' if check else '--follow'],dict(local.os.environ,CUDA_VISIBLE_DEVICES='')
    monkeypatch.setattr(local,'_reporter_command',command)
    return dict(kind='hypergan-develop',runtime_python=sys.executable,runtime_path=str(tmp_path/'runtime'),
                runtime_commit='a'*40,runtime_sha256='b'*64)


def test_profile_reporting_is_separate_from_training_identity(product,tmp_path,reporting):
    plain=request_for(product,tmp_path)
    profile=deepcopy(product[1]);profile['reporting']=reporting
    with_reporting=resolve_request({'steps':5},base=tmp_path,profile=profile)
    assert plain['resolved']['semantic_sha256']==with_reporting['resolved']['semantic_sha256']
    assert plain['resolved']['runtime_sha256']==with_reporting['resolved']['runtime_sha256']
    assert 'reporting_sha256' in with_reporting['resolved']
    changed=deepcopy(with_reporting);changed['reporting']['runtime_sha256']='c'*64
    with pytest.raises(ValueError,match='reporting configuration changed'):validate_resolved(changed)
    disabled=resolve_request({'steps':5},base=tmp_path,profile=profile,no_reporting=True)
    assert disabled['reporting'] is None
    assert disabled['resolved']['semantic_sha256']==plain['resolved']['semantic_sha256']


def test_reporter_ready_before_native_and_flushed_after_result(product,tmp_path,reporting):
    worker=tmp_path/'worker.py'
    worker.write_text(worker.read_text().replace("o=pathlib.Path(a.output);o.mkdir()", "o=pathlib.Path(a.output);assert json.loads((o.parent/'reporting/hypergan/reporter-state.json').read_text())['status']=='ready';o.mkdir()"))
    request=request_for(product,tmp_path,reporting=reporting)
    checkpoints=[]
    result=execute_resolved_request(request,on_checkpoint=checkpoints.append)
    output=Path(request['output']['root'])
    assert result['reporting']['status']=='complete'
    assert json.loads((output/'reporting/hypergan/observed-result.json').read_text())['status']=='completed'
    assert json.loads((output/'reporting/hypergan/reporter-state.json').read_text())['status']=='complete'
    assert len(checkpoints)==len(set(checkpoints))


def test_missing_reporting_runtime_fails_before_gpu_or_output(product,tmp_path,reporting):
    reporting['runtime_path']=str(tmp_path/'missing')
    request=request_for(product,tmp_path,reporting=reporting)
    with pytest.raises(RuntimeError,match='reporting runtime check failed'):execute_resolved_request(request)
    assert not Path(request['output']['root']).exists()


def test_reporting_can_be_disabled_without_runtime(product,tmp_path,reporting):
    profile=deepcopy(product[1]);profile['reporting']=dict(reporting,runtime_path=str(tmp_path/'missing'))
    request=resolve_request({'steps':5},base=tmp_path,profile=profile,output=tmp_path/'run',no_reporting=True)
    assert execute_resolved_request(request)['status']=='completed'


def test_cancel_closes_reporter_and_retains_terminal_truth(product,tmp_path,reporting):
    (tmp_path/'worker.py').write_text('import time;time.sleep(60)')
    request=request_for(product,tmp_path,reporting=reporting)
    with patch.object(local.queue.Queue,'get',side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):execute_resolved_request(request)
    output=Path(request['output']['root'])
    assert json.loads((output/'result.json').read_text())['status']=='cancelled'
    assert json.loads((output/'reporting/hypergan/observed-result.json').read_text())['status']=='cancelled'


def test_reporter_failure_stops_owned_native(product,tmp_path,reporting):
    (tmp_path/'worker.py').write_text('import time;time.sleep(60)')
    reporting['runtime_path']=str(tmp_path/'dies')
    request=request_for(product,tmp_path,reporting=reporting)
    with pytest.raises(RuntimeError,match='reporter exited'):execute_resolved_request(request)
    final=json.loads((Path(request['output']['root'])/'result.json').read_text())
    assert final['status']=='failed' and final['worker_reaped'] is True


def test_host_default_is_used_without_changing_recipe(product,tmp_path,reporting,monkeypatch,capsys):
    from particle_sliders.execution.cli import main
    config=tmp_path/'recipe.json';config.write_text('{"steps":5}')
    defaults=tmp_path/'host-reporting.json';defaults.write_text(json.dumps(reporting))
    monkeypatch.setenv('PARTICLE_SLIDER_REPORTING_CONFIG',str(defaults))
    assert main(['plan','--config',str(config)])==0
    planned=json.loads(capsys.readouterr().out)
    assert planned['reporting']==reporting
    assert main(['plan','--config',str(config),'--no-reporting'])==0
    disabled=json.loads(capsys.readouterr().out)
    assert disabled['reporting'] is None
    assert disabled['resolved']['semantic_sha256']==planned['resolved']['semantic_sha256']


def test_explicit_reporting_overrides_host_default(product,tmp_path,reporting,monkeypatch,capsys):
    from particle_sliders.execution.cli import main
    config=tmp_path/'recipe.json';config.write_text('{"steps":5}')
    defaults=tmp_path/'default.json';defaults.write_text(json.dumps(reporting))
    custom=tmp_path/'custom.json';custom.write_text(json.dumps(dict(reporting,runtime_commit='c'*40)))
    monkeypatch.setenv('PARTICLE_SLIDER_REPORTING_CONFIG',str(defaults))
    assert main(['plan','--config',str(config),'--reporting-config',str(custom)])==0
    assert json.loads(capsys.readouterr().out)['reporting']['runtime_commit']=='c'*40


def test_reporter_close_failure_keeps_completed_training_evidence(product,tmp_path,reporting):
    script=tmp_path/'fake_reporter.py'
    script.write_text(script.read_text().replace("(o/'observed-result.json').write_text", "if a.runtime_path.endswith('fail_close'):sys.exit(8)\n(o/'observed-result.json').write_text"))
    reporting['runtime_path']=str(tmp_path/'fail_close')
    request=request_for(product,tmp_path,reporting=reporting)
    with pytest.raises(local.ReportingError,match='code 8'):execute_resolved_request(request)
    output=Path(request['output']['root'])
    terminal=json.loads((output/'result.json').read_text())
    assert terminal['training_status']=='completed'
    assert terminal['completed_steps']==5
    assert terminal['reporting']['status']=='failed'
    assert Path(terminal['checkpoint']).is_dir()
    assert json.loads((output/'reporting/hypergan/reporter-state.json').read_text())['status']=='failed'


def test_one_command_uses_actual_hypergan_from_host_configuration(product,tmp_path,monkeypatch,capsys):
    import os
    runtime=os.environ.get('HYPERGAN_TEST_RUNTIME_PATH')
    if not runtime:
        pytest.skip('set HYPERGAN_TEST_RUNTIME_PATH for real controller/HyperGAN integration')
    from particle_sliders.execution.reporting import HYPERGAN_COMMIT,HYPERGAN_SOURCE_SHA256
    from particle_sliders.session_cli import main
    worker=tmp_path/'worker.py'
    code=worker.read_text().replace('import argparse,json,pathlib,sys','import argparse,json,pathlib,sys,hashlib')
    code=code.replace("json.dumps({'step':step})", "json.dumps({'step':step,'sha256':hashlib.sha256((cp/'training.pt').read_bytes()).hexdigest()})")
    code=code.replace("(o/'adapter.bin').write_bytes", "(o/'metrics.jsonl').write_text(''.join(json.dumps({'step':s,'g_loss':1.0/s,'d_loss':2.0/s})+'\\n' for s in range(1,c['steps']+1)))\n(o/'adapter.bin').write_bytes")
    worker.write_text(code)
    reporter=dict(kind='hypergan-develop',runtime_python=os.environ.get('HYPERGAN_TEST_PYTHON',sys.executable),
        runtime_path=runtime,runtime_commit=HYPERGAN_COMMIT,runtime_sha256=HYPERGAN_SOURCE_SHA256)
    config=tmp_path/'recipe.json';config.write_text('{"steps":5}')
    defaults=tmp_path/'reporting.json';defaults.write_text(json.dumps(reporter))
    monkeypatch.setenv('PARTICLE_SLIDER_REPORTING_CONFIG',str(defaults))
    output=tmp_path/'command-run'
    assert main(['train','--config',str(config),'--out',str(output)])==0
    result=json.loads(capsys.readouterr().out)
    assert result['reporting']['status']=='complete'
    events=[json.loads(line) for line in (output/'reporting/hypergan/events.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events if e['event']=='train']==[1,2,3,4,5]
    state=json.loads((output/'reporting/hypergan/reporter-state.json').read_text())
    assert state['status']=='complete' and state['last_durable_step']==5
