"""No native imports, CUDA queries, subprocesses or signals in these fixtures."""
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
import sys
sys.path.insert(0, str(SCRIPTS))
import guarded_audio_next_pg09 as guard


def fixture(phase='Y20'):
    fixed = guard.CARDS[phase]
    limits = dict(compute_seconds=720, shutdown_grace_seconds=180, maximum_wall_seconds=900,
        maximum_concurrent_cuda_children=1, host_running_available_gib_minimum=8,
        gpu_running_free_mib_minimum=6144, nvme_free_gib_minimum=100,
        gpu_abort_temperature_c=82, gpu_cooling_start_c=76, gpu_cooling_target_c=70,
        gpu_cooling_maximum_seconds=180, power_abort_fraction_of_limit=1.15, retry=False,
        automatic_next_phase=False, startup_gpu_free_gib_minimum=fixed['startup_gpu'],
        startup_host_available_gib_minimum=fixed['startup_host'],
        sequential_cuda_children=1 if phase=='Y20' else 3)
    output = guard.ROOT/'runs/audio-pg09-reviewed-20261017'/fixed['output']
    fields = {'--out':str(output), '--family':fixed['family'], '--total-steps':'20',
        '--device':'cuda:0', '--seed':'7', '--preset':'bispf_optimized', '--max-seq-len':'512',
        '--pg-source-profile':'alz-continuous-v1', '--penalty-input-domain':'continuous',
        '--minimum-free-gib':str(fixed['startup_gpu']), '--admission-id':'<ROOT_ASSIGN_NEW_PHASE>',
        '--manifest':'/fixture/manifest.json', '--train-prompts':'/fixture/train.yaml',
        '--heldout-prompts':'/fixture/heldout.yaml', '--model-dir':'/fixture/model'}
    if phase=='Y20':
        fields.update({'--until':'20', '--product':str(guard.YUE_PRODUCT),
            '--resume':str(guard.ROOT/'runs/audio-pg09-reviewed-20261017/yue2-smoke/full3/checkpoint.pt')})
    source_names=[guard.AUDIO/'scripts/qualify_audio_pg09_live.py',
                  guard.AUDIO/'scripts/run_audio_pg09_smoke.py',
                  guard.AUDIO/'conceptmod/textsliders/pg09_audio.py',
                  guard.ROOT/'ops-music3/music3_resources.py']
    if phase=='Y20':
        source_names += [guard.YUE_PRODUCT/'yue2'/name for name in guard.YUE_PRODUCT_FILES]
    proposal=dict(schema='audio-pg09-next-stage-proposal-v1', phase=phase, family=fixed['family'],
        status='PROPOSED_NOT_ADMITTED', dispatch_allowed=False, admission_id=None,
        core_commit=guard.CORE, source_profile='alz-continuous-v1', penalty_input_domain='continuous',
        patch_sha256=guard.PATCH, pg_sources={str(i):'0'*64 for i in range(49)}, limits=limits,
        output_directory=str(output), command=[fixed['executable'],fixed['script'],
            *[part for pair in fields.items() for part in pair]],
        input_pins={v:'0'*64 for k,v in fields.items() if k in ('--manifest','--train-prompts','--heldout-prompts','--resume')},
        source_pins={str(p):'0'*64 for p in source_names},
        environment=dict(CUDA_VISIBLE_DEVICES='<ROOT_ASSIGNED_SINGLE_PHYSICAL_GPU_AFTER_REVIEW>',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTHONPATH='/tmp/alz-pg09-domain-proposal:/fixture/core:/fixture/audio'),
        initial_completed_steps=3, additional_updates=17)
    card=dict(schema='root-audio-next-admission-v1', status='REVIEWED_PENDING_FRESH_RESOURCES',
        phase=phase, physical_gpu=fixed['gpu'], uuid=fixed['uuid'], admission_id='audio-phase-fixture-v1',
        lease_paths=[str(guard.ROOT/f"admission-gpu-{fixed['gpu']}.lock"),
                     f"/mnt/nvme/sliders/locks/gpu-{fixed['gpu']}.lock"])
    return card, proposal


@pytest.mark.parametrize('phase',['Y20','M3S'])
def test_exact_new_scope_binds_card_command_and_physical_gpu(phase,monkeypatch):
    monkeypatch.setattr(guard.resources,'status',lambda:pytest.fail('no GPU query allowed'))
    card, proposal=fixture(phase)
    _,_,_,command,environment=guard.validate_scope(card,proposal)
    assert guard.argument(command,'--admission-id')==card['admission_id']
    assert environment['CUDA_VISIBLE_DEVICES']==str(card['physical_gpu'])
    assert proposal['environment']['CUDA_VISIBLE_DEVICES'].startswith('<ROOT_')


@pytest.mark.parametrize('kind',['renderer','wrong_gpu','wrong_uuid','wrong_lease','placeholder_id',
    'four_gib','retry','parallel','site_overlay','fixture','new_flag','missing_input','missing_source',
    'wrong_resume','wrong_product','missing_product_source'])
def test_scope_changes_fail_before_any_native_action(kind):
    card,p=fixture()
    if kind=='renderer':card['phase']='YR0'
    elif kind=='wrong_gpu':card['physical_gpu']=0
    elif kind=='wrong_uuid':card['uuid']='other'
    elif kind=='wrong_lease':card['lease_paths'][0]='/fixture/unowned.lock'
    elif kind=='placeholder_id':card['admission_id']='<ROOT_ASSIGN>'
    elif kind=='four_gib':p['limits']['host_running_available_gib_minimum']=4
    elif kind=='retry':p['limits']['retry']=True
    elif kind=='parallel':p['limits']['maximum_concurrent_cuda_children']=2
    elif kind=='site_overlay':p['environment']['PYTHONPATH']+=':/fixture/site-packages'
    elif kind=='fixture':p['command']+=['--fixture','1']
    elif kind=='new_flag':p['command']+=['--render','1']
    elif kind=='missing_input':p['input_pins'].pop('/fixture/manifest.json')
    elif kind=='missing_source':p['source_pins'].pop(str(guard.AUDIO/'scripts/qualify_audio_pg09_live.py'))
    elif kind=='wrong_product':p['command'][p['command'].index('--product')+1]='/fixture/other-product'
    elif kind=='missing_product_source':p['source_pins'].pop(str(guard.YUE_PRODUCT/'yue2/backend.py'))
    else:p['command'][p['command'].index('--resume')+1]='/fixture/wrong.pt'
    with pytest.raises(ValueError):guard.validate_scope(card,p)


@pytest.mark.parametrize('value',[4096,8191,float('nan'),float('inf'),None])
def test_new_host_floor_rejects_old_floor_and_invalid_telemetry(value):
    with pytest.raises(RuntimeError):guard.host_floor({'host_free_mib':value})


def test_eight_gib_host_floor_is_accepted():
    guard.host_floor({'host_free_mib':8192})


def test_owned_child_filter_preserves_residents_and_rejects_unowned_or_parallel(monkeypatch):
    groups={20:100,21:100,30:300}
    def pgid(pid):
        if pid==40:raise ProcessLookupError
        return groups[pid]
    monkeypatch.setattr(guard.os,'getpgid',pgid)
    assert guard.owned_children({10,20,40},{10},100)==[{'pid':20,'pgid':100}]
    with pytest.raises(RuntimeError,match='unowned'):guard.owned_children({10,30},{10},100)
    with pytest.raises(RuntimeError,match='concurrent'):guard.owned_children({20,21},set(),100)


def proof(step=20):
    return dict(status='bounded_training_software_checks_passed',admission_id='fixture-v1',
        completed_steps=step,frozen_base_unchanged=True,zero_replay_exact=True,export_replay_exact=True,
        actual_bank_weights_changed=True,changed_adapter_tensors=4,changed_critic_tensors=2,
        actual_bank_delta_rms=.01,bank_response_gradient_l1=.1)


@pytest.mark.parametrize('change',['none','step','admission','bank_nan','critic_zero','off'])
def test_terminal_continuation_requires_actual_learning_and_exact_update_count(tmp_path,change):
    value=proof()
    if change=='step':value['completed_steps']=3
    elif change=='admission':value['admission_id']='another'
    elif change=='bank_nan':value['actual_bank_delta_rms']=float('nan')
    elif change=='critic_zero':value['changed_critic_tensors']=0
    elif change=='off':value['zero_replay_exact']=False
    (tmp_path/'progress.json').write_text(json.dumps(value))
    if change=='none':guard.terminal_proof('Y20',tmp_path,'fixture-v1')
    else:
        with pytest.raises(RuntimeError):guard.terminal_proof('Y20',tmp_path,'fixture-v1')


def test_music3_requires_all_three_stages_and_exact_comparison(tmp_path):
    for name,step in [('full3',3),('first1',1),('resumed3',3)]:
        folder=tmp_path/name;folder.mkdir();(folder/'progress.json').write_text(json.dumps(proof(step)))
    comparison=tmp_path/'comparison.json'
    comparison.write_text(json.dumps(dict(status='exact_smoke_passed',exact_full_state_metrics_probes_export=True)))
    guard.terminal_proof('M3S',tmp_path,'fixture-v1')
    comparison.write_text(json.dumps(dict(status='exact_smoke_failed',exact_full_state_metrics_probes_export=False)))
    with pytest.raises(RuntimeError):guard.terminal_proof('M3S',tmp_path,'fixture-v1')


@pytest.mark.parametrize('change',['none','training','wrong_recipe','missing_case','changed_checksum'])
def test_new_audio_cannot_bypass_prioritized_moss_matched_evaluations(tmp_path,monkeypatch,change):
    monkeypatch.setattr(guard,'ROOT',tmp_path)
    card={'completed_moss_evaluations':{}}
    for case,recipe in guard.MOSS_RECIPES.items():
        path=tmp_path/'runs/moss-pg09-original-runtime-v2-20261017'/case/'progress.json'
        path.parent.mkdir(parents=True)
        value=dict(recipe_sha256=recipe,mode='evaluate',status='evaluated',latest_step=3840,
                   frozen_base_unchanged=True)
        if case=='bad-intent':
            if change=='training':value.update(mode='train',status='training',latest_step=80)
            elif change=='wrong_recipe':value['recipe_sha256']='wrong'
        path.write_text(json.dumps(value))
        card['completed_moss_evaluations'][case]=dict(path=str(path),sha256=guard.sha(path))
    if change=='missing_case':card['completed_moss_evaluations'].pop('afterimage')
    elif change=='changed_checksum':card['completed_moss_evaluations']['afterimage']['sha256']='0'*64
    if change=='none':guard.priority_proofs(card)
    else:
        with pytest.raises(ValueError):guard.priority_proofs(card)
