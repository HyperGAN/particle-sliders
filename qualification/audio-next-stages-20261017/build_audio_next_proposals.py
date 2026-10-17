"""Freeze non-dispatchable audio proposals from CPU/offline evidence only."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/mnt/nvme/sliders/audio-adapters-pg09-20261017')
CORE = Path('/ml2/codex-storage/future-workspaces/2026-10-16/task-6/particle-sliders-pg09-core')
WORK = Path('/ml2/codex-storage/future-workspaces/2026-10-17/task-4')
OUT = WORK / 'audio-next-stage-proposals'
OUT.mkdir(exist_ok=True)
RUNS = Path('/mnt/nvme/sliders/qualification/runs/audio-pg09-reviewed-20261017')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


smoke = load(CORE / 'qualification/audio-reruns/yue2-smoke.json')
smoke_receipt_path = RUNS / 'yue2-smoke-root-admission/receipt.json'
smoke_receipt = load(smoke_receipt_path)
assert smoke_receipt['status'] == 'completed' and smoke_receipt['exit_code'] == 0
assert smoke_receipt['proof']['exact_full_state_metrics_probes_export']
for category in ('source_pins', 'input_pins'):
    assert all(sha(path) == expected for path, expected in smoke[category].items())
assets = load(WORK / 'audio-next-assets.json')
runtime = {family: load(WORK / f'audio-next-{family}-runtime.json') for family in ('yue2', 'music3')}
for family, value in runtime.items():
    assert not value['cuda_initialized'] and not value['models_constructed']
    assert not value['pretrained_tensors_loaded'] and not value['audio_generated']
    assert len(value['runtime']['pg_sources']) == 49
    assert value['domain']['resolved_penalty_domain'] == 'continuous'
    shutil.copy2(WORK / f'audio-next-{family}-runtime.json', OUT / f'{family}-offline-runtime.json')
shutil.copy2(WORK / 'audio-next-assets.json', OUT / 'actual-asset-identities.json')
for name in ('audio_offline_identity.py', 'audio_next_runtime.py', 'build_audio_next_proposals.py'):
    shutil.copy2(WORK / name, OUT / name)
checkpoint = RUNS / 'yue2-smoke/full3/checkpoint.pt'
checkpoint_hash = sha(checkpoint)
common_sources = {p:h for p,h in smoke['source_pins'].items()
    if '/audio-adapters-' in p or '/audio-core-pin-' in p or '/ops-music3/' in p or '/audio-yue2-source-' in p}
base_env = deepcopy(smoke['environment'])
base_env['CUDA_VISIBLE_DEVICES'] = '<ROOT_ASSIGNED_SINGLE_PHYSICAL_GPU_AFTER_REVIEW>'
gates = dict(moss_priority='All existing MOSS jobs and source pins retain priority; root alone admits a free window.',
    execution='No dispatch from these proposal files. Root must freeze a phase-specific guardian/card and grant explicit admission.',
    ownership='Both physical-card leases, fresh UUID/resident identity, one owned PGID, at most one owned CUDA child, no unowned CUDA PIDs.',
    failure='Root guardian SIGINT only the owned group; up to180 seconds grace then owned-group kill; no retry, automatic fallback or next phase.',
    kernel='Use the existing pinned kernel telemetry/resource checks throughout; no stale admission snapshot.',
    cleanup='Only root controls worker/guardian processes; no service stops, permission changes or installed-package writes.')
limits = dict(compute_seconds=720, shutdown_grace_seconds=180, maximum_wall_seconds=900,
    maximum_concurrent_cuda_children=1, gpu_running_free_mib_minimum=6144,
    gpu_abort_temperature_c=82, gpu_cooling_start_c=76, gpu_cooling_target_c=70,
    gpu_cooling_maximum_seconds=180, startup_temperature_c_below=76,
    host_running_available_gib_minimum=8, nvme_free_gib_minimum=100,
    power_abort_fraction_of_limit=1.15, retry=False, automatic_next_phase=False)


def base(phase, family, output):
    assert not output.exists(), 'proposal output already exists'
    value = dict(schema='audio-pg09-next-stage-proposal-v1', phase=phase, family=family,
        status='PROPOSED_NOT_ADMITTED', dispatch_allowed=False, admission_id=None,
        core_commit='3e94b13eb5d6cd1ceee3f018f019a68603f35696',
        worker_commit='27ae2cbd880fd98f05dc373209dccd1f0a306990',
        native_environment_commit='225c29df220c89001338a959b48c7cadc5f818fd',
        source_profile='alz-continuous-v1', penalty_input_domain='continuous',
        patch_sha256='c6531a21f62fa96682f5aece517a6a862bb849ab971aa2766b3946b9a7543571',
        output_directory=str(output), output_absent_at_proposal=True,
        environment=deepcopy(base_env), limits=deepcopy(limits), external_gates=deepcopy(gates),
        runtime_receipt=f'{family}-offline-runtime.json', asset_receipt='actual-asset-identities.json',
        source_pins=dict(common_sources, **runtime[family]['sources']),
        pg_sources=runtime[family]['runtime']['pg_sources'],
        input_pins={f['path']:f['sha256'] for f in assets['families'][family]['files']},
        capacity_status='Fresh root capacity/temperature/UUID/resident checks pending; no reservation or current-capacity claim.',
        assets_status='All required existing bytes independently SHA256/size verified; no download needed.',
        runtime_status='Native interpreter imports, tokenizers and public domain validator passed without model construction.',
        native_audio_generated=False, quality_qualified=False)
    manifest = assets['families'][family]
    value['input_pins'][manifest['manifest_path']] = manifest['manifest_sha256']
    for flag in ('--train-prompts', '--heldout-prompts'):
        path = smoke['command'][smoke['command'].index(flag)+1]
        value['input_pins'][path] = sha(path)
    return value


continuation = base('Y20', 'yue2', RUNS / 'yue2-continue20')
command = deepcopy(smoke['command'])
command[1] = str(ROOT / 'scripts/qualify_audio_pg09_live.py')
command[command.index('--out')+1] = continuation['output_directory']
command[command.index('--admission-id')+1] = '<ROOT_ASSIGN_NEW_YUE2_CONTINUE20_ADMISSION>'
command += ['--resume', str(checkpoint), '--until', '20']
continuation.update(command=command, input_pins=dict(continuation['input_pins'], **{
    str(checkpoint):checkpoint_hash, str(smoke_receipt_path):sha(smoke_receipt_path),
    str(RUNS / 'yue2-smoke/comparison.json'):sha(RUNS / 'yue2-smoke/comparison.json')}),
    required_previous_phase='Completed independently qualified YuE2 native full3 == fresh1+2 smoke.',
    implementation_status='Existing source-identical native worker can continue3to20; phase-specific ROOT guardian/card scope remains to be implemented/reviewed by its owner.',
    required_guardian_change='Existing guarded_audio_pg09.py accepts only the three-child YuE smoke. Do not pass this continuation to that unchanged guardian; root must bind this one-child command in a separately reviewed phase scope.',
    total_recipe_steps=20, initial_completed_steps=3, additional_updates=17,
    acceptance_gates=['Source/input/runtime/domain/profile hashes unchanged; public transactional checkpoint restore succeeds.',
        'Exactly20 completed updates in the original20-step recipe, no restart, loss/optimizer or target fallback.',
        'Every step finite; Gadv+required public prior and Dadv+positive public continuous penalty only.',
        'Actual participating128x4 bank changes from step3; positive bank response-direction gradient; critic/adapter update.',
        'Frozen base SHA d3365aeba830e1af83ed5852d7564c305eb3e3ce3cd06716b461f65f6801dfbc unchanged; exact heldout scale0 replay.',
        'Both heldout scale1 hidden-response effects are finite and nonzero; report magnitude without a semantic-quality claim.',
        'Perturb then public adapter export/reload yields exact heldout hidden probes; restore full training state afterward.',
        'Final balanced D/G exposures40 each; all4 training rows exposed10times in each phase. No heldout row training.',
        'Save full checkpoint.pt, adapter-export.pt, qualification.pt, fixed rows/progress, resource/kernel/guardian receipts.',
        'No audio, no additional update, no implicit admission of renderer or Music3.'])
continuation['limits'].update(startup_gpu_free_gib_minimum=16, startup_host_available_gib_minimum=12,
    sequential_cuda_children=1, observed_prior_peak_allocated_bytes=7491249664)
save('yue2-continue20.proposed.json', continuation)

render_phases = []
renderer_path = ROOT / 'scripts/render_yue2_pg09.py'
renderer_tree = ast.parse(renderer_path.read_text())
protocol_node = next(node.value for node in renderer_tree.body if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == 'PROTOCOL' for target in node.targets))
renderer_protocol = {keyword.arg:ast.literal_eval(keyword.value) for keyword in protocol_node.keywords}
for index in (0, 1):
    render = base(f'YR{index}', 'yue2', RUNS / f'yue2-matched-case-{index}')
    render.update(command=[smoke['command'][0], str(renderer_path), '--card',
            str(CORE / f'qualification/audio-reruns/yue2-matched-case-{index}.json')],
        executable_card_ready=False, renderer_sha256=sha(renderer_path),
        implementation_status='Separate bounded public-PG09-export renderer implemented and CPU qualified with native-random AR and synthetic acoustics; no pretrained/native-audio/GPU result.',
        implementation_gates=['Root independently reviews frozen renderer source and native-random CPU receipt.',
            'Accepted Y20 producer export AND qualification artifacts must exist and their exact hashes must be frozen.',
            'Root separately implements/reviews the render phase guardian and reviewed root card with continuous8GiB host reserve.',
            'Native base/off token/latent/waveform parity is a runtime gate before half/on/reference; no fallback.'],
        producer_input=dict(path=str(RUNS / 'yue2-continue20/adapter-export.pt'), sha256=None,
            status='NOT_YET_PRODUCED: freeze actual output hash only after acceptedY20 receipt; do not admit an unbound export.'),
        producer_qualification_input=dict(path=str(RUNS / 'yue2-continue20/qualification.pt'), sha256=None,
            status='NOT_YET_PRODUCED: bind the matching20-step qualification, probes and frozen-base identity.'),
        pipeline=dict(class_name='yue2.pipeline.YuE2Pipeline', model_id='m-a-p/YuE2-3B',
            revision='14fc6c6f146441b1dd6363fcb2e01e82a6914cb7', vae_id='m-a-p/YuE2-Vae',
            vae_revision='9a94e1d0ea9f8087e98f77fa88df4a4068104d2a', local_files_only=True,
            device='cuda:0', backend='torch-eager', quantization='none', offload_ar=False,
            memory_budget_gib=16, vae_core_frames=1024, vae_halo_frames=16,
            model_dtype='bfloat16', vae_dtype='float32', ode_steps=32, ode_method='midpoint'),
        protocol=renderer_protocol, heldout_case=index,
        heldout_prefix_identity=runtime['yue2']['heldout_prefixes'][index], seed=1709+index,
        variants=[dict(name='base', caption='neutral', adapter_attached=False),
            dict(name='off', caption='neutral', adapter_scale=0),
            dict(name='half', caption='neutral', adapter_scale=0.5),
            dict(name='on', caption='neutral', adapter_scale=1),
            dict(name='positive_caption', caption='positive', adapter_scale=0)],
        scale_policy='AdapterParameters.scaled at declared value through native plan/semantic AR; scale0 throughout native NAR and VAE. No adapter on negative CFG branch: explicit cfg_scale1.0 yields one conditional branch.',
        cfg_protocol_change='Explicit1.0 is a reviewed-proposal choice matching the single conditional training path; native cot=off default1.01 must never be silently used.',
        checkpoint_steps=20, maximum_clips=5, maximum_semantic_tokens_per_clip=200,
        maximum_audio_seconds_per_clip=8.1,
        duration_note='200 semantic/latent frames times staged VAE ratio1920/sample_rate48000=8s nominal;0.1s conservative convolution-boundary allowance. Validate actual natural decoded length; no crop/stretch.',
        acceptance_gates=['Exact neutral caption/lyrics bytes and predeclared seed per case; whole prefixes fit native context without truncation.',
            'Base/off exact native semantic tokens, latents and FLOAT/FLAC waveform before other variants; adapter RNG does not mutate training state.',
            'Correct public20-step export, exact hidden-probe replay before audio; frozen base unchanged and actual bank hash unchanged during inference.',
            'Finite nonempty stereo48kHz arrays, max8.1s, all semantic/ABC truncation flags recorded, no hidden crop/pad/normalization.',
            'Run matched_waveform_checks for base/off/half/on/positive-caption, record RMS/peak/clipping/silence and difference only on equal shapes.',
            'Retain exact requests/configs/source/model/export hashes, semantic tokens, NAR latents, audio and output hashes, perclip times/memory and guardian/kernel telemetry.',
            'Heldout listening review separately assesses concept direction, voice/lyrics and artifacts. Waveform difference is not a quality pass.',
            'Stop this case on any failed gate; case1 requires case0 accepted plus a new root lease. No automatic case2 or retry.'])
    render['limits'].update(startup_gpu_free_gib_minimum=20, startup_host_available_gib_minimum=16,
        sequential_cuda_children=1, maximum_resident_native_models=1)
    render['source_pins'][str(ROOT / 'conceptmod/textsliders/pg09_audio_checks.py')] = sha(ROOT / 'conceptmod/textsliders/pg09_audio_checks.py')
    render['source_pins'][str(renderer_path)] = sha(renderer_path)
    renderer_inputs = dict(render['input_pins'])
    renderer_inputs[render['producer_input']['path']] = None
    renderer_inputs[render['producer_qualification_input']['path']] = None
    assets_manifest = load(assets['families']['yue2']['manifest_path'])
    runtime_keys = ('sys_executable','torch_version','torch_file','transformers_version',
                    'transformers_file','particlegan_version','particlegan_file')
    render['root_card_template'] = dict(schema='root-yue2-pg09-render-admission-v1',
        status='UNREVIEWED_TEMPLATE_NOT_ADMITTED', admission_id=None, case=index,
        output_directory=render['output_directory'], protocol=renderer_protocol,
        limits=render['limits'], source_pins=render['source_pins'], input_pins=renderer_inputs,
        runtime_expected={key:runtime['yue2']['runtime'][key] for key in runtime_keys},
        manifest=assets['families']['yue2']['manifest_path'],
        model_dir=assets_manifest['records']['model']['path'], vae_dir=assets_manifest['records']['vae']['path'],
        product='/mnt/nvme/sliders/audio-yue2-source-20261017',
        train_prompts=smoke['command'][smoke['command'].index('--train-prompts')+1],
        heldout_prompts=smoke['command'][smoke['command'].index('--heldout-prompts')+1],
        adapter_export=render['producer_input']['path'], qualification=render['producer_qualification_input']['path'])
    render['required_previous_phase'] = 'AcceptedY20 plus separately qualified renderer; ' + ('case0 accepted for case1.' if index else 'first native render parity gate.')
    save(f'yue2-matched-case-{index}.proposed.json', render)
    render_phases.append(render['phase'])

music = base('M3S', 'music3', RUNS / 'music3-smoke')
draft = load(ROOT / 'evidence/pg09-audio/audio-admission-drafts.json')
command = deepcopy(next(c['argv'] for c in draft['cards'] if c['family']=='music3'))
command[command.index('--out')+1] = music['output_directory']
command[command.index('--minimum-free-gib')+1] = '30'
command[command.index('--admission-id')+1] = '<ROOT_ASSIGN_NEW_MUSIC3_SMOKE_ADMISSION>'
music.update(command=command, implementation_status='Shared native LM worker/supervisor and fullwidth4096 CPU gates exist; no pretrainedMusic3 GPU result. Root guardian family/interpreter/source/output scope must be separately implemented/reviewed.',
    required_guardian_change='The unchanged existing root guardian hard-codes YuE family/interpreter/output; it rejects Music3. Root must freeze a distinct reviewedMusic3 phase before admission.',
    smoke_steps=dict(total_recipe_steps=20, uninterrupted=3, split=[1,2]),
    language_model_only=True, native_audio=False,
    acceptance_gates=deepcopy(continuation['acceptance_gates'][:1])+[
        'Exactly three serial native CUDA workers: full3, first1, resumed3. Same20-update recipe and fourtrain/twoheldout rows.',
        'Exact entire checkpoint/optimizer/prior/controller/row/noise/model-RNG/loss/probe/export equality full3==fresh1+2 using CPU safe loader.',
        'All stages show actual128x4 bank weight delta>0, directional bank gradient>0, critic/adapter changes and positive public continuous penalties.',
        'Native Qwen3 last-real-token AR response[1,4096]; complete prompt ends audio_start151669; no truncation/projection/field reshape.',
        'Frozen LM tensor hash unchanged in each stage; heldout scale0 exact; perturb then public inference export reload exact.',
        'Peak allocated/reserved memory and runtime/source identities saved; guardian/resource/kernel checks remain clear.',
        'No Music3 continuation or downstream condition-encoder/RVQ/flow/vocoder generation admitted by this smoke.'])
music['limits'].update(startup_gpu_free_gib_minimum=30, startup_host_available_gib_minimum=24,
    sequential_cuda_children=3, memory_note='Conservative new proposal increases prior26GiB draft to30GiB freshfree; actual pretrained memory remains an engineering smoke gate.')
save('music3-smoke.proposed.json', music)

index = dict(schema='audio-pg09-sequential-plan-v1', status='CPU_OFFLINE_PROPOSALS_ONLY',
    dispatch_allowed=False, automation_created=False, no_already_admitted_source_changes=True,
    phase_order=['Y20', 'YR0', 'YR1', 'M3S'],
    alternate_order='Root may review a separate Music3 smoke window after Y20 while renderer independent review/producer/guard gates remain pending; every phase stays serial and subordinate to MOSS priority. No automatic dispatch.',
    root_owner_thread='01a147d7-f540-778e-ad79-4d8eb267403d',
    phase_cards=['yue2-continue20.proposed.json','yue2-matched-case-0.proposed.json',
                 'yue2-matched-case-1.proposed.json','music3-smoke.proposed.json'],
    capacity_only=False,
    capacity_note='No proposal runs under the unchanged smoke-only guardian. Phase-specific root guards and new capacity/admission remain mandatory. The separate renderer now exists and passes native-random/synthetic CPU gates; producer20-step artifacts, independent review and first native render parity still remain.',
    readiness=dict(Y20=dict(native_implementation_missing=False, assets_missing=False,
        runtime_missing=False, phase_specific_root_guardian_pending=True,
        new_admission_and_fresh_capacity_pending=True),
        M3S=dict(native_worker_missing=False, assets_missing=False, runtime_missing=False,
        phase_specific_root_guardian_pending=True, first_native_gpu_qualification_pending=True,
        new_admission_and_fresh_capacity_pending=True)),
    actual_missing=dict(YR0=['Independent renderer review', 'Accepted20-step export+qualification producer hashes',
        'Renderer-specific root guardian/card and first native render parity'],
        YR1=['Same review/producer/guardian gates plus acceptedcase0 and new rootlease']),
    not_missing=['Existing YuE2 model/tokenizer/NAR/VAE bytes and native runtime APIs',
        'Existing Music3 LM/tokenizer/condition/RVQ/flow/vocoder bytes and native modular API imports'],
    asset_total_verified_bytes=sum(f['total_bytes'] for f in assets['families'].values()),
    pretrained_model_tensors_loaded_by_preparation=False, cuda_initialized_by_preparation=False,
    source_integrity_checked_against_admitted_card=True,
    proposal_hashes={name:sha(OUT/name) for name in ('yue2-continue20.proposed.json',
        'yue2-matched-case-0.proposed.json','yue2-matched-case-1.proposed.json','music3-smoke.proposed.json')})
save('sequential-plan.json', index)
print(json.dumps(index, indent=2))
