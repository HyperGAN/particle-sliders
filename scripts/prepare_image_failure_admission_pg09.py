"""Metadata/source-only asset validation and unexecuted image admission proposals.

Never imports Torch/model packages, opens tensor packets, runs a child process,
or accesses GPU. Model/checkpoint tensors are stat-only; old declared checksums
are labeled separately from fresh metadata/source checksum verification.
"""
import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path

ROOT = Path('/mnt/nvme/sliders/qualification')
CHECKOUT = Path(__file__).resolve().parents[1]
BINARY = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.model', '.npz', '.npy'}
NATIVE = [
 'diffusers/pipelines/sana/pipeline_sana.py',
 'diffusers/models/transformers/sana_transformer.py',
 'diffusers/models/autoencoders/autoencoder_dc.py',
 'diffusers/schedulers/scheduling_dpmsolver_multistep.py',
 'diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py',
 'diffusers/models/transformers/transformer_sd3.py',
 'diffusers/models/autoencoders/autoencoder_kl.py',
 'diffusers/schedulers/scheduling_flow_match_euler_discrete.py',
 'transformers/models/gemma2/modeling_gemma2.py',
 'transformers/models/gemma/tokenization_gemma.py',
 'transformers/models/gemma/__init__.py',
 'transformers/tokenization_utils_tokenizers.py',
 'transformers/models/clip/modeling_clip.py',
 'transformers/models/clip/tokenization_clip.py',
]


def source_record(path):
    path = Path(path)
    if path.suffix in BINARY:
        raise ValueError('metadata audit refuses tensor/checkpoint/tokenizer-binary content')
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def stat_record(path):
    path = Path(path)
    result = dict(path=str(path), exists=path.is_file(), content_read=False)
    if result['exists']:
        result.update(bytes=path.stat().st_size, resolved_path=str(path.resolve()))
    return result


def runtime_metadata():
    packages = {}
    for name in ('torch','diffusers','transformers','safetensors','numpy','Pillow','tokenizers','accelerate','peft','particlegan'):
        dist = metadata.distribution(name)
        direct = dist.read_text('direct_url.json')
        packages[name] = dict(version=dist.version, distribution_path=str(dist._path),
            package_origin=str(dist.locate_file('')), direct_url=json.loads(direct) if direct else None)
    origin = Path(packages['diffusers']['package_origin'])
    native = [source_record(origin/path) for path in NATIVE]
    return dict(packages=packages, native_module_sources=native,
        archive_interpreter=str(ROOT/'envs/cpu/bin/python'),
        pg_interpreter=str(ROOT/'envs/pg090/bin/python'),
        native_origins_shared=True,
        shared_origin_basis='both --system-site-packages venv configs plus independent metadata probes; no native class imported',
        venv_configs=[source_record(ROOT/f'envs/{env}/pyvenv.cfg') for env in ('cpu','pg090')],
        archive_pg_absent=True, native_runtime_parity='source/metadata only; actual import and numerical parity pending')


def inventory(proposal):
    verified = []
    for record in proposal['source_bindings'] + [r for c in proposal['campaigns']
            for r in c['archive_bindings']+c['frozen_baseline_assets']]:
        actual = source_record(record['path'])
        if actual['sha256'] != record['sha256'] or actual['bytes'] != record['bytes']:
            raise ValueError('frozen source/input drift: '+record['path'])
        verified.append(actual)
    cases, packets = [], []
    for case in proposal['campaigns']:
        run, model = Path(case['archived_run']), Path(case['model']['directory'])
        family = 'sana' if case['id'].startswith('sana') else 'sd3'
        manifest_path = run/'model-manifest.json' if family == 'sana' else ROOT/'models/sd3-ea42f8c/qualification-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        declared = manifest if family == 'sana' else [dict(path=str(model/r['file']),**r) for r in manifest['files']]
        weights = []
        for row in declared:
            if Path(row['path']).suffix not in {'.safetensors','.bin'}:
                continue
            stat = stat_record(row['path'])
            stat.update(declared_sha256=row['sha256'], declared_bytes=row['bytes'], fresh_payload_checksum_verified=False)
            stat['size_matches_manifest'] = stat.get('bytes') == row['bytes']
            if not stat['exists'] or not stat['size_matches_manifest']:
                raise ValueError('declared cached weight absent or size drift: '+row['path'])
            weights.append(stat)
        config_files = [source_record(p) for p in sorted(model.rglob('*'))
                        if p.is_file() and p.suffix in {'.json','.txt'}]
        for row in declared:
            if Path(row['path']).suffix not in {'.json','.txt'}:
                continue
            actual = source_record(row['path'])
            if actual['sha256'] != row['sha256'] or actual['bytes'] != row['bytes']:
                raise ValueError('model metadata checksum drift: '+row['path'])
        model_index = json.loads((model/'model_index.json').read_text())
        scheduler = json.loads((model/'scheduler/scheduler_config.json').read_text())
        all_files = [p for p in sorted(run.rglob('*')) if p.is_file()]
        candidates = [stat_record(p) for p in all_files if any(k in p.name.lower() for k in
            ('noise','embed','condition','latent','descriptor','cache','rng'))]
        for step in (0,500):
            sidecar = run/f'checkpoints/step-{step:06d}.json'
            record = json.loads(sidecar.read_text())
            path = sidecar.parent/record['file']
            packets.append(dict(**stat_record(path), sidecar=source_record(sidecar),
                declared_sha256=record['sha256'], identity=record['identity'],step=step,
                checksum_refreshed=False, packet_opened=False))
        cases.append(dict(id=case['id'], archive=str(run), scanned_archive_files=len(all_files),
            archive_name_scan_scope='this exact run tree only; binary file bodies never opened',
            standalone_draw_or_condition_candidates=candidates,
            standalone_assets_found=bool(candidates), cached_weights=weights,
            model_metadata_files=config_files, model_index=model_index, scheduler_config=scheduler,
            tokenizer_binary=[stat_record(p) for p in sorted(model.rglob('tokenizer.model'))],
            available=['exact prompt text/heldout/seed/grid recipe metadata',
                'local native model/tokenizer/scheduler config files',
                'declared prior model/initial-checkpoint checksums and present tensor-file sizes',
                'frozen native baseline PNGs and archive source'],
            missing=['persisted initial-noise bytes/draw trace and selected-row checksum index',
                'native numeric text embeddings/masks (Sana) or embeddings/pooled vectors (SD3)',
                'complete fixed condition table and collision receipt',
                'numeric training-only response_scale and frozen final calibrated recipe',
                'production selected-row descriptor reader/worker, lease/guardian binding, staged native parity and resume harness'],
            historical_replay_qualified=False,
            caveat='No standalone assets in audited tree does not rule out opaque RNG fields inside untouched packets; encodings/noise are not saved by source TrainingState schema.',
            observed_old_mse_timing={k:v for k,v in json.loads((run/'progress.json').read_text()).items()
                if k in ('mean_update_wall_seconds','projected_500_update_wall_hours','peak_allocated_bytes')},
            old_mse_cost_is_gan_bound=False))
    extra_sources = [source_record(ROOT/'checkouts'/checkout/relative) for checkout, relative in (
        ('particle-sliders-sana-cached-negative','conceptmod/textsliders/qualification_state.py'),
        ('particle-sliders-sd3','conceptmod/textsliders/qualification_state.py'),
        ('particle-sliders-sd3','conceptmod/textsliders/lora.py'),
        ('particle-sliders-sd3','scripts/sd3_qualification_images.py'),
    )]
    return dict(schema='observed-image-offline-assets-v1', metadata_only=True,
        additional_source_bindings=extra_sources,
        verified_frozen_sources_and_inputs=verified, cases=cases, known_packets=packets,
        runtime=runtime_metadata(), active_moss_files_modified=False, gpu_queries_or_calls=0,
        checkpoint_or_weight_bodies_opened=0, native_model_imports_or_forwards=0,
        archive_schema_source='TrainingState.save: identity,step,trainable,optimizer,torch_rng,python_rng,cuda_rng; no descriptor/cache payload',
        source_save_timing=dict(sana='step0 after model/optimizer setup+text/render encoding, before diagnostics/render; every20 after optimizer, before checkpoint diagnostics/render',
            sd3='step0 after model/optimizer setup+training/heldout encoding+restore, before evaluate; later checkpoints after optimizer',
            limitation='RNG hash presence alone does not establish draw replay. Baseline/render/source/runtime RNG consumption must be audited; CUDA RNG bytes cannot be replaced with CPU seed7.'))


def admission(asset, audit_sha):
    native_worker = CHECKOUT/'scripts/run_image_failure_campaigns_pg09.py'
    root = ROOT/'runs/image-failure-pg09-proposed-20261017'
    def stage(name, depends, caps, gate, outputs, *, argv=None):
        return dict(id=name, depends_on=depends, status='not executed; root admission pending',
            caps_proposed_not_authorized=caps, required_gate=gate, outputs=outputs,
            execution_argv=argv, worker_implementation_available=native_worker.exists() if argv is None else True,
            blocked_on=['root authorization','MOSS-priority capacity']+(['native worker implementation/source freeze'] if argv is None else []))
    common=dict(threads=1, gpu_count=1, host_available_min_gib=16, host_rss_max_gib=8,
        gpu_peak_allocated_max_gib=24, gpu_free_reserve_min_gib=6, disk_free_min_gib=100,
        thermal_soft_c=76,thermal_resume_c=70,thermal_hard_c=82, max_cooling_seconds=180,
        retries=0, concurrent_image_workers=1, network=False, guardian='root-owned; implementation binding required')
    stages=[
        stage('native-assets', ['reviewed-static-rng-audit','separate root lease'], dict(common,wall_seconds=1800,optimizer_updates=0),
            'local-only exact model revisions; native encoder arguments incl Sana clean_caption=False/complex_human_instruction=None and SD3 noT5; no old MSE trainer; source/module origin gate before loading',
            ['conditions/native prompt encodings+masks/pooled tensors and checksums','500 training draw descriptors/noise hashes/time-index plan','native scheduler50 timesteps/51 sigmas (SD3)','full condition collision receipt']),
        stage('native-off-teacher-parity', ['native-assets'], dict(common,wall_seconds=1800,optimizer_updates=0),
            'exact archived15 frozen baseline PNG replay with true512 decoder; same CFG/sampler/masks/empty negative; zero-strength native model response and teacher endpoint equality; fail closed on any mismatch',
            ['off-teacher-native-parity.json','matched baseline PNG checksums','actual imported module architecture/source/runtime receipt']),
        stage('calibrate-and-probe', ['native-off-teacher-parity'], dict(common,wall_seconds=1800,optimizer_updates_max=5),
            'training-only complete500 descriptor initial G target-neutral RMS scalar, finite and strictly positive; freeze scalar before any update; both presets preflight D/G connectivity and4th public penalty; measured snapshot/hash/identity validation cost fits caps',
            ['calibration.json','final calibrated recipes+SHA','probe metrics/selected-row receipts/gradient norms and GPU+RSS+hash/snapshot phase timings']),
        stage('fresh-process-resume', ['calibrate-and-probe'], dict(common,wall_seconds_per_process=900,optimizer_updates_per_preset=5),
            'separate disk save/exit/load: uninterrupted5 vs2+3 for each preset; exact next selected row/startG/buffer/prefix/target receipts, losses, adapter/critic/optimizer/regularizer/RNG; export/reload native response and heldout pixels; evaluation restores all RNG',
            ['uninterrupted and resumed receipt+checkpoint hashes','fresh-resume-exact.json','native export-reload parity.json']),
        stage('full1000-arms', ['fresh-process-resume','root accepts calibrated final recipe and measured wall projection'], dict(common,wall_seconds_per_arm=21600,g_updates_per_arm=1000,d_updates_per_arm=1000,batch_size=2,descriptor_count=500),
            'root separately admits each case/preset sequentially; both bispf and bispf_optimized; no tiny gate substitutes for full horizon;1000 steps/2000exposures/4complete D+Gcycles; no automatic extension or retries',
            ['full shared-core checkpoints at0/20/250/500/750/1000','all row receipts/coverage/finite metrics/telemetry','adapter export+complete source/input/output checksum manifest']),
        stage('matched-heldout-review', ['completed full1000 case arm'], dict(common,wall_seconds=1800,optimizer_updates=0),
            'exact archived prompts/seeds/noise policy/native CFG/steps/strengths; preserve fruit controls and frozen positive teacher; baseline and GAN images plus source-bound old MSE exports only after separate permission to load them; independent quality judgment',
            ['matched full resolution images and panels','fruit/control/heldout provenance audit','independent quality report; failure retained honestly'])]
    return dict(schema='image-failure-sequential-admission-v1', sole_admission_owner='01a147d7-f540-778e-ad79-4d8eb267403d',
        root_reported_canonical_integration='42ba82f6', immutable_core=asset.get('core'),
        dispatch_allowed=False, automation=False, all_actual_content_stages_not_run=True,
        ordering='Sana stages to qualification then each full arm sequentially, then SD3 stages/arms; root may reorder by capacity through a newly frozen plan; never overlap an image stage with prioritized MOSS',
        native_worker_path=str(native_worker), native_worker_exists=native_worker.exists(),
        native_stage_command_status='execution_argv null truthfully: native stage CLI/reader/guardian orchestration is not implemented; commands cannot be frozen as executable until source reviewed',
        required_future_cli_contract=[str(ROOT/'envs/pg090/bin/python'),str(native_worker),'--plan','<final reviewed calibrated plan path>','--plan-sha256','<frozen SHA>','--case','sana|sd3','--stage','native-assets|native-off-teacher-parity|calibrate-and-probe|fresh-process-resume|full1000-arms|matched-heldout-review','--preset','bispf|bispf_optimized','--root-lease','<root-issued exact process/GPU guardian lease>','--output',str(root/'<case>/<preset>/<stage>')],
        stages=stages, audit_script_sha256=audit_sha,
        blocker_classes=dict(capacity='both MOSS GPUs prioritized; freeing capacity alone does not resolve below',
            authorization='no native loading/forward/cache/GPU/worker/audit packet execution admitted',
            assets='noise/selected-time/numeric condition/scale assets missing in audited scopes',
            implementation='source-bound production reader/native stage worker/guardian+persistence+resume orchestration not implemented',
            qualification='native original parity, fullstate cost measurements, GPU fresh resume and adequately trained quality not yet observed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    proposal_path=CHECKOUT/'evidence/pg09-image-failure-campaigns/proposals.json'
    proposal=json.loads(proposal_path.read_text())
    result=inventory(proposal)
    for name, version in proposal['runtime_versions'].items():
        if result['runtime']['packages'][name]['version'] != version:
            raise ValueError('frozen runtime version drift: '+name)
    provenance=result['runtime']['packages']['diffusers']['direct_url']
    if provenance['vcs_info']['commit_id'] != 'dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d':
        raise ValueError('original Diffusers source commit drift')
    result['core']=proposal['core']
    audit_path=CHECKOUT/'scripts/audit_image_checkpoint_rng_pg09.py'
    audit_sha=source_record(audit_path)['sha256']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    audit=dict(schema='image-known-checkpoint-rng-audit-proposal-v1',execution_admitted=False,
        audit_script_sha256=audit_sha,known_packets=result['known_packets'],
        caps=dict(threads=1,CUDA_VISIBLE_DEVICES='',wall_seconds=120,rss_max_gib=8,
                  address_space_hard_cap_gib=8,host_available_min_gib=16,packet_max_bytes=2*1024**3,
                  caveat='RLIMIT_AS is stronger than RSS cap and may reject Torch import; fail closed; owner must review/import-qualify before actual packet audit'),
        next_stage='static ZIP/pickle opcode inventory before any restricted tensor load; root must review static receipt separately',
        packet_payloads_opened=False,static_executed=False,rng_extracted=False,
        output_policy='schema/RNG hashes only; no trainable/optimizer values, model instantiation, native forward or RNG restoration',
        source_schema_and_timing={k:result[k] for k in ('archive_schema_source','source_save_timing')},
        source_bindings=result['verified_frozen_sources_and_inputs']+result['additional_source_bindings'])
    for name,doc in [('offline-assets.json',result),('rng-audit-plan.json',audit),('sequential-admission-plan.json',admission(result,audit_sha))]:
        (args.output_dir/name).write_text(json.dumps(doc,indent=2)+'\n')
    plan_sha=source_record(args.output_dir/'rng-audit-plan.json')['sha256']
    commands=dict(environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'),
        cwd=str(CHECKOUT),metadata_validation=[str(ROOT/'envs/pg090/bin/python'),str(Path(__file__).resolve()),'--output-dir',str(args.output_dir)],
        unexecuted_static_audit=[str(ROOT/'envs/pg090/bin/python'),str(audit_path),'--plan',str(args.output_dir/'rng-audit-plan.json'),'--plan-sha256',plan_sha,'--mode','static','--output',str(args.output_dir/'actual-static-audit.json')],
        unexecuted_rng_audit_template=[str(ROOT/'envs/pg090/bin/python'),str(audit_path),'--plan',str(args.output_dir/'rng-audit-plan.json'),'--plan-sha256',plan_sha,'--mode','rng','--reviewed-static-receipt',str(args.output_dir/'actual-static-audit.json'),'--reviewed-static-sha256','<ROOT-REVIEWED-STATIC-RECEIPT-SHA256>','--output',str(args.output_dir/'actual-rng-schema-audit.json')],
        actual_packet_commands_admitted=False,native_worker_commands_available=False,
        frozen_inputs=[source_record(proposal_path),source_record(audit_path),source_record(Path(__file__))])
    (args.output_dir/'stage-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(json.dumps(dict(cases=len(result['cases']),metadata_checks=len(result['verified_frozen_sources_and_inputs']),
        packet_stat_bytes=sum(p['bytes'] for p in result['known_packets']),standalone_assets=[c['standalone_assets_found'] for c in result['cases']],
        packet_content_read=False,gpu_used=False,native_worker_implemented=False,output=str(args.output_dir))))


if __name__=='__main__':
    main()
