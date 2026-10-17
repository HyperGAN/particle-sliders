"""Read archived Sana Happy/SD3 Smile metadata and freeze CPU-only proposals.

No checkpoint/weight reads, model loading, subprocesses, GPU calls or dispatch.
"""
import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path

import yaml

CORE = '3e94b13eb5d6cd1ceee3f018f019a68603f35696'
CORE_SHA = '426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256'
ROOT = Path('/mnt/nvme/sliders/qualification')


def file_record(path):
    path = Path(path)
    if path.suffix in ('.pt','.safetensors','.bin'):
        raise ValueError('campaign preparation refuses checkpoint/weight content reads')
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read_json(path):
    record = file_record(path)
    return json.loads(Path(path).read_text()), record


def proposals():
    sana = ROOT/'runs/sana-happy-20261017'
    sd3 = ROOT/'runs/sd3-smile-20261017'
    sana_recipe, sana_bind = read_json(sana/'recipe.json')
    sd3_recipe, sd3_bind = read_json(sd3/'recipe.json')
    training = yaml.safe_load(Path(sana_recipe['prompts_file']).read_text())
    assert file_record(sana_recipe['prompts_file'])['sha256'] == sana_recipe['prompts_sha256']
    sana_heldout = [
        dict(neutral='a portrait photograph of an adult woman with a neutral expression, plain background',
             positive='a portrait photograph of an adult woman smiling, happy expression, plain background',seeds=[123,456]),
        dict(neutral='a portrait photograph of an adult man with a neutral expression, plain background',
             positive='a portrait photograph of an adult man smiling, happy expression, plain background',seeds=[123,456]),
        dict(neutral='a bowl of fruit on a table',positive='a bowl of fruit on a table',seeds=[789],control=True)]
    result = dict(schema='observed-image-failure-pg09-campaign-proposals-v1',
        status='CPU-prepared proposals; no live admission',dispatch_allowed=False,
        scope=['sana-happy-20261017','sd3-smile-20261017'],
        core=dict(commit=CORE,module_sha256=CORE_SHA,pg_source_profile='release',penalty_input_domain='auto',
                  response_domain='continuous_field',pg_math_changed=False), campaigns=[])
    for family,base,recipe,binding in [('sana',sana,sana_recipe,sana_bind),('sd3',sd3,sd3_recipe,sd3_bind)]:
        initial,initial_bind = read_json(base/'checkpoints/step-000000.json')
        disposition,disp_bind = read_json(base/'objective-disposition.json')
        progress,progress_bind = read_json(base/'progress.json')
        assert disposition['classification']=='wrong-objective MSE baseline' and progress['latest_step']==500
        heldout = sana_heldout if family=='sana' else [dict(row,seeds=[123+i],control=i==2)
                                                    for i,row in enumerate(recipe['heldout'])]
        snapshots = [binding, initial_bind, disp_bind, progress_bind]
        if family=='sd3':
            snapshots.append(file_record(ROOT/'models/sd3-ea42f8c/qualification-manifest.json'))
        for name in (['quality-review.json','real-baseline-validation.json','implementation-repair-20261017.json','model-manifest.json']
                     if family=='sana' else ['receipts/full.json']):
            snapshots.append(file_record(base/name))
        baseline = []
        if family=='sana':
            for path in sorted((base/'samples/baselines').glob('*.png')):
                baseline.append(file_record(path))
        else:
            for i in range(3):
                for name in ('Off','Frozen_positive_CFG3'):
                    baseline.append(file_record(base/f'samples/step-000000/case-{i}-{name}.png'))
        lr = 2e-5 if family=='sana' else 1e-4
        arms = []
        from particlegan import get_recipe
        for preset in ('bispf','bispf_optimized'):
            resolved = asdict(get_recipe(preset,batch_size=2,total_steps=1000,lr=lr))
            arms.append(dict(preset=preset,resolved_public_recipe=resolved,
                generator_objective='public adversarial loss only; no prior without a participating bank',
                critic_objective='public adversarial loss + public critic penalty',
                particles=None,arm='adapter GAN; no disconnected particle-bank claim',
                g_updates=1000,batch_size=2,descriptor_count=500,full_g_cycles=4,full_d_cycles=4,
                g_exposures=2000,d_exposures=2000,
                budget_basis='historical500-update floor plus4complete independent balanced cycles of500descriptors; proposal not convergence proof',
                evaluation_updates=[0,20,250,500,750,1000],
                stopping='full authorized horizon; invariant/nonfinite/resource failure may stop; no canary substitutes for trained quality'))
        model_dir = (recipe['model_path'] if family=='sana' else
            json.loads((ROOT/'models/sd3-ea42f8c/qualification-manifest.json').read_text())['model_dir'])
        response_shape = [1,32,16,16] if family=='sana' else [1,16,64,64]
        result['campaigns'].append(dict(id=family+'-'+('happy' if family=='sana' else 'smile')+'-pg09-proposal',
            archived_run=str(base), archive_bindings=snapshots, frozen_baseline_assets=baseline,
            model=dict(id=recipe['model_id'] if family=='sana' else 'stabilityai/stable-diffusion-3-medium-diffusers',
                       revision=recipe['model_revision'],directory=model_dir,
                       weight_content_read=False,manifest_hashes_retained=True),
            initial_checkpoint_metadata=initial, checkpoint_content_loaded=False,
            training_prompts=training if family=='sana' else recipe['training_prompts'],
            heldout=heldout,evaluation=dict(resolution=512,steps=20 if family=='sana' else 28,
                cfg=4.5 if family=='sana' else 3.,strength_grid=[0.,.5,1.] if family=='sana' else [0.,-1.,.5,1.],
                frozen_positive_teacher=True,fruit_is_evaluation_only=True,
                sampler='original native pipeline, original cached conditioning, explicit empty negative',
                adapted_cfg_branches='both, matching archive',generator='CPU torch.Generator with per-case fixed seed'),
            training_protocol=dict(seed=7,initial_noise='original device-RNG descriptor plan; exact tensor checksums required after root admission',
                source_rng_extraction='not done; no checkpoint loaded',
                selected_times='fixed raw t500' if family=='sana' else 'recorded index1..49 of original50-step scheduler',
                state='raw initial noise' if family=='sana' else 'CURRENT-adapter native no-grad prefix, detached before endpoint',
                target='frozen empty +4.5*(frozen positive-empty)' if family=='sana' else
                       'native BF16 neutral_endpoint + (positive_endpoint-negative_endpoint)',
                student='raw neutral at CFG1' if family=='sana' else 'native neutral BF16 scheduler-step endpoint at CFG1',
                native_response_shape=response_shape,
                adapter='same complete attn2 parameter set; frozen W + FP32delta' if family=='sana' else
                        'same Attention/full rank4 alpha1 LoRA; FP32 trainables required by PG, BF16 frozen host',
                precision_delta_from_archive=None if family=='sana' else 'old LoRA leaves BF16; PG requires FP32 leaves, explicitly declared',
                original_training_updates=500,original_training_batch=1,
                fruit_training_added=False,conditioning='full numeric tensor fixed source-bound projection + global descriptor collision gate'),
            critic=dict(class_name='ImageFailureCritic',native_shape=response_shape,hidden_sizes=[256,128],
                conditioning_dim=106,raw_logits=True,all_native_coordinates=True,flatten_only_inside_critic=True),
            arms=arms,response_coordinates=dict(domain='continuous_field',shape=response_shape,
                response_scale=dict(value=None,rule='freeze one finite positive RMS(target-neutral) scalar from training-only scale0 calibration',
                    heldout_used=False,optimized_loss=False),sigma=1.,wide_vector_overlay=False),
            online_provider=dict(protocol='online-current-adapter-snapshot-proposal-v1',
                immutable='descriptor/noise/context/time/guidance/source/condition-index identities',
                dynamic='prefix and paired targets regenerated from actual start-of-step G',
                selected_row_receipt=['logical_step','G_state_sha256','descriptor_sha256','initial_noise_sha256','prefix_sha256','target_sha256','neutral_sha256'],
                same_G_version='physical parameter version guard across preflight/G/D fetches; full adapter-state digest at step start',
                memory='selected unique D/G union only; no full response target concatenation',
                state_hash_cost='full FP32 generator adapter-state readback/hash per logical step; Sana148M leaves cost unmeasured and must enter runtime admission budget',
                resume='load coreG+completedstep, re-arm and deterministically regenerate selected rows; no provider RNG',
                semantic_review='accepted for CPU contract prototype only; not old immutable-cache qualification'),
            pending_admission_gates=['root GPU/process admission (MOSS currently owns actual model slots)',
                'exact initial-noise/selected-time descriptor asset manifest and global condition collision audit',
                'source/runtime/model-manifest revalidation without downloads',
                'native no-update frozen teacher/Off pixel replay, masks/CFG/sampler and true512decode parity',
                'online provider independent review and selected-row/G-version provenance audit',
                'actual model gradient and GPU memory/runtime measurement',
                'separate-process file-backed GPU checkpoint resume with exact next row/state/loss',
                'full authorized1000-update run and matched heldout/fruit panels before any quality claim'],
            alternatives=[dict(name='frozen-prefix cache',selected=False,
                tradeoff='changes SD3 original CURRENT-adapter state distribution; requires explicit separate protocol/approval'),
                dict(name='raw velocity for SD3 or conditional-only CFG rendering',selected=False,
                tradeoff='changes archived endpoint teacher or rendering; not silently substituted')]))
    dist = metadata.distribution('diffusers')
    runtime_sources = [dist.locate_file(name) for name in (
        'diffusers/pipelines/sana/pipeline_sana.py','diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py',
        'diffusers/schedulers/scheduling_flow_match_euler_discrete.py','diffusers/schedulers/scheduling_dpmsolver_multistep.py')]
    source_files = [ROOT/'checkouts/particle-sliders-sana-cached-negative'/name for name in (
        'scripts/qualify_sana.py','conceptmod/textsliders/train_lora_sana.py','conceptmod/textsliders/slider_targets.py')]
    source_files += [ROOT/'checkouts/particle-sliders-sd3'/name for name in (
        'scripts/qualify_sd3.py','scripts/sd3_qualification_images.py','conceptmod/textsliders/train_lora_sd3.py',
        'conceptmod/textsliders/train_util.py','conceptmod/textsliders/prompt_util.py','conceptmod/textsliders/model_util.py')]
    result['source_bindings']=[file_record(path) for path in source_files+runtime_sources]
    core_path=Path('/mnt/nvme/sliders/image-campaign-core-pin-3e94/packages/particle-sliders-core/src/particle_sliders')
    assert file_record(core_path/'pg09.py')['sha256']==CORE_SHA
    expected=json.loads((core_path/'pg09_release_sources.json').read_text())
    pg_root=metadata.distribution('particlegan').locate_file('particlegan')
    actual={str(path.relative_to(pg_root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(pg_root.rglob('*.py'))}
    assert actual==expected and len(actual)==49
    result['core']['verified_pg_python_files']=49
    result['core']['pg_sources_sha256']=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()
    result['source_bindings'].append(file_record(core_path/'pg09.py'))
    result['runtime_versions']={name:metadata.version(name) for name in
        ('torch','diffusers','transformers','particlegan','safetensors','numpy','Pillow')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    document=proposals()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),campaigns=len(document['campaigns']),dispatch_allowed=False,
                         weight_content_read=False,quality_claim=False)))


if __name__=='__main__':
    main()
