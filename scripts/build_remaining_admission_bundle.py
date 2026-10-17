#!/usr/bin/env python3
"""Freeze draft sequential commands/contracts only; never admit or dispatch work."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from conceptmod.textsliders.pg09_remaining_admission import CASES,CORE,OUTPUT_ROOT,MODEL_REV,TEXT_REV,VAE_REV,MOSS_REV,CONFIG_SHA,RemainingRecipe,digest,sha
from conceptmod.textsliders.pg09_remaining_offline import MOSS,SUPRA,MOSS_SNAPSHOT


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2)
        stream.write('\n')


def build(offline_path,source_root,output_dir):
    from particlegan import get_recipe
    offline=json.loads(Path(offline_path).read_text())
    if offline.get('status')!='offline_validated_not_admitted':
        raise ValueError('complete read-only offline qualification required')
    source_root=Path(source_root).resolve()
    output_dir=Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    native=Path('/mnt/nvme/sliders/qualification/checkouts/supra-product')
    moss_native=Path('/ml2/codex-storage/future-workspaces/2026-10-16/task-6/moss-debug-dev/src/moss_sliders')
    filenames=['pg09_remaining_admission.py','pg09_remaining_worker.py','pg09_remaining_bind.py','pg09_remaining_campaigns.py',
               'pg09_backends.py','pg09_moss_campaign.py','pg09_moss_worker.py']
    common_sources=[source_root/'conceptmod/textsliders'/f for f in filenames]
    common_sources += [CORE/'packages/particle-sliders-core/src/particle_sliders/pg09.py',CORE/'scripts/moss_original_runtime.py']
    guard=Path('/mnt/nvme/sliders/qualification/ops-sana-kernel-fixed/music3_resources.py')
    recipes={}
    for case in CASES:
        settings=deepcopy(CASES[case])
        root=OUTPUT_ROOT/case/'v1'
        sources=common_sources+([moss_native/'vendor/pipeline_moss_image.py',moss_native/'lora.py'] if case=='final-form' else
                                [native/'supra/runtime.py',native/'vendor/particle-sliders/conceptmod/textsliders/supra_model.py'])
        origins={name:str(Path(importlib.util.find_spec(name).origin).resolve()) for name in settings['versions']}
        if case=='final-form':
            site=Path('/mnt/nvme/sliders/moss-image-sliders/.venv/lib/python3.12/site-packages')
            origins={name:str(site/name/'__init__.py') for name in settings['versions']}
        value=dict(schema='remaining-pg09-worker-v1',case=case,objective='pg09_gan_only',
                   preset='bispf_optimized',sampling='balanced',particles=None,**settings,
                   cohort_id='remaining-sequential-20261017-'+case+'-v1',cohort_root=str(root),
                   output_directory=str(root/'full-gan'),admission='DRAFT_NOT_ADMITTED',physical_gpu=0,
                   lease_paths=['/mnt/nvme/sliders/qualification/admission-gpu-0.lock','/mnt/nvme/sliders/locks/gpu-0.lock'],
                   owner_priority_gate=dict(path=str(OUTPUT_ROOT/'owner-priority-handoff.json'),sha256=None),
                   source_pins={str(p):sha(p) for p in sources},
                   resource_guard=dict(path=str(guard),sha256=sha(guard)),package_origins=origins,
                   package_source_sha256={name:sha(p) for name,p in origins.items()},
                   resolved_recipe_sha256=digest(asdict(get_recipe('bispf_optimized',total_steps=settings['total_steps'],batch_size=settings['batch_size']))),
                   offline_review_path=str(Path(offline_path).resolve()),offline_review_sha256=sha(offline_path))
        inputs={str(Path(offline_path).resolve()):sha(offline_path)}
        if case=='final-form':
            old={}
            for step in (200,254):
                path=MOSS/'checkpoints'/f'step-{step:06d}'/'adapter'
                old['MSE'+str(step)]=dict(path=str(path/'adapter.safetensors'),sha256=sha(path/'adapter.safetensors'),
                                         manifest=str(path/'slider.json'),manifest_sha256=sha(path/'slider.json'),completed_steps=step)
            for name in ('recipe.json','target-index.json','embeddings.json','embeddings.safetensors'):
                inputs[str(MOSS/name)]=sha(MOSS/name)
            for path in (MOSS/'targets').glob('*.json'):
                inputs[str(path)]=sha(path)
            # Provider identity is pure CPU metadata, not a pretrained load.
            from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows
            provider=MossCampaignRows(MOSS)
            proof_path=Path('/mnt/nvme/sliders/qualification/runs/moss-pg09-original-runtime-v2-20261017/bad-intent/fresh-process-resume.json')
            proof=json.loads(proof_path.read_text())
            if not proof.get('exact_full_state') or proof.get('verified_step')!=3:
                raise ValueError('prior original-runtime native MOSS parameter proof is incomplete')
            inputs[str(proof_path)]=sha(proof_path)
            value.update(expected_base_parameter_sha256=proof['frozen_base_sha256'],model_revision=MOSS_REV,model_snapshot=str(MOSS_SNAPSHOT),cache_directory=str(MOSS),
                         cache_identity_sha256=digest(provider.identity()),signed_rows=15360,teacher_indices=[0,39],old_adapters=old)
            value['model_assets']=[a for a in offline['model_assets'] if 'models--OpenMOSS' in a['path']]
        else:
            old={}
            for label,folder,step in [('MSE1600','benchmark',1600),('MSE-selected27200','converged',27200)]:
                path=SUPRA/folder
                old[label]=dict(path=str(path/'final-boss-supra.safetensors'),sha256=sha(path/'final-boss-supra.safetensors'),
                                manifest=str(path/'final-boss-supra.json'),manifest_sha256=sha(path/'final-boss-supra.json'),completed_steps=step)
            config=SUPRA/'benchmark/source/prompts-final-boss.yaml'
            for path in (config,SUPRA/'recipe.json',SUPRA/'benchmark/teacher_cache.pt',SUPRA/'converged/validation_cache.pt'):
                inputs[str(path)]=sha(path)
            value.update(expected_base_parameter_sha256=None,model_revision=MODEL_REV,text_revision=TEXT_REV,vae_revision=VAE_REV,
                         config_path=str(config),config_canonical_sha256=CONFIG_SHA,
                         teacher_cache=str(SUPRA/'benchmark/teacher_cache.pt'),
                         teacher_cache_sha256=sha(SUPRA/'benchmark/teacher_cache.pt'),
                         logical_rows=600,teacher_indices=[0,239],old_adapters=old,
                         contexts=dict(path=str(root/'context-preparation/frozen-contexts.safetensors'),sha256=None,
                                       receipt_sha256=None,preparation_phase_path=str(root/'context-preparation/phase.json'),
                                       preparation_phase_sha256=None,case_budget_sha256=None,unique_prompts=19))
            value['model_assets']=[a for a in offline['model_assets'] if 'models--OpenMOSS' not in a['path']]
        for old in value['old_adapters'].values():
            inputs[old['path']]=old['sha256']
            inputs[old['manifest']]=old['manifest_sha256']
        value['input_sha256']=inputs
        RemainingRecipe.validate(value)
        recipes[case]=value
    preparation=deepcopy(recipes['supra-final-boss'])
    preparation['output_directory']=str(Path(preparation['cohort_root'])/'context-preparation')
    recipes['supra-contexts']=preparation
    for name,value in recipes.items():
        write(output_dir/(name+'.draft.json'),value)
    phases=[('FF01','final-form','probe'),('FF02','final-form','resume-check'),('FF03','final-form','train'),('FF04','final-form','evaluate'),
            ('S01','supra-contexts','contexts'),('S02','supra-final-boss','probe'),('S03','supra-final-boss','resume-check'),
            ('S04','supra-final-boss','train'),('S05','supra-final-boss','evaluate')]
    commands=[]
    for phase,key,mode in phases:
        recipe=recipes[key]
        path=output_dir/(key+'.draft.json')
        environment=dict(CUDA_VISIBLE_DEVICES='0',HF_HOME=recipe['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                         TOKENIZERS_PARALLELISM='false',PARTICLE_SLIDERS_ROOT=str(native/'vendor/particle-sliders'),
                         PYTHONPATH=':'.join([str(source_root),str(CORE/'packages/particle-sliders-core/src'),str(CORE/'scripts'),
                                               str(moss_native.parent),str(native),'/mnt/nvme/sliders/qualification/pg09-audit/wheel']))
        commands.append(dict(phase=phase,case=key,mode=mode,environment=environment,
                             argv=[recipe['python'],'-m','conceptmod.textsliders.pg09_remaining_worker','--recipe',str(path),
                                   '--output',recipe['output_directory'],'--mode',mode,
                                   '--lease-fd','{owner_admission_fd}','--lease-fd','{owner_gpu_fd}'],
                             executed=False,root_guardian_required=True,
                             blocked_by=['owner priority safe-handoff','capacity/leases','reviewed admission']+
                                        (['reviewed19-context hashes must be rebound afterS01'] if key=='supra-final-boss' else [])))
    plan=dict(schema='remaining-sequential-admission-v1',status='DRAFT_NOT_ADMITTED',automatic_dispatch=False,
              prefix='Existing BI/Afterimage full budgets and matched heldouts retain priority; owner must record safe handoff',
              source_root=str(source_root),offline_review_sha256=sha(offline_path),commands=commands,
              stop_after_S01='Owner reviews actual native19-context receipt; bind hashes into NEW full Supra recipe beforeS02',
              total_case_wall_includes_all_phases_and_waits=True,model_weights_loaded=False,native_forward=False,T5_encode=False)
    write(output_dir/'sequential-plan.json',plan)
    return plan


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline-review',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    build(args.offline_review,args.source_root,args.output)


if __name__=='__main__':
    main()
