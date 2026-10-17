"""Owner-invoked FF/Supra phases. Default fail closed; no guardian or admission.

Import/plan are stdlib-only. Every live mode requires inherited leases and exact
reviewed imports/runtime/inputs. All optimization is delegated to ConditionalPG09.
"""
import argparse
from dataclasses import asdict
import importlib
import importlib.util
import inspect
import json
import math
import os
import resource
from pathlib import Path
import signal
import sys
import time
import uuid

from .pg09_remaining_admission import RemainingRecipe, digest, sha
from .pg09_moss_worker import (atomic_json, inherited_leases, parameter_sha, tree_equal,
                              save_checkpoint, restore_latest, verify_checkpoint, no_optimized_mse, GradientWitness)


def immutable_json(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError('immutable JSON differs: '+str(path))
        return
    atomic_json(path,value)


def immutable_packet(path,packet):
    import torch
    path=Path(path)
    if path.exists():
        if not tree_equal(torch.load(path,map_location='cpu',weights_only=True),packet):
            raise ValueError('immutable file reference differs')
        return sha(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name('.pending-'+uuid.uuid4().hex)
    with pending.open('xb') as stream:
        torch.save(packet,stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)
    return sha(path)


def frozen_identity(model,guard):
    """Stream only frozen base parameters and buffers, never clone a full base."""
    import hashlib
    import torch
    class FrozenView:
        def named_parameters(self):
            return ((n,p) for n,p in model.named_parameters() if not p.requires_grad)
    parameters=parameter_sha(FrozenView(),frozen_only=True,guard=guard)
    buffers=hashlib.sha256()
    with torch.no_grad():
        for name,value in model.named_buffers():
            guard()
            x=value.detach().cpu().contiguous()
            buffers.update(str((name,list(x.shape),str(x.dtype))).encode())
            buffers.update(memoryview(x.reshape(-1).view(torch.uint8).numpy()))
    result=dict(parameters_sha256=parameters,buffers_sha256=buffers.hexdigest(),
                parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    result['combined_sha256']=digest(result)
    return result


def import_and_attest(recipe):
    """Actual definitions must originate in pinned files, before any weights load."""
    names=['conceptmod.textsliders.pg09_remaining_admission','conceptmod.textsliders.pg09_remaining_worker',
           'conceptmod.textsliders.pg09_remaining_campaigns','conceptmod.textsliders.pg09_backends',
           'conceptmod.textsliders.pg09_moss_campaign','conceptmod.textsliders.pg09_moss_worker','particle_sliders.pg09',
           'moss_original_runtime']
    if recipe.data['case']=='final-form':
        names+=['moss_sliders.vendor.pipeline_moss_image','moss_sliders.lora']
    else:
        names+=['supra.runtime']
    origins={name:recipe.verify_origin(importlib.import_module(name)) for name in names}
    packages={}
    for name,version in recipe.data['versions'].items():
        module=importlib.import_module(name)
        origin=str(Path(module.__file__).resolve())
        if str(module.__version__)!=version or recipe.data['package_origins'].get(name)!=origin:
            raise ValueError('runtime version/origin differs: '+name)
        if recipe.data['package_source_sha256'].get(name)!=sha(origin):
            raise ValueError('runtime package source changed: '+name)
        packages[name]=dict(version=version,path=origin,sha256=sha(origin))
    if sys.executable!=recipe.data['python']:
        raise ValueError('exact reviewed Python interpreter required')
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('CUDA initialized before source/runtime attestation')
    from particle_sliders.pg09 import validate_pg09_domain
    shape=(1,128,32,32) if recipe.data['case']=='final-form' else (1,4,32,32)
    domain=validate_pg09_domain(shape,response_domain='continuous_field')
    return dict(imports=origins,packages=packages,pg_domain=domain,python=sys.executable)


def resource_guard(recipe):
    source=recipe.data['resource_guard']
    spec=importlib.util.spec_from_file_location('remaining_reviewed_guard',source['path'])
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if str(Path(module.__file__).resolve())!=str(Path(source['path']).resolve()) or sha(module.__file__)!=source['sha256']:
        raise ValueError('actual ResourceGuard origin differs')
    return module.ResourceGuard()


def verify_inputs(recipe):
    """Small reviewed source/cache/sidecar identities; model weights are not opened."""
    handoff=recipe.data['owner_priority_gate']
    if sha(handoff['path'])!=handoff['sha256']:
        raise ValueError('owner priority handoff checksum differs')
    priority=json.loads(Path(handoff['path']).read_text())
    if (priority.get('schema')!='remaining-priority-handoff-v1' or priority.get('physical_gpu')!=recipe.data['physical_gpu'] or
            priority.get('owner_admitted') is not True or priority.get('active_moss_pins_unchanged') is not True):
        raise ValueError('owner safe MOSS priority handoff is incomplete')
    for path,expected in recipe.data['input_sha256'].items():
        if sha(path)!=expected:
            raise ValueError('input/metadata identity changed: '+path)
    for asset in recipe.data['model_assets']:
        if not Path(asset['path']).is_file() or Path(asset['path']).stat().st_size!=asset['bytes']:
            raise ValueError('local model asset missing/size differs: '+asset['path'])
    if recipe.data['case']=='supra-final-boss':
        import yaml
        from .pg09_remaining_campaigns import validate_live_supra_context_preparation
        config=yaml.safe_load(Path(recipe.data['config_path']).read_text())
        validate_live_supra_context_preparation(config,recipe.data)
        contexts=recipe.data['contexts']
        if contexts['sha256'] is not None:
            if sha(contexts['path'])!=contexts['sha256'] or sha(Path(contexts['path']).with_suffix('.json'))!=contexts['receipt_sha256']:
                raise ValueError('reviewed19-context export/receipt differs')
            phase_path=Path(contexts['preparation_phase_path'])
            if sha(phase_path)!=contexts['preparation_phase_sha256']:
                raise ValueError('native context preparation phase proof changed')
            preparation=json.loads(phase_path.read_text())
            if (preparation.get('status')!='contexts_prepared' or preparation.get('training_updates')!=0 or
                    preparation.get('native_encode_calls_at_most')!=19 or
                    preparation.get('context_sha256')!=contexts['sha256'] or
                    preparation.get('context_receipt_sha256')!=contexts['receipt_sha256'] or
                    preparation.get('cohort_budget_sha256')!=contexts['case_budget_sha256']):
                raise ValueError('native19-context preparation proof is incomplete')
            receipt=json.loads(Path(contexts['path']).with_suffix('.json').read_text())
            runtime_source=receipt.get('runtime_source','')
            if (receipt.get('unique_contexts')!=19 or receipt.get('config_sha256')!=recipe.data['config_canonical_sha256'] or
                    receipt.get('runtime_identity',{}).get('model_revision')!=recipe.data['model_revision'] or
                    receipt.get('runtime_identity',{}).get('text_revision')!=recipe.data['text_revision'] or
                    recipe.data['source_pins'].get(runtime_source)!=receipt.get('runtime_source_sha256') or
                    not receipt.get('stored_unconditional')):
                raise ValueError('actual native context runtime/revision/config identity differs')
        return config
    return None


def cuda_setup():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('admitted CUDA unavailable; no CPU or MSE fallback')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False



def register_native_supra_source(module,recipe):
    """Register the product's verified dynamic import for class-source inspection.

    The native factory executes a module without registering it in sys.modules.
    Core requires inspectable native classes. This registers a NEW import name;
    it does not replace native methods, mutate source, construct weights or forward.
    """
    expected_source=str(module.MODEL_SOURCE.resolve())
    expected_sha=recipe.data['source_pins'].get(expected_source)
    if expected_sha is None or sha(expected_source)!=expected_sha:
        raise ValueError('dynamic native module source is not pinned before import')
    namespace='supra_particle_model'
    old=sys.modules.get(namespace)
    if old is not None:
        if (str(Path(getattr(old,'__file__','')).resolve())!=expected_source or
                getattr(old,'_pg09_verified_source_sha256',None)!=expected_sha):
            raise ValueError('dynamic native module name already has another origin/hash')
    try:
        native=module.model_module()
        if native.__name__!=namespace:
            raise ValueError('native dynamic namespace must be deterministic supra_particle_model')
        source=str(Path(native.__file__).resolve())
        if source!=expected_source or sha(source)!=expected_sha:
            raise ValueError('dynamic native module source differs')
        if sys.modules.get(namespace) not in (None,old):
            raise ValueError('native factory mutated the reserved module registry')
        if old is None:
            native._pg09_verified_source_sha256=expected_sha
            sys.modules[namespace]=native
        return source
    except BaseException:
        if old is None:
            sys.modules.pop(namespace,None)
        else:
            sys.modules[namespace]=old
        raise


def load_native(recipe,guard,config):
    """Only reached after owner admission. CPU tests replace this entire boundary."""
    import torch
    values=recipe.data
    guard()
    if values['case']=='final-form':
        from moss_sliders.vendor.pipeline_moss_image import MOSSImageTransformer
        model=MOSSImageTransformer.from_pretrained(values['model_snapshot'],subfolder='transformer',
                                                  local_files_only=True,torch_dtype=torch.bfloat16)
        model.requires_grad_(False).eval().to('cuda:0')
        guard.attach(model)
        return model
    import supra.runtime as native_runtime
    from supra.runtime import SupraRuntime,MODEL_SOURCE
    if values['source_pins'].get(str(MODEL_SOURCE.resolve()))!=sha(MODEL_SOURCE):
        raise ValueError('actual native Supra model implementation is not pinned')
    registered=register_native_supra_source(native_runtime,recipe)
    runtime=SupraRuntime(device='cuda:0',rank=16,allow_hub=False)
    if (str(Path(inspect.getsourcefile(type(runtime.model))).resolve())!=registered or
            str(Path(type(runtime.model).forward.__code__.co_filename).resolve())!=registered):
        raise ValueError('actual native forward implementation differs from verified import')
    for model in (runtime.model,runtime.text_encoder,runtime.vae):
        guard.attach(model)
    return runtime


def bind_run(recipe,native,guard,config):
    from .pg09_remaining_campaigns import (build_final_form_campaign,build_supra_campaign,
                                          load_supra_contexts,SupraCampaignRows)
    from .pg09_moss_campaign import MossCampaignRows
    base=frozen_identity(native if recipe.data['case']=='final-form' else native.model,guard)
    if base['parameters_sha256']!=recipe.data['expected_base_parameter_sha256']:
        raise ValueError('actual frozen base parameter identity differs from admitted native proof')
    if recipe.data['case']=='supra-final-boss' and base['parameters']!=native.base_params:
        raise ValueError('native base parameter ownership/count differs')
    identity=dict(model_revision=recipe.data['model_revision'],revision=recipe.data['model_revision'],
                  model=recipe.data['case'],frozen_parameter_sha256=base['parameters_sha256'],
                  frozen_buffers_sha256=base['buffers_sha256'])
    if recipe.data['case']=='final-form':
        rows=MossCampaignRows(recipe.data['cache_directory'],split='train',device='cuda:0')
        if digest(rows.identity())!=recipe.data['cache_identity_sha256']:
            raise ValueError('FinalForm cache/index/conditioning policy differs')
        run=build_final_form_campaign(native,rows,model_identity=identity)
    else:
        contexts=recipe.data['contexts']
        bank=load_supra_contexts(contexts['path'],contexts['sha256'],config)
        native.cache={p:tuple(t.to(native.device) for t in pair) for p,pair in bank.items()}
        # Cached encoding is complete; no T5 forward is permitted in any later phase.
        def forbidden_encode(*args,**kwargs):
            raise RuntimeError('T5 forward forbidden after reviewed context preparation')
        native.text_encoder.register_forward_pre_hook(forbidden_encode)
        native.text_encoder.to('cpu')
        rows=SupraCampaignRows(recipe.data['teacher_cache'],cache_sha256=recipe.data['teacher_cache_sha256'],
                               config=config,contexts=bank,split='train')
        if len(rows)!=600 or rows.condition_width!=774 or len(rows.records)!=240 or len(rows.holds)!=30:
            raise ValueError('full native Supra dataset/condition dimensions differ')
        run=build_supra_campaign(native,rows,model_identity=identity,response_scale=recipe.data['response_scale'])
    if digest(asdict(run.game.recipe))!=recipe.data['resolved_recipe_sha256']:
        raise ValueError('actual selected public PG recipe differs')
    guard.attach(run.game.critic)
    return run,base


def measure_resources(output):
    import torch
    rss=next(int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    total=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file() and not p.is_symlink())
    return dict(rss_bytes=rss,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),output_bytes=total)


def check_caps(recipe,output,started,mode,*,clock=time.monotonic,measure=measure_resources,cohort_deadline=None):
    caps=recipe.data['caps']
    deadline={'contexts':caps.get('phase_contexts_seconds',0),'probe':caps['phase_probe_seconds'],
              'resume-check':caps['phase_resume_seconds'],'train':caps['wall_seconds'],
              'evaluate':caps['phase_evaluate_seconds']}[mode]
    stats=measure(output)
    if ((cohort_deadline is not None and time.time()>=cohort_deadline) or clock()-started>=deadline or max(stats['rss_bytes'],stats.get('rss_peak_bytes',0))>caps['max_rss_gib']*1024**3 or
            max(stats['cuda_reserved_bytes'],stats.get('cuda_peak_reserved_bytes',0))>caps['max_reserved_gib']*1024**3 or
            stats['output_bytes']>caps['output_gib']*1024**3):
        raise RuntimeError('reviewed phase wall/RAM/VRAM/disk cap exceeded: '+json.dumps(stats))
    return stats


def export_and_verify(run,path):
    """Portable inference export plus destructive-adapter reload test; no base copy."""
    import torch
    from .pg09_backends import save_adapter,load_adapter
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    state=run.game.state_dict()
    packet=run.game.export_state_dict()
    expected=run.game.evaluate(0,seed=29001)
    fingerprint=digest(run.backend.model_identity)
    pending=path.parent/('.pending-'+uuid.uuid4().hex)
    pending.mkdir(parents=True)
    try:
        save_adapter(run.backend,pending/'adapter.safetensors',model_fingerprint=fingerprint)
        immutable_json(pending/'inference.json',dict(schema='remaining-gan-inference-v1',config=packet['config'],
                      completed_steps=packet['completed_steps'],sha256=sha(pending/'adapter.safetensors')))
        with torch.no_grad():
            for parameter in run.backend.network.parameters():
                parameter.zero_()
        load_adapter(run.backend,pending/'adapter.safetensors',model_fingerprint=fingerprint)
        run.game.load_export_state_dict(dict(schema='conditional-pg09-adapter-v1',config=packet['config'],
                                            completed_steps=packet['completed_steps'],network=run.backend.network.state_dict()))
        if not tree_equal(expected,run.game.evaluate(0,seed=29001)):
            raise ValueError('inference export/reload sample differs')
        pending.rename(path)
    finally:
        run.game.load_state_dict(state)
    return dict(exact_export_reload=True,export_sha256=sha(path/'adapter.safetensors'))


def check_supra_protocol_measurements(noise,velocity):
    """Pure diagnostic gate; synthetic CPU fixtures supply measurements directly."""
    if len(noise)!=2 or any(r.get('exact') is not True for r in noise):
        raise ValueError('native CUDA seeded noise differs from archived t0: '+json.dumps(noise))
    if not velocity or any(not math.isfinite(r['relative_rms']) or r['relative_rms']>.001 for r in velocity):
        error=ValueError('native batched/split CFG rounding parity failed: '+json.dumps(velocity))
        error.diagnostics=dict(noise=noise,velocity=velocity)
        raise error
    return dict(native_cuda_noise=noise,native_batched_vs_split_cfg=velocity,relative_rms_limit=.001,
                optimized_loss=False)


def supra_native_protocol_gate(run,indices,*,emit):
    """Future admitted native measurements, strictly before the first update."""
    import torch
    from .pg09_remaining_campaigns import tensor_identity
    if run.runtime.device.type!='cuda' or run.game.completed_steps!=0:
        raise ValueError('native protocol gate requires admitted CUDA and zero updates')
    seeds=sorted({r['seed'] for r in run.rows.records})
    noise=[]
    velocity=[]
    with torch.no_grad():
        for seed in seeds:
            first=next(r for r in run.rows.records if r['seed']==seed and r['t']==0.)
            actual=run.runtime.noise(seed)
            if actual.device!=run.runtime.device or actual.dtype!=torch.float32:
                raise ValueError('native noise device/dtype differs')
            noise.append(dict(seed=seed,exact=torch.equal(actual.cpu(),first['z']),
                              actual=tensor_identity(actual),archived=tensor_identity(first['z'])))
        for index in indices:
            cached=run.rows[index]
            actual=run.backend.prepare(cached.row)
            denominator=(cached.target-cached.neutral).float().square().mean().sqrt().clamp_min(1e-20)
            for role,prompt,split in [('neutral',cached.row.neutral,actual.neutral),
                                      ('positive',cached.row.positive,actual.target)]:
                batched=run.runtime.velocity(prompt,cached.row.latent,cached.row.timestep,scale=0.,cfg=3.)
                velocity.append(dict(index=index,role=role,exact=torch.equal(batched,split),
                    relative_rms=float((batched-split).square().mean().sqrt()/denominator.to(batched)),
                    max_absolute=float((batched-split).abs().max()),relative_rms_limit=.001))
    emit('native_supra_protocol',noise=noise,velocity=velocity)
    return check_supra_protocol_measurements(noise,velocity)


def qualify_probe(run,recipe,output,base,*,emit,guard,cap_check,base_check,initial_gate,stop_requested=lambda:False):
    """Exactly three updates on the full-budget recipe; raw metrics only in proof."""
    from .pg09_remaining_campaigns import teacher_parity
    if run.game.completed_steps!=0:
        raise ValueError('probe requires fresh zero-step GAN state')
    if recipe.data.get('case')=='supra-final-boss':
        supra_native_protocol_gate(run,recipe.data['teacher_indices'],emit=emit)
    teacher=teacher_parity(run,recipe.data['teacher_indices'])
    initial_gate(run,emit=emit,indices=recipe.data['teacher_indices'])
    emit('teacher_and_gradient_parity',teacher=teacher)
    g_before,d_before=parameter_sha(run.backend.network),parameter_sha(run.game.critic)
    witnesses=[GradientWitness(run.backend.network),GradientWitness(run.game.critic)]
    before=None
    try:
        save_checkpoint(output,recipe,run,base['combined_sha256'])
        for _ in range(3):
            if stop_requested():
                save_checkpoint(output,recipe,run,base['combined_sha256'])
                return dict(status='stopped',completed_steps=run.game.completed_steps,
                            frozen_base_unchanged=base_check(),file_reference_written=False,
                            exact_in_memory_replay=False,exact_export_reload=False)
            guard()
            cap_check()
            for witness in witnesses:
                witness.reset()
            if run.game.completed_steps==2:
                before=run.game.state_dict()
            with no_optimized_mse():
                raw=run.step()
            norms=[w.norm() for w in witnesses]
            if not all(math.isfinite(v) and v>0 for v in norms):
                raise ValueError('actual G/D backward norm is zero or nonfinite')
            emit('probe_step',completed_steps=run.game.completed_steps,metrics=raw,gradient_norms=norms,resources=cap_check())
        expected=run.game.state_dict()
        sample=run.game.evaluate(0,seed=29001)
        reference=dict(schema='remaining-gan-file-reference-v1',recipe_sha256=recipe.identity,
                       frozen_base=base,probe_pid=os.getpid(),before=before,metrics=raw,expected=expected,sample=sample)
        reference_sha=immutable_packet(Path(output)/'probe-reference.pt',reference)
        run.game.load_state_dict(before)
        with no_optimized_mse():
            replay=run.step()
        if not tree_equal(replay,raw) or not tree_equal(run.game.state_dict(),expected) or not tree_equal(run.game.evaluate(0,seed=29001),sample):
            raise ValueError('in-memory exact loss/state/sample replay differs')
        export=export_and_verify(run,Path(output)/'exports/probe-000003')
        save_checkpoint(output,recipe,run,base['combined_sha256'])
        if not base_check():
            raise ValueError('frozen base parameters or buffers changed')
        changed=(g_before!=parameter_sha(run.backend.network),d_before!=parameter_sha(run.game.critic))
        if not all(changed):
            raise ValueError('generator or critic parameters did not change')
        return dict(status='probe_passed',completed_steps=3,frozen_base_unchanged=True,
                    generator_changed=True,critic_changed=True,exact_in_memory_replay=True,
                    file_reference_written=True,reference_sha256=reference_sha,**export)
    finally:
        for witness in witnesses:
            witness.close()


def resume_disk_reference(run,recipe,output,base,previous,*,base_check):
    import torch
    path=Path(output)/'probe-reference.pt'
    if sha(path)!=previous.get('reference_sha256'):
        raise ValueError('disk reference checksum differs')
    packet=torch.load(path,map_location='cpu',weights_only=True)
    if (packet.get('schema')!='remaining-gan-file-reference-v1' or packet.get('recipe_sha256')!=recipe.identity or
            packet.get('frozen_base')!=base or packet.get('probe_pid')==os.getpid()):
        raise ValueError('resume requires a separate process and exact source/recipe/frozen base')
    checkpoint=Path(output)/'checkpoints/step-000003'
    verify_checkpoint(checkpoint,recipe)
    saved=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=True)
    if (saved.get('frozen_base_sha256')!=base['combined_sha256'] or
            not tree_equal(saved.get('game'),packet['expected'])):
        raise ValueError('immutable checkpoint3 differs from probe reference state')
    checkpoint_before=sha(checkpoint/'state.pt')
    run.game.load_state_dict(packet['before'])
    with no_optimized_mse():
        actual=run.step()
    if (not tree_equal(actual,packet['metrics']) or not tree_equal(run.game.state_dict(),packet['expected']) or
            not tree_equal(run.game.evaluate(0,seed=29001),packet['sample']) or not base_check()):
        raise ValueError('fresh-process exact disk loss/state/sample/frozen-base proof failed')
    if sha(Path(output)/'checkpoints/step-000003/state.pt')!=checkpoint_before:
        raise ValueError('resume qualification changed immutable training checkpoint')
    proof=dict(schema='remaining-gan-fresh-disk-proof-v1',recipe_sha256=recipe.identity,
               reference_sha256=sha(path),checkpoint_sha256=checkpoint_before,
               frozen_base=base,probe_pid=packet['probe_pid'],resume_pid=os.getpid(),
               replayed_from_step=2,verified_step=3,checkpoint_advanced=False,
               exact_loss=True,exact_full_state=True,exact_sample=True,frozen_base_unchanged=True)
    immutable_json(Path(output)/'fresh-disk-resume.json',proof)
    return dict(fresh_disk_resume_exact=True,fresh_disk_proof_sha256=sha(Path(output)/'fresh-disk-resume.json'))


def verify_disk_proof(recipe,output,previous,base):
    path=Path(output)/'fresh-disk-resume.json'
    if sha(path)!=previous.get('fresh_disk_proof_sha256'):
        raise ValueError('fresh-process proof checksum differs')
    proof=json.loads(path.read_text())
    verify_checkpoint(Path(output)/'checkpoints/step-000003',recipe)
    if (proof.get('recipe_sha256')!=recipe.identity or proof.get('frozen_base')!=base or
            proof.get('probe_pid')==proof.get('resume_pid') or proof.get('checkpoint_advanced') is not False or
            proof.get('reference_sha256')!=sha(Path(output)/'probe-reference.pt') or
            proof.get('checkpoint_sha256')!=sha(Path(output)/'checkpoints/step-000003/state.pt') or
            not all(proof.get(k) is True for k in ('exact_loss','exact_full_state','exact_sample','frozen_base_unchanged'))):
        raise ValueError('separate-process disk proof is incomplete or belongs to different inputs')


def train_full(run,recipe,output,base,*,emit,guard,cap_check,base_check,stop_requested):
    if run.game.completed_steps!=3:
        raise ValueError('sustained training must resume qualified GAN checkpoint3')
    witnesses=[GradientWitness(run.backend.network),GradientWitness(run.game.critic)]
    try:
        while run.game.completed_steps<recipe.data['total_steps'] and not stop_requested():
            guard()
            cap_check()
            for witness in witnesses:
                witness.reset()
            tick=time.monotonic()
            with no_optimized_mse():
                raw=run.step()
            norms=[w.norm() for w in witnesses]
            if not all(math.isfinite(v) and v>0 for v in norms):
                raise ValueError('actual generator/critic backward witness failed')
            emit('training_step',completed_steps=run.game.completed_steps,metrics=raw,
                 gradient_norms=norms,step_seconds=time.monotonic()-tick,resources=cap_check())
            if run.game.completed_steps%recipe.data['checkpoint_every']==0 or stop_requested():
                save_checkpoint(output,recipe,run,base['combined_sha256'])
        save_checkpoint(output,recipe,run,base['combined_sha256'])
        if not base_check():
            raise ValueError('frozen base changed during sustained training')
        return dict(status='stopped' if stop_requested() else 'trained',completed_steps=run.game.completed_steps,
                    frozen_base_unchanged=True)
    finally:
        for witness in witnesses:
            witness.close()


def evaluate_heldouts(run,recipe,output,*,guard,cap_check,emit):
    import torch
    from safetensors.torch import load_file
    from .pg09_backends import load_adapter
    if run.game.completed_steps!=recipe.data['total_steps']:
        raise ValueError('heldouts require the complete fixed GAN budget')
    state=run.game.state_dict()
    export_dir=Path(output)/'exports'/f'full-{run.game.completed_steps:06d}'
    export_and_verify(run,export_dir)
    pipeline=None
    if recipe.data['case']=='final-form':
        from diffusers import AutoencoderKLFlux2
        from moss_sliders.vendor.pipeline_moss_image import MOSSImagePipeline,MOSSImageScheduler
        guard()
        vae=AutoencoderKLFlux2.from_pretrained(recipe.data['model_snapshot'],subfolder='vae',
                                              local_files_only=True,torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
        guard.attach(vae)
        pipeline=MOSSImagePipeline(run.backend.transformer,MOSSImageScheduler(),vae=vae)
        pipeline.set_progress_bar_config(disable=True)
    arms={'GAN'+str(recipe.data['total_steps']):None,**recipe.data['old_adapters']}
    count=0
    try:
        for arm,old in arms.items():
            if old is None:
                load_adapter(run.backend,export_dir/'adapter.safetensors',model_fingerprint=digest(run.backend.model_identity))
                adapter_sha=sha(export_dir/'adapter.safetensors')
            else:
                if sha(old['path'])!=old['sha256'] or sha(old['manifest'])!=old['manifest_sha256']:
                    raise ValueError('preserved old export identity differs')
                adapter_sha=old['sha256']
                if recipe.data['case']=='final-form':
                    manifest=json.loads(Path(old['manifest']).read_text())
                    slider=run.backend.slider
                    if (manifest.get('revision')!=recipe.data['model_revision'] or manifest.get('rank')!=4 or
                            manifest.get('alpha')!=4 or manifest.get('targets')!=slider.targets or
                            manifest.get('training',{}).get('step')!=old['completed_steps']):
                        raise ValueError('actual old FinalForm200/254 model/architecture/step differs')
                    weights=load_file(old['path'],device='cpu')
                    current=slider.state_dict()
                    if weights.keys()!=current.keys() or any(weights[k].shape!=current[k].shape or
                            not weights[k].isfinite().all() for k in current):
                        raise ValueError('old MOSS adapter shape or finite values differ')
                    slider.load_state_dict(weights,strict=True)
                else:
                    # Native load validates exact rank/targets/model/T5/VAE revisions.
                    weights=load_file(old['path'],device='cpu')
                    if any(not tensor.isfinite().all() for tensor in weights.values()):
                        raise ValueError('nonfinite old Supra adapter')
                    run.runtime.load(old['path'])
            for case in run.render_cases():
                guard()
                cap_check()
                name=f"{case['row_id']}-seed{case['seed']}-{case['label']}"
                target=Path(output)/'evaluation'/arm/(name+'.png')
                if target.exists() or target.with_suffix('.json').exists():
                    raise FileExistsError('immutable heldout output already exists: '+str(target))
                image=run.render(pipeline,case) if pipeline is not None else run.render(case)
                if isinstance(image,list):
                    image=image[0]
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name('.pending-'+uuid.uuid4().hex+'.png')
                image.save(temporary,format='PNG')
                temporary.rename(target)
                immutable_json(target.with_suffix('.json'),dict(case,arm=arm,adapter_sha256=adapter_sha,
                               recipe_sha256=recipe.identity,sha256=sha(target),quality_qualified=False,
                               teacher_cfg_branch='frozen',adapter_cfg_branch='conditional_only',
                               native_noise='CUDA SupraRuntime.noise' if pipeline is None else 'CPU MOSS seeded noise'))
                count+=1
                emit('heldout_saved',file=str(target),arm=arm,images=count)
    finally:
        run.game.load_state_dict(state)
    expected=30 if recipe.data['case']=='final-form' else 96
    if count!=expected:
        raise ValueError('matched heldout row/seed/arm coverage differs')
    return dict(status='evaluated',images=count,completed_steps=run.game.completed_steps,quality_qualified=False)


def execute(recipe,output,mode,lease_fds):
    """No process dispatch/admission here; parent guardian must hold both leases."""
    recipe.verify_sources()
    output=Path(output)
    if output.resolve()!=Path(recipe.data['output_directory']).resolve():
        raise ValueError('output differs from reviewed new run')
    previous=json.loads((output/'phase.json').read_text()) if (output/'phase.json').exists() else {}
    reason=recipe.phase_reason(mode,previous)
    if reason:
        raise ValueError(reason)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(recipe.data['physical_gpu']):
        raise ValueError('exact owner-bound physical card must be exposed')
    if os.environ.get('HF_HOME')!=recipe.data['hf_home'] or os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
        raise ValueError('exact native offline HuggingFace cache environment required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('reviewed deterministic cuBLAS workspace required')
    inherited_leases(lease_fds,recipe.data['lease_paths'])
    attestation=import_and_attest(recipe)
    config=verify_inputs(recipe)
    output.mkdir(parents=True,exist_ok=True)
    immutable_json(output/'recipe.json',recipe.data)
    immutable_json(output/'runtime-attestation.json',dict(recipe_sha256=recipe.identity,**attestation))
    budget_path=Path(recipe.data['cohort_root'])/'case-budget.json'
    if budget_path.exists():
        budget=json.loads(budget_path.read_text())
        if (budget.get('cohort_id')!=recipe.data['cohort_id'] or budget.get('case')!=recipe.data['case'] or
                budget.get('wall_seconds')!=recipe.data['caps']['wall_seconds'] or
                budget.get('deadline_unix')!=budget.get('started_unix',0)+budget['wall_seconds']):
            raise ValueError('immutable cohort wall budget belongs to different case')
    else:
        now=time.time()
        budget=dict(schema='remaining-cohort-wall-v1',cohort_id=recipe.data['cohort_id'],case=recipe.data['case'],
                    wall_seconds=recipe.data['caps']['wall_seconds'],started_unix=now,
                    deadline_unix=now+recipe.data['caps']['wall_seconds'])
        immutable_json(budget_path,budget)
    budget_sha=sha(budget_path)
    if previous and previous.get('cohort_budget_sha256')!=budget_sha:
        raise ValueError('cumulative cohort wall budget changed between phases')
    if recipe.data['case']=='supra-final-boss' and mode!='contexts':
        if recipe.data['contexts'].get('case_budget_sha256')!=budget_sha:
            raise ValueError('Supra full phase must preserve its context preparation wall budget')
    if time.time()>=budget['deadline_unix']:
        raise RuntimeError('original cumulative cohort wall budget expired; no automatic extension')
    started=time.monotonic()
    record=dict(previous,recipe_sha256=recipe.identity,case=recipe.data['case'],mode=mode,pid=os.getpid(),
                cohort_budget_sha256=budget_sha,
                status='loading',completed_steps=previous.get('completed_steps',0))
    def emit(event,**fields):
        record.update(event=event,elapsed_seconds=time.monotonic()-started,**fields)
        atomic_json(output/'phase.json',record)
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
        print(json.dumps(record,allow_nan=False),flush=True)
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signals={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    run=None
    try:
        emit('loading')
        cuda_setup()
        import torch
        import shutil
        available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
        if (torch.cuda.mem_get_info()[0]<recipe.data['caps']['admission_free_vram_gib']*1024**3 or
                available<recipe.data['caps']['admission_host_available_gib']*1024**3 or
                shutil.disk_usage('/mnt/nvme').free<100*1024**3):
            raise RuntimeError('owner-admitted phase lacks conservative VRAM/RAM/disk headroom')
        torch.manual_seed(recipe.data['seed'])
        torch.cuda.manual_seed_all(recipe.data['seed'])
        guard=resource_guard(recipe)
        def cap_check():
            return check_caps(recipe,Path(recipe.data['cohort_root']),started,mode,cohort_deadline=budget['deadline_unix'])
        cap_check()
        native=load_native(recipe,guard,config)
        if recipe.data['case']=='supra-final-boss':
            def stop_native_encode(*_):
                if stopped:
                    raise InterruptedError('stopped at safe native encoding boundary')
            native.text_encoder.register_forward_pre_hook(stop_native_encode)
        cap_check()
        if mode=='contexts':
            from .pg09_remaining_campaigns import export_live_supra_contexts,load_supra_contexts
            contexts=recipe.data['contexts']
            destination=Path(contexts['path'])
            if not destination.resolve().is_relative_to(output.resolve()):
                raise ValueError('preparation output must stay within its capped fresh phase directory')
            base=frozen_identity(native.model,guard)
            receipt=export_live_supra_contexts(native,config,destination,runtime_identity=dict(
                model_revision=recipe.data['model_revision'],text_revision=recipe.data['text_revision'],
                preparation_recipe_sha256=recipe.identity))
            bank=load_supra_contexts(destination,receipt['sha256'],config)
            if len(bank)!=19 or any(a.shape!=(1,128,768) or b.shape!=(1,128) for a,b in bank.values()):
                raise ValueError('native full T5 context count/shape differs')
            if not native.stored_unconditional or base!=frozen_identity(native.model,guard):
                raise ValueError('stored unconditional or frozen base changed during preparation')
            cap_check()
            if stopped:
                raise InterruptedError('context preparation stopped; immutable partial outputs require owner review')
            emit('contexts_prepared',status='contexts_prepared',context_sha256=receipt['sha256'],
                 context_receipt_sha256=sha(destination.with_suffix('.json')),native_encode_calls_at_most=19,
                 training_updates=0,frozen_base=base)
            return record
        run,base=bind_run(recipe,native,guard,config)
        base_model=run.backend.transformer if recipe.data['case']=='final-form' else native.model
        def base_check():
            return base==frozen_identity(base_model,guard)
        immutable_json(output/'base-identity.json',base)
        if mode=='probe':
            from moss_original_runtime import initial_gradient_parity
            result=qualify_probe(run,recipe,output,base,emit=emit,guard=guard,cap_check=cap_check,
                                 base_check=base_check,initial_gate=initial_gradient_parity,stop_requested=lambda:stopped)
        elif mode=='resume-check':
            result=resume_disk_reference(run,recipe,output,base,previous,base_check=base_check)
            result.update(status='probe_passed',completed_steps=3)
        else:
            if mode=='train':
                verify_disk_proof(recipe,output,previous,base)
            restore_latest(output,recipe,run,base['combined_sha256'])
            if mode=='train':
                result=train_full(run,recipe,output,base,emit=emit,guard=guard,cap_check=cap_check,
                                  base_check=base_check,stop_requested=lambda:stopped)
            else:
                result=evaluate_heldouts(run,recipe,output,guard=guard,cap_check=cap_check,emit=emit)
                if not base_check():
                    raise ValueError('frozen base changed during heldout evaluation')
                result['frozen_base_unchanged']=True
        cap_check()
        emit('phase_complete',**result)
        return record
    except BaseException as error:
        emit('failed',status='failed_no_retry',error=repr(error),diagnostics=getattr(error,'diagnostics',None),
             completed_steps=run.game.completed_steps if run is not None else 0)
        raise
    finally:
        for s,handler in signals.items():
            signal.signal(s,handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mode',choices=['plan','contexts','probe','resume-check','train','evaluate'],default='plan')
    parser.add_argument('--lease-fd',action='append',type=int,default=[])
    args=parser.parse_args(argv)
    recipe=RemainingRecipe.read(args.recipe)
    if args.mode=='plan':
        recipe.verify_sources()
        print(json.dumps(dict(recipe_sha256=recipe.identity,case=recipe.data['case'],
                             admission=recipe.data['admission'],live_executed=False,
                             context_prerequisite=recipe.data.get('contexts'),caps=recipe.data['caps']),indent=2))
        return
    if args.output is None:
        parser.error('live phases require the exact --output and two inherited --lease-fd arguments')
    execute(recipe,args.output,args.mode,args.lease_fd)


if __name__=='__main__':
    main()
