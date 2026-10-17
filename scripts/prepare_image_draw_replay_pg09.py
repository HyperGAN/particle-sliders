"""NEW metadata-only private-draw cohort proposal; never reloads archive packets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
STATIC_SHA='843b6966b7501068d8baf982fcd6f52561975684b0a690a6eee4cffac4c8aae1'
RNG_SHA='1ee854f66d368389b39c0cd08399a16b2ee0df4e873330c437e656c591cf9ed2'


def record(path):
    path=Path(path)
    if path.suffix in {'.pt','.safetensors','.bin','.ckpt','.model','.npy','.npz'}:
        raise ValueError('private-draw proposal rejects binary input bodies')
    data=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def visible_training_draw_calls(path):
    text=Path(path).read_text()
    tree=ast.parse(text)
    train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
    calls=[n for n in ast.walk(train) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
           and isinstance(n.func.value,ast.Name) and n.func.value.id=='torch' and n.func.attr in ('randint','randn')]
    return [dict(line=n.lineno,call=ast.get_source_segment(text,n)) for n in sorted(calls,key=lambda n:n.lineno)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    prior=ROOT/'evidence/pg09-image-failure-campaigns/offline-admission-v3/rng-audit-plan.json'
    old=json.loads(prior.read_text())
    static=Path('/tmp/image-checkpoint-static-root-v3-20261017.json')
    rng=Path('/tmp/image-checkpoint-rng-root-v3-20261017.json')
    if record(static)['sha256']!=STATIC_SHA or record(rng)['sha256']!=RNG_SHA:
        raise ValueError('root completed v3 metadata receipts changed')
    recovered=json.loads(rng.read_text())
    if recovered['mode']!='rng' or recovered['plan_sha256']!=record(prior)['sha256']:
        raise ValueError('root recovered v3 plan mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    copies=[]
    for source,name in [(static,'root-v3-static.json'),(rng,'root-v3-rng-schema.json')]:
        dest=args.output_dir/name
        if dest.exists() and dest.read_bytes()!=source.read_bytes():
            raise FileExistsError('preserved root metadata receipt differs')
        if not dest.exists():
            shutil.copy2(source,dest)
        copies.append(record(dest))
    q=Path('/mnt/nvme/sliders/qualification/checkouts')
    sources=dict(sana=q/'particle-sliders-sana-cached-negative/conceptmod/textsliders/train_lora_sana.py',
                 sd3=q/'particle-sliders-sd3/conceptmod/textsliders/train_lora_sd3.py')
    calls={case:visible_training_draw_calls(path) for case,path in sources.items()}
    if len(calls['sana'])!=1 or len(calls['sd3'])!=2:
        raise ValueError('visible source draw call count changed')
    observations={}
    for case in ('sana','sd3'):
        rows=sorted([p for p in recovered['packets'] if p['rng_schema']['case']==case],key=lambda p:p['step'])
        if [p['step'] for p in rows]!=[0,500]:
            raise ValueError('root initial/final case metadata missing')
        a,b=[p['rng_schema'] for p in rows]
        observations[case]=dict(initial_cpu=a['torch_rng'],final_cpu=b['torch_rng'],
            initial_python=a['python_rng'],final_python=b['python_rng'],
            cpu_hash_unchanged=a['torch_rng']==b['torch_rng'],python_hash_unchanged=a['python_rng']==b['python_rng'],
            cuda_hash_unchanged=a['cuda_rng']==b['cuda_rng'],actual_private_draw_schedule_executed=False)
    source_bindings=old['source_bindings']+[record(ROOT/name) for name in (
        'conceptmod/textsliders/pg09_image_draw_replay.py','scripts/verify_image_draw_replay_pg09.py',
        'scripts/prepare_image_draw_replay_pg09.py','scripts/audit_image_checkpoint_rng_pg09.py')]
    for binding in source_bindings:
        if record(binding['path'])!=binding:
            raise ValueError('frozen source drift before private draw plan')
    plan=dict(schema='image-private-draw-plan-v1',execution_admitted=False,
        driver_sha256=record(ROOT/'scripts/verify_image_draw_replay_pg09.py')['sha256'],
        replay_module_sha256=record(ROOT/'conceptmod/textsliders/pg09_image_draw_replay.py')['sha256'],
        prior_v3_plan_sha256=record(prior)['sha256'],known_packets=old['known_packets'],
        rng_audit=copies[1],audit_receipts=copies,source_bindings=source_bindings,
        source_draw_calls=calls,metadata_observations=observations,
        algorithm=dict(sana='500alternating2prompt rows/fixedt500; noCPU/Python stochastic draws',
            sd3='500times private CPU torch.randint0..3 then1..49; noPython draws',
            rng='private torch.GeneratorCPU.set_state + private random.Random.setstate only; never global seed/setstate/restore',
            outputs='ordered draw indices and source-only CPU/Python exact endstate comparison; noCUDA noise or text',
            mismatch='report mismatch; no seed7 substitution, hidden-consumer search, extra draws or repair'),
        caps=dict(threads=1,CUDA_VISIBLE_DEVICES='',wall_seconds=120,rss_max_gib=8,address_space_max_gib=8,host_available_min_gib=16),
        historical_draw_replay_qualified=False,actual_packet_reload_executed=False,
        pending=['root source/plan review and separate actualCPUcohort admission','possible hidden CPU/Python consumers/save-render-resume timing',
            'tiny separate GPU-only CUDA-noise proposal and source/runtime state match','numeric native text encoding/nativeparity',
            'full reader-to-native-worker/lease/guardian and online target integration qualification'])
    dest=args.output_dir/'private-draw-plan.json'
    dest.write_text(json.dumps(plan,indent=2)+'\n')
    commands=dict(cwd=str(ROOT),environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT)),
        execution_admitted=False,argv=[str(Path('/mnt/nvme/sliders/qualification/envs/pg090/bin/python')),
            str(ROOT/'scripts/verify_image_draw_replay_pg09.py'),'--plan',str(dest.resolve()),'--plan-sha256',record(dest)['sha256'],
            '--output',str((args.output_dir/'actual-private-draw-check.json').resolve())])
    (args.output_dir/'stage-command.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(json.dumps(dict(output=str(dest),plan_sha256=record(dest)['sha256'],packet_reload=False,private_state_replay=False)))


if __name__=='__main__':
    main()
