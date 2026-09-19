"""Two frozen rank-8 student recipes on the expanded shared-history targets."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from types import SimpleNamespace

import torch

from ..reward_sliders.specs import MODEL, WORKSPACE, sha, digest, read_json, write_json, save_tensor
from ..reward_sliders.experiment import verify
from ..reward_sliders.directions import RewardTeacher, causal_gate
from ..reward_sliders.data import FamilySampler, generation_hidden, set_scale
from ..reward_sliders.train import audit_adapter
from ..gan_v2 import state
from ..gan_v2.train import arm_settings
from ..gan_v2.critic import SpanCritic, pad_sequences
from ..gan_v2.engine import GANEngine
from .data import prepare_window_rows, WindowStudentForward, diagnostics, slices


def verify_gate(run,manifest):
    gate=read_json(run/'causal-result.json')
    paths=sorted((run/'stages/causal/observations').glob('*.json'))
    expected={f"{f['family']}-s{seed}-{arm}" for f in manifest['families'] if f.get('group')=='causal'
              for seed in manifest['search']['causal_seeds'] for arm in ('off','positive','reversed','random')}
    rows=[read_json(p) for p in paths]
    if {r['id'] for r in rows}!=expected:raise ValueError('Incomplete declared causal evidence')
    declared={a['name']:a for a in read_json(run/'stages/causal/protocol.json')['arms']}
    for row in rows:
        if row['status']!='complete' or sha(row['audio'])!=row['audio_sha256']:raise ValueError('Invalid causal output')
        if row['provenance']['manifest_sha256']!=digest(manifest):raise ValueError('Causal protocol mismatch')
        if row['arm']!=declared.get(row['arm']['name']):raise ValueError('Causal arm changed')
    actual=causal_gate(rows)
    if not actual['passed'] or any(gate[k]!=v for k,v in actual.items()):raise ValueError('Causal evidence does not support this teacher')
    if gate['teacher_sha256']!=sha(run/'teacher.pt'):raise ValueError('Teacher changed after causal evaluation')
    return {str(p):sha(p) for p in paths}


def train(run,arm,gpu):
    run=Path(run)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise ValueError('Expose only the assigned GPU')
    manifest=read_json(run/'manifest.json');verify(manifest)
    if arm not in manifest['search']['student_arms']:raise ValueError('Undeclared training arm')
    causal_sources=verify_gate(run,manifest)
    folder=run/'students'/arm;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.cuda.set_device(0);device=torch.device('cuda:0')
    seed=manifest['search']['student_seed'];torch.manual_seed(seed);random.seed(seed)
    sources={str(p):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name in ('train.py','data.py')}
    sources.update(state.code_fingerprints())
    recipe,critic_config=arm_settings(arm,origin=0,horizon=600,diagnostics_every=25)
    windows=manifest['search']['training_windows'];stride=manifest['search']['training_stride']
    observations=read_json(run/'training-observations.json')
    if len(observations)!=96 or len({r['family'] for r in observations})!=24:raise ValueError('Expected the frozen expanded training set')
    row_signature=dict(manifest_sha256=digest(manifest),teacher_sha256=sha(run/'teacher.pt'),
                       observations_sha256=digest(observations),windows=windows,stride=stride,
                       data_source_sha256=sha(Path(__file__).with_name('data.py')))
    signature=dict(row_signature=row_signature,causal_sources=causal_sources,sources=sources,
                   recipe=asdict(recipe),critic=critic_config,rank=8,alpha=8,seed=seed,
                   checkpoints=manifest['search']['student_checkpoints'],
                   gross_divergence_stop=dict(prompt_relative_rms=2.,generation_relative_error=20.))
    frozen=folder/'manifest.json'
    if frozen.exists() and read_json(frozen)!=signature:raise ValueError('Training recipe changed')
    write_json(frozen,signature)
    for source,expected in sources.items():
        target=folder/'provenance'/Path(source).relative_to('/')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(source).read_bytes());assert sha(target)==expected
    sys.path.insert(0,str(WORKSPACE))
    from app import generator
    from app.lora_runtime import LoRANetwork
    from transformers import AutoModelForCausalLM
    teacher=RewardTeacher(**torch.load(run/'teacher.pt',map_location='cpu',weights_only=True))
    lm=AutoModelForCausalLM.from_pretrained(str(MODEL/'language_model'),torch_dtype=torch.bfloat16,
            local_files_only=True).to(device).eval().requires_grad_(False)
    lm.config.use_cache=False;pipe=SimpleNamespace(language_model=lm)
    def apply_styles(family):
        with torch.random.fork_rng(devices=[0]):
            generator._merge_sliders(pipe,'cuda:0',manifest['style_components'][family])
    cache=run/'student-rows.pt'
    with (run/'student-rows.lock').open('a') as cache_lock:
        fcntl.flock(cache_lock,fcntl.LOCK_EX)
        if cache.exists():
            cached=torch.load(cache,map_location='cpu',weights_only=True)
            if cached['signature']!=digest(row_signature):raise ValueError('Row cache changed')
            rows=cached['rows'];del cached
        else:
            rows=prepare_window_rows(lm,observations,teacher,apply_styles,device=device,windows=windows,stride=stride)
            save_tensor(cache,dict(signature=digest(row_signature),rows=rows))
    torch.manual_seed(seed);random.seed(seed)
    network=LoRANetwork(lm,rank=8,alpha=8.,multiplier=0.,target_replace=['Qwen3Attention'],
                        prefix='lora_te',delimiter='-',train_method='full').to(device).requires_grad_(True)
    topology=audit_adapter(network,lm)
    if any(torch.count_nonzero(m.lora_up.weight) for m in network.unet_loras):raise ValueError('Student did not start at exact zero')
    for start in sorted({r['window_start'] for r in rows}):
        row=next(r for r in rows if r['window_start']==start)
        apply_styles(row['family']);set_scale(network,0.)
        with torch.no_grad():zero=generation_hidden(lm,row['embeds'].to(device))[:,slices(row)[0]].float().cpu()
        if not torch.equal(zero,row['neutral_span']):raise ValueError('Zero student differs on a saved history')
    critic=SpanCritic(4096,**critic_config).to(device)
    real,mask=pad_sequences([r['real'].to(device) for r in rows])
    input_scale=critic.calibrate_input_scale(real,mask);del real,mask
    engine=GANEngine(network,critic,WindowStudentForward(lm,network,apply_styles,device),rows,recipe)
    sampler=FamilySampler(rows)
    history=state.restore(folder/'state.pt',engine,sampler,None,signature) if (folder/'state.pt').exists() else []
    write_json(folder/'geometry-audit.json',dict(passed=True,topology=topology,row_count=len(rows),
        training_families=24,windows=windows,stride=stride,zero_parity=True,critic_input_scale=input_scale,
        physical_gpu=gpu,exact_preceding_feedback_retained=True))
    stop=False
    def cancel(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)

    def export():
        from safetensors.torch import save_file,load_file
        path=folder/f'reward-ce-v2-{arm}_step{engine.completed}.safetensors'
        audit_adapter(network,lm)
        tensors={k:v.detach().cpu().contiguous() for k,v in network.state_dict().items()}
        save_file(tensors,str(path));saved=load_file(str(path))
        if set(saved)!=set(tensors) or any(not torch.equal(saved[k],tensors[k]) for k in saved):raise ValueError('Export differs')
        metadata=dict(kind='language_model',rank=8,alpha=8,target_replace=['Qwen3Attention'],prefix='lora_te',
            delimiter='-',train_method='full',unit_scale=1.,plus_label='CE',minus_label='Off',
            recommended_range=[0.,1.],unipolar=True,steps=engine.completed,prompts_file=str(run/'manifest.json'),
            weights_sha256=sha(path),reward=dict(name='reward-ce-v2-'+arm,spec=manifest['reward_spec'],
                teacher_sha256=sha(run/'teacher.pt'),manifest_sha256=digest(manifest),
                student_signature_sha256=digest(signature),interpretation='research candidate; audio selection pending'))
        write_json(path.with_suffix('.json'),metadata)
        results=diagnostics(lm,network,rows,apply_styles,device)
        write_json(folder/f'diagnostics-step{engine.completed}.json',results)
        state.save(folder/'state.pt',engine,sampler,None,signature,history)
        state.save(folder/f'step{engine.completed}_state.pt',engine,sampler,None,signature,history)
        write_json(folder/'latest.json',dict(step=engine.completed,weights=str(path)))
        if any(r['prompt_relative_rms']>2. or r['generation_relative_error']>20. for r in results):
            raise FloatingPointError('Gross divergence at checkpoint')
    try:
        with (folder/f'train-from-{engine.completed}.jsonl').open('a') as log:
            while engine.completed<max(signature['checkpoints']) and not stop:
                began=time.monotonic();update=engine.update(sampler.next());update['seconds']=time.monotonic()-began
                history.append(update);log.write(json.dumps(update,allow_nan=False)+'\n');log.flush()
                write_json(folder/'status.json',dict(stage='training',step=engine.completed,total=600,updated_unix=time.time()))
                print(f"UPDATE {arm} {engine.completed}/600 losses={update['losses']} seconds={update['seconds']:.2f}",flush=True)
                if engine.completed in signature['checkpoints']:export()
                elif engine.completed%30==0:state.save(folder/'state.pt',engine,sampler,None,signature,history)
            if stop:
                state.save(folder/'state.pt',engine,sampler,None,signature,history)
                raise SystemExit(130)
            write_json(folder/'status.json',dict(stage='complete',step=engine.completed,updated_unix=time.time()))
    except BaseException as error:
        state.save(folder/'state.pt',engine,sampler,None,signature,history)
        write_json(folder/'status.json',dict(stage='error',step=engine.completed,error=repr(error)))
        raise
    finally:
        set_scale(network,0.);generator._merge_sliders(pipe,'cuda:0',[])
        exact=all(torch.equal(m.weight.detach().cpu(),p) for m,p in generator._merge_state('cuda:0').pristine.items())
        write_json(folder/'off-restoration.json',dict(exact=exact,physical_gpu=gpu))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--arm',required=True);parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args();train(args.run_dir,args.arm,args.gpu)
