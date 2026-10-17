"""UNADMITTED next CPU cohort: private-generator source draw-order comparison.

Actual known-packet reload occurs only if root separately admits this exact plan.
No global RNG restoration, CUDA generator, model/optimizer/native forward or noise.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import signal
import sys
import time

if __package__:
    from .audit_image_checkpoint_rng_pg09 import FreshReceiptOutput, sha_file, available_bytes, expected_case_schema, scheduler_metadata
else:
    from audit_image_checkpoint_rng_pg09 import FreshReceiptOutput, sha_file, available_bytes, expected_case_schema, scheduler_metadata


def load_private_inputs(packet,expected_rng,torch):
    """Called only inside the separately admitted bounded main; no global setstate."""
    state=torch.load(packet['path'],weights_only=True,map_location='cpu',mmap=True)
    case=packet['case']
    if (type(state) is not dict or set(state)!=set(expected_case_schema(case)['exact_fields']) or
            type(state['identity']) is not str or type(state['step']) is not int or
            state['identity']!=packet['identity'] or state['step']!=packet['step']):
        raise ValueError('private draw input exact case/identity/schema mismatch')
    if type(state['trainable']) is not dict or type(state['optimizer']) is not dict:
        raise ValueError('private draw trainable/optimizer schema mismatch')
    if case=='sd3':
        scheduler_metadata(state['lr_scheduler'],step=state['step'])
    def rng(value):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=torch.uint8 or
                value.ndim!=1 or value.numel()>1024**2):
            raise ValueError('private draw RNG input is not a bounded CPU byte vector')
        return dict(shape=list(value.shape),dtype=str(value.dtype),
            sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
    if rng(state['torch_rng'])!=expected_rng['torch_rng']:
        raise ValueError('recovered CPU RNG receipt mismatch')
    if type(state['cuda_rng']) is not list or [rng(x) for x in state['cuda_rng']]!=expected_rng['cuda_rng']:
        raise ValueError('recovered CUDA RNG byte identity mismatch')
    python=state['python_rng']
    if (type(python) is not tuple or len(python)!=3 or type(python[0]) is not int or
            type(python[1]) is not tuple or len(python[1])>10000 or
            not all(type(x) is int for x in python[1]) or
            (python[2] is not None and type(python[2]) is not float)):
        raise ValueError('private Python RNG input schema mismatch')
    identity=dict(version=python[0],integer_count=len(python[1]),
                  sha256=hashlib.sha256(json.dumps(python,separators=(',',':')).encode()).hexdigest())
    if identity!=expected_rng['python_rng']:
        raise ValueError('recovered Python RNG receipt mismatch')
    cpu=state['torch_rng'].clone()
    del state
    gc.collect()
    return cpu,python


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with FreshReceiptOutput(args.output,[args.plan,__file__]) as receipt:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in
                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
            raise RuntimeError('CUDA-hidden single-thread environment required')
        resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,(120,120))
        signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('120s private draw cap')))
        signal.alarm(120)
        start=time.monotonic()
        if sha_file(args.plan)!=args.plan_sha256:
            raise ValueError('private draw plan checksum mismatch')
        plan=json.loads(args.plan.read_text())
        if sha_file(__file__)!=plan['driver_sha256']:
            raise ValueError('private draw driver source mismatch')
        protected=[r['path'] for r in plan['source_bindings']]+[r['path'] for r in plan['audit_receipts']]
        protected += [p['path'] for p in plan['known_packets']]+[p['sidecar']['path'] for p in plan['known_packets']]
        receipt.protect(protected)
        for binding in plan['source_bindings']+plan['audit_receipts']:
            if sha_file(binding['path'])!=binding['sha256']:
                raise ValueError('private draw source/recovered receipt changed')
        audit=json.loads(Path(plan['rng_audit']['path']).read_text())
        if audit['mode']!='rng' or audit['plan_sha256']!=plan['prior_v3_plan_sha256']:
            raise ValueError('required recovered RNG audit plan mismatch')
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized() or str(torch.get_default_device())!='cpu':
            raise RuntimeError('CPU default device and uninitialized CUDA required')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from conceptmod.textsliders import pg09_image_draw_replay as replay
        if sha_file(replay.__file__)!=plan['replay_module_sha256']:
            raise ValueError('actual imported private replay module source mismatch')
        before_cpu=torch.get_rng_state().clone()
        before_python=random.getstate()
        before_numpy=np.random.get_state()
        results=[]
        for case in ('sana','sd3'):
            private=[]
            for step in (0,500):
                receipt.check()
                if available_bytes()<16*1024**3:
                    raise RuntimeError('16GiB host available required')
                packets=[p for p in plan['known_packets'] if p['case']==case and p['step']==step]
                if len(packets)!=1:
                    raise ValueError('exact case/step packet required')
                packet=packets[0]
                if (Path(packet['path']).stat().st_size!=packet['bytes'] or packet['bytes']>2*1024**3 or
                        sha_file(packet['sidecar']['path'])!=packet['sidecar']['sha256'] or
                        sha_file(packet['path'])!=packet['declared_sha256']):
                    raise ValueError('private draw known packet changed')
                bound=[p for p in audit['packets'] if p['path']==packet['path'] and p['sha256']==packet['declared_sha256']]
                if len(bound)!=1:
                    raise ValueError('recovered RNG packet identity mismatch')
                private.append(load_private_inputs(packet,bound[0]['rng_schema'],torch))
            results.append(replay.cpu_draw_reconstruction(case,private[0][0],private[1][0],private[0][1],private[1][1]))
            del private
        now_numpy=np.random.get_state()
        if (not torch.equal(before_cpu,torch.get_rng_state()) or before_python!=random.getstate() or
                before_numpy[0]!=now_numpy[0] or not np.array_equal(before_numpy[1],now_numpy[1]) or
                before_numpy[2:]!=now_numpy[2:] or torch.cuda.is_initialized()):
            raise RuntimeError('private draw cohort changed global RNG or initialized CUDA; no restore attempted')
        output=dict(schema='image-private-cpu-draw-check-v1',plan_sha256=args.plan_sha256,cases=results,
            global_rng_unchanged=True,global_rng_restored=False,cuda_noise_generated=False,
            model_optimizer_scheduler_instantiated=False,native_forward=False,historical_draw_replay_qualified=False,
            elapsed_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if output['peak_rss_bytes']>8*1024**3:
            raise RuntimeError('8GiB RSS cap exceeded')
        receipt.publish_json(output)
        print(json.dumps(dict(output=str(args.output),cases=len(results),historical_draw_replay_qualified=False)))


if __name__=='__main__':
    main()
