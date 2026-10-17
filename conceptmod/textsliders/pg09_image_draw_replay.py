"""Private CPU draw-order checks and lazy descriptor assets; no native model path.

Recovered archive states must be supplied only by a separately admitted reader.
CPU substream agreement never qualifies CUDA noise, text, prefixes or full replay.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import torch


def tensor_sha(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
        raise ValueError('CPU tensor required')
    data=value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def python_rng_sha(state):
    return hashlib.sha256(json.dumps(state,separators=(',',':')).encode()).hexdigest()


def cpu_draw_reconstruction(case,initial_cpu,final_cpu,initial_python,final_python,*,updates=500):
    """Private state assignment only. No manual_seed/global setstate/RNG restore.

    Source calls: Sana alternates2 rows with no CPU stochastic choice; SD3 draws
    prompt randint(0,4,(1,)), then prefix randint(1,50,(1,)), both on CPU. Native
    Gaussian CUDA noise is deliberately absent, not replaced with CPU noise.
    """
    if case not in ('sana','sd3') or type(updates) is not int or not 1<=updates<=500:
        raise ValueError('bounded known case/update policy required')
    for value in (initial_cpu,final_cpu):
        if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=torch.uint8 or
                value.ndim!=1 or value.numel()>1024**2):
            raise ValueError('bounded CPU uint8 generator state required')
    generator=torch.Generator(device='cpu')
    generator.set_state(initial_cpu.detach().clone())
    python=random.Random(0)
    python.setstate(deepcopy(initial_python))
    draws=[]
    for i in range(updates):
        if case=='sana':
            row=dict(update=i+1,prompt_index=i%2,fixed_training_timestep=500.)
        else:
            row=dict(update=i+1,
                prompt_index=int(torch.randint(0,4,(1,),device='cpu',generator=generator).item()),
                selected_prefix_index=int(torch.randint(1,50,(1,),device='cpu',generator=generator).item()),
                native_training_step_count=50)
        draws.append(row)
    produced_cpu=generator.get_state()
    produced_python=python.getstate()  # No source Python-random draws after step0.
    cpu_match=torch.equal(produced_cpu,final_cpu)
    python_match=produced_python==final_python
    return dict(case=case,updates=updates,draws=draws,schedule_sha256=json_sha(draws),
        initial_cpu_sha256=tensor_sha(initial_cpu),produced_final_cpu_sha256=tensor_sha(produced_cpu),
        archived_final_cpu_sha256=tensor_sha(final_cpu),cpu_end_state_exact=cpu_match,
        initial_python_sha256=python_rng_sha(initial_python),
        produced_final_python_sha256=python_rng_sha(produced_python),
        archived_final_python_sha256=python_rng_sha(final_python),python_end_state_exact=python_match,
        cpu_substream_status='source-only end-state match' if cpu_match and python_match else 'MISMATCH; no hidden-consumer search or repair',
        schedule_candidate_usable=cpu_match and python_match,
        cuda_noise_generated=False,numeric_text_conditions_reconstructed=False,
        historical_draw_replay_qualified=False,
        limits=['CPU/Python endpoint agreement is not proof of every intermediate draw or absence of compensating consumers',
            'all CUDA noise/timestep native values/conditions/prefixes remain separately unqualified',
            'checkpoint observer render/diagnostic consumption and resume timing require separate review'])


class FileDescriptorAssets:
    """Lazy CPU asset reader prototype: selected noise + native text packets only.

    No teacher responses, targets, models, GPU moves or evolving provider state.
    Root must separately stage actual source-bound assets and qualify native paths.
    """
    def __init__(self,index,*,expected_sha256,expected_bindings):
        self.path=Path(index).absolute()
        if self.path.is_symlink():
            raise ValueError('asset index symlink forbidden')
        raw=self.path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError('asset index checksum mismatch')
        self._manifest=raw
        self._sha=expected_sha256
        value=json.loads(raw)
        if (set(value)!={'schema','case','bindings','rows'} or value['schema']!='image-descriptor-assets-cpu-prototype-v1'
                or value['case'] not in ('sana','sd3') or value['bindings']!=expected_bindings
                or not isinstance(value['rows'],list) or not 1<=len(value['rows'])<=500):
            raise ValueError('asset index schema/source bindings mismatch')
        keys={'draw_schedule_sha256','conditions_policy_sha256','native_source_sha256','complete_condition_gate_sha256'}
        if (type(value['bindings']) is not dict or set(value['bindings'])!=keys or
                any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in value['bindings'].values())):
            raise ValueError('complete asset source/policy/hash bindings required')
        if json_sha([row['draw'] for row in value['rows']])!=value['bindings']['draw_schedule_sha256']:
            raise ValueError('asset draw schedule checksum mismatch')
        self.root=self.path.parent.resolve()
        for i,row in enumerate(value['rows']):
            if (set(row)!={'draw','noise','conditions'} or type(row['draw']['update']) is not int or
                    row['draw']['update']!=i+1):
                raise ValueError('asset draw index schema/order mismatch')
            draw=row['draw']
            if value['case']=='sana':
                if (set(draw)!={'update','prompt_index','fixed_training_timestep'} or type(draw['prompt_index']) is not int or draw['prompt_index']!=i%2 or
                        type(draw['fixed_training_timestep']) is not float or draw['fixed_training_timestep']!=500.):
                    raise ValueError('Sana asset source draw policy differs')
            elif (set(draw)!={'update','prompt_index','selected_prefix_index','native_training_step_count'} or
                    type(draw['prompt_index']) is not int or not 0<=draw['prompt_index']<4 or
                    type(draw['selected_prefix_index']) is not int or not 1<=draw['selected_prefix_index']<=49 or
                    type(draw['native_training_step_count']) is not int or draw['native_training_step_count']!=50):
                raise ValueError('SD3 asset source draw policy differs')
            for key in ('noise','conditions'):
                if set(row[key])!={'path','bytes','sha256'} or type(row[key]['bytes']) is not int or not 0<row[key]['bytes']<=64*1024**2:
                    raise ValueError('bounded asset file descriptor required')
                self._path(row[key]['path'])

    def _path(self,relative):
        part=Path(relative)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError('asset path must be inside index directory')
        full=self.root/part
        for component in (full,*full.parents):
            if component==self.root:
                break
            if component.is_symlink():
                raise ValueError('asset symlink forbidden')
        if not full.resolve().is_relative_to(self.root):
            raise ValueError('asset escaped index root')
        return full

    def _bound_index(self):
        if hashlib.sha256(self._manifest).hexdigest()!=self._sha:
            raise ValueError('private asset index changed')
        return json.loads(self._manifest)

    def identity(self):
        index=self._bound_index()
        return dict(schema=index['schema'],case=index['case'],index_sha256=self._sha,
            bindings=deepcopy(index['bindings']),qualification='CPU asset-reader prototype only',
            native_targets_cached=False,historical_draw_replay_qualified=False)

    def __len__(self):
        return len(self._bound_index()['rows'])

    def _load(self,record):
        path=self._path(record['path'])
        if path.stat().st_size!=record['bytes']:
            raise ValueError('asset size changed')
        with path.open('rb') as stream:
            data=stream.read(record['bytes']+1)
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('asset checksum changed')
        # Load the exact checksum-verified bytes, not a path reopened after hash.
        import io
        return torch.load(io.BytesIO(data),weights_only=True,map_location='cpu')

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):
            raise IndexError('asset row index out of range')
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self._sha:
            raise ValueError('asset index changed')
        bound=self._bound_index()
        row=bound['rows'][index]
        noise=self._load(row['noise'])
        conditions=self._load(row['conditions'])
        case=bound['case']
        shape=(1,32,16,16) if case=='sana' else (1,16,64,64)
        if (type(noise) is not torch.Tensor or noise.device.type!='cpu' or noise.dtype!=torch.float32 or
                tuple(noise.shape)!=shape or noise.requires_grad or not torch.isfinite(noise).all()):
            raise ValueError('native initial FP32 noise asset schema mismatch')
        if type(conditions) is not dict or set(conditions)!={'neutral','positive','negative','unconditional'}:
            raise ValueError('native condition role schema mismatch')
        for condition in conditions.values():
            keys={'embeds','mask'} if case=='sana' else {'embeds','pooled'}
            if type(condition) is not dict or set(condition)!=keys:
                raise ValueError('native condition entry schema mismatch')
            for key,value in condition.items():
                dtype=torch.int64 if key=='mask' else (torch.float32 if case=='sana' else torch.bfloat16)
                dims=3 if key=='embeds' else 2
                if (type(value) is not torch.Tensor or value.device.type!='cpu' or value.dtype!=dtype or
                        value.ndim!=dims or not all(size>0 for size in value.shape) or value.shape[0]!=1 or value.requires_grad or not torch.isfinite(value).all()):
                    raise ValueError('native numeric condition schema mismatch')
            if case=='sana' and not torch.all((condition['mask']==0)|(condition['mask']==1)):
                raise ValueError('native attention mask must be binary')
            if case=='sana' and condition['mask'].shape!=condition['embeds'].shape[:2]:
                raise ValueError('native mask/embedding geometry mismatch')
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=self._sha:
            raise ValueError('asset index changed during selected-row read')
        return dict(draw=deepcopy(row['draw']),initial_noise_fp32=noise,conditions=conditions,
            noise_fp32_bytes_sha256=tensor_sha(noise),source_identity=self.identity())
