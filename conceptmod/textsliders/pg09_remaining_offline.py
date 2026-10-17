"""Bounded offline validation. Pretrained model files are stat-ed, never opened."""
import json
from pathlib import Path
import struct

from .pg09_remaining_admission import CORE,MODEL_REV,TEXT_REV,VAE_REV,MOSS_REV,CONFIG_SHA,digest,sha

QUAL=Path('/mnt/nvme/sliders/qualification')
MOSS=Path('/mnt/nvme/sliders/runs/moss-nonlighting-campaign-v1/final-form')
SUPRA=QUAL/'runs/supra-final-boss-20261017'
SUPRA_HUB=Path('/mnt/nvme/ai/caches/huggingface/main/hub')
MOSS_SNAPSHOT=Path('/mnt/nvme/sliders/cache/huggingface/hub/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots')/MOSS_REV


def safe_header(path):
    """Read bounded safetensors JSON only, checking shapes and complete offsets."""
    path=Path(path)
    with path.open('rb') as stream:
        raw=stream.read(8)
        if len(raw)!=8:
            raise ValueError('short safetensors prefix')
        size=struct.unpack('<Q',raw)[0]
        if size>16*1024*1024 or size+8>path.stat().st_size:
            raise ValueError('unbounded or truncated safetensors header')
        header=json.loads(stream.read(size))
    bytes_per={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
    intervals=[]
    for name,tensor in header.items():
        if name=='__metadata__':
            continue
        if set(tensor)!={'dtype','shape','data_offsets'} or tensor['dtype'] not in bytes_per:
            raise ValueError('safetensors dtype/keys differ')
        count=1
        for n in tensor['shape']:
            if type(n) is not int or n<0:
                raise ValueError('invalid tensor shape')
            count*=n
        a,b=tensor['data_offsets']
        if type(a) is not int or type(b) is not int or a<0 or b-a!=count*bytes_per[tensor['dtype']]:
            raise ValueError('invalid tensor byte range')
        intervals.append((a,b))
    end=0
    for a,b in sorted(intervals):
        if a!=end:
            raise ValueError('tensor offsets overlap or have a gap')
        end=b
    if end+size+8!=path.stat().st_size:
        raise ValueError('tensor payload size differs from complete header')
    return header


def validate_moss_cache(root=MOSS,*,progress=lambda _:None):
    root=Path(root)
    recipe=json.loads((root/'recipe.json').read_text())
    identity=digest(recipe)
    if recipe.get('concept')!='final-form' or recipe.get('model_revision')!=MOSS_REV:
        raise ValueError('FinalForm recipe identity differs')
    training,dev=recipe['training_rows'],recipe['development_rows']
    if len(training)!=24 or len(dev)!=2 or set(r['id'] for r in training)&set(r['id'] for r in dev):
        raise ValueError('exact disjoint24train/2heldout rows required')
    expected={(r['id'],s,'train') for r in training for s in r['seeds']}
    expected|={(r['id'],recipe['evaluation_seed'],'dev') for r in dev}
    if len(expected)!=386 or recipe['settings']['target_steps']!=10:
        raise ValueError('full384train+2dev bundle index required')
    index=json.loads((root/'target-index.json').read_text())
    if index['identity']!=identity or len(index['records'])!=len(expected):
        raise ValueError('index/recipe identity differs')
    found=set()
    total=0
    digests={}
    for i,record in enumerate(index['records']):
        path=Path(record['path']).resolve()
        if path.parent!=(root/'targets').resolve():
            raise ValueError('indexed path escapes original target directory')
        key=(record['row'],record['seed'],record['split'])
        if key in found:
            raise ValueError('duplicate teacher bundle')
        found.add(key)
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['identity']!=digest(dict(identity=identity,row=key[0],seed=key[1],split=key[2])) or sha(path)!=meta['sha256']:
            raise ValueError('target sidecar/hash identity differs')
        header=safe_header(path)
        if set(header)!={'base','delta','latent','time'}:
            raise ValueError('native teacher tensor keys differ')
        for name in ('base','delta','latent'):
            if header[name]['shape']!=[2,10,1,128,32,32] or header[name]['dtype']!='F32':
                raise ValueError('native teacher field shape/dtype differs')
        if header['time']['shape']!=[10] or header['time']['dtype']!='F32':
            raise ValueError('native timestep header differs')
        total+=path.stat().st_size
        digests[str(path)]=meta['sha256']
        if (i+1)%40==0:
            progress(dict(checked_bundles=i+1,total_bundles=386))
    if found!=expected:
        raise ValueError('cache Cartesian coverage differs')
    prompts=list(dict.fromkeys([r[k] for r in training+dev for k in ('neutral','positive')]+['']))
    embeds=json.loads((root/'embeddings.json').read_text())
    if len(prompts)!=53 or embeds['prompts_sha256']!=digest(prompts) or sha(root/'embeddings.safetensors')!=embeds['sha256']:
        raise ValueError('frozen Qwen prompt-order/hash differs')
    header=safe_header(root/'embeddings.safetensors')
    if set(header)!={f'{k}{i}' for i in range(53) for k in ('e','m')}:
        raise ValueError('frozen Qwen context keys differ')
    for i in range(53):
        a,b=header[f'e{i}'],header[f'm{i}']
        if (a['dtype']!='BF16' or len(a['shape'])!=3 or a['shape'][0]!=1 or a['shape'][2]!=4096 or
                not 0<a['shape'][1]<=512 or b['dtype']!='BOOL' or b['shape']!=a['shape'][:2]):
            raise ValueError('Qwen context/mask header differs')
    return dict(schema_pass=True,recipe_identity=identity,recipe_sha256=sha(root/'recipe.json'),
                index_sha256=sha(root/'target-index.json'),embeddings_sha256=embeds['sha256'],
                train_bundles=384,heldout_bundles=2,signed_train_rows=15360,
                total_teacher_bytes=total,all_target_hashes=digests,contexts=53,
                values_finite_previously_qualified=True,model_weights_opened=False)


def validate_supra_cache(root=SUPRA):
    """Only the small CPU response caches, never checkpoint/model weights."""
    import torch
    import yaml
    torch.set_num_threads(1)
    root=Path(root)
    config=yaml.safe_load((root/'benchmark/source/prompts-final-boss.yaml').read_text())
    if digest(config)!=CONFIG_SHA:
        raise ValueError('exact19-context config identity differs')
    from .pg09_remaining_campaigns import required_supra_prompts
    if len(required_supra_prompts(config))!=19:
        raise ValueError('exact missing19-context prerequisite differs')
    result={}
    for split,name,counts in [('train','benchmark/teacher_cache.pt',(240,30)),
                              ('validation','converged/validation_cache.pt',(240,60))]:
        path=root/name
        if path.stat().st_size>64*1024*1024:
            raise ValueError('CPU response cache exceeds offline load bound')
        cache=torch.load(path,map_location='cpu',weights_only=True)
        if (len(cache['records']),len(cache['holds']))!=counts:
            raise ValueError('full native cache row counts differ')
        for hold,pool in [(False,cache['records']),(True,cache['holds'])]:
            for row in pool:
                if not 0<=row['t']<1:
                    raise ValueError('native time outside [0,1)')
                if any(tuple(row[k].shape)!=(1,4,32,32) or row[k].dtype!=torch.float32 or
                       row[k].device.type!='cpu' or not torch.isfinite(row[k]).all() for k in ('z','neutral','target')):
                    raise ValueError('native field/dtype/finite schema differs')
                if hold and not torch.equal(row['neutral'],row['target']):
                    raise ValueError('preservation target differs from frozen base')
        if split=='train':
            if sorted({r['seed'] for r in cache['records']})!=[7063,7064] or sorted({r['t'] for r in cache['records']})!=[i/10 for i in range(10)]:
                raise ValueError('training native seed/time coverage differs')
            energy=sum(float((r['target']-r['neutral']).double().square().sum()) for r in cache['records'])
            scale=(energy/sum(r['target'].numel() for r in cache['records']))**.5
        else:
            if cache['seeds']!=(39001,39002) and cache['seeds']!=[39001,39002]:
                raise ValueError('validation seeds differ')
            if len(cache['endpoints'])!=12:
                raise ValueError('validation endpoint count differs')
        result[split]=dict(schema_pass=True,path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                           edited=counts[0],preservation=counts[1])
    if torch.cuda.is_initialized():
        raise RuntimeError('offline qualification unexpectedly initialized CUDA')
    result.update(response_scale=scale,required_unique_contexts=19,config_canonical_sha256=CONFIG_SHA,
                  archived_context_export=False,cuda_initialized=False,model_weights_opened=False)
    return result


def stat_model_assets():
    """Metadata/prior-hash provenance only: never open pretrained weight files."""
    manifest=json.loads((SUPRA/'model-manifest.json').read_text())
    names={'SupraLabs/Supra2-IMG':('models--SupraLabs--Supra2-IMG',MODEL_REV),
           'google/flan-t5-base':('models--google--flan-t5-base',TEXT_REV),
           'stabilityai/sd-vae-ft-mse':('models--stabilityai--sd-vae-ft-mse',VAE_REV)}
    result=[]
    for item in manifest:
        directory,revision=names[item['model']]
        path=SUPRA_HUB/directory/'snapshots'/revision/item['filename']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or path.resolve().name!=item['sha256']:
            raise ValueError('Supra local asset stat/prior content-address identity differs')
        result.append(dict(path=str(path),bytes=item['bytes'],prior_verified_sha256=item['sha256'],
                           weight_bytes_opened=False,sha_reverified=False))
    index=json.loads((MOSS_SNAPSHOT/'transformer/diffusion_pytorch_model.safetensors.index.json').read_text())
    for name in sorted(set(index['weight_map'].values())):
        path=MOSS_SNAPSHOT/'transformer'/name
        if not path.is_file():
            raise ValueError('local MOSS shard absent')
        result.append(dict(path=str(path),bytes=path.stat().st_size,prior_content_address=path.resolve().name,
                           weight_bytes_opened=False,sha_reverified=False))
    vae=MOSS_SNAPSHOT/'vae/diffusion_pytorch_model.safetensors'
    if not vae.is_file():
        raise ValueError('MOSS local FP32 VAE asset absent')
    result.append(dict(path=str(vae),bytes=vae.stat().st_size,prior_content_address=vae.resolve().name,
                       weight_bytes_opened=False,sha_reverified=False))
    return result


def validate_active_pins():
    result={}
    for case in ('bad-intent','afterimage'):
        path=CORE/'qualification/moss-reruns'/f'{case}-original-runtime-v2.json'
        data=json.loads(path.read_text())
        mismatches=[p for p,h in data['source_pins'].items() if sha(p)!=h]
        if mismatches:
            raise ValueError('active MOSS source drift: '+repr(mismatches))
        result[case]=dict(recipe_sha256=sha(path),sources=len(data['source_pins']),all_match=True,edited=False)
    return result
