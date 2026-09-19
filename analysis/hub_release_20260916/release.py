"""Validate a reviewable package, publish atomically, and verify remote hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote,urlparse
import yaml
from build_assets import WORK,ROOT,PACKAGE,VERSION,REPO,read,write,sha,names

URL=f'https://huggingface.co/{REPO}/resolve/main/'

def inventory():
    files={str(p.relative_to(PACKAGE)):dict(sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='release-manifest.json'}
    write(PACKAGE/'release-manifest.json',dict(release='2026-09-16-'+VERSION,repo=REPO,
        parent_commit=read(WORK/'remote-before.json')['revision'],version=VERSION,
        native_checkpoints=16,comfyui_checkpoints=16,concept_pairs=64,
        selected_from_candidates=80,selection_policy='quality-later-v2',quality_tolerance=.2,
        audio=dict(original_wavs=192,mp3_previews=192,arms=['off','on','reference'],strength=1,
            normalization='none',first_draw=True,featured_case=dict(row=2,seed=1709)),
        conversion=dict(repository='https://github.com/mikkel/conceptmod',revision='a8a9e898ea618d83f05505c5ece7c8e4ffa9c3df',
            factors='bf16',alpha='float32',native_tensor_count=432,converted_tensor_count=432),
        validation=read(WORK/'asset-validation.json'),
        previous_releases='Prior files remain at their existing paths; current catalog and card select this release.',files=files))

def validate():
    manifest=read(PACKAGE/'release-manifest.json');remote=set(read(WORK/'remote-before.json')['files'])
    for path,record in manifest['files'].items():assert sha(PACKAGE/path)==record['sha256'],path
    catalog=read(PACKAGE/'catalog.json')['sliders'];snapshot=read(WORK/'audit-snapshot.json')
    expected={m['id']:m['recommendation'] for m in snapshot['sliders']}
    assert len(catalog)==16 and len({c['id'] for c in catalog})==16
    for c in catalog:
        assert c['sha256']==expected[c['id']]['weights_sha256']
        assert c['training_steps']==expected[c['id']]['step']
        assert sha(PACKAGE/c['weights'])==c['sha256']
        assert sha(PACKAGE/c['comfyui_weights'])==c['comfyui_sha256']
    pairs=read(PACKAGE/f'samples/{VERSION}/pairs.json')['pairs'];assert len(pairs)==64
    assert len({(p['id'],p['row'],p['seed']) for p in pairs})==64
    assert all(p['weights_sha256']==expected[p['id']]['weights_sha256'] for p in pairs)
    native=list((PACKAGE/f'weights/{VERSION}').rglob('*.safetensors'))
    comfy=list((PACKAGE/f'comfyui/{VERSION}').glob('*.safetensors'))
    assert len(native)==len(comfy)==16 and all('comfyui' not in p.name for p in native)
    card=(PACKAGE/'README.md').read_text();metadata=yaml.safe_load(card.split('---',2)[1]);assert metadata['base_model']=='MiniMaxAI/MiniMax-Music-3'
    featured={p['id']:p for p in pairs if p['featured']}
    audio_links=re.findall(r'<audio[^>]+src="([^"]+)"',card)
    first=[URL+featured[s][a]['mp3'] for s in ('female','metal','house','disco-funk','pop-punk','lofi') for a in ('off','on')]
    assert audio_links[:12]==first
    assert len(audio_links)==12, 'Six current concept pairs'
    checked=0
    for p in PACKAGE.rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.md','.json','.yaml','.svg','.csv'):
            text=p.read_text();names(text,str(p.relative_to(PACKAGE)))
            assert '/ml2/' not in text and '/home/mikkel/' not in text and '100.90.104.57' not in text,p
        if p.suffix!='.md':continue
        links=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'src="([^"]+)"',text)
        for link in links:
            link=unquote(link)
            if link.startswith(URL):rel=link[len(URL):].split('#')[0]
            elif link.startswith(('https:','http:','#')):continue
            else:
                rel=os.path.relpath(p.parent/link.split('#')[0],PACKAGE)
            target=PACKAGE/rel
            assert target.exists() or rel in remote or any(r.startswith(rel.rstrip('/')+'/') for r in remote),(p.name,link)
            checked+=1
    result=dict(files=len(manifest['files'])+1,bytes=sum(r['bytes'] for r in manifest['files'].values()),
        selected_ids=len(catalog),native_files=len(native),comfyui_files=len(comfy),
        sample_pairs=len(pairs),links_checked=checked,all_local_hashes_verified=True,
        selected_steps={c['id']:c['training_steps'] for c in catalog})
    write(WORK/'package-validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result

def publish():
    from huggingface_hub import HfApi,CommitOperationAdd,ModelCard
    validate();ModelCard((PACKAGE/'README.md').read_text()).validate()
    api=HfApi();parent=read(WORK/'remote-before.json')['revision']
    assert api.model_info(REPO).sha==parent,'Remote head changed; review before committing'
    operations=[CommitOperationAdd(path_in_repo=str(p.relative_to(PACKAGE)),path_or_fileobj=str(p))
        for p in sorted(PACKAGE.rglob('*')) if p.is_file()]
    result=api.create_commit(repo_id=REPO,repo_type='model',operations=operations,parent_commit=parent,
        commit_message='Release audit-selected Music 3 checkpoints, ComfyUI exports and 64 matched sample pairs',num_threads=4)
    write(WORK/'published.json',dict(repo=REPO,commit=result.oid,url=result.commit_url,parent=parent))
    print(result.commit_url,flush=True)

def verify_remote():
    from huggingface_hub import HfApi,hf_hub_download
    published=read(WORK/'published.json');revision=published['commit'];api=HfApi()
    info=api.model_info(REPO,revision=revision,files_metadata=True);assert info.sha==revision
    remote={f.rfilename:f for f in info.siblings};manifest=read(PACKAGE/'release-manifest.json')
    files=dict(manifest['files']);files['release-manifest.json']=dict(sha256=sha(PACKAGE/'release-manifest.json'),bytes=(PACKAGE/'release-manifest.json').stat().st_size)
    checks=[]
    for path,expected in files.items():
        got=remote[path];assert got.size==expected['bytes'],path
        if got.lfs:
            assert got.lfs.sha256==expected['sha256'],path;method='Hub LFS SHA-256'
        else:
            data=(PACKAGE/path).read_bytes();gitsha=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            assert got.blob_id==gitsha,path;method='Hub Git blob SHA-1 and size'
        checks.append(dict(path=path,method=method,verified=True))
    for name in ('README.md','catalog.json','release-manifest.json','comfyui/README.md'):
        path=hf_hub_download(REPO,name,revision=revision);assert sha(path)==sha(PACKAGE/name)
    previous=set(read(WORK/'remote-before.json')['files']);assert previous<=set(remote),'Existing remote files were removed'
    write(WORK/'remote-validation.json',dict(revision=revision,files_verified=len(checks),previous_files_retained=True,checks=checks))
    print(f'Verified {len(checks)} remote file hashes/sizes at {revision}; all prior paths retained.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inventory','validate','publish','verify-remote']);args=parser.parse_args()
    {'inventory':inventory,'validate':validate,'publish':publish,'verify-remote':verify_remote}[args.action]()
