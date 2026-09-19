"""Verify completed research states, exported weights and captured sources."""
from pathlib import Path
import hashlib
import json
import torch
from safetensors.torch import load_file

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    return True


def main():
    torch.set_num_threads(4)
    results=[]
    for name in ['conditional-energy-research-20260905','conditional-energy-cfg-research-20260905-attempt2','conditional-mmd-cfg-research-20260905']:
        folder=ROOT/'models'/name
        complete=json.loads((folder/'completion.json').read_text())
        manifest=json.loads((folder/'manifest.json').read_text())
        state=torch.load(folder/'state.pt',map_location='cpu',weights_only=True)
        history=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
        assert state['step']==complete['step'] and state['manifest']==manifest and state['history']==history
        assert finite(state)
        assert len(history)==manifest['arguments']['updates']
        assert complete['accepted']==sum(r['line_search']['accepted'] for r in history)
        assert all(r['line_search']['loss']<r['loss_before'] for r in history if r['line_search']['accepted'])
        assert all(r['parameter_step']==0 for r in history if not r['line_search']['accepted'])
        weights=load_file(str(folder/f"{name}_step{state['step']}.safetensors"))
        assert weights.keys()==state['network'].keys()
        assert all(torch.equal(v,state['network'][k]) for k,v in weights.items())
        for filename,digest in manifest['sources'].items():
            assert sha(folder/'provenance'/Path(filename).relative_to(ROOT))==digest
        assert sha(Path(manifest['arguments']['source_state']))==manifest['source_sha256']
        fixed=manifest['arguments'].get('fixed_kernel',False)
        monotone=all(b['line_search']['loss']<=a['line_search']['loss']+1e-7 for a,b in zip(history,history[1:])) if fixed else None
        if fixed:assert monotone and state['metric'] is None and state['d_optimizer'] is None
        results.append(dict(name=name,step=state['step'],accepted=complete['accepted'],
            rejections=len(history)-complete['accepted'],all_tensors_finite=True,
            export_equals_full_state=True,captured_source_hashes_valid=True,
            fixed_objective_monotone=monotone))
        del state,weights
    (HERE/'live-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
