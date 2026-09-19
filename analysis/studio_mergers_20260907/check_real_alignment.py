"""Check thin-SVD reconstruction on real full-size q/k projections."""
from contextlib import ExitStack
import json
import torch
from safetensors import safe_open
from common import WORK,PAIRS,sha,write
from mergers import joint_basis


def main():
    torch.set_num_threads(4)
    checks=[]
    for pair in PAIRS:
        manifest=json.loads((WORK/'artifacts'/'+'.join(pair)/'knots_ties/manifest.json').read_text())
        names=[next(m['name'] for m in manifest['modules'] if m['name'].endswith(suffix))
               for suffix in ('q_proj','k_proj')]
        comps=manifest['source_components']
        with ExitStack() as stack:
            files=[stack.enter_context(safe_open(c['weights'],framework='pt',device='cpu')) for c in comps]
            for name in names:
                ups,downs,deltas=[],[],[]
                for f,c in zip(files,comps):
                    a=f.get_tensor(name+'.lora_down.weight').float()
                    b=f.get_tensor(name+'.lora_up.weight').float()
                    scale=c['multiplier']*float(f.get_tensor(name+'.alpha'))/a.shape[0]
                    deltas.append((b@a)*scale);ups.append(b*scale);downs.append(a)
                left,blocks,s=joint_basis(ups,downs)
                ordinary=sum(deltas)
                aligned=left@sum(blocks)
                error=float(torch.linalg.vector_norm(aligned-ordinary,dtype=torch.float64)/torch.linalg.vector_norm(ordinary,dtype=torch.float64))
                assert error < 1e-6
                checks.append(dict(pair=pair,module=name,joint_rank=len(s),shape=list(ordinary.shape),relative_linear_reconstruction_error=error))
    write(WORK/'real-alignment-checks.json',dict(status='passed',algorithm_sha256=sha(WORK/'mergers.py'),checks=checks))
    print('Real alignment checks passed:',len(checks),'maximum relative error',max(x['relative_linear_reconstruction_error'] for x in checks))


if __name__=='__main__':main()
