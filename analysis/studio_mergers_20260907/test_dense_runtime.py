import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from common import sha,write
from dense_runtime import install
from app import generator as G


class DenseRuntimeTests(unittest.TestCase):
    def test_swap_cache_restore_and_partial_failure(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            hosts={n:torch.nn.Linear(4,3,bias=False,dtype=torch.bfloat16) for n in ('a','b')}
            bases={n:h.weight.detach().clone() for n,h in hosts.items()}
            modules=[]
            for i,(n,h) in enumerate(hosts.items()):
                path=tmp/f'{n}.safetensors';save_file({'delta':torch.ones(3,4)*(i+1)/10},str(path))
                modules.append(dict(name=n,path=str(path),sha256=sha(path),shape=[3,4]))
            manifest=tmp/'manifest.json';write(manifest,dict(method='test',source_components=[{}],modules=modules))
            comp=dict(weights=str(manifest),mtime=manifest.stat().st_mtime,multiplier=1.,format='dense_merge',sha256=sha(manifest))
            state=G._MergeState()
            def ordinary(pipe,device,comps):
                G._restore_pristine(state)
                for h in hosts.values():
                    p=G._snapshot_pristine(state,h)
                    if comps:h.weight.data.copy_((p.float()+.25).to(p.dtype))
                state.signature=G._merge_signature(comps)
            with patch.object(G,'_merge_sliders',ordinary),patch.object(G,'_merge_state',return_value=state),patch.object(G,'_apply_mode',return_value='merge'),patch.object(G,'_slider_network',return_value=SimpleNamespace(hosts=hosts)):
                events=install(G)
                G._merge_sliders(None,'cpu',[dict(weights='ordinary',mtime=1,multiplier=1)])
                G._merge_sliders(None,'cpu',[comp])
                for i,(n,h) in enumerate(hosts.items()):
                    torch.testing.assert_close(h.weight,(bases[n].float()+(i+1)/10).bfloat16(),atol=0,rtol=0)
                G._merge_sliders(None,'cpu',[comp]);self.assertEqual(events[-1]['mode'],'dense_cached')
                G._merge_sliders(None,'cpu',[])
                for n,h in hosts.items():torch.testing.assert_close(h.weight,bases[n],atol=0,rtol=0)
                # Fail after the first host has changed, then ensure every base is exact.
                modules[1]['sha256']='bad';write(manifest,dict(method='test',source_components=[{}],modules=modules))
                comp.update(sha256=sha(manifest),mtime=comp['mtime']+1)
                with self.assertRaises(ValueError):G._merge_sliders(None,'cpu',[comp])
                self.assertIs(state.signature,G._MERGE_FAILED)
                for n,h in hosts.items():torch.testing.assert_close(h.weight,bases[n],atol=0,rtol=0)
                G._merge_sliders(None,'cpu',[dict(weights='ordinary',mtime=1,multiplier=1)])
                for n,h in hosts.items():torch.testing.assert_close(h.weight,(bases[n].float()+.25).bfloat16(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
