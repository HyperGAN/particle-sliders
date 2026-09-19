"""Bind the actual selected conversions through ComfyUI's CPU LoRA loader."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import torch
from safetensors.torch import load_file

WORK=Path(__file__).resolve().parent
CONVERTER=Path('/tmp/music3-hf-release-20260916/conceptmod')
spec=importlib.util.spec_from_file_location('comfy_apply_tests',CONVERTER/'tests/test_comfyui_lora_apply.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
comfy=m.comfy.__wrapped__()
torch.set_num_threads(4)
host=m._clip_host(comfy,merged_qkv=False)
key_map=comfy.lora.model_lora_keys_clip(host,{})
shapes={k:tuple(v.shape) for k,v in host.state_dict().items()}
package=WORK/'package';catalog=json.loads((package/'catalog.json').read_text());rows=[]
for item in catalog['sliders']:
 tensors=load_file(str(package/item['comfyui_weights']))
 stems=m._lora_stems(tensors);assert len(stems)==144 and stems<=set(key_map)
 patches,unused=m._load_with_unused(comfy,tensors,key_map)
 assert not unused and len(patches)==144,(item['id'],len(patches),unused)
 for stem in stems:
  down=tensors[stem+'.lora_A.weight'];up=tensors[stem+'.lora_B.weight']
  assert (up.shape[0],down.shape[1])==shapes[key_map[stem]],stem
  assert down.shape[0]==up.shape[1]==8 and float(tensors[stem+'.alpha'])==8.
 errors=[]
 for proj in ('q_proj','k_proj','v_proj','o_proj'):
  stem=f'text_encoders.model.layers.0.self_attn.{proj}'
  changed,error=m._apply_delta_error(comfy,patches,key_map[stem],tensors[stem+'.lora_A.weight'],
   tensors[stem+'.lora_B.weight'],float(tensors[stem+'.alpha']),strength=1.)
  assert changed and error<1e-5,(item['id'],proj,error)
  errors.append(error)
 rows.append(dict(id=item['id'],mapped_projections=144,unused_keys=0,all_shapes_match=True,
  delta_checks=4,max_absolute_delta_error=max(errors)))
 print(item['id'],'144 projections bound; q/k/v/o deltas verified',flush=True)
result=dict(comfyui_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/tmp/comfyui',text=True).strip(),
 converter_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CONVERTER,text=True).strip(),
 mode='CPU LoRA loader and meta-device unmerged Music 3 text-encoder topology',
 no_audio_generation=True,checkpoints=rows,all_projections_bound=True,unused_keys=0)
(WORK/'comfyui-loader-validation.json').write_text(json.dumps(result,indent=2)+'\n')
(package/'evidence/uni16-fresh-selected-v2/comfyui-loader.json').write_text(json.dumps(result,indent=2)+'\n')
