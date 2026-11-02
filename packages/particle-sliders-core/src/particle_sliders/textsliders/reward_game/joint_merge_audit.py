"""Actual native loading and weight routing for a pair of ordinary host LoRAs."""
from pathlib import Path
from .core import write,IntegrityError
from .joint_artifact import checkpoint,component


def audit(renderer,bundle,folder):
    import torch
    from safetensors.torch import load_file
    pair=checkpoint(bundle,1.);parts=component(bundle,1.);host=renderer.host;device=renderer.device
    result=dict(passed=False,pair=pair,physical_gpu=renderer.gpu,local_optimizer_updates=0)
    try:
        host._merge_sliders(renderer.pipe,device,parts)
        state=host._merge_state(device);counts={}
        for part in parts:
            network=host._slider_network(renderer.pipe,device,part,attach=False)
            source=load_file(part['weights']);count=0
            for lora in network.unet_loras:
                name=lora.lora_name;projection=network.hosts[name]
                up=source[name+'.lora_up.weight'].float();down=source[name+'.lora_down.weight'].float()
                alpha=float(source[name+'.alpha']);expected=(state.pristine[projection].float()+up@down*(alpha/down.shape[0])*part['multiplier']).to(projection.weight.dtype)
                if not torch.equal(expected,projection.weight.detach().cpu()):
                    raise IntegrityError('Joint native merge mismatch: '+name)
                count+=1
            if count!=144:raise IntegrityError('Joint host did not load all projections')
            counts[part['kind']]=count
        if counts!=dict(language_model=144,transformer=144):raise IntegrityError('Joint host routing changed')
        result.update(passed=True,exact_projection_counts=counts,ordinary_native_loading=True)
    except BaseException as exc:
        result['error']=repr(exc);raise
    finally:
        host._merge_sliders(renderer.pipe,device,[])
        exact=all(torch.equal(mod.weight.detach().cpu(),saved) for mod,saved in host._merge_state(device).pristine.items())
        result['off_exact']=exact;result['passed'] &= exact
        write(Path(folder)/'joint-merge.json',result)
        if not exact:raise IntegrityError('Joint merge failed Off restoration')
    return result
