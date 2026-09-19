"""Research-only dense updates; install solely in the dedicated render process."""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from common import sha


def install(generator):
    original = generator._merge_sliders
    events = []

    def merge(pipe, device, comps):
        dense = [c for c in comps if c.get('format') == 'dense_merge']
        if not dense:
            original(pipe, device, comps)
            events.append(dict(mode='ordinary', components=len(comps)))
            return
        if len(dense) != 1 or len(comps) != 1:
            raise ValueError('Study supports one precomputed dense mixture')
        comp = dense[0]
        if generator._apply_mode(device) != 'merge':
            raise RuntimeError('Dense study requires a fully loaded merge-mode pipeline')
        state = generator._merge_state(device)
        sig = generator._merge_signature(comps)
        if state.signature == sig:
            events.append(dict(mode='dense_cached', manifest_sha256=comp['sha256']))
            return
        state.signature = generator._MERGE_FAILED
        generator._restore_pristine(state)
        try:
            if sha(comp['weights']) != comp['sha256']:
                raise ValueError('Dense manifest changed')
            manifest = json.loads(Path(comp['weights']).read_text())
            with torch.no_grad():
                network = generator._slider_network(pipe, device, manifest['source_components'][0], attach=False)
                hosts = network.hosts
                names = [m['name'] for m in manifest['modules']]
                if len(set(names)) != len(names) or set(names) != set(hosts):
                    raise ValueError('Dense update and runtime host module sets differ')
                for module in manifest['modules']:
                    if sha(module['path']) != module['sha256']:
                        raise ValueError('Dense update changed: '+module['name'])
                    host = hosts[module['name']]
                    delta = load_file(module['path'], device='cpu')['delta']
                    if list(delta.shape) != module['shape'] or delta.shape != host.weight.shape:
                        raise ValueError('Dense update shape mismatch')
                    if delta.dtype != torch.float32 or not torch.isfinite(delta).all():
                        raise ValueError('Invalid dense update values')
                    pristine = generator._snapshot_pristine(state, host)
                    merged = (pristine.float()+delta*float(comp['multiplier'])).to(pristine.dtype)
                    host.weight.data.copy_(merged)
                    del delta, merged
            state.signature = sig
            events.append(dict(mode='dense', method=manifest['method'], modules=len(names),
                               manifest_sha256=comp['sha256'], pristine_bytes=state.bytes))
        except BaseException:
            generator._restore_pristine(state)
            raise

    generator._merge_sliders = merge
    return events
