"""Verify trained block tensors and all optimizer states against their parent."""
import argparse
import json
from pathlib import Path
from .core import read, immutable, sha, digest, IntegrityError
from .block_artifact import checkpoint


def audit(home, folder):
    import torch
    from safetensors.torch import load_file
    from diffusers.models.transformers.transformer_minimax_music3 import MiniMaxMusic3Transformer1DModel
    from app.lora_runtime import LoRANetwork
    from ..reward_sliders.specs import MODEL
    home = Path(home).resolve(); folder = Path(folder).resolve()
    recipe = read(folder/'recipe.json'); status = read(folder/'status.json')
    if status['state'] != 'checkpoint_ready' or status['actual_updates'] != 2:
        raise IntegrityError('Need two completed joint full-batch updates')
    if not read(folder/'off-restoration-2.json')['exact']:
        raise IntegrityError('Joint training Off restoration failed')
    for step in (1, 2):
        if not read(folder/f'native-step{step}-audit.json')['passed']:
            raise IntegrityError('Updated joint block native audit failed')
    candidate = checkpoint(status['checkpoint'], 1.)
    if checkpoint(recipe['parent']['path'], 1.) != recipe['parent']:
        raise IntegrityError('Declared block parent changed')
    values = load_file(candidate['path']); parent = load_file(recipe['parent']['path'])
    state_path = folder/'state-step2.pt'; state = torch.load(state_path, map_location='cpu', weights_only=True)
    updates = [json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines() if line]
    if state['recipe_sha256'] != digest(recipe) or state['completed'] != 2 or state['pending'] or state['history'] != updates:
        raise IntegrityError('Saved joint block history differs from training records')
    if [u['step'] for u in updates] != [1, 2] or any(u['examples'] != 8 or len(u['cases']) != 8 for u in updates):
        raise IntegrityError('Each update must contain every balanced training case')
    if values.keys() != state['network'].keys() or not all(torch.equal(v, state['network'][k]) for k, v in values.items()):
        raise IntegrityError('Joint export differs from actual saved factors')
    torch.set_num_threads(2)
    with torch.device('meta'):
        host = MiniMaxMusic3Transformer1DModel.from_config(read(MODEL/'transformer/config.json')).to(torch.bfloat16).requires_grad_(False)
    network = LoRANetwork(host, rank=8, alpha=8., multiplier=1., target_replace=['MiniMaxMusic3TransformerBlock'],
        prefix='lora_unet', delimiter='-', train_method='full', attach=False).requires_grad_(True)
    parameters = list(network.named_parameters()); groups = state['optimizer']['param_groups']
    if len(groups) != 1 or len(groups[0]['params']) != 432 or len(parameters) != 432:
        raise IntegrityError('Wrong joint block optimizer groups')
    group = groups[0]; optimizer = state['optimizer']['state']
    if group['lr'] != .0005 or tuple(group['betas']) != (.9, .999) or group['weight_decay'] != 0.:
        raise IntegrityError('Joint optimizer settings changed')
    if set(optimizer) != set(group['params']):
        raise IntegrityError('Missing or extra joint optimizer state')
    for ident, (name, parameter) in zip(group['params'], parameters):
        opt = optimizer[ident]
        if float(opt['step']) != 2:
            raise IntegrityError('A joint factor has the wrong optimizer step')
        for key in ('exp_avg', 'exp_avg_sq'):
            if opt[key].shape != values[name].shape or not torch.isfinite(opt[key]).all():
                raise IntegrityError('Invalid joint optimizer moment')
        if bool((opt['exp_avg_sq'] < 0).any()):
            raise IntegrityError('Negative joint second moment')
    alpha = [k for k in values if k.endswith('.alpha')]
    if len(alpha) != 216 or not all(torch.equal(values[k], parent[k]) for k in alpha):
        raise IntegrityError('Joint alpha buffers changed')
    changed = {kind: [k for k in values if token in k and not k.endswith('.alpha') and not torch.equal(values[k], parent[k])]
               for kind, token in (('attention', '-attn-'), ('feed_forward', '-ff_'))}
    if not all(changed.values()):
        raise IntegrityError('Both joint parameter groups must change')
    result = dict(passed=True, checkpoint=candidate, parent_sha256=recipe['parent']['weights_sha256'],
        state_sha256=sha(state_path), actual_updates=2, full_batch_examples=16, tensor_count=648,
        export_matches_state_exactly=True, unchanged_alpha_tensors=216, optimizer_factor_states=432,
        all_optimizer_steps=2, changed_factors={k: len(v) for k, v in changed.items()},
        native_audits_passed=True, off_restoration_exact=True, source_sha256=sha(__file__),
        new_audio_generated=0, new_optimizer_updates=0, interpretation='Training/export integrity only; ordinary development determines improvement')
    immutable(home/'audit/acoustic-robust-block-export-v1.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); p.add_argument('--folder', required=True)
    a = p.parse_args(); print(json.dumps(audit(a.home, a.folder), indent=2))
