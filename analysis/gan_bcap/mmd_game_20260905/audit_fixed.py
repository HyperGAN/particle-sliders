"""Verify the saved round-two tensors and summarize precision/optimization evidence."""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from conceptmod.textsliders.gan_v2.data import sha


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent/'rounds'/args.round
    training = folder/args.candidate
    completion = json.loads((training/'completion.json').read_text())
    manifest = json.loads((training/'manifest.json').read_text())
    final = torch.load(training/'state.pt', map_location='cpu', weights_only=True)
    initial = torch.load(training/f"state-step{manifest['source_step']}.pt", map_location='cpu', weights_only=True)
    source = torch.load(training/'initial-state.pt', map_location='cpu', weights_only=True)
    weights = load_file(completion['final_weights'])
    assert weights.keys() == final['network'].keys()
    assert all(torch.equal(value, final['network'][key]) for key, value in weights.items())
    assert all(torch.equal(value, initial['network'][key]) for key, value in source['network'].items())
    assert not initial['optimizer']['state']
    assert sha(training/'prepared.pt') == manifest['prepared_sha256']
    history = [json.loads(line) for line in (training/'train.jsonl').read_text().splitlines()]
    assert final['history'][:len(source['history'])] == source['history']
    assert final['history'][len(source['history']):] == history
    assert final['step'] == history[-1]['step'] == completion['step']
    for row in history:
        result = row['line_search']
        assert result['loss'] <= row['loss_before']
        assert result['accepted'] == (result['parameter_step'] > 0)
    moment_steps = sorted(set(int(value['step']) for value in final['optimizer']['state'].values()))
    adam_accepts = sum(row['line_search']['method'] == 'adam' for row in history)
    assert moment_steps == ([adam_accepts] if adam_accepts else [])
    movement = math.sqrt(sum(float((value.double()-initial['network'][key].double()).square().sum())
                             for key, value in final['network'].items()))
    source_root = Path(__file__).resolve().parents[3]
    for filename, expected in manifest['sources'].items():
        archived = training/'provenance'/Path(filename).relative_to(source_root)
        assert sha(archived) == expected
    probes = {}
    for precision in ('bf16', 'float32'):
        report = json.loads((training/f'precision-{precision}.json').read_text())
        probes[precision] = dict(
            repeated_loss_range=max(report['no_grad_repeats'])-min(report['no_grad_repeats']),
            backward_minus_no_grad=report['backward_loss']-report['no_grad_repeats'][0],
            gradient_norm=report['gradient_norm'],
            directions=[dict(length=t['length'], decrease=t['sides']['1']['change'],
                             derivative_ratio=t['finite_difference']/t['predicted_derivative'])
                        for t in report['trials']])
    prepared = torch.load(training/'prepared.pt', map_location='cpu', weights_only=True)
    assert all(sha(row['history_cache']) == row['history_cache_sha256'] for row in prepared['train']+prepared['heldout'])
    groups = {}
    for row in prepared['train']:
        kind = row.get('history_origin', 'original frozen neutral history')
        groups[kind] = groups.get(kind, 0)+1
    result = dict(history_groups=groups, status='passed', checkpoint_state='bitwise identical',
        initial_source_weights='bitwise identical', optimizer_initialization='empty Adam state',
        saved_moment_steps=moment_steps, adam_accepted_updates=adam_accepts,
        final_parameter_l2_from_parent=movement, attempted_updates=len(history),
        accepted=sum(row['line_search']['accepted'] for row in history),
        loss_before=history[0]['loss_before'], loss_after=history[-1]['line_search']['loss'],
        objective_decrease_fraction=1-history[-1]['line_search']['loss']/history[0]['loss_before'],
        objective_evaluations=completion['objective_evaluations'],
        peak_cuda_gib=completion['peak_cuda_bytes']/2**30,
        source_snapshot='all manifest hashes verified', probes=probes,
        rejection_count=sum(not row['line_search']['accepted'] for row in history))
    (folder/'state-and-precision-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
