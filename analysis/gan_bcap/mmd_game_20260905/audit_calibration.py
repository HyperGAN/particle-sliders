"""Verify a real strength candidate, its derivation state and archived sources."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .game import ROOT, sha, load_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    home = Path(__file__).resolve().parent
    folder = home/'rounds'/args.round
    data = folder/args.candidate
    manifest = json.loads((data/'manifest.json').read_text())
    completion = json.loads((data/'completion.json').read_text())
    state = torch.load(data/'state.pt', map_location='cpu', weights_only=True)
    parent_state = torch.load(data/'parent-state.pt', map_location='cpu', weights_only=True)
    initial = load_file(str(data/'initial.safetensors'))
    final = load_file(completion['final_weights'])
    original_meta = json.loads((data/'initial.json').read_text())
    metadata = json.loads(Path(completion['final_weights']).with_suffix('.json').read_text())
    assert sha(data/'parent-state.pt') == manifest['parent_state_sha256'] == state['parent_state_sha256']
    assert sha(data/'initial.safetensors') == manifest['parent_weights_sha256']
    assert sha(completion['final_weights']) == manifest['final_weights_sha256']
    assert final.keys() == initial.keys() == state['network'].keys() == parent_state['network'].keys()
    alpha_count = matrix_count = 0
    for key, value in initial.items():
        assert torch.equal(value, parent_state['network'][key])
        assert torch.equal(final[key], state['network'][key])
        if key.endswith('.alpha'):
            assert float(final[key]) == float(value)*manifest['factor'] == metadata['alpha']
            alpha_count += 1
        else:
            assert torch.equal(value, final[key])
            matrix_count += 1
    assert metadata['alpha'] == original_meta['alpha']*manifest['factor']
    assert metadata['rank'] == original_meta['rank'] and metadata['unit_scale'] == 1.
    assert alpha_count == manifest['alpha_buffer_count']
    assert state['schema'] == 'derived-strength-1' and state['optimizer'] is None
    assert completion['attempted_updates'] == manifest['attempted_updates'] == state['completed_optimizer_updates'] == 0
    assert sha(data/'prompts.yaml') == manifest['prompts_sha256']
    for filename, expected in manifest['sources'].items():
        assert sha(data/'provenance'/Path(filename).relative_to(ROOT)) == expected
    bookkeeping = json.loads((folder/'bookkeeping-provenance/manifest.json').read_text())
    for filename, expected in bookkeeping['sources'].items():
        assert sha(folder/'bookkeeping-provenance'/Path(filename).name) == expected
    protocol, _ = load_game(home)
    assert bookkeeping['frozen_protocol_sha256'] == sha(home/'protocol.json')
    assert bookkeeping['frozen_judge_sources'] == protocol['sources']
    report = dict(status='passed', alpha_buffers_changed=alpha_count,
        trained_tensors_bitwise_retained=matrix_count, checkpoint_and_derived_state='bitwise identical',
        full_parent_state='hash verified; all original network tensors match', optimizer_updates=0,
        factor=manifest['factor'], alpha_before=original_meta['alpha'], alpha_after=metadata['alpha'],
        source_snapshots='all declared hashes verified', frozen_judge_and_protocol='unchanged',
        final_weights_sha256=manifest['final_weights_sha256'])
    (folder/'state-and-source-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
