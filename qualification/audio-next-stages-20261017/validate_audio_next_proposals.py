"""CPU/read-only validation of exact proposal scope, identity and phase gates."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


root = args.directory
index = load(root / 'sequential-plan.json')
assert not index['dispatch_allowed'] and not index['automation_created']
assert index['phase_order'] == ['Y20', 'YR0', 'YR1', 'M3S']
asset_receipt = load(root / 'actual-asset-identities.json')
large = {f['path']:f for r in asset_receipt['families'].values() for f in r['files']}
checked_sources = set()
phases = []
for name in index['phase_cards']:
    card = load(root / name)
    assert sha(root / name) == index['proposal_hashes'][name]
    assert card['status'] == 'PROPOSED_NOT_ADMITTED' and not card['dispatch_allowed']
    assert card['admission_id'] is None
    assert not Path(card['output_directory']).exists()
    assert all('site-packages' not in p for p in card['environment']['PYTHONPATH'].split(':'))
    assert card['limits']['compute_seconds'] == 720
    assert card['limits']['shutdown_grace_seconds'] == 180
    assert card['limits']['maximum_wall_seconds'] == 900
    assert card['limits']['maximum_concurrent_cuda_children'] == 1
    assert card['limits']['host_running_available_gib_minimum'] >= 8
    assert not card['limits']['retry'] and not card['limits']['automatic_next_phase']
    assert card['source_profile'] == 'alz-continuous-v1' and card['penalty_input_domain'] == 'continuous'
    assert len(card['pg_sources']) == 49
    if card['family'] == 'yue2':
        product = Path('/mnt/nvme/sliders/audio-yue2-source-20261017')
        product_files = sorted((product/'yue2').glob('*.py'))
        assert len(product_files) == 9
        assert all(str(path) in card['source_pins'] for path in product_files), 'missing fixed product source'
        assert all(card['source_pins'][str(path)] == sha(path) for path in product_files)
    runtime = load(root / card['runtime_receipt'])
    assert not runtime['cuda_initialized'] and not runtime['models_constructed']
    assert runtime['domain']['resolved_penalty_domain'] == 'continuous'
    for path, expected in card['source_pins'].items():
        assert sha(path) == expected, path
        checked_sources.add(path)
    for path, expected in card['input_pins'].items():
        if path in large and large[path]['bytes'] > 32 * 1024**2:
            assert Path(path).stat().st_size == large[path]['bytes']
            assert large[path]['sha256'] == expected and large[path]['verified']
        else:
            assert sha(path) == expected, path
    phase = card['phase']
    if phase == 'Y20':
        command = card['command']
        assert command[1].endswith('/scripts/qualify_audio_pg09_live.py')
        assert command[command.index('--until')+1] == '20'
        assert command[command.index('--total-steps')+1] == '20'
        assert command[command.index('--product')+1] == '/mnt/nvme/sliders/audio-yue2-source-20261017'
        assert card['initial_completed_steps'] + card['additional_updates'] == 20
        assert card['input_pins'][command[command.index('--resume')+1]]
    elif phase == 'M3S':
        command = card['command']
        assert command[1].endswith('/scripts/run_audio_pg09_smoke.py')
        assert '--resume' not in command and '--until' not in command
        assert command[command.index('--family')+1] == 'music3'
        assert command[command.index('--minimum-free-gib')+1] == '30'
        assert card['limits']['sequential_cuda_children'] == 3
    else:
        assert card['command'][1].endswith('/scripts/render_yue2_pg09.py')
        assert not card['executable_card_ready']
        assert card['producer_input']['sha256'] is None
        assert card['producer_qualification_input']['sha256'] is None
        assert card['root_card_template']['admission_id'] is None
        assert card['root_card_template']['product'] == '/mnt/nvme/sliders/audio-yue2-source-20261017'
        assert card['root_card_template']['limits']['host_running_available_gib_minimum'] >= 8
        assert sha(card['command'][1]) == card['renderer_sha256']
        assert card['protocol']['cfg_scale'] == 1.0 and card['protocol']['cot'] == 'off'
        assert card['maximum_clips'] == 5 and card['maximum_semantic_tokens_per_clip'] == 200
        assert card['maximum_audio_seconds_per_clip'] == 8.1
        assert card['implementation_gates'] and card['pipeline']['offload_ar'] is False
    assert '--fixture' not in (card['command'] or [])
    phases.append(dict(phase=phase, status=card['status'], output_absent=True,
                       command_frozen=card['command'] is not None))
print(json.dumps(dict(status='CPU_PROPOSAL_VALIDATION_PASSED', phases=phases,
    source_files_rehashed=len(checked_sources), asset_bytes_independently_verified_in_receipt=index['asset_total_verified_bytes'],
    model_tensors_loaded=False, gpu_actions=False, dispatch=False), indent=2))
