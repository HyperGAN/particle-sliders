"""Three CPU-only negative policy checks on temporary proposal copies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
results = {}
for fault in ('old_four_gib_floor', 'missing_product_source', 'wrong_product_root'):
    with tempfile.TemporaryDirectory(prefix='audio-proposal-negative-') as temporary:
        target = Path(temporary) / 'proposal'
        shutil.copytree(args.directory, target)
        name = 'yue2-continue20.proposed.json'
        card = json.loads((target / name).read_text())
        if fault == 'old_four_gib_floor':
            card['limits']['host_running_available_gib_minimum'] = 4
        elif fault == 'missing_product_source':
            del card['source_pins']['/mnt/nvme/sliders/audio-yue2-source-20261017/yue2/backend.py']
        else:
            card['command'][card['command'].index('--product')+1] = '/tmp/not-the-frozen-product'
        (target / name).write_text(json.dumps(card, indent=2)+'\n')
        index = json.loads((target / 'sequential-plan.json').read_text())
        index['proposal_hashes'][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
        (target / 'sequential-plan.json').write_text(json.dumps(index, indent=2)+'\n')
        result = subprocess.run([sys.executable, str(target / 'validate_audio_next_proposals.py'),
                                 str(target)], capture_output=True, text=True)
        assert result.returncode != 0 and 'AssertionError' in result.stderr
        results[fault] = dict(rejected=True, exit_code=result.returncode)
print(json.dumps(dict(status='CPU_NEGATIVE_POLICIES_PASSED', cases=results,
    gpu_actions=False, admitted_sources_changed=False), indent=2))
