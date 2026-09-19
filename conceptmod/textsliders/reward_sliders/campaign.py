"""Serialize GPU stages in child processes, releasing models between stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    prefix = 'conceptmod.textsliders.reward_sliders.'
    subprocess.run([sys.executable, '-u', '-m', prefix+'experiment', 'run', '--run-dir', str(run)], check=True)
    gate_path = run/'causal-result.json'
    if not gate_path.exists() or not json.loads(gate_path.read_text())['passed']:
        print('Causal gate closed; stopped before LoRA training.', flush=True)
        return
    subprocess.run([sys.executable, '-u', '-m', prefix+'train', '--run-dir', str(run)], check=True)
    subprocess.run([sys.executable, '-u', '-m', prefix+'evaluate', '--run-dir', str(run)], check=True)


if __name__ == '__main__':
    main()
