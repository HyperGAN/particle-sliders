"""Explicit shared lifecycle CLI; no historical config/checkpoint coercion."""
import argparse
import json
from pathlib import Path

from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import export_inference


def train_shared(config, *, resume=None, checkpoint=None, export=None, selection='raw', **staged):
    config = SessionConfig.from_dict(config) if isinstance(config, dict) else config
    with open_response(config, resume=resume, **staged) as session:
        session.train()
        packet = session.checkpoint(checkpoint)
        if export is not None:
            export_inference(session, export, selection=selection)
        return packet


def main():
    parser = argparse.ArgumentParser(description='Opt-in SDXL native response GAN; local staging only')
    parser.add_argument('--config', required=True, help='Shared SessionConfig JSON with a new cohort name')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--export')
    parser.add_argument('--selection', choices=('raw', 'ema'), default='raw')
    args = parser.parse_args()
    train_shared(json.loads(Path(args.config).read_text()), resume=args.resume,
                 checkpoint=args.checkpoint, export=args.export, selection=args.selection)


if __name__ == '__main__':
    main()
