#!/usr/bin/env python3
"""Two-step synthetic CPU connector GAN smoke, requiring the explicit PG profile.

This calls the installed shared CLI and never loads model weights or writes AV.
The caller must provision the exact reviewed ParticleGAN 0.9.1 continuous
profile; no installed source is patched and release-auto fails closed.
"""
from particle_sliders.ltx25_cli import main as train_main


def main(argv=None):
    return train_main(['--dummy', '--steps', '2', '--rank', '2', '--alpha', '2',
                       '--te_last_n', '1', '--seed', '0',
                       '--save_dir', 'models/ltx25-smoke',
                       '--pg_source_profile', 'alz-continuous-v1',
                       '--penalty_input_domain', 'continuous'] + list(argv or []))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
