"""Explicit opt-in shared SD3 cohort. Historical train_lora_sd3.py remains unchanged.

Install packages/sd3-shared-slider in its own environment. Run this module with
shared lifecycle CLI arguments, e.g. train --config cohort.json --out fresh-run
--export-format both. This path never calls the historical prompt_pair.loss trainer.
"""
from particle_sliders_sd3_shared.cli import main

if __name__ == '__main__':
    main()
