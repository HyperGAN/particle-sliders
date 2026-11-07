#!/usr/bin/env python3
"""Source-checkout entrypoint; installed users run particle-slider-search."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packages/particle-sliders-core/src'))
from particle_sliders.parameter_search.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main())
