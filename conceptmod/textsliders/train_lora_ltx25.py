#!/usr/bin/env python3
"""Compatibility entrypoint for the installable shared LTX GAN CLI."""
from __future__ import annotations

if __name__ == '__main__':
    import sys
    if '--dummy' not in sys.argv[1:]:
        raise SystemExit('LTX production runtime is not qualified; only --dummy CPU software fixtures are admitted')
    from particle_sliders.ltx25_cli import main
    main()
else:
    import sys
    from particle_sliders import ltx25_cli as _implementation
    sys.modules[__name__] = _implementation
