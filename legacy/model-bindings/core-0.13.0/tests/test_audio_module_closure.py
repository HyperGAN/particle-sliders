"""Package closure and explicit legacy engine boundary; no native model loads."""
import hashlib
import importlib
import json
from pathlib import Path
import pytest
import particle_sliders
from particle_sliders import pg09


def test_audio_modules_have_closed_shared_imports():
    for name in ('pg09_audio', 'music3_native', 'native_lora', 'yue2_native', 'pg09_audio_cli'):
        assert importlib.import_module('particle_sliders.' + name)
    root = Path(particle_sliders.__file__).parent
    receipt = json.loads((root / 'audio_closure_provenance.json').read_text())
    for name, entry in receipt['sources'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == entry['packaged_sha256']


def test_legacy_cli_does_not_relabel_other_pg_engines():
    from particle_sliders import pg09_audio_cli as legacy
    if pg09.PG_VERSION != '0.9.1':
        with pytest.raises(RuntimeError, match='exact 0.9.1'):
            legacy.make_audio_game()
        with pytest.raises(RuntimeError, match='exact 0.9.1'):
            legacy.probe_export('/missing-training-and-model-data')
    else:
        assert legacy.capabilities('music3')['cpu_random_native_ar']
        assert not legacy.capabilities('yue2')['cpu_random_native_ar']
