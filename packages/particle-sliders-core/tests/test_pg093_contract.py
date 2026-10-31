"""The previous explicit engine name must never alias the new PG source cohort."""
import json
from pathlib import Path

import pytest
from particle_sliders import pg093, pg094


def test_old_explicit_engine_refuses_instead_of_rebinding():
    assert pg093.PG_VERSION == '0.9.3' and pg093.PG_SOURCE_PROFILE == 'release-0.9.3'
    assert pg094.PG_VERSION == '0.9.4'
    with pytest.raises(RuntimeError, match='pinned core0.11.0'):
        pg093.ConditionalPG093(object(), object(), object())
    with pytest.raises(ValueError, match='cannot rebind old checkpoints'):
        pg093.engine_options({'pg_version': '0.9.3'})
    with pytest.raises(ValueError, match='pinned core0.11.0'):
        pg093.validate_pg09_domain((1, 32), response_domain='continuous_vector')


def test_original_release_manifest_identity_is_preserved():
    doc = json.loads(Path(pg093.__file__).with_name('pg093_release.json').read_text())
    assert doc['version'] == pg093.PG_VERSION
    assert doc['source_profile'] == pg093.PG_SOURCE_PROFILE
    assert doc['wheel_sha256'] == pg093.PG_WHEEL_SHA256
