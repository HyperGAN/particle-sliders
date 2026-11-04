"""Controller tests do not consume the developer's host reporting setup."""
import pytest


@pytest.fixture(autouse=True)
def isolated_reporting_config(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path/'config-home'))
    monkeypatch.delenv('PARTICLE_SLIDER_REPORTING_CONFIG', raising=False)
