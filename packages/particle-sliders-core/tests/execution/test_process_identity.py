import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from particle_sliders.execution import process_identity as identity


def test_local_process_and_own_child_death_are_distinguished():
    assert identity.process_liveness(identity.capture_process_identity()) == 'live'
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        owner = identity.capture_process_identity(proc.pid)
        assert owner['pid'] == proc.pid
        assert identity.process_liveness(owner) == 'live'
    finally:
        proc.terminate()
        proc.wait()
    assert identity.process_liveness(owner) == 'dead'


@pytest.mark.parametrize('change', [
    lambda x: x.update(pid=True), lambda x: x.update(pid=0),
    lambda x: x.update(start='not-decimal'), lambda x: x.update(boot='not-a-uuid'),
    lambda x: x.update(extra='unexpected'), lambda x: x.pop('host'),
])
def test_corrupt_metadata_never_proves_death(change):
    owner = identity.capture_process_identity()
    change(owner)
    assert identity.process_liveness(owner) == 'unknown'


def test_foreign_host_is_unknown_but_proven_local_boot_and_pid_reuse_are_dead():
    owner = identity.capture_process_identity()
    assert identity.process_liveness(dict(owner, host=owner['host'] + '-foreign')) == 'unknown'
    assert identity.process_liveness(dict(owner, boot=str(uuid.uuid4()))) == 'dead'
    assert identity.process_liveness(dict(owner, start=str(int(owner['start']) + 1))) == 'dead'


def test_unreadable_target_is_unknown(monkeypatch):
    owner = identity.capture_process_identity()
    original = identity._stat
    monkeypatch.setattr(identity, '_stat', lambda pid: original(pid) if pid == os.getpid() else (_ for _ in ()).throw(PermissionError()))
    assert identity.process_liveness(dict(owner, pid=owner['pid'] + 100000)) == 'unknown'


def test_stat_parser_handles_spaces_and_parentheses_in_comm(monkeypatch):
    values = ['S'] + ['0'] * 18 + ['4242'] + ['0'] * 5
    monkeypatch.setattr(Path, 'read_text', lambda self: '123 (worker ) with spaces) ' + ' '.join(values))
    assert identity._stat(123) == ('S', '4242')
