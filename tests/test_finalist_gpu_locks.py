import fcntl
from pathlib import Path

import pytest

from scripts import search_yue2_critic as search


def setup_search(tmp_path, monkeypatch, *, fail_spawn=None, exit_code=0):
    native_flock = fcntl.flock
    monkeypatch.setattr(search.fcntl, 'flock', lambda fd, flags: native_flock(fd, flags | fcntl.LOCK_NB))
    monkeypatch.setattr(search, 'ROOT', tmp_path / 'repo')
    monkeypatch.setattr(search, 'environment', lambda gpu: {})
    monkeypatch.setattr(search, 'eval_command', lambda candidate, run, output: [candidate['name']])
    events, processes = [], []

    class Process:
        def __init__(self, name, descriptor):
            self.name, self.descriptor, self.code = name, descriptor, None

        def wait(self, timeout=None):
            events.append(('wait', self.name))
            self.code = exit_code
            return self.code

        def poll(self):
            return self.code

        def terminate(self):
            events.append(('terminate', self.name))

    def spawn(argv, **kwargs):
        events.append(('spawn', argv[0]))
        if argv[0] == fail_spawn:
            raise OSError('spawn failed')
        process = Process(argv[0], kwargs['pass_fds'][0])
        processes.append(process)
        return process

    monkeypatch.setattr(search.subprocess, 'Popen', spawn)
    return events, processes


def assert_leases_released(tmp_path, gpus):
    for gpu in gpus:
        with (tmp_path / f'.music-gpu-{gpu}.lock').open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_more_finalists_than_gpus_reaps_before_reacquiring(tmp_path, monkeypatch):
    events, processes = setup_search(tmp_path, monkeypatch)
    search.render_finalists(tmp_path, [{'name': str(i)} for i in range(3)], [7])
    assert events == [('spawn', '0'), ('wait', '0'), ('spawn', '1'), ('wait', '1'), ('spawn', '2'), ('wait', '2')]
    assert all(process.code == 0 for process in processes)
    assert_leases_released(tmp_path, [7])


def test_spawn_failure_reaps_owned_children_and_releases_all_leases(tmp_path, monkeypatch):
    events, processes = setup_search(tmp_path, monkeypatch, fail_spawn='1')
    with pytest.raises(OSError, match='spawn failed'):
        search.render_finalists(tmp_path, [{'name': '0'}, {'name': '1'}], [7, 8])
    assert ('terminate', '0') in events
    assert processes[0].code == 0
    assert_leases_released(tmp_path, [7, 8])


def test_child_failure_reaps_batch_and_does_not_launch_successor(tmp_path, monkeypatch):
    events, _ = setup_search(tmp_path, monkeypatch, exit_code=3)
    with pytest.raises(RuntimeError, match='exited 3'):
        search.render_finalists(tmp_path, [{'name': str(i)} for i in range(3)], [7, 8])
    assert events.count(('wait', '0')) == 1 and events.count(('wait', '1')) == 1
    assert ('spawn', '2') not in events
    assert_leases_released(tmp_path, [7, 8])


@pytest.mark.parametrize('gpus', [[], [7, 7]])
def test_invalid_gpu_lists_refused_before_launch(tmp_path, monkeypatch, gpus):
    events, _ = setup_search(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='distinct GPUs'):
        search.render_finalists(tmp_path, [{'name': '0'}], gpus)
    assert events == []
