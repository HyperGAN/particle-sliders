import json
import multiprocessing
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from particle_sliders.textsliders.reward_search import queue


def jobs(count=1):
    return [{'family': {'family': str(i)}, 'seed': i, 'arms': [{'name': 'off'}]} for i in range(count)]


def claim_and_exit(run):
    queue.claim(run, 0)


def test_dead_owner_is_failed_without_rerendering_or_requeue(tmp_path):
    queue.add(tmp_path, 'screen', jobs())
    child = multiprocessing.get_context('fork').Process(target=claim_and_exit, args=(tmp_path,))
    child.start()
    child.join(timeout=10)
    assert not child.is_alive() and child.exitcode == 0
    recovery = queue.recover_abandoned(tmp_path, 'screen')
    assert recovery == {'failed': ['screen/0/0'], 'unverified': []}
    assert queue.counts(tmp_path) == {'failed': 1}
    assert queue.claim(tmp_path, 1) is None
    queue.add(tmp_path, 'screen', jobs())
    assert queue.counts(tmp_path) == {'failed': 1}


def test_live_claims_remain_owned_and_complete_under_parallel_claims(tmp_path):
    queue.add(tmp_path, 'screen', jobs(8))
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda gpu: queue.claim(tmp_path, gpu), range(8)))
    assert len({job['id'] for job in claimed}) == 8
    assert queue.recover_abandoned(tmp_path) == {'failed': [], 'unverified': []}
    for job in claimed:
        queue.finish(tmp_path, job['id'])
    assert queue.counts(tmp_path) == {'complete': 8}
    with pytest.raises(ValueError, match='current claim owner'):
        queue.finish(tmp_path, claimed[0]['id'])


def test_pid_reuse_cannot_adopt_or_finish_another_claim(tmp_path):
    queue.add(tmp_path, 'screen', jobs())
    job = queue.claim(tmp_path, 0)
    stale = dict(queue._owner(), start='0')
    with queue.connect(tmp_path) as db:
        db.execute('UPDATE jobs SET owner=? WHERE id=?', (json.dumps(stale, sort_keys=True), job['id']))
    with pytest.raises(ValueError, match='current claim owner'):
        queue.finish(tmp_path, job['id'])
    assert queue.recover_abandoned(tmp_path)['failed'] == [job['id']]


def test_legacy_running_rows_are_preserved_and_report_unknown_ownership(tmp_path):
    with sqlite3.connect(tmp_path / 'queue.sqlite') as db:
        db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, stage TEXT, payload TEXT, status TEXT, gpu INTEGER, updated REAL, error TEXT)')
        db.execute("INSERT INTO jobs VALUES ('old','screen','{}','running',0,0,NULL)")
    assert queue.recover_abandoned(tmp_path) == {'failed': [], 'unverified': ['old']}
    assert queue.counts(tmp_path) == {'running': 1}
    queue.add(tmp_path, 'screen', jobs())
    assert queue.claim(tmp_path, 1)['id'] == 'screen/0/0'


def test_campaign_refuses_unverifiable_claims_before_polling_services(tmp_path, monkeypatch):
    from particle_sliders.textsliders.reward_search import campaign
    monkeypatch.setattr(campaign.queue, 'recover_abandoned', lambda *a: {'failed': [], 'unverified': ['old']})
    monkeypatch.setattr(campaign.subprocess, 'check_output', lambda *a, **k: pytest.fail('must not poll services'))
    with pytest.raises(RuntimeError, match='explicit reconciliation'):
        campaign.wait_stage(tmp_path, 'screen')
