import io
import json

import pytest

from slider_selection import model, server
from slider_selection.features import file_hash


def make_submission(tmp_path, monkeypatch, *, fail_score=False):
    public = tmp_path / 'public'
    public.mkdir()
    (public / 'session.json').write_text(json.dumps({'trials': [
        {'id': 't', 'kind': 'ladder', 'original': 'o', 'settings': [{'url': 's'}]},
    ]}))
    (tmp_path / 'session-key.json').write_text(json.dumps({
        'trials': {'t': {}}, 'public_sha256': file_hash(public / 'session.json'),
    }))
    features = tmp_path / 'features.json'
    monkeypatch.setattr(server, 'ThreadingHTTPServer', lambda address, handler: handler)
    monkeypatch.setattr(server, 'validate_response', lambda *args: None)
    monkeypatch.setattr(server, 'read_labels', lambda *args: ({'a': 1}, {}))
    calls = []
    monkeypatch.setattr(model, 'fit', lambda *args: calls.append('fit') or {'fitted': True})

    def score(*args):
        calls.append('score')
        if fail_score and calls.count('score') == 1:
            raise ValueError('temporary scoring failure')
        return {'score': 1}

    monkeypatch.setattr(model, 'score', score)
    handler_type = server.make_server(tmp_path, '127.0.0.1', 0, features)

    def submit():
        body = json.dumps({'trial_id': 't', 'played': ['o', 's']}).encode()
        handler = handler_type.__new__(handler_type)
        handler.path = '/api/response'
        handler.headers = {'Content-Length': str(len(body))}
        handler.rfile = io.BytesIO(body)
        replies = []
        handler.respond = lambda status, payload: replies.append((status, payload))
        handler.do_POST()
        assert replies == [(200, {'saved': True})]
        return json.loads((tmp_path / 'calibration-status.json').read_text())

    return features, calls, submit


def test_identical_answers_retry_after_missing_features_are_repaired(tmp_path, monkeypatch):
    features, calls, submit = make_submission(tmp_path, monkeypatch)
    assert submit()['state'] == 'needs_more_evidence'
    assert calls == []
    features.write_text('{}')
    assert submit()['state'] == 'exploratory_fit_complete'
    assert calls == ['fit', 'score']
    submit()
    assert calls == ['fit', 'score']  # Successful identical fit is still deduplicated.


def test_partial_model_publication_does_not_suppress_retry(tmp_path, monkeypatch):
    features, calls, submit = make_submission(tmp_path, monkeypatch, fail_score=True)
    features.write_text('{}')
    assert submit()['state'] == 'needs_more_evidence'
    assert (tmp_path / 'model.json').exists()
    assert not (tmp_path / 'score-preview.json').exists()
    assert submit()['state'] == 'exploratory_fit_complete'
    assert calls == ['fit', 'score', 'fit', 'score']
    submit()
    assert calls == ['fit', 'score', 'fit', 'score']
