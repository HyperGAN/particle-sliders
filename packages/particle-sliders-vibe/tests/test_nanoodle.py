import json

import pytest

pytest.importorskip('nanoodle')

from particle_sliders_vibe.nanoodle import NanoodleWorkflows


def workflows(monkeypatch, answer):
    from nanoodle.transport import HttpResponse
    monkeypatch.setenv('NANOGPT_API_KEY', 'fixture-not-a-real-key')
    monkeypatch.setattr('nanoodle.transport.default_http', lambda *a, **kw:
        HttpResponse(200, {}, json.dumps({'answers': {'answer': answer}}).encode()))
    prompts = {'nodes': [], 'links': []}
    scores = {'nodes': [
        {'id': 'criterion', 'name': 'Criterion', 'type': 'text', 'fields': {'text': ''}},
        {'id': 'image', 'name': 'Image', 'type': 'upload', 'fields': {}},
        {'id': 'judge', 'name': 'Score', 'type': 'decide', 'fields': {
            'mode': 'score', 'model': 'typesafe/jev-1.14', 'question': 'How visible is the requested effect?',
            'levels': 'Absent\nWeak\nClear\nDominant'}}],
        'links': [
            {'id': 'text', 'from': {'node': 'criterion', 'port': 'text'}, 'to': {'node': 'judge', 'port': 'text'}},
            {'id': 'image', 'from': {'node': 'image', 'port': 'image'}, 'to': {'node': 'judge', 'port': 'img1'}}]}
    return NanoodleWorkflows(prompts, scores, judge_provider='nanogpt')


def image_path(tmp_path):
    from PIL import Image
    path = tmp_path / 'image.png'
    Image.new('RGB', (16, 16)).save(path)
    return {'path': str(path)}


def test_live_jev_independently_rounded_score_preserves_reported_measurement(monkeypatch, tmp_path):
    # Actual NanoGPT Jev 1.14 response: rounded rows imply 2.63, reported score is 2.62.
    answer = {'type': 'score', 'score': 2.62,
              'probabilities': {'0': 0, '1': .03, '2': .31, '3': .66}}
    result = workflows(monkeypatch, answer).score('final boss', image_path(tmp_path), request_id='rounded')
    assert result['score'] == pytest.approx(2.62 / 3)
    assert result['raw']['answers']['answer'] == answer
    assert result['score_validation']['expected_score'] == pytest.approx(2.63)


@pytest.mark.parametrize('answer', [
    {'type': 'score', 'score': 2.62, 'probabilities': {'0': 0, '1': .03, '2': .31, '3': .60}},
    {'type': 'score', 'score': 2, 'probabilities': {'0': 0, '1': .03, '2': .31, '3': .66}},
    {'type': 'score', 'score': True, 'probabilities': {'0': 0, '1': .03, '2': .31, '3': .66}},
    {'type': 'score', 'score': 2.62, 'probabilities': {'0': False, '1': .03, '2': .31, '3': .66}},
])
def test_invalid_measurements_still_fail(monkeypatch, tmp_path, answer):
    with pytest.raises(ValueError, match='no valid measured score'):
        workflows(monkeypatch, answer).score('final boss', image_path(tmp_path), request_id='invalid')
