import json

import pytest

pytest.importorskip('nanoodle')

from particle_sliders_vibe import collect
from particle_sliders_vibe.nanoodle import NanoodleWorkflows


def score_graph():
    return {'nodes': [
        {'id': 'criterion', 'name': 'Criterion', 'type': 'text', 'fields': {'text': ''}},
        {'id': 'image', 'name': 'Image', 'type': 'upload', 'fields': {}},
        {'id': 'judge', 'name': 'Score', 'type': 'decide', 'fields': {
            'mode': 'score', 'model': 'typesafe/jev-1.14', 'question': 'How visible is the requested effect?',
            'levels': 'Absent\nWeak\nClear\nDominant'}}],
        'links': [
            {'id': 'text', 'from': {'node': 'criterion', 'port': 'text'}, 'to': {'node': 'judge', 'port': 'text'}},
            {'id': 'image', 'from': {'node': 'image', 'port': 'image'}, 'to': {'node': 'judge', 'port': 'img1'}}]}


def workflows(monkeypatch, answer):
    from nanoodle.transport import HttpResponse
    monkeypatch.setenv('NANOGPT_API_KEY', 'fixture-not-a-real-key')
    monkeypatch.setattr('nanoodle.transport.default_http', lambda *a, **kw:
        HttpResponse(200, {}, json.dumps({'answers': {'answer': answer}}).encode()))
    return NanoodleWorkflows({'nodes': [], 'links': []}, score_graph(), judge_provider='nanogpt')


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


def test_additional_decide_node_is_refused_before_execution():
    graph = score_graph()
    graph['nodes'].append(dict(graph['nodes'][-1], id='other', name='Other'))
    with pytest.raises(ValueError, match='exactly one Decide node'):
        NanoodleWorkflows({'nodes': [], 'links': []}, graph)


@pytest.mark.parametrize('provider', ['nanogpt', 'typesafe'])
def test_collection_keeps_unbundled_judgments_and_confidence_cannot_break_ties(monkeypatch, tmp_path, provider):
    from nanoodle.transport import HttpResponse
    from PIL import Image

    monkeypatch.setenv('NANOGPT_API_KEY', 'fixture-not-a-real-key')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'fixture-not-a-real-key')
    rubric = score_graph()
    rubric['nodes'][-1]['fields']['question'] = 'Score final-boss presence. Armor alone does not qualify.'
    workflows = NanoodleWorkflows({'nodes': [], 'links': []}, rubric, judge_provider=provider)
    calls = []
    scores = iter([0, 3, 0, 0, 0, 1.5, 0, 1.5])

    def http(method, url, *, headers, body, timeout):
        payload = json.loads(body)
        assert method == 'POST'
        assert payload['questions'] == {'answer': {
            'type': 'score', 'instructions': 'Score final-boss presence. Armor alone does not qualify.',
            'criteria': ['Absent', 'Weak', 'Clear', 'Dominant'],
        }}
        assert 'final boss' in payload['state']
        calls.append(payload)
        score = next(scores)
        probabilities = {str(i): {0: [1, 0, 0, 0], 1.5: [0, .5, .5, 0], 3: [0, 0, 0, 1]}[score][i]
                         for i in range(4)}
        answer = {'type': 'score', 'score': score, 'probabilities': probabilities,
                  'confidence': .1 if len(calls) <= 4 else .99}
        return HttpResponse(200, {}, json.dumps({'answers': {'answer': answer}}).encode())

    class ImageFixture:
        identity = {'judge': workflows.identity}
        score = staticmethod(workflows.score)

        def propose(self, vibe, count, history, *, request_id):
            return [{'neutral': 'castle', 'positive': 'boss in a castle'},
                    {'neutral': 'forest', 'positive': 'boss in a forest'}]

        def render(self, prompt, seed, directory, *, request_id):
            directory.mkdir(parents=True)
            path = directory / 'image.png'
            Image.new('RGB', (16, 16)).save(path)
            return {'path': str(path), 'prompt': prompt, 'seed': seed}

    monkeypatch.setattr('nanoodle.transport.default_http', http)
    state = collect(ImageFixture(), vibe='final boss', directory=tmp_path, pool_size=2, replace=1, best_of=2)
    first, second = state['rounds'][0]['ranking']
    assert len(calls) == 8
    assert first['prompt']['neutral'] == 'castle'
    assert first['fitness'] == second['fitness'] == .5
    assert [row['gain'] for row in first['cases']] == [1, 0]
    assert [row['gain'] for row in second['cases']] == [.5, .5]
    assert [row['seed'] for row in state['selected']] == [7, 7]


@pytest.mark.parametrize('change, error', [
    ('bundle', 'single typed score question'), ('reverse', 'frozen ordered rubric'),
])
def test_transport_refuses_changed_question_before_sending(monkeypatch, change, error):
    adapter = NanoodleWorkflows({'nodes': [], 'links': []}, score_graph())
    payload = {'questions': {'answer': {'type': 'score', 'instructions': adapter.question,
                                         'criteria': adapter.anchors}}}
    if change == 'bundle':
        payload['questions']['second'] = payload['questions']['answer']
    else:
        payload['questions']['answer']['criteria'] = list(reversed(adapter.anchors))

    def unexpected_http(*args, **kwargs):
        pytest.fail('invalid request reached the provider')

    monkeypatch.setattr('nanoodle.transport.default_http', unexpected_http)
    with pytest.raises(ValueError, match=error):
        adapter._nanogpt_score_http('POST', 'https://nano-gpt.com/api/v1/decisions', body=json.dumps(payload))
