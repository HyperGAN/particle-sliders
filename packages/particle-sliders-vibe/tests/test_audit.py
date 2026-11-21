import pytest

from particle_sliders_vibe import collect
from particle_sliders_vibe.audit import audit_collection


class Product:
    identity = {'fixture': 'matched finished-output scores'}

    def propose(self, vibe, count, history, *, request_id):
        return [{'neutral': 'terrain', 'positive': 'canyon'},
                {'neutral': 'fish', 'positive': 'shark'}]

    def render(self, prompt, seed, directory, *, request_id):
        directory.mkdir(parents=True)
        output = directory / 'finished.txt'
        output.write_text(prompt)
        return {'path': str(output), 'prompt': prompt, 'seed': seed}

    def score(self, vibe, output, *, request_id):
        score = {'terrain': 0, 'canyon': int(output['seed'] == 7), 'fish': 0, 'shark': .5}[output['prompt']]
        confidence = .99 if output['prompt'] in {'fish', 'shark'} else .01
        return {'score': score, 'raw': {'answers': {'answer': {'confidence': confidence}}}}


def test_audit_compares_full_and_selected_receipts_without_reranking(tmp_path):
    state = collect(Product(), vibe='rugged', directory=tmp_path, pool_size=2, replace=1, best_of=2)
    audit = audit_collection(state, verify_media=True)
    assert audit['all_matched_cases'] == {'pairs': 4, 'mean_gain': .5}
    assert audit['best_of_n_contexts'] == {'pairs': 2, 'mean_gain': .75}
    assert audit['rounds'] == [{'round': 0, 'candidates': 2, 'top_effect_gain': .5, 'tied_top_candidates': 2}]
    assert audit['media_files_checked'] == 8


@pytest.mark.parametrize('change, error', [
    ('score', 'saved case gain differs'), ('seed', 'frozen matched seeds'), ('selection', 'selected contexts differ'),
])
def test_audit_refuses_inconsistent_evidence(tmp_path, change, error):
    state = collect(Product(), vibe='rugged', directory=tmp_path, pool_size=2, replace=1, best_of=2)
    case = state['rounds'][0]['ranking'][0]['cases'][0]
    if change == 'score':
        case['positive']['judgment']['score'] = 0
    elif change == 'seed':
        case['seed'] += 1
    else:
        state['selected'][0] = {}
    with pytest.raises(ValueError, match=error):
        audit_collection(state)


@pytest.mark.parametrize('field, error', [
    ('round', 'frozen Vibe cohort'), ('seed', 'frozen matched seeds'),
    ('gain', 'saved case gain differs'), ('best', 'best-of-N index differs'),
    ('selected-round', 'selected contexts differ'),
])
def test_audit_refuses_boolean_impostors(tmp_path, field, error):
    state = collect(Product(), vibe='rugged', directory=tmp_path, pool_size=2, replace=1, best_of=2, seed=1)
    round_ = state['rounds'][0]
    row = round_['ranking'][0]
    case = row['cases'][0]
    if field == 'round':
        round_['index'] = False
    elif field == 'seed':
        case['seed'] = True
    elif field == 'gain':
        round_['ranking'][1]['cases'][0]['gain'] = False
    elif field == 'best':
        row['best'] = False
    else:
        state['selected'][0]['round'] = False
    with pytest.raises(ValueError, match=error):
        audit_collection(state)
