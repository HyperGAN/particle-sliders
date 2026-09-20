from collections import Counter
import json
from pathlib import Path

import pytest

from lumen_studio.candidates import compile_candidate
from lumen_studio.contracts import Shared, VARIATIONS, file_hash
from lumen_studio.dataset import DATA, compile_manifest


@pytest.mark.parametrize('candidate', ['neon-v1', 'dusk-v1', 'dusk-v2'])
def test_draft_pairs_preserve_shared_fields_and_isolate_training_inputs(candidate):
    source = DATA / f'candidates/{candidate}.json'
    original = file_hash(DATA / 'definitions.json')
    before = {s: compile_manifest(s) for s in ('train', 'dev')}
    bundle = compile_candidate(source)
    assert not bundle['registered_slider'] and candidate not in VARIATIONS
    assert file_hash(DATA / 'definitions.json') == original
    for split in ('train', 'dev'):
        assert compile_manifest(split) == before[split]
        templates = [r for r in before[split]['rows'] if r['variation'] == 'candlelit']
        for row, template in zip(bundle[split]['rows'], templates, strict=True):
            assert row['shared'] == template['shared']
            shared = Shared(**row['shared']).prompt()
            assert row['neutral'] == shared + (', ' + row['neutral_lighting'] if row['neutral_lighting'] else '')
            assert row['positive'] == shared + ', ' + row['positive_lighting']
            assert row['seeds'] == template['seeds'] and row['character'] == template['character']
    train, dev = bundle['train']['rows'], bundle['dev']['rows']
    assert set(Counter(r['definition'] for r in train).values()) == {4}
    assert set(Counter(r['character'] for r in train).values()) == {2}
    assert not {r['character'] for r in train} & {r['character'] for r in dev}
    assert len([r for r in dev if r['bare']]) == 4
    assert 'test' not in bundle


def test_recorded_neutral_clause_survives_compilation_except_bare_prompts(tmp_path):
    spec = json.loads((DATA / 'candidates/neon-v1.json').read_text())
    for d in spec['definitions']:
        d['neutral_lighting'] = 'neutral diffuse illumination'
    path = tmp_path / 'draft.json'
    path.write_text(json.dumps(spec))
    bundle = compile_candidate(path)
    for split in ('train', 'dev'):
        for row in bundle[split]['rows']:
            assert row['neutral_lighting'] == ('' if row['bare'] else 'neutral diffuse illumination')
            if not row['bare']:
                assert row['neutral'].endswith(', neutral diffuse illumination')
    spec['definitions'].pop()
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='six training definitions'):
        compile_candidate(path)
