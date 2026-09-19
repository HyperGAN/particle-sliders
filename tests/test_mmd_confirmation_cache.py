import json

import pytest

from analysis.gan_bcap.mmd_game_20260905.confirmation_cache import reuse_confirmation
from conceptmod.textsliders.gan_v2.data import sha


def fixture(tmp_path):
    source = tmp_path/'source'
    (source/'row-2').mkdir(parents=True)
    renderer = tmp_path/'renderer.py'
    renderer.write_text('frozen renderer')
    weight = tmp_path/'reference.safetensors'
    weight.write_bytes(b'reference weights')
    entry = dict(weights=str(weight), weights_sha256=sha(weight))
    plan = dict(prompts_sha256='prompts', seeds=[101, 303], duration=30., gpu=1, rows=[2, 3])
    records = []
    for seed in plan['seeds']:
        record = dict(row=2, seed=seed, checkpoint=dict(path=str(weight), sha256=sha(weight)))
        for key, filename in [('baseline', '01_off.wav'), ('positive_reference', '03_positive.wav'),
                              ('candidate', '02_slider.wav')]:
            path = source/f'{seed}-{filename}'
            path.write_bytes(f'{key}-{seed}'.encode())
            record[key] = dict(audio=str(path), sha256=sha(path))
        records.append(record)
    (source/'summary.json').write_text(json.dumps(dict(status='complete', plan=plan, records=records)))
    spec = dict(plan, row=2, scales=[0., 1.], seed_retries=0, renderer_sha256=sha(renderer),
                checkpoints=[dict(path=str(weight), sha256=sha(weight))])
    (source/'row-2/render_spec.json').write_text(json.dumps(spec))
    return source, plan, entry, renderer


def test_matching_reference_and_controls_reuse_exact_bytes(tmp_path):
    source, plan, entry, renderer = fixture(tmp_path)
    copies = reuse_confirmation(source, tmp_path/'destination', plan, [entry], 2, renderer)
    assert len(copies) == 6
    assert all(sha(row['destination']) == row['sha256'] for row in copies)
    assert sum(row['kind'] == 'identical_checkpoint' for row in copies) == 2


def test_changed_fixture_or_audio_is_rejected(tmp_path):
    source, plan, entry, renderer = fixture(tmp_path)
    with pytest.raises(ValueError, match='duration'):
        reuse_confirmation(source, tmp_path/'destination', dict(plan, duration=20.), [entry], 2, renderer)
    assert not (tmp_path/'destination').exists()
    (source/'101-01_off.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='audio changed'):
        reuse_confirmation(source, tmp_path/'destination', plan, [entry], 2, renderer)
