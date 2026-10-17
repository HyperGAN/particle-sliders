import json
from pathlib import Path

import pytest
import torch

from diagnose_moss_runtime import error_metrics, tensor_sha, validate_card, RUNTIMES, sha, REPO


def test_raw_metrics_detect_small_edit_relative_error_and_nonfinite():
    base = torch.ones(1, 2, 2, 2)
    edit = base * .01
    got = error_metrics(base + .005, base, edit)
    assert got['base_relative_rms'] == pytest.approx(.005, rel=1e-5)
    assert got['edit_relative_rms'] == pytest.approx(.5, rel=1e-5)
    assert not got['exact']
    with pytest.raises(FloatingPointError):
        error_metrics(base * float('nan'), base, edit)


def test_tensor_identity_binds_dtype_shape_and_values():
    value = torch.ones(1, 8, dtype=torch.bfloat16)
    assert tensor_sha(value) != tensor_sha(value.float())
    assert tensor_sha(value) != tensor_sha(value.reshape(2, 4))
    assert tensor_sha(value) == tensor_sha(value.clone())


def test_diagnostic_card_scope_and_source_binding(tmp_path, monkeypatch):
    # Test this card gate independently of historical completed-run recipe pins.
    from conceptmod.textsliders.pg09_moss_worker import BoundRecipe
    monkeypatch.setattr(BoundRecipe, "verify_sources", lambda self: None)
    recipe = REPO/'qualification/moss-reruns/bad-intent.json'
    script = REPO/'scripts/diagnose_moss_runtime.py'
    card = dict(schema='moss-readonly-runtime-diagnostic-v1', runtimes=RUNTIMES,
        indices=[0,39], optimizer_updates=0, wall_seconds=900,
        admission='REVIEWED_READONLY_DIAGNOSTIC', source_pins={str(script):sha(script)},
        training_recipe_file_sha256=sha(recipe),
        output_directory='/mnt/nvme/sliders/qualification/runs/moss-runtime-diagnostic-test')
    path = tmp_path/'card.json'
    path.write_text(json.dumps(card))
    assert validate_card(path)[0] == card
    for field, value in [('optimizer_updates', 1), ('indices', [0]),
                         ('runtimes', dict(original='/tmp/python')), ('wall_seconds', 901),
                         ('source_pins', {str(script): 'a'*64})]:
        wrong = dict(card, **{field:value})
        path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError):
            validate_card(path)
