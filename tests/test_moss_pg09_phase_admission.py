"""No models/processes: reject drafts and incomplete actual-pretrained gates."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('moss_guard', Path(__file__).parents[1] / 'scripts/guarded_moss_pg09.py')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def passed_probe(recipe=None):
    recipe = recipe or dict(admission='REVIEWED_GAN_RECIPE')
    return dict(status='probe_passed', latest_step=3, frozen_base_unchanged=True,
                recipe_sha256=hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                generator_changed=True, critic_changed=True, exact_adapter_reload=True,
                exact_resume=dict(exact_loss=True, exact_full_state=True, exact_sample=True))


def test_draft_or_previous_probe_cannot_dispatch():
    assert guard.phase_reason({'admission': 'DRAFT_NOT_ADMITTED'}, 'probe', {})
    assert guard.phase_reason({'admission': 'REVIEWED_GAN_RECIPE'}, 'probe', {'status': 'failed'})


def test_full_training_requires_every_real_probe_gate():
    recipe = dict(admission='REVIEWED_GAN_RECIPE')
    assert guard.phase_reason(recipe, 'train', {})
    good = passed_probe()
    assert guard.phase_reason(recipe, 'train', good) is None
    for key in good:
        bad = dict(good)
        bad.pop(key)
        assert guard.phase_reason(recipe, 'train', bad), key


def test_evaluation_requires_completed_full_budget():
    recipe = dict(admission='REVIEWED_GAN_RECIPE')
    assert guard.phase_reason(recipe, 'evaluate', passed_probe())
    progress = passed_probe(recipe)
    progress.update(status='trained', latest_step=3840)
    assert guard.phase_reason(recipe, 'evaluate', progress) is None


def test_prior_phase_identity_cannot_be_copied_to_different_recipe():
    old = dict(admission='REVIEWED_GAN_RECIPE', case='old')
    new = dict(admission='REVIEWED_GAN_RECIPE', case='new')
    assert guard.phase_reason(new, 'train', passed_probe(old))
    progress = passed_probe(old)
    progress.update(status='trained', latest_step=3840)
    assert guard.phase_reason(new, 'evaluate', progress)


def test_checkpoint_pointer_and_base_must_match_qualified_probe(tmp_path):
    from types import SimpleNamespace
    recipe = SimpleNamespace(identity='a' * 64)
    metadata = dict(step=3, frozen_base_sha256='b' * 64)
    contract = SimpleNamespace(verify_checkpoint=lambda directory, bound: metadata)
    progress = dict(frozen_base_sha256='b' * 64, frozen_base_after_sha256='b' * 64)
    pointer = tmp_path / 'LATEST.json'
    valid = dict(recipe_sha256=recipe.identity, step=3, directory='step-000003')
    pointer.write_text(json.dumps(valid))
    guard.verify_prior_checkpoint(contract, recipe, 'train', progress, tmp_path)
    for key, wrong in [('recipe_sha256', 'bad'), ('step', 80), ('directory', '../escape')]:
        changed = dict(valid)
        changed[key] = wrong
        pointer.write_text(json.dumps(changed))
        import pytest
        with pytest.raises(ValueError):
            guard.verify_prior_checkpoint(contract, recipe, 'train', progress, tmp_path)
    pointer.write_text(json.dumps(valid))
    metadata['frozen_base_sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        guard.verify_prior_checkpoint(contract, recipe, 'train', progress, tmp_path)
