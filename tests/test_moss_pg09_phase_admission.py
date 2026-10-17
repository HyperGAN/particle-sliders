"""No models/processes: reject drafts and incomplete actual-pretrained gates."""
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('moss_guard', Path(__file__).parents[1] / 'scripts/guarded_moss_pg09.py')
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def passed_probe():
    return dict(status='probe_passed', latest_step=3, frozen_base_unchanged=True,
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
    assert guard.phase_reason(recipe, 'evaluate', dict(status='trained', latest_step=3840, frozen_base_unchanged=True)) is None
