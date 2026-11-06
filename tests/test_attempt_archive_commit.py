import importlib

import pytest

from particle_sliders.textsliders.reward_game import attempt_commit
from particle_sliders.textsliders.reward_game.core import IntegrityError, read, write


@pytest.mark.parametrize('prefix', ['', 'ff_', 'joint_', 'acoustic_', 'robust_block_', 'block_'])
@pytest.mark.parametrize('legacy_decision', [False, True])
def test_archive_failure_remains_pending_without_repeating_evaluation(tmp_path, monkeypatch, prefix, legacy_decision):
    controller = importlib.import_module('particle_sliders.textsliders.reward_game.' + prefix + 'controller')
    artifacts = importlib.import_module('particle_sliders.textsliders.reward_game.' + (prefix + 'artifacts' if prefix else 'artifacts'))
    recipe = dict(hypothesis='bounded', failure_mechanism='fault', method='checkpoint', parent_checkpoint=None,
                  changed_variables={}, budget={'new_clips': 4}, sources={}, checkpoint='nonexistent-fixture')
    source = tmp_path / 'recipe.json'
    write(source, recipe)
    ident = controller.register(tmp_path, source)['attempt_id']
    folder = tmp_path / 'attempts' / ident
    evaluations, archives = [], []
    card = dict(decision='rejected', advance=False, stage=4, run_id='fixed', render_keys=[], new_clips=0)
    def evaluate(*args):
        evaluations.append(True)
        return card
    def archive(*args):
        archives.append(True)
        if len(archives) == 1:
            raise OSError('archive copy failed')
    monkeypatch.setattr(artifacts, 'archive', archive)
    if legacy_decision:
        write(folder / 'scorecard.json', card)
        write(folder / 'decision.json', dict(advance=False, decision='rejected', run_id='fixed', next_action='try_next_registered_attempt'))
    assert controller.search(tmp_path, evaluator=evaluate)['state'] == 'engineering_failure'
    assert not (folder / 'archive-complete.json').exists()
    assert (folder / 'decision.json').exists() == legacy_decision
    assert controller.search(tmp_path, evaluator=evaluate)['state'] == 'method_selection_required'
    assert len(evaluations) == (0 if legacy_decision else 1)
    assert len(archives) == 2
    assert attempt_commit.decision_committed(folder)
    controller.search(tmp_path, evaluator=evaluate)
    assert len(archives) == 2


def test_selection_archive_failure_never_commits_decision(tmp_path):
    decision = {'decision': 'rejected_training_selection'}
    def fail():
        raise OSError('selection archive')
    with pytest.raises(OSError, match='selection archive'):
        attempt_commit.commit_decision(tmp_path, decision, fail)
    assert not (tmp_path / 'decision.json').exists()
    attempt_commit.commit_decision(tmp_path, decision, lambda: None)
    assert attempt_commit.decision_committed(tmp_path)
    write(tmp_path / 'archive-complete.json', {'decision_sha256': 'changed'})
    with pytest.raises(IntegrityError, match='differs'):
        attempt_commit.decision_committed(tmp_path)
