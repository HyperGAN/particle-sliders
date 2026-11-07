"""Real common-lifecycle/PG CPU bridge conformance; no pretrained model evidence."""
from dataclasses import replace

import pytest
import torch

from particle_sliders.lifecycle import read_packet
from particle_sliders.parameter_search import LifecycleTrial, score_chunks
from particle_sliders.pg09 import _hash
from test_image_lifecycle import session


class Observer:
    def __init__(self, active):
        state = active.game.state_dict()
        self.initial_state_sha256 = _hash({k: state[k] for k in
            ('network', 'critic', 'prior', 'streams', 'model_cpu_rng', 'model_cuda_rng')})

    def native_state(self):
        return {'external_cache': 'no mutable external training state in this CPU fixture'}

    def effective_parameters(self, active):
        recipe = active.game.recipe
        result = {'lr.generator': recipe.lr, 'lr.critic': recipe.lr * recipe.d_lr_mult}
        if active.game.prior is not None:
            result['lr.particle'] = recipe.lr * recipe.prior_lr_mult
        return result

    def optimizer_groups(self, active):
        result = []
        prior = active.game.prior
        for phase, optimizer in enumerate((active.game.opt_g, active.game.opt_d)):
            for index, group in enumerate(optimizer.param_groups):
                is_prior = prior is not None and any(p is prior.z for p in group['params'])
                role = 'critic_lr' if phase else 'particle_lr' if is_prior else 'generator_lr'
                result.append(dict(role=role, base_lr=active.game.base_lrs[phase][index],
                    lr=group['lr'], parameter_count=sum(p.numel() for p in group['params'])))
        return result

    def frozen_identity(self, active):
        return {'recipe_horizon': active.game.recipe.total_steps, 'target': 'native prepared target'}

    def chunks(self, active, split, selection, *, off=False):
        # Exercise the existing raw observation transaction and scaler.
        with active._observation(9, {'split': split}):
            for index, prepared in enumerate(active.game.rows):
                row = replace(prepared, row=replace(prepared.row, scale=0. if off else 1.))
                student = active.game.backend.generated(row).detach().cpu().reshape(-1)
                target, neutral = prepared.target.cpu().reshape(-1), prepared.neutral.cpu().reshape(-1)
                for offset in range(0, student.numel(), 64):
                    yield dict(example=dict(scene='cpu', seed=9, path=str(index)),
                        position={'surface': 'image', 'timestep': 500}, shape=list(prepared.target.shape), offset=offset,
                        student=student[offset:offset + 64].tolist(), target=target[offset:offset + 64].tolist(),
                        neutral=neutral[offset:offset + 64].tolist())


@pytest.mark.parametrize('particles', [False, True])
def test_actual_lifecycle_bridge_preserves_raw_evaluation_export_and_replay(tmp_path, particles):
    active = session(particles=particles, render=False)
    observer = Observer(active)
    wrapped = LifecycleTrial(active, observer)
    initial = wrapped.initial_state_sha256
    grid = [dict(example=dict(scene='cpu', seed=9, path=str(index)), positions=[
        dict(position={'surface': 'image', 'timestep': 500}, shape=list(row.target.shape))])
        for index, row in enumerate(active.game.rows)]
    wrapped.checkpoint(tmp_path / 'initial.pt')
    wrapped.step()
    after = wrapped.state_digest()
    rng = torch.get_rng_state().clone()
    for selection in ('raw',):
        result = score_chunks(wrapped.responses('dev', selection), grid)
        off = score_chunks(wrapped.responses('dev', selection, off=True), grid)
        assert off['error'] == pytest.approx(1.) and off['signed_edit_gain'] == pytest.approx(0.)
        assert result['examples'][0]['elements'] == 200
        wrapped.export(tmp_path / (selection + '.pt'), selection)
        assert wrapped.state_digest() == after
        assert torch.equal(torch.get_rng_state(), rng)
    wrapped.close()
    resumed = session(particles=particles, render=False)
    replay = LifecycleTrial(resumed, Observer(resumed))
    assert replay.initial_state_sha256 == initial
    resumed.restore(read_packet(tmp_path / 'initial.pt'))
    replay.step()
    assert replay.state_digest() == after
    assert {v['role'] for v in replay.optimizer_groups()} == (
        {'generator_lr', 'critic_lr', 'particle_lr'} if particles else {'generator_lr', 'critic_lr'})
    replay.close()
    assert not torch.cuda.is_initialized()
