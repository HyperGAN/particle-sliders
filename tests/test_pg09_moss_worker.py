import os

import pytest
import torch

from conceptmod.textsliders.pg09_moss_worker import (
    BoundRecipe, REVISION, SNAPSHOT, OLD_ROOT, SCALES, inherited_leases,
    digest, parameter_sha, save_checkpoint, restore_latest, probe_replay,
    no_optimized_mse, GradientWitness,
)


def recipe_values():
    return dict(schema='moss-pg09-worker-v1', concept='bad-intent', objective='pg09_gan_only',
                model_revision=REVISION, model_snapshot=str(SNAPSHOT),
                cache_directory=str(OLD_ROOT / 'bad-intent'),
                old_mse_adapter=str(OLD_ROOT / 'bad-intent/checkpoints/step-001600/adapter'),
                output_directory='/mnt/nvme/sliders/qualification/runs/new-gan-only-worker',
                pg_version='0.9.0', preset='bispf_optimized', total_steps=3840, batch_size=8,
                training_seed=1000, checkpoint_every=80, wall_seconds=604800,
                signed_rows=15360, response_domain='continuous_field', sampling='balanced',
                response_scale=SCALES['bad-intent'], cache_identity_sha256='a' * 64,
                resolved_recipe_sha256='b' * 64, old_adapter_sha256='c' * 64,
                old_manifest_sha256='d' * 64,
                resource_guard=dict(path='/reviewed/music3_resources.py', sha256='e' * 64),
                source_pins={f'/reviewed/{n}': 'f' * 64 for n in
                             ('pg09_moss_worker.py', 'pg09_moss_campaign.py', 'pg09_backends.py',
                              'pg09.py', 'pipeline_moss_image.py', 'lora.py')},
                lease_paths=['/reviewed/global.lock', '/reviewed/gpu0.lock'])


def test_bound_recipe_strict_fixed_objective_budget_paths_and_source_coverage():
    values = recipe_values()
    bound = BoundRecipe.validate(values)
    assert bound.identity == digest(values)
    values['objective'] = 'MSE'
    assert bound.data['objective'] == 'pg09_gan_only'
    for key, value in [('objective', 'MSE'), ('total_steps', 3), ('batch_size', 2),
                       ('checkpoint_every', 1), ('wall_seconds', 1), ('signed_rows', 7680),
                       ('sampling', 'replacement'), ('model_revision', 'wrong'),
                       ('model_snapshot', '/tmp/remote'), ('response_scale', 1.),
                       ('output_directory', str(OLD_ROOT / 'bad-intent/new-output'))]:
        wrong = recipe_values()
        wrong[key] = value
        with pytest.raises(ValueError):
            BoundRecipe.validate(wrong)
    missing = recipe_values()
    missing['source_pins'].pop('/reviewed/lora.py')
    with pytest.raises(ValueError, match='source pins'):
        BoundRecipe.validate(missing)
    with pytest.raises(ValueError, match='output'):
        bound.verify_output('/tmp/not-reviewed')


def test_inherited_lease_validation_never_takes_or_releases_locks(tmp_path):
    paths = [tmp_path / 'global.lock', tmp_path / 'gpu.lock']
    streams = [p.open('a+') for p in paths]
    try:
        fds = [s.fileno() for s in streams]
        assert inherited_leases(fds, paths) == [str(p) for p in paths]
        with pytest.raises(ValueError):
            inherited_leases([fds[0]], paths)
        with pytest.raises(ValueError):
            inherited_leases(fds, [tmp_path / 'wrong', paths[1]])
        assert all(os.fstat(fd).st_size == 0 for fd in fds)
    finally:
        for stream in streams:
            stream.close()


def test_frozen_parameter_stream_hash_dtype_and_no_state_clone():
    model = torch.nn.Linear(4, 3).to(dtype=torch.bfloat16).requires_grad_(False)
    before = parameter_sha(model, frozen_only=True)
    assert before == parameter_sha(model, frozen_only=True)
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert before != parameter_sha(model, frozen_only=True)
    model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match='trainable'):
        parameter_sha(model, frozen_only=True)


def test_gradient_witness_and_mse_function_ban_are_diagnostic_only():
    model = torch.nn.Linear(3, 2)
    witness = GradientWitness(model)
    model(torch.ones(1, 3)).sum().backward()
    assert witness.norm() > 0
    witness.reset()
    assert witness.norm() == 0
    witness.close()
    original = torch.nn.functional.mse_loss
    with no_optimized_mse():
        with pytest.raises(RuntimeError, match='MSE'):
            torch.nn.functional.mse_loss(torch.zeros(1), torch.ones(1))
        assert float((torch.ones(1) - torch.zeros(1)).square().mean()) == 1
    assert torch.nn.functional.mse_loss is original


def test_worker_checkpoint_probe_replay_export_and_no_mse_resume(tmp_path):
    from test_pg09_moss_campaign import make_cache, new_model
    from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows, build_campaign_run
    torch.set_num_threads(1)
    original, recipe = make_cache(tmp_path / 'cache')
    rows = MossCampaignRows(tmp_path / 'cache')
    run = build_campaign_run(new_model(original), rows,
                             model_identity=dict(model='tiny CPU MOSS', revision=recipe['model_revision']),
                             response_scale=rows.response_scale(), batch_size=4, total_steps=14,
                             checkpointing=True, critic_width=8)
    bound = BoundRecipe.validate(recipe_values())
    base_sha = parameter_sha(run.backend.transformer, frozen_only=True)
    with no_optimized_mse():
        run.step()
        run.step()
        before = run.game.state_dict()
        metrics = run.step()
        expected = run.game.state_dict()
        assert probe_replay(run, before, metrics, expected)['exact_full_state']
    destination = save_checkpoint(tmp_path / 'output', bound, run, base_sha)
    run.step()
    assert restore_latest(tmp_path / 'output', bound, run, base_sha) == destination
    assert run.game.completed_steps == 3
    from test_pg09_backends import equal
    equal(run.game.state_dict(), expected)
    export = tmp_path / 'worker-export.safetensors'
    run.export_adapter(export)
    expected_sample = run.game.evaluate(0)
    run.load_adapter(export)
    assert torch.equal(expected_sample, run.game.evaluate(0))
    run.game.load_state_dict(expected)
    checkpoint = destination / 'state.pt'
    checkpoint.write_bytes(b'MSE optimizer checkpoint')
    with pytest.raises(ValueError, match='MSE resume forbidden'):
        restore_latest(tmp_path / 'output', bound, run, base_sha)
    assert run.game.completed_steps == 3
    assert not torch.cuda.is_initialized()


def tiny_run(tmp_path):
    from test_pg09_moss_campaign import make_cache, new_model
    from conceptmod.textsliders.pg09_moss_campaign import MossCampaignRows, build_campaign_run
    original, recipe = make_cache(tmp_path / 'cache')
    rows = MossCampaignRows(tmp_path / 'cache')
    return build_campaign_run(new_model(original), rows,
                              model_identity=dict(model='tiny CPU MOSS', revision=recipe['model_revision']),
                              response_scale=rows.response_scale(), batch_size=4, total_steps=14,
                              checkpointing=True, critic_width=8)


def test_same_step_checkpoint_rejects_divergent_live_state_without_latest_write(tmp_path):
    from conceptmod.textsliders.pg09_moss_worker import tree_equal
    run = tiny_run(tmp_path)
    bound = BoundRecipe.validate(recipe_values())
    output = tmp_path / 'output'
    base = parameter_sha(run.backend.transformer, frozen_only=True)
    run.step()
    destination = save_checkpoint(output, bound, run, base)
    latest = (output / 'LATEST.json').read_bytes()
    saved = (destination / 'state.pt').read_bytes()
    assert save_checkpoint(output, bound, run, base) == destination
    with torch.no_grad():
        next(run.backend.network.parameters()).add_(1.)
    live = run.game.state_dict()
    with pytest.raises(ValueError, match='divergent same-step'):
        save_checkpoint(output, bound, run, base)
    assert (output / 'LATEST.json').read_bytes() == latest
    assert (destination / 'state.pt').read_bytes() == saved
    assert tree_equal(live, run.game.state_dict())
    with pytest.raises(ValueError, match='divergent same-step'):
        save_checkpoint(output, bound, run, 'changed-base')
    assert (output / 'LATEST.json').read_bytes() == latest


def test_latest_recovers_completed_checkpoint_directory_after_pointer_interruption(tmp_path):
    from conceptmod.textsliders.pg09_moss_worker import tree_equal
    run = tiny_run(tmp_path)
    bound = BoundRecipe.validate(recipe_values())
    output = tmp_path / 'output'
    base = parameter_sha(run.backend.transformer, frozen_only=True)
    run.step()
    save_checkpoint(output, bound, run, base)
    stale = (output / 'LATEST.json').read_bytes()
    run.step()
    destination = save_checkpoint(output, bound, run, base)
    expected = run.game.state_dict()
    for pointer_bytes in (stale, None):
        if pointer_bytes is None:
            (output / 'LATEST.json').unlink()
        else:
            (output / 'LATEST.json').write_bytes(pointer_bytes)
        run.step()
        assert restore_latest(output, bound, run, base) == destination
        assert tree_equal(expected, run.game.state_dict())
        assert 'step-000002' in (output / 'LATEST.json').read_text()


def test_teacher_parity_emits_actual_relative_rms_for_both_states_before_reject(tmp_path, monkeypatch):
    from dataclasses import replace
    from conceptmod.textsliders.pg09_moss_worker import parity_diagnostics, TeacherParityError
    run = tiny_run(tmp_path)
    original = run.backend.prepare
    def wrong(row):
        prepared = original(row)
        return replace(prepared, target=prepared.target + 1.)
    monkeypatch.setattr(run.backend, 'prepare', wrong)
    events = []
    with pytest.raises(TeacherParityError) as error:
        parity_diagnostics(run, indices=(0, 1), emit=lambda event, **fields: events.append((event, fields)))
    assert len(events) == 2
    assert all(e[0] == 'teacher_parity_state' for e in events)
    assert error.value.diagnostics == [e[1]['diagnostic'] for e in events]
    for diagnostic in error.value.diagnostics:
        assert diagnostic['relative_target_rms'] > diagnostic['threshold_relative_rms']
        assert diagnostic['relative_target_rms'] ** 2 == pytest.approx(diagnostic['relative_target_mse'])
        assert diagnostic['teacher_negative'] == 'neutral'
        assert diagnostic['cfg_negative'] == 'empty'
    assert run.game.completed_steps == 0


def test_actual_loaded_origins_are_pinned_not_just_arbitrary_listed_files(tmp_path):
    import importlib
    from types import SimpleNamespace
    from conceptmod.textsliders.pg09_moss_worker import sha_file
    values = recipe_values()
    names = ('conceptmod.textsliders.pg09_moss_worker', 'conceptmod.textsliders.pg09_moss_campaign',
             'conceptmod.textsliders.pg09_backends', 'particle_sliders.pg09',
             'moss_sliders.vendor.pipeline_moss_image', 'moss_sliders.lora')
    modules = [importlib.import_module(n) for n in names]
    values['source_pins'] = {str(__import__('pathlib').Path(m.__file__).resolve()): sha_file(m.__file__) for m in modules}
    guard = tmp_path / 'guard.py'
    guard.write_text('class ResourceGuard: pass\n')
    values['resource_guard'] = dict(path=str(guard), sha256=sha_file(guard))
    bound = BoundRecipe.validate(values)
    bound.verify_sources()
    assert len(bound.verify_loaded_sources()) == 5
    wrong = tmp_path / 'pg09_backends.py'
    wrong.write_text('# unrelated copy with a perfectly valid listed hash\n')
    copied = dict(values['source_pins'])
    actual = modules[2].__file__
    copied.pop(str(__import__('pathlib').Path(actual).resolve()))
    copied[str(wrong)] = sha_file(wrong)
    values['source_pins'] = copied
    bad = BoundRecipe.validate(values)
    bad.verify_sources()  # Every listed file really matches, yet Python loaded another copy.
    with pytest.raises(ValueError, match='actual loaded module source is not pinned'):
        bad.verify_loaded_sources()
    with pytest.raises(ValueError, match='no verifiable'):
        bound.verify_module_origin(SimpleNamespace())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_native_decoder_rejects_nonfinite_before_vae_or_image_conversion(bad):
    from types import SimpleNamespace
    from conceptmod.textsliders.pg09_backends import moss_pipeline_decoder
    calls = []
    def decode(z, return_dict=False):
        calls.append('decode')
        return (z.expand(-1, 3, -1, -1),)
    def postprocess(images, **kwargs):
        calls.append('postprocess')
        return images
    vae = SimpleNamespace(dtype=torch.float32, device=torch.device('cpu'),
                          config=SimpleNamespace(batch_norm_eps=1e-5),
                          bn=SimpleNamespace(running_var=torch.ones(4), running_mean=torch.zeros(4)),
                          decode=decode)
    pipe = SimpleNamespace(vae=vae, image_processor=SimpleNamespace(postprocess=postprocess))
    decoder = moss_pipeline_decoder(pipe)
    latent = torch.ones(1, 4, 2, 2)
    latent[0, 0, 0, 0] = bad
    with pytest.raises(FloatingPointError, match='latent before'):
        decoder(latent)
    assert calls == []
    vae.decode = lambda z, **kwargs: (torch.full((1, 3, 4, 4), bad),)
    with pytest.raises(FloatingPointError, match='float output before clamp'):
        decoder(torch.ones(1, 4, 2, 2))
    assert calls == []
