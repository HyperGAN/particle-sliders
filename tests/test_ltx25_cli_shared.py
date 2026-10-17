"""CPU software qualification of the public packaged LTX connector GAN path.

Random fixtures establish software contracts only, never native AV quality.
"""
from copy import deepcopy
import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from particle_sliders import ltx25_backend as native
from particle_sliders import ltx25_cli as cli
from particle_sliders import ltx25_uni as uni
from particle_sliders.pg09 import ConditionalPG09


def options(path, *extra):
    return cli.parse_args([
        '--dummy', '--steps', '4', '--rank', '2', '--alpha', '2', '--te_last_n', '1',
        '--seed', '21', '--save_dir', str(path),
        '--pg_source_profile', 'alz-continuous-v1', '--penalty_input_domain', 'continuous',
        *extra,
    ])


def equal_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            equal_tree(value, reference)
    else:
        assert actual == expected


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('module', ['ltx25_backend', 'ltx25_uni', 'pg09_ltx_embed'])
def test_compatibility_modules_share_canonical_objects(module):
    canonical = importlib.import_module('particle_sliders.' + module)
    legacy = importlib.import_module('conceptmod.textsliders.' + module)
    assert canonical is legacy


def test_training_entrypoint_is_the_packaged_cli():
    assert importlib.import_module('conceptmod.textsliders.train_lora_ltx25') is cli


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('sampling', ['balanced', 'replacement'])
def test_actual_shared_gan_objective_ownership_and_neutral_conditions(preset, sampling, tmp_path, monkeypatch):
    caller_rng = torch.get_rng_state().clone()
    adapter, prepared, game, rows = cli.create_session(
        options(tmp_path, '--preset', preset, '--sampling', sampling))
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert isinstance(game, ConditionalPG09)
    frozen = [(parameter, parameter.detach().clone())
              for root in (adapter.backend.encoder, adapter.backend.connectors, adapter.backend.transformer)
              for parameter in root.parameters() if not parameter.requires_grad]
    before = deepcopy(adapter.network.state_dict())
    critic_before = deepcopy(game.critic.state_dict())
    expected_conditions = torch.cat([
        torch.cat((row.neutral.flatten(), row.row.common_mask.flatten().float(),
                   torch.tensor([1., 0.])))[None]
        for row in prepared
    ])
    assert torch.equal(game.conditions, expected_conditions)
    assert game.conditions.requires_grad is False
    assert [row.row.caption.neutral for row in prepared] == [row['neutral'] for row in rows]
    assert all(row.target.requires_grad is False and row.neutral.requires_grad is False for row in prepared)
    adapter_parameters = list(adapter.network.parameters())
    owned = [parameter for group in game.opt_g.param_groups for parameter in group['params']]
    assert len(owned) == len({id(parameter) for parameter in owned}) == len(adapter_parameters)
    assert {id(parameter) for parameter in owned} == {id(parameter) for parameter in adapter_parameters}
    assert all(parameter.dtype == torch.float32 and parameter.requires_grad for parameter in owned)
    assert game.prior is None and game.spread is None

    def forbidden(*args, **kwargs):
        raise AssertionError('historical reconstruction/hold objective entered the shared GAN path')

    monkeypatch.setattr(torch.nn.functional, 'mse_loss', forbidden)
    for name in ('ltx25_embed_match_loss', 'ltx25_uni_total_loss', 'ltx25_unused_hold_loss'):
        monkeypatch.setattr(uni, name, forbidden)
    for _ in range(2):
        result = game.step()
        assert result['g_loss'] == result['g_adv']
        assert result['prior_regularization'] == 0
        assert result['d_loss'] == pytest.approx(result['d_adv'] + result['d_pen'])
        assert result['d_pen'] > 0
        assert all(torch.isfinite(torch.tensor(value)) for value in result.values()
                   if isinstance(value, (float, int)))
        assert game.penalty.last_stats['resolved_domain'] == 'continuous'
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in adapter_parameters)
    assert any(not torch.equal(value, adapter.network.state_dict()[key]) for key, value in before.items())
    assert any(not torch.equal(value, game.critic.state_dict()[key]) for key, value in critic_before.items())
    assert all(torch.equal(parameter, original) for parameter, original in frozen)
    for row in prepared:
        assert torch.equal(adapter.render(row, scale=0), row.neutral)
        assert not torch.equal(adapter.render(row, scale=1), row.neutral)
    state = game.state_dict()
    assert state['config']['generator_arm'] == 'adapter_gan'
    assert state['config']['pg_version'] == '0.9.1'
    assert state['config']['pg_source_profile'] == 'alz-continuous-v1'
    assert state['config']['sampling'] == sampling
    for counts in state['exposures'].values():
        assert int(counts.sum()) == 4
        if sampling == 'balanced':
            assert counts.tolist() == [2, 2]
    metadata = adapter.metadata()
    assert metadata['hold_objective'].startswith('none')
    assert metadata['quality_qualified'] is False
    assert metadata['audiovisual_render_qualified'] is False
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_cli_full_resume_and_steps_zero_export_inference(preset, tmp_path, monkeypatch):
    full_dir, split_dir, resume_dir = (tmp_path / name for name in ('full', 'split', 'resume'))
    common = ('--preset', preset)
    full = cli.train(options(full_dir, *common))
    first = cli.train(options(split_dir, *common, '--stop_after', '2'))
    resumed = cli.train(options(resume_dir, *common, '--resume', str(split_dir / 'checkpoint.pt')))
    assert first['completed_steps'] == 2
    assert full['completed_steps'] == resumed['completed_steps'] == 4
    assert first['history'] + resumed['history'] == full['history']
    for name in ('checkpoint.pt', 'adapter-export.pt', 'response-probes.pt'):
        equal_tree(torch.load(full_dir / name, weights_only=True),
                   torch.load(resume_dir / name, weights_only=True))
    checkpoint = torch.load(full_dir / 'checkpoint.pt', weights_only=True)
    exported = torch.load(full_dir / 'adapter-export.pt', weights_only=True)
    assert set(checkpoint) == set(exported) == {'schema', 'kind', 'options', 'rows', 'game'}
    assert checkpoint['schema'] == exported['schema'] == cli.SCHEMA
    assert checkpoint['kind'] == 'training' and exported['kind'] == 'inference'
    assert checkpoint['game']['schema'] == 'conditional-pg09-v1'
    assert exported['game']['schema'] == 'conditional-pg09-adapter-v1'
    assert set(exported['game']) == {'schema', 'config', 'network', 'completed_steps'}
    assert exported['game']['completed_steps'] == 4
    assert 'opt_g' in checkpoint['game'] and 'opt_d' in checkpoint['game']
    assert 'opt_g' not in exported['game'] and 'critic' not in exported['game']
    assert json.loads((full_dir / 'result.json').read_text()) == full
    probes = torch.load(full_dir / 'response-probes.pt', weights_only=True)
    assert [(probe['row'], probe['scale']) for probe in probes] == [
        (row, scale) for row in range(2) for scale in (0., .5, 1.)]
    assert all(probe['response'].shape == (1, 8208) for probe in probes)
    assert all(torch.isfinite(probe['response']).all() for probe in probes)
    assert not list(tmp_path.rglob('*.mp4')) and not list(tmp_path.rglob('*.safetensors'))

    def no_training(*args, **kwargs):
        raise AssertionError('inference export must never run a GAN update')

    monkeypatch.setattr(ConditionalPG09, 'step', no_training)
    inference_dir = tmp_path / 'inference'
    inferred = cli.train(cli.parse_args([
        '--dummy', '--steps', '0', '--load_export', str(full_dir / 'adapter-export.pt'),
        '--save_dir', str(inference_dir),
    ]))
    assert inferred['mode'] == 'inference' and inferred['completed_steps'] == 4
    assert inferred['quality_qualified'] is inferred['audiovisual_render_qualified'] is False
    equal_tree(torch.load(inference_dir / 'response-probes.pt', weights_only=True), probes)
    assert {path.name for path in inference_dir.iterdir()} == {'response-probes.pt', 'result.json'}
    assert not torch.cuda.is_initialized()


def test_release_profile_rejects_before_backend_or_artifact_access(tmp_path, monkeypatch):
    def no_backend(*args, **kwargs):
        raise AssertionError('source rejection must precede native construction')

    monkeypatch.setattr(cli, 'build_backend', no_backend)
    output = tmp_path / 'closed'
    with pytest.raises(ValueError, match='released PG09.*width 8208'):
        cli.train(cli.parse_args(['--dummy', '--steps', '1', '--save_dir', str(output)]))
    assert not output.exists()


def test_packet_source_and_kind_bindings_reject_without_output(tmp_path):
    source_dir = tmp_path / 'source'
    cli.train(options(source_dir, '--stop_after', '1', '--no_sample'))
    export = torch.load(source_dir / 'adapter-export.pt', weights_only=True)
    bad_source = deepcopy(export)
    bad_source['game']['config']['pg_sources_sha256'] = {'altered-source': '0' * 64}
    bad_path = tmp_path / 'altered-source.pt'
    torch.save(bad_source, bad_path)
    output = tmp_path / 'closed'
    with pytest.raises(ValueError, match='adapter export protocol differs'):
        cli.train(cli.parse_args([
            '--dummy', '--steps', '0', '--load_export', str(bad_path), '--save_dir', str(output),
        ]))
    assert not output.exists()
    with pytest.raises(ValueError, match='full checkpoint and inference export are distinct'):
        cli.train(options(output, '--resume', str(source_dir / 'adapter-export.pt')))
    assert not output.exists()
    with pytest.raises(ValueError, match='full checkpoint and inference export are distinct'):
        cli.train(cli.parse_args([
            '--dummy', '--steps', '0', '--load_export', str(source_dir / 'checkpoint.pt'),
            '--save_dir', str(output),
        ]))
    assert not output.exists()
    malformed = deepcopy(export)
    malformed['options']['unknown_override'] = True
    malformed_path = tmp_path / 'unknown-options.pt'
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match='invalid LTX session options'):
        cli.train(cli.parse_args([
            '--dummy', '--steps', '0', '--load_export', str(malformed_path), '--save_dir', str(output),
        ]))
    assert not output.exists()


def test_resume_cannot_change_rows_or_fixed_horizon(tmp_path):
    source_dir = tmp_path / 'source'
    cli.train(options(source_dir, '--stop_after', '1', '--no_sample'))
    output = tmp_path / 'closed'
    with pytest.raises(ValueError, match='resume options differ'):
        cli.train(options(output, '--steps', '5', '--resume', str(source_dir / 'checkpoint.pt')))
    prompts = tmp_path / 'changed.json'
    prompts.write_text(json.dumps([
        {'neutral': 'different adult neutral', 'positive': 'different adult smile', 'concept_words': 'smile'},
        {'neutral': 'different child neutral', 'positive': 'different child happy', 'concept_words': 'happy'},
    ]))
    with pytest.raises(ValueError, match='resume captions differ'):
        cli.train(options(output, '--prompts_file', str(prompts), '--resume', str(source_dir / 'checkpoint.pt')))
    assert not output.exists()


def test_no_sample_skips_probe_artifacts(tmp_path):
    result = cli.train(options(tmp_path / 'run', '--stop_after', '0', '--no_sample'))
    assert result['completed_steps'] == 0 and result['history'] == []
    assert {path.name for path in (tmp_path / 'run').iterdir()} == {
        'checkpoint.pt', 'adapter-export.pt', 'result.json'}


@pytest.mark.parametrize('std', [0., .02])
def test_local_native_lora_keeps_float32_leaves_on_bfloat16_host(std):
    host = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    original = host.weight.detach().clone()
    lora = native._AttnLoRA('bf16-cpu', host, rank=2, alpha=2., up_init_std=std)
    assert all(parameter.dtype == torch.float32 for parameter in lora.parameters())
    result = host(torch.ones(2, 4, dtype=torch.bfloat16))
    assert result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    result.float().sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in lora.parameters())
    assert host.weight.dtype == torch.bfloat16 and host.weight.grad is None
    assert torch.equal(host.weight, original)


@pytest.mark.parametrize('std', [0., .02])
def test_peft_initialization_casts_only_trainable_leaves(std):
    fixture = nn.Module()
    fixture.base = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16).requires_grad_(False)
    fixture.lora_A = nn.ModuleDict({'default': nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)})
    fixture.lora_B = nn.ModuleDict({'default': nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)})
    fixture.lora_B['default'].weight.data.zero_()
    original_base = fixture.base.weight.detach().clone()
    identities = [id(parameter) for parameter in fixture.parameters()]
    native._init_peft_lora_up(fixture, std=std)
    assert [id(parameter) for parameter in fixture.parameters()] == identities
    assert fixture.base.weight.dtype == torch.bfloat16
    assert torch.equal(fixture.base.weight, original_base)
    assert all(parameter.dtype == torch.float32 for parameter in fixture.parameters() if parameter.requires_grad)
    assert bool(torch.count_nonzero(fixture.lora_B['default'].weight)) == (std > 0)


def test_native_lora_reload_preserves_encoder_placement(tmp_path):
    backend = native.LTX25Backend(device='cpu', dummy=True, lora_rank=2, lora_alpha=2)
    backend.save_trained(str(tmp_path / 'saved_lora'))
    expected = deepcopy(backend.network.state_dict())
    # A meta DiT device is a CPU-only trap for the old network.to(self.device).
    backend.device = torch.device('meta')
    backend.load_trained(str(tmp_path / 'saved_lora.safetensors'))
    assert all(parameter.device.type == 'cpu' for parameter in backend.network.parameters())
    equal_tree(backend.network.state_dict(), expected)


@pytest.mark.parametrize('frames_present', [True, False])
def test_live_output_tensor_is_selected_without_truth_value_coercion(frames_present):
    backend = native.LTX25Backend(device='cpu', dummy=True, lora_rank=2, lora_alpha=2)
    frames = torch.ones(1, 2, 8, 8, 3)
    fallback = torch.zeros_like(frames)
    calls = []

    class Pipeline:
        vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(frames=frames if frames_present else None, videos=fallback,
                                   audio=torch.zeros(1, 64))

    backend.pipe = Pipeline()
    result = backend._live_generate_t2v('neutral caption', num_frames=49, height=544, width=960, seed=0)
    assert result['videos'] is (frames if frames_present else fallback)
    assert result['sampling_rate'] == 24000
    assert calls[0]['generator'].initial_seed() == 0
    assert calls[0]['prompt'] == 'neutral caption'
    assert not torch.cuda.is_initialized()
