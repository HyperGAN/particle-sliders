"""Held-out distribution probe: metrics, seed split, and the training hook."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from conceptmod.textsliders import distribution_probe as probe
from conceptmod.textsliders.train_lora_yue2_arm_b import parse_args, train


def _groups(n, vocal='female'):
    return [dict(
        template=f't{i % 2}', vocal=vocal, tempo_range='110_to_130',
        sample_seed=100 + i, template_row=i % 2, tempo_bpm=112,
    ) for i in range(n)]


def _record(step, strength, rms, *, p95=None, swd=0.1, gain=None, subgroup_rms=None, kind='ema'):
    gain = float(strength if gain is None else gain)
    sub = float(rms if subgroup_rms is None else subgroup_rms)
    p95 = float(rms if p95 is None else p95)
    return dict(
        step=step, weights_kind=kind, strength=float(strength), examples=4,
        residual_rms=float(rms), residual_p95=p95, teacher_swd=float(swd),
        game_swd_sigma_1=float(swd), gain_mean=gain, gain_p95_error=abs(gain - float(strength)),
        orthogonal_error=0.0, evaluator_auc=None, subgroups={
            'template': {'t0': {'residual_rms': sub, 'n': 4, 'flagged': False}},
            'vocal': {}, 'tempo_range': {}, 'flagged_seeds': [],
        })


def _trio(step, rms, **kwargs):
    return [_record(step, strength, rms, **kwargs) for strength in probe.STRENGTHS]


def test_probe_seeds_stay_out_of_training_and_listening():
    training = list(range(10, 30))
    first = probe.plan_probe_seeds(4, training, probe_per=3, listen_per=2, seed=7)
    second = probe.plan_probe_seeds(4, training, probe_per=3, listen_per=2, seed=7)
    assert first == second
    probe_seeds = [item['sample_seed'] for item in first['probe']]
    listen_seeds = [item['sample_seed'] for item in first['listening']]
    assert len(probe_seeds) == 12 and len(set(probe_seeds)) == 12
    assert len(listen_seeds) == 8 and len(set(listen_seeds)) == 8
    assert set(probe_seeds).isdisjoint(training)
    assert set(listen_seeds).isdisjoint(training)
    assert set(probe_seeds).isdisjoint(listen_seeds)
    assert [item['template_row'] for item in first['probe']] == [row for row in range(4) for _ in range(3)]


def test_prompt_subgroup_reads_vocal_and_tempo_bucket():
    female = probe.prompt_subgroup(
        'English. BPM 112. One adult female lead vocalist. A guitar-led band song.')
    assert female == dict(vocal='female', tempo_bpm=112, tempo_range='110_to_130')
    assert probe.prompt_subgroup('English. BPM 100. One adult male lead vocalist.')['tempo_range'] == 'under_110'
    assert probe.prompt_subgroup('English. BPM 148. One adult male lead vocalist.')['tempo_range'] == 'over_130'
    assert probe.prompt_subgroup('No tempo and no voice.')['vocal'] == 'unknown'


def test_perfect_prediction_is_zero_error_at_the_requested_gain():
    neutral = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
    positive = torch.tensor([[0., 2., 1.], [1., 0., 4.]])
    strength = 0.25
    prediction = neutral + strength * (positive - neutral)
    metrics = probe.score_heldout(
        neutral, positive, prediction, torch.tensor([0.5, 1., 2.]),
        torch.eye(3), torch.randn(2, 3), strength=strength, groups=_groups(2))
    assert metrics['residual_rms'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['residual_p95'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['gain_mean'] == pytest.approx(strength)
    assert metrics['gain_p95_error'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['orthogonal_error'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['cosine_mean'] == pytest.approx(1.0)
    assert metrics['teacher_swd'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['game_swd_sigma_1'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['game_swd_sigma_0_5'] == pytest.approx(0.0, abs=1e-6)
    assert metrics['evaluator_auc'] is None
    assert metrics['evaluator_status'] == 'not_trained'
    assert metrics['subgroups']['flagged_seeds'] == []


def test_orthogonal_leak_and_fixed_eval_sigma_are_visible():
    neutral = torch.zeros(1, 3)
    positive = torch.tensor([[3., 0., 0.]])
    prediction = torch.tensor([[3., 4., 0.]])
    metrics = probe.score_heldout(
        neutral, positive, prediction, torch.ones(3), torch.eye(3), torch.zeros(1, 3),
        strength=1.0, groups=_groups(1))
    assert metrics['gain_mean'] == pytest.approx(1.0)
    assert metrics['orthogonal_error'] == pytest.approx(4 / 3)
    assert metrics['cosine_mean'] == pytest.approx(0.6)
    assert metrics['residual_rms'] == pytest.approx((16 / 3) ** 0.5)
    assert metrics['teacher_swd'] > 0

    positive = torch.tensor([[1., 0.], [0., 1.]])
    prediction = torch.tensor([[0.75, 0.], [1., 1.]])
    noise = torch.tensor([[1., 0.], [0., 0.]])
    compared = probe.score_heldout(
        torch.zeros(2, 2), positive, prediction, torch.ones(2),
        torch.tensor([[1., 0.]]), noise, strength=1.0, groups=_groups(2))
    assert compared['game_swd_sigma_0_5'] == pytest.approx(0.375)
    assert compared['game_swd_sigma_2'] == pytest.approx(0.625)
    assert compared['game_swd_sigma_0_5'] != compared['game_swd_sigma_2']


def test_fixtures_and_bank_are_reused_without_touching_listening_seeds(tmp_path):
    scale = torch.linspace(0.1, 1.0, 4)
    path = tmp_path / 'probe-fixtures.pt'
    first = probe.ensure_fixtures(path, dim=4, rows=3, scale=scale)
    second = probe.ensure_fixtures(path, dim=4, rows=3, scale=scale)
    assert torch.equal(first['projections'], second['projections'])
    assert torch.equal(first['projections'], probe.projections_for(4))
    assert torch.allclose(first['projections'].norm(dim=-1), torch.ones(probe.PROJECTION_COUNT), atol=1e-4)
    assert tuple(first['noise'].shape) == (3, 4)
    with pytest.raises(ValueError, match='noise'):
        probe.ensure_fixtures(path, dim=4, rows=5, scale=scale)

    class Backend:
        def __init__(self):
            self.calls = []
            self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64))

        def prefix(self, style, lyrics, cot='off'):
            return [len(style), len(lyrics)]

        def continuation(self, prefix, count, seed):
            self.calls.append(int(seed))
            return [int(seed) % 17 + i for i in range(count)]

        def hidden(self, ids, checkpointing=False):
            generator = torch.Generator().manual_seed(sum(int(token) for token in ids) + len(ids))
            return torch.randn(1, len(ids), 4, generator=generator)

    backend = Backend()
    templates = [
        dict(neutral='English. BPM 112. One adult female lead vocalist. Guitar band.',
             positive='English. BPM 112. One adult female lead vocalist. Saturated guitars.',
             lyrics='[verse]\nWe carry crate 0'),
        dict(neutral='English. BPM 148. One adult male lead vocalist. Guitar band.',
             positive='English. BPM 148. One adult male lead vocalist. Saturated guitars.',
             lyrics='[verse]\nWe carry crate 1'),
    ]
    training = [dict(sample_seed=seed, neutral=torch.zeros(4), targets=torch.ones(4)) for seed in (10, 11, 12, 13)]
    bank_path = tmp_path / 'prepared-eval.pt'
    bank = probe.ensure_bank(
        bank_path, backend, templates, training, max_seq_len=32, history_tokens=3,
        probe_per=2, listen_per=1, seed=7, model_id='dummy', dummy=True)
    again = probe.ensure_bank(
        bank_path, backend, templates, training, max_seq_len=32, history_tokens=3,
        probe_per=2, listen_per=1, seed=7, model_id='dummy', dummy=True)
    assert len(backend.calls) == 4
    assert set(backend.calls).isdisjoint({10, 11, 12, 13})
    listen = {item['sample_seed'] for item in bank['listening']}
    assert listen.isdisjoint(backend.calls)
    assert [row['sample_seed'] for row in again['prepared']] == [row['sample_seed'] for row in bank['prepared']]
    assert bank['prepared'][0]['vocal'] == 'female'
    assert bank['prepared'][2]['tempo_range'] == 'over_130'
    with pytest.raises(ValueError, match='does not match'):
        probe.ensure_bank(
            bank_path, backend, templates, training, max_seq_len=32, history_tokens=3,
            probe_per=3, listen_per=1, seed=7, model_id='dummy', dummy=True)


def test_plateau_waits_for_the_evaluator_and_selection_prefers_the_earliest_peer():
    flat = []
    for step in (100, 200, 300):
        flat.extend(_trio(step, 0.5, p95=0.8, swd=0.2))
    summary = probe.build_summary(flat, stride=100, training_finite=True)
    assert summary['stopping']['status'] == 'plateau_pending_evaluator'
    assert summary['stopping']['plateau'] is True
    assert summary['stopping']['finished'] is False
    assert summary['stopping']['evaluator_auc'] is None
    assert summary['best_step'] == 100

    moving = []
    for step, rms in ((100, 1.0), (200, 0.5), (300, 0.2)):
        moving.extend(_trio(step, rms))
    assert probe.build_summary(moving, stride=100)['stopping']['status'] == 'not_plateaued'

    worsened = []
    for step, sub in ((100, 0.5), (200, 0.5), (300, 0.8)):
        worsened.extend(_trio(step, 0.5, p95=0.8, swd=0.2, subgroup_rms=sub))
    assert probe.build_summary(worsened, stride=100)['stopping']['plateau'] is False

    peers = []
    peers.extend(_trio(100, 0.401, p95=0.50, swd=0.20))
    peers.extend(_trio(200, 0.400, p95=0.50, swd=0.20))
    peers.append(_record(50, 1.0, 0.0, kind='live'))
    chosen = probe.build_summary(peers, stride=100)
    assert chosen['best_step'] == 100
    assert chosen['quality_best_step'] == 200
    assert chosen['gain_ok'] is True

    collected = probe.build_summary(_trio(100, 0.4), stride=100)
    assert collected['stopping']['status'] == 'collecting'
    assert collected['best_step'] == 100


def test_rng_and_weights_are_restored_and_milestones_ignore_live_files(tmp_path):
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    with probe.preserve_rng(torch.device('cpu')):
        torch.randn(8)
    assert torch.equal(torch.get_rng_state(), before)

    layer = nn.Linear(2, 2, bias=False)
    original = layer.weight.detach().clone()
    other = {'weight': torch.ones_like(layer.weight)}
    with probe.use_state(layer, other):
        assert torch.equal(layer.weight, torch.ones_like(layer.weight))
    assert torch.equal(layer.weight, original)
    with pytest.raises(RuntimeError):
        with probe.use_state(layer, other):
            raise RuntimeError('probe failed')
    assert torch.equal(layer.weight, original)

    (tmp_path / 'run_step20.safetensors').write_bytes(b'x')
    (tmp_path / 'run_step100.safetensors').write_bytes(b'x')
    (tmp_path / 'run_live_step100.safetensors').write_bytes(b'x')
    assert [step for step, _path in probe.milestone_steps(tmp_path)] == [20, 100]
    assert probe.needs_probe(tmp_path, 100)
    probe._upsert_jsonl(tmp_path / 'probe.jsonl', _trio(100, 0.2))
    assert not probe.needs_probe(tmp_path, 100)
    assert probe.needs_probe(tmp_path, 20)


def test_probe_flags_are_particle_only():
    args = parse_args(['--recipe', 'particle_bridge', '--save_dir', 'unused', '--no_probe', '--probe_seeds', '16'])
    assert args.no_probe and args.probe_seeds == 16 and args.probe_listening_seeds == 4
    with pytest.raises(SystemExit):
        parse_args(['--save_dir', 'unused', '--probe_seeds', '8'])


def test_training_writes_one_ema_probe_and_keeps_it_out_of_the_training_bank(tmp_path):
    pytest.importorskip('yue2')
    prompts = tmp_path / 'p.yaml'
    prompts.write_text(yaml.safe_dump(dict(rows=[
        dict(neutral='English. BPM 112. One adult female lead vocalist. Guitar band clear vocal.',
             positive='English. BPM 112. One adult female lead vocalist. Saturated high-gain guitars.',
             lyrics='[verse]\nWe carry crate 0'),
        dict(neutral='English. BPM 148. One adult male lead vocalist. Guitar band clear vocal.',
             positive='English. BPM 148. One adult male lead vocalist. Saturated high-gain guitars.',
             lyrics='[verse]\nWe carry crate 1'),
    ])))
    save = tmp_path / 'run'
    common = [
        '--recipe', 'particle_bridge', '--dummy', '--name', 'probe-test', '--steps', '1',
        '--save_every', '1', '--prompts_file', str(prompts), '--save_dir', str(save),
        '--sample_seeds', '2', '--history_tokens', '2', '--probe_seeds', '1',
        '--probe_listening_seeds', '1',
    ]
    train(parse_args(common))
    training = torch.load(save / 'prepared.pt', map_location='cpu', weights_only=True)['prepared']
    bank = torch.load(save / 'prepared-eval.pt', map_location='cpu', weights_only=True)
    train_seeds = {int(row['sample_seed']) for row in training}
    eval_seeds = {int(row['sample_seed']) for row in bank['prepared']}
    listen_seeds = {int(row['sample_seed']) for row in bank['listening']}
    assert len(training) == 4
    assert len(bank['prepared']) == 2
    assert train_seeds.isdisjoint(eval_seeds)
    assert train_seeds.isdisjoint(listen_seeds)
    assert eval_seeds.isdisjoint(listen_seeds)
    fixtures = torch.load(save / 'probe-fixtures.pt', map_location='cpu', weights_only=True)
    assert tuple(fixtures['projections'].shape) == (probe.PROJECTION_COUNT, training[0]['neutral'].reshape(-1).numel())
    lines = [json.loads(line) for line in (save / 'probe.jsonl').read_text().splitlines()]
    assert [row['strength'] for row in lines] == [0.25, 0.5, 1.0]
    assert all(row['weights_kind'] == 'ema' and row['evaluator_auc'] is None for row in lines)
    assert all(row['evaluator_status'] == 'not_trained' and row['examples'] == 2 for row in lines)
    full = lines[-1]
    assert 'female' in full['subgroups']['vocal'] and 'male' in full['subgroups']['vocal']
    summary = json.loads((save / 'probe-summary.json').read_text())
    assert summary['best_step'] == 1
    assert summary['stopping']['status'] == 'collecting'
    assert summary['stopping']['finished'] is False
    text = (save / 'probe.jsonl').read_text()
    train(parse_args(common))
    assert (save / 'probe.jsonl').read_text() == text
