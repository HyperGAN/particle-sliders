import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from scripts import verify_bonsai_pg09_cached as worker
from scripts.caption_quality import HELDOUT, CANARIES


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
@pytest.mark.parametrize('planned', [False, True])
def test_cached_worker_complete_flow_with_synthetic_cache_only(tmp_path, preset, planned, monkeypatch):
    # This fixture never opens the official cache. It validates worker control,
    # safe serialization, diagnostics and replay using synthetic native readouts.
    identity = 'synthetic-native-5120-readouts-only'
    cache = tmp_path / 'synthetic-cache'
    cache.mkdir()
    data = yaml.safe_load((worker.ROOT / 'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml').read_text())
    texts = set(CANARIES)
    for row in data + HELDOUT:
        texts.update((row['neutral'], row['positive']))
    for text in texts:
        key = hashlib.sha256((identity + '\0' + text).encode()).hexdigest()
        path = cache / (key + '.pt')
        generator = torch.Generator().manual_seed(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        torch.save(torch.randn(1, 5120, generator=generator), path)
        path.with_suffix('.json').write_text(json.dumps(dict(identity=identity, text=text,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), independent_requests_exact=True)))
        (cache / (key + '-tokens.json')).write_text(json.dumps(dict(identity=identity, text=text, ids=[1, 2])))
    receipt = dict(cache=str(cache), readout_identity=identity, qualification='synthetic cache only',
                   permitted_scope='unit test', model_loaded=False, tensor_payload_loaded=False, gpu_used=False)
    args = argparse.Namespace(core=Path('/mnt/nvme/sliders/text-video-core-3e94b13-20261017'),
                              overlay=Path('/tmp/alz-pg09-domain-proposal'), output=tmp_path / preset,
                              mode='produce', preset=preset)
    if planned:
        from dataclasses import asdict
        from particlegan import get_recipe
        # Reduced six-update unit schedule exercises the exact same worker,
        # checkpoints/evaluations and final replay on synthetic data only.
        # The production CLI separately enforces exactly1200 in load_plan.
        args.plan_sha256 = 'synthetic-test-plan'
        fs = worker.inspect_output_filesystem(args.output)
        args.plan_data = dict(output=str(args.output),
            output_filesystem={k: fs[k] for k in ('device', 'mountpoint')},
            training=dict(total_steps=6, batch_size=2, seed=7,
            sigma=1., response_scale=1., sampling='balanced', checkpoint_every=2,
            replay_start=4, evaluation_steps=[0, 2, 6]),
            limits=dict(cpu_threads=1, interop_threads=1, rss_cap_gib=8,
                        host_reserve_gib=0, nvme_reserve_gib=0,
                        produce_seconds=1800, restore_seconds=180),
            resolved_public_recipe=json.loads(json.dumps(asdict(get_recipe(preset,
                      batch_size=2, total_steps=6, num_particles=128, z_dim=4)))))
    monkeypatch.setattr(torch.nn.functional, 'mse_loss', torch.nn.functional.mse_loss)
    worker.execute(args, receipt)
    args.mode = 'restore'
    worker.execute(args, receipt)
    result = json.loads((args.output / 'restore-receipt.json').read_text())
    assert result['exact_full_state'] and result['exact_losses'] and result['exact_export_reload']
    assert result['quality_qualified'] is False and result['gpu_used'] is False
    assert not torch.cuda.is_initialized()
    if planned:
        produced = json.loads((args.output / 'produce-receipt.json').read_text())
        assert produced['completed_steps'] == 6 and len(produced['metrics']) == 6
        assert sorted(produced['evaluations']) == ['0', '2', '6']
        assert len(list(args.output.glob('checkpoint-*.pt'))) == 3
        assert result['replayed_updates'] == 2
        assert (args.output / 'evaluation/step-000006/inference-export.pt').is_file()


def test_long_plan_rejects_changed_budget_before_cache_or_torch(tmp_path):
    from importlib.metadata import version
    import sys
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(training=dict(total_steps=4), runtime_versions=dict(
        python=sys.version.split()[0], **{name: version(name) for name in
        ('torch', 'numpy', 'transformers', 'diffusers', 'safetensors')}))))
    args = argparse.Namespace(preset='bispf_optimized')
    with pytest.raises(ValueError, match='1200-update'):
        worker.load_plan(plan, args)


def test_resource_limit_stops_before_payload_access(monkeypatch):
    monkeypatch.setattr(worker.resource, 'getrusage', lambda *_: type('Usage', (), {'ru_maxrss': 9 * 1024**2})())
    plan = dict(limits=dict(produce_seconds=1800, rss_cap_gib=8))
    with pytest.raises(MemoryError, match='peak RSS'):
        worker.check_limits(plan, worker.time.monotonic(), 'produce', startup=True)


@pytest.mark.parametrize('symlink', [False, True])
def test_disk_guard_resolves_output_parent_and_retains_reserve(tmp_path, monkeypatch, symlink):
    parent = tmp_path / 'volume'
    parent.mkdir()
    path = parent
    if symlink:
        path = tmp_path / 'volume-link'
        path.symlink_to(parent, target_is_directory=True)
    output = path / 'not-created' / 'preset'
    actual = worker.inspect_output_filesystem(output)
    assert actual['existing_parent'] == str(parent)
    seen = []

    def usage(path):
        seen.append(path)
        return type('Usage', (), {'free': 101 * 1024**3})()

    monkeypatch.setattr(worker.shutil, 'disk_usage', usage)
    plan = dict(output=str(output), output_filesystem={k: actual[k] for k in ('device', 'mountpoint')},
                limits=dict(produce_seconds=1800, rss_cap_gib=8, host_reserve_gib=0, nvme_reserve_gib=100))
    worker.check_limits(plan, worker.time.monotonic(), 'produce', startup=True)
    assert seen == [parent] and not output.exists()
    monkeypatch.setattr(worker.shutil, 'disk_usage', lambda _: type('Usage', (), {'free': 99 * 1024**3})())
    with pytest.raises(OSError, match='reserve unavailable'):
        worker.check_limits(plan, worker.time.monotonic(), 'produce')
    # A large-enough fallback filesystem cannot satisfy the admitted device.
    monkeypatch.setattr(worker.shutil, 'disk_usage', usage)
    plan['output_filesystem']['device'] = '/dev/not-the-admitted-device'
    with pytest.raises(OSError, match='device/mount differs'):
        worker.check_limits(plan, worker.time.monotonic(), 'produce')
