"""Frozen CPU-only actual-Qwen qualification proposal; never self-admits a run.

Only ConditionalPG09 owns the objective, optimizers and update. Preflight uses
stdlib metadata only. A coordinator must admit the exact produce/restore pair
after independent source review and before the first pretrained-model load.
"""
import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA = '426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256'
TRAINING = dict(preset='bispf_optimized', total_steps=1200, stop_after=4,
                replay_start=2, batch_size=2, seed=7, sigma=1.0,
                response_scale=1.0, sampling='balanced', recipe_overrides={})
GENERATION = dict(do_sample=False, num_beams=1, max_new_tokens=32, use_cache=True)
LIMITS = dict(cpu_threads=1, interop_threads=1, rss_cap_gib=8,
              host_reserve_gib=8, nvme_reserve_gib=100,
              produce_seconds=900, restore_seconds=600)
DOMAIN = dict(response_domain='continuous_vector',
              pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')


def digest(path):
    """Stream even the 1.19GB weight file; never allocate a weight-sized copy."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


def runtime_versions():
    return dict(python=sys.version.split()[0], **{name: version(name) for name in
                ('torch', 'numpy', 'transformers', 'diffusers', 'safetensors')})


def validate_dataset(data):
    if set(data) != {'train', 'validation', 'final_holdout', 'canaries', 'fluency'}:
        raise ValueError('explicit train/validation/final/canary/fluency split required')
    pairs = []
    for split in ('train', 'validation', 'final_holdout'):
        if len(data[split]) != 3:
            raise ValueError('three fixed pairs per split required')
        for row in data[split]:
            if (set(row) != {'neutral', 'positive'} or not all(row.values())
                    or row['neutral'] == row['positive']):
                raise ValueError('distinct nonempty neutral/positive captions required')
            pairs.extend(row.values())
    if len(set(pairs)) != len(pairs):
        raise ValueError('train/validation/final texts must be disjoint')
    if len(data['canaries']) != 3 or len(data['fluency']) != 1:
        raise ValueError('three canaries and one fixed fluency prompt required')
    if len(set(pairs + data['canaries'] + data['fluency'])) != len(pairs) + 4:
        raise ValueError('canaries/fluency must not overlap paired splits')
    return data


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan['schema'] != 'tiny-qwen-native-pg09-cpu-smoke-v1'
            or plan['training'] != TRAINING or plan['generation'] != GENERATION
            or plan['limits'] != LIMITS or plan['domain'] != DOMAIN
            or plan['runtime_versions'] != runtime_versions()
            or plan['output_filesystem'] != dict(device='/dev/nvme0n1p2', mountpoint='/mnt/nvme')):
        raise ValueError('fixed recipe/domain/runtime/resource proposal differs')
    if Path(plan['checkout']).resolve() != ROOT:
        raise ValueError('worker checkout differs')
    if Path(sys.executable).resolve() != Path(plan['interpreter']).resolve():
        raise ValueError('interpreter path differs')
    if not Path(plan['output']).resolve().is_relative_to('/mnt/nvme/sliders/qualification/runs'):
        raise ValueError('isolated qualification NVMe output required')
    required = [Path(__file__).resolve(), Path(plan['dataset']).resolve()]
    required += [ROOT / p for p in (
        'scripts/verify_bonsai_pg09_cached.py', 'scripts/prepare_text_video_pg09_verification.py',
        'conceptmod/textsliders/tiny_llm_backend.py', 'conceptmod/textsliders/tiny_llm_particle.py',
        'conceptmod/textsliders/particle_bridge_gan.py', 'conceptmod/textsliders/pg09_text_video.py',
        'conceptmod/textsliders/pg09_backends.py', 'conceptmod/textsliders/pg09_readout_critic.py')]
    core_sources = Path(plan['core']) / 'packages/particle-sliders-core/src/particle_sliders'
    required += [core_sources / p for p in ('pg09.py', 'pg09_continuous_profile.json')]
    profile = json.loads((core_sources / 'pg09_continuous_profile.json').read_text())
    if plan['source_profile'] != profile:
        raise ValueError('reviewed source profile differs')
    required += [Path(plan['overlay']) / 'particlegan' / p for p in profile['python_sources']]
    for p, expected in profile['python_sources'].items():
        if digest(Path(plan['overlay']) / 'particlegan' / p) != expected:
            raise ValueError('reviewed PG overlay differs')
    if not set(map(str, required)) <= plan['source_sha256'].keys():
        raise ValueError('proposal omits required source/data pins')
    for name, value in plan['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('environment differs: ' + name)
    for name, expected in plan['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError('source/metadata differs: ' + name)
    if digest(Path(plan['core']) / 'packages/particle-sliders-core/src/particle_sliders/pg09.py') != CORE_SHA:
        raise ValueError('immutable core differs')
    validate_dataset(json.loads(Path(plan['dataset']).read_text()))
    return plan


def preflight(plan):
    from prepare_text_video_pg09_verification import audit_manifest, ASSETS
    from verify_bonsai_pg09_cached import check_limits
    resource_receipt = check_limits(plan, time.monotonic(), 'produce', startup=True)
    asset = audit_manifest(Path(plan['model']), ASSETS['tiny'])
    config = json.loads((Path(plan['model']) / 'config.json').read_text())
    expected = dict(model_type='qwen3', hidden_size=1024, num_hidden_layers=28,
                    num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                    intermediate_size=3072, vocab_size=151936, max_position_embeddings=32768)
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('actual staged Qwen architecture differs')
    if plan['model_revision'] != asset['revision']:
        raise ValueError('model revision differs')
    return dict(resource_preflight=resource_receipt, assets=asset,
                torch_imported=False, model_loaded=False, gpu_used=False,
                final_holdout_forwarded=False)


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def base_digest(model):
    import torch
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        assert tensor.device.type == 'cpu' and tensor.is_contiguous()
        result.update(name.encode())
        result.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        result.update(memoryview(tensor.detach().view(torch.uint8).numpy()))
    return result.hexdigest()


def generate_text(backend, slider, prompt, scale, controls):
    """Actual autoregressive HF generation, including emitted-token diagnostics."""
    import torch
    ids = backend.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device='cpu')
    scope = slider.scaled(scale) if slider is not None else nullcontext()
    with torch.no_grad(), scope:
        output = backend.model.generate(
            input_ids=tokens, attention_mask=torch.ones_like(tokens),
            pad_token_id=backend.tokenizer.eos_token_id,
            eos_token_id=backend.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True, **controls)
    emitted = output.sequences[0, len(ids):].tolist()
    nll, entropy = [], []
    for score, token in zip(output.scores, emitted):
        assert torch.isfinite(score).all()
        logp = score.float().log_softmax(-1)
        nll.append(float(-logp[0, token]))
        entropy.append(float(-(logp.exp() * logp).sum()))
    assert emitted and len(emitted) == len(output.scores)
    return dict(prompt=prompt, scale=scale, input_ids=ids, generated_ids=emitted,
                text=backend.tokenizer.decode(output.sequences[0], skip_special_tokens=True),
                continuation=backend.tokenizer.decode(emitted, skip_special_tokens=True),
                mean_self_nll=sum(nll) / len(nll), mean_entropy=sum(entropy) / len(entropy),
                repeated_token_fraction=1 - len(set(emitted)) / len(emitted))


def evaluation_rows(backend, data):
    """Final holdout deliberately never encoded, forwarded or generated here."""
    from conceptmod.textsliders.pg09_text_video import TextResponseRow
    result = {}
    for split in ('train', 'validation'):
        result[split] = [(TextResponseRow(tuple(backend.encode(r['neutral']).ids),
                                        tuple(backend.encode(r['positive']).ids)), r)
                         for r in data[split]]
    return result


def sample(adapter, fixed, canaries, prompts, controls):
    import torch
    result = dict(vectors={}, diagnostics={}, text=[])
    with torch.no_grad():
        for split, rows in fixed.items():
            result['vectors'][split] = []
            diagnostics = []
            for row in rows:
                values = {str(s): adapter.render(row.row, scale=s) for s in (0, -1, .5, 1)}
                equal(values['0'], row.neutral)
                assert all(torch.isfinite(v).all() for v in values.values())
                edit, target = values['1'] - row.neutral, row.target - row.neutral
                diagnostics.append(dict(relative_edit_mse=float((edit-target).square().mean() /
                                       target.square().mean().clamp_min(1e-12)),
                                        cosine=float(torch.nn.functional.cosine_similarity(edit, target).mean()),
                                        edit_rms=float(edit.square().mean().sqrt())))
                result['vectors'][split].append(values)
            result['diagnostics'][split] = diagnostics
        result['vectors']['canaries'] = []
        for row in canaries:
            values = {str(s): adapter.render(row, scale=s) for s in (0, -1, .5, 1)}
            assert all(torch.isfinite(v).all() for v in values.values())
            result['vectors']['canaries'].append(values)
        for prompt in prompts:
            for scale in (0, -1, .5, 1):
                result['text'].append(generate_text(adapter.backend, adapter.slider,
                                                    prompt, scale, controls))
    return result


def execute(plan, plan_sha, mode, receipt):
    started = time.monotonic()
    from verify_bonsai_pg09_cached import check_limits
    check = lambda startup=False: check_limits(plan, started, mode, startup=startup)
    check(True)
    output = Path(plan['output'])
    if mode == 'produce':
        output.mkdir(parents=True, exist_ok=False)
    else:
        # Marker remains even if a later step fails: no automatic retry.
        assert (output / 'produce-receipt.json').is_file()
    with (output / (mode + '-started.json')).open('x') as stream:
        json.dump(dict(plan_sha256=plan_sha, pid=os.getpid()), stream)
    # Full fresh artifact hash precedes model allocation and Torch import.
    manifest = json.loads((Path(plan['model']) / 'qualification-manifest.json').read_text())
    assert digest(Path(plan['model']) / 'model.safetensors') == manifest['weight_sha256']
    check()
    sys.path[:0] = [plan['overlay'], str(Path(plan['core']) / 'packages/particle-sliders-core/src'), str(ROOT)]
    import torch
    import random
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
    validate_pg09_domain((1, 1024), **DOMAIN)
    from conceptmod.textsliders.tiny_llm_backend import TinyLLMBackend
    from conceptmod.textsliders.tiny_llm_particle import TinyParticleSlider
    from conceptmod.textsliders.pg09_text_video import TinyQwenResponseAdapter, TextResponseRow
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic
    data = validate_dataset(json.loads(Path(plan['dataset']).read_text()))
    backend = TinyLLMBackend(device='cpu', model_id=plan['model'], allow_hub=False, dummy=False)
    assert sum(p.numel() for p in backend.model.parameters()) == 596049920
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in backend.model.parameters())
    check()
    original_base = base_digest(backend.model)
    selected = evaluation_rows(backend, data)
    raw = {split: [(backend.hidden(row.neutral_ids)[:, -1].detach(),
                   backend.hidden(row.positive_ids)[:, -1].detach()) for row, _ in rows]
           for split, rows in selected.items()}
    prompts = [data['train'][0]['neutral'], data['validation'][0]['neutral'],
               data['canaries'][0], data['fluency'][0]]
    baseline_text = [generate_text(backend, None, prompt, 0, GENERATION) for prompt in prompts]
    check()
    slider = TinyParticleSlider(backend.model).float()
    assert len(slider.target_names) == 112
    identity = dict(model_id='Qwen/Qwen3-0.6B-Base', revision=plan['model_revision'],
                    weight_sha256=manifest['weight_sha256'], base_state_sha256=original_base,
                    proposal_sha256=plan_sha, dataset_sha256=digest(plan['dataset']),
                    generation=GENERATION, final_holdout_policy='unused during smoke')
    adapter = TinyQwenResponseAdapter(backend, slider, model_identity=identity,
                    pg_source_profile=DOMAIN['pg_source_profile'],
                    penalty_input_domain=DOMAIN['penalty_input_domain'])
    fixed = {split: [adapter.prepare(row) for row, _ in rows] for split, rows in selected.items()}
    for split, rows in fixed.items():
        for row, (neutral, target) in zip(rows, raw[split]):
            equal((row.neutral, row.target), (neutral, target))
    canaries = [TextResponseRow(tuple(backend.encode(text).ids), (1,)) for text in data['canaries']]
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([1., 0.])))[None]
                            for r in fixed['train']])
    critic = ConditionalReadoutCritic(1024, conditions.shape[1], hidden=16)
    game = ConditionalPG09(adapter, fixed['train'], critic, conditions=conditions,
                          particles=adapter.particles, **DOMAIN,
                          **{k: v for k, v in TRAINING.items() if k not in ('stop_after', 'replay_start')})
    from dataclasses import asdict
    equal(json.loads(json.dumps(asdict(game.recipe))), plan['resolved_public_recipe'])
    initial_g = {k: v.clone() for k, v in adapter.network.state_dict().items()}
    initial_d = {k: v.clone() for k, v in critic.state_dict().items()}
    initial_bank = slider.particles.detach().clone()

    def save(value, name):
        pending = output / (name + '.pending')
        torch.save(value, pending)
        pending.replace(output / name)

    def evaluate():
        before, rng = game.state_dict(), torch.get_rng_state().clone()
        value = sample(adapter, fixed, canaries, prompts, GENERATION)
        equal(before, game.state_dict())
        equal(rng, torch.get_rng_state())
        for expected, actual in zip(baseline_text, value['text'][::4]):
            equal(expected, actual)
        check()
        return value

    def step():
        value = game.step()
        assert abs(value['g_loss'] - value['g_adv'] - value['prior_regularization']) < 1e-5
        assert abs(value['d_loss'] - value['d_adv'] - value['d_pen']) < 1e-5
        assert value['d_pen'] > 0 and game.penalty.last_stats['input_domain'] == 'continuous'
        check()
        return value

    def forbidden(*args, **kwargs):
        raise AssertionError('optimized MSE forbidden')
    torch.nn.functional.mse_loss = forbidden
    if mode == 'produce':
        first = evaluate()
        save(first, 'initial-sample.pt')
        later = []
        with (output / 'metrics.jsonl').open('x') as stream:
            while game.completed_steps < TRAINING['stop_after']:
                value = step()
                stream.write(json.dumps(value) + '\n')
                stream.flush()
                if game.completed_steps == TRAINING['replay_start']:
                    save(game.state_dict(), 'checkpoint-000002.pt')
                if game.completed_steps > TRAINING['replay_start']:
                    later.append(value)
        final = evaluate()
        assert any(r['edit_rms'] > 0 for r in final['diagnostics']['train'])
        moved = dict(adapter=any(not torch.equal(initial_g[k], v) for k, v in adapter.network.state_dict().items()),
                     critic=any(not torch.equal(initial_d[k], v) for k, v in critic.state_dict().items()),
                     actual_bank=not torch.equal(initial_bank, slider.particles))
        assert all(moved.values())
        save(dict(state=game.state_dict(), metrics=later, sample=final), 'expected.pt')
        save(game.export_state_dict(), 'inference-export.pt')
        (output / 'initial-text.json').write_text(json.dumps(first['text'], indent=2) + '\n')
        (output / 'final-text.json').write_text(json.dumps(final['text'], indent=2) + '\n')
        result = dict(movement=moved, initial_diagnostics=first['diagnostics'],
                      final_diagnostics=final['diagnostics'])
    else:
        game.load_state_dict(torch.load(output / 'checkpoint-000002.pt', weights_only=True))
        assert game.completed_steps == 2
        later = [step(), step()]
        expected = torch.load(output / 'expected.pt', weights_only=True)
        equal(later, expected['metrics'])
        equal(game.state_dict(), expected['state'])
        equal(evaluate(), expected['sample'])
        save(game.state_dict(), 'restored-final-state.pt')
        # Reset trainables before loading export: an actual reload, not no-op.
        with torch.no_grad():
            for p in adapter.network.parameters():
                p.zero_()
        game.load_export_state_dict(torch.load(output / 'inference-export.pt', weights_only=True))
        equal(evaluate(), expected['sample'])
        result = dict(exact_full_state=True, exact_losses=True, exact_samples=True,
                      exact_text=True, exact_export_reload=True, replayed_updates=2)
    assert original_base == base_digest(backend.model)
    assert all(not p.requires_grad and p.grad is None for p in backend.model.parameters())
    assert not torch.cuda.is_initialized()
    check()
    result.update(mode=mode, proposal_sha256=plan_sha, completed_steps=game.completed_steps,
                  horizon=TRAINING['total_steps'], frozen_base_sha256=original_base,
                  actual_attention_projections=112, base_unchanged=True, zero_scale_exact=True,
                  fresh_weight_sha256=manifest['weight_sha256'], model_loaded=True,
                  final_holdout_forwarded=False, negative_scale='extrapolation; no negative teacher',
                  gpu_used=False, quality_qualified=False, elapsed_seconds=time.monotonic()-started,
                  peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)
    (output / (mode + '-receipt.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'produce', 'restore'), required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    receipt = preflight(plan)
    assert 'torch' not in sys.modules
    if args.mode == 'preflight':
        print(json.dumps(receipt, indent=2))
    else:
        execute(plan, digest(args.plan), args.mode, receipt)


if __name__ == '__main__':
    main()
