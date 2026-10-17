"""Read-only saved-checkpoint diagnosis; never constructs an optimizer/game."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    code, run = args.code.resolve(), args.run.resolve()
    plan = json.loads((code / 'evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal-v2.json').read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(path) == expected, path
    indexed = json.loads((code / 'evidence/pg09-text-video/bonsai-1200-v2-actual/artifact-index.json').read_text())
    assert Path(indexed['artifact_root']) == run
    artifact_hashes = {f['path']: f['sha256'] for f in indexed['files']}
    opened = {}
    for name in ('CUDA_VISIBLE_DEVICES',):
        assert os.environ.get(name) == ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ.get(name) == '1'
    sys.path.insert(0, str(code))
    import torch
    import yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    def forbidden(*a, **kw):
        raise AssertionError('read-only diagnosis forbids optimizer construction')
    torch.optim.Optimizer.__init__ = forbidden
    from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic

    def packet(relative):
        path = run / relative
        assert digest(path) == artifact_hashes[relative], relative
        opened[str(path)] = artifact_hashes[relative]
        return torch.load(path, map_location='cpu', weights_only=True)

    identity = plan['comparison']['readout_identity']
    cache = Path('/mnt/nvme/sliders/qualification/models/bonsai-caption-readouts-20261017')
    values = {}

    def hidden(text):
        if text not in values:
            key = hashlib.sha256((identity + '\0' + text).encode()).hexdigest()
            metadata = cache / (key + '.json')
            assert digest(metadata) == plan['source_sha256'][str(metadata)]
            meta = json.loads(metadata.read_text())
            assert meta['identity'] == identity and meta['text'] == text
            path = cache / (key + '.pt')
            assert digest(path) == meta['sha256']
            opened[str(path)] = meta['sha256']
            value = torch.load(path, map_location='cpu', weights_only=True)
            assert value.shape == (1, 5120) and value.dtype == torch.float32 and value.isfinite().all()
            values[text] = value
        return values[text]

    training = yaml.safe_load((code / 'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml').read_text())
    heldout = plan['comparison']['heldout_pairs']
    x = torch.cat([hidden(row['neutral']) for row in training])
    target = torch.cat([hidden(row['positive']) for row in training])
    teacher_edit = target - x
    conditions = torch.cat((x, torch.tensor([[1., 0.]]).repeat(3, 1)), dim=1)
    torch.manual_seed(7)
    head = BonsaiParticleHead().float().eval().requires_grad_(False)
    critic = ConditionalReadoutCritic(5120, 5122).float().eval().requires_grad_(False)

    def predict(value, scale=1.):
        # Preserve native singleton calls used by the original worker/evaluator.
        with head.scaled(scale):
            return torch.cat([head(row[None]) for row in value])

    def stats(rows, scale=1.):
        neutral = torch.cat([hidden(r['neutral']) for r in rows])
        positive = torch.cat([hidden(r['positive']) for r in rows])
        edit = positive - neutral
        prediction = predict(neutral, scale)
        delta = prediction - neutral
        result = []
        for i in range(len(rows)):
            energy = float(edit[i:i+1].square().sum())
            norm = float(delta[i:i+1].square().sum())
            dot = float((delta[i:i+1] * edit[i:i+1]).sum())
            result.append(dict(relative_edit_mse=float((delta[i:i+1]-edit[i:i+1]).square().sum())/energy,
                               cosine=dot/max((energy*norm)**.5, 1e-30),
                               edit_rms_ratio=(norm/energy)**.5))
        return dict(mean_relative_edit_mse=sum(r['relative_edit_mse'] for r in result) / len(result),
                    successful_cases=sum(r['relative_edit_mse'] <= .9 and r['cosine'] >= .25 for r in result),
                    cases=result)

    curves = [dict(step=0, training=stats(training), heldout=stats(heldout))]
    signatures = None
    final = None
    for step in range(20, 1201, 20):
        state = packet(f'bispf_optimized/checkpoint-{step:06d}.pt')
        if signatures is None:
            signatures = state['config']
        else:
            assert state['config'] == signatures
        assert state['completed_steps'] == step
        names = state['config']['backend']['adapter_names']
        assert names == list(dict(head.named_parameters()))
        head.load_state_dict({name: state['network'][f'weights.{i}'] for i, name in enumerate(names)}, strict=True)
        assert torch.equal(head.particles, state['prior']['z'])
        critic.load_state_dict(state['critic'], strict=True)
        record = dict(step=step, training=stats(training))
        if step in (20, 100, 400, 800, 1200):
            record['heldout'] = stats(heldout)
        curves.append(record)
        if step == 1200:
            final = state

    saved = packet('bispf_optimized/evaluation/step-001200/vectors.pt')
    held_x = torch.cat([hidden(r['neutral']) for r in heldout])
    assert torch.equal(predict(held_x, 0), torch.cat(saved['zero']))
    assert torch.equal(predict(held_x, 1), torch.cat(saved['plus']))
    assert signatures['response_scale'] == 1 and signatures['sigma'] == 1
    delta = predict(x) - x
    sign = {str(scale): stats(training, scale) for scale in (-1., 0., .5, 1.)}
    linear_errors = {str(scale): float((predict(x, scale) - (x + scale * delta)).abs().max())
                     for scale in (-1., 0., .5, 1.)}
    assert torch.equal(predict(x, 0), x)
    teacher_cos = torch.nn.functional.normalize(teacher_edit, dim=1) @ torch.nn.functional.normalize(teacher_edit, dim=1).T
    generated_cos = torch.nn.functional.normalize(delta, dim=1) @ torch.nn.functional.normalize(delta, dim=1).T
    cross_error = (delta[:, None] - teacher_edit[None]).square().sum(-1) / teacher_edit.square().sum(-1)[None]

    # Training contexts only: no choice of scale/checkpoint is made on heldouts.
    noise = torch.randn(256, 5120, generator=torch.Generator().manual_seed(123))
    conditioning = []
    for i in range(3):
        context = conditions[i:i+1].expand(256, -1)
        swapped = conditions[(i+1) % 3:(i+1) % 3+1].expand(256, -1)
        residual = predict(x[i:i+1]) - target[i:i+1]
        real = critic(noise, context)
        fake = critic(noise + residual, context)
        fake_swap = critic(noise + residual, swapped)
        real_swap = critic(noise, swapped)
        fake_zero_context = critic(noise + residual, torch.zeros_like(context))
        off = critic(noise + x[i:i+1] - target[i:i+1], context)
        conditioning.append(dict(row=i, real_logit_mean=float(real.mean()),
            learned_fake_minus_real_mean=float((fake-real).mean()),
            learned_fake_minus_real_std=float((fake-real).std()),
            no_edit_fake_minus_real_mean=float((off-real).mean()),
            swapped_context_abs_logit_change_mean=float((fake_swap-fake).abs().mean()),
            swapped_context_abs_gap_change_mean=float(((fake_swap-real_swap)-(fake-real)).abs().mean()),
            zero_context_abs_logit_change_mean=float((fake_zero_context-fake).abs().mean()),
            critic_real_logit_std=float(real.std()),
            response_weight_norm=float(critic.net[0].weight[:, :5120].norm()),
            context_weight_norm=float(critic.net[0].weight[:, 5120:10240].norm())))

    canary_energy = sum(float((predict(hidden(t))-hidden(t)).square().sum()) for t in plan['comparison']['canaries'])
    heldout_energy = sum(float((hidden(r['positive'])-hidden(r['neutral'])).square().sum()) for r in heldout)
    positive_error = sum(float((predict(row[None])-row[None]).square().sum()) for row in target)
    identity_error = positive_error / sum(float(row.square().sum()) for row in target)
    for path, expected in opened.items():
        assert digest(path) == expected, path
    for path, expected in plan['source_sha256'].items():
        assert digest(path) == expected, path
    assert not torch.cuda.is_initialized()
    assert not torch.is_grad_enabled()
    report = dict(scope='read-only no_grad native-head/critic checkpoint forwards; no optimizer/game',
        script_sha256=digest(Path(__file__)), source_commit='1865a07ae9e147f2bb3a1178a0dcf7587036cd0e',
        source_sha256=plan['source_sha256'], all96_sources_unchanged=True,
        recipe_sha256=digest(code/'evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal-v2.json'),
        read_artifact_hashes=opened, curves=curves, final_training=curves[-1]['training'],
        final_heldout=curves[-1]['heldout'], training_only_sign_strength=sign,
        strength_linearity_max_abs_error=linear_errors, training_target_pair_cosine=teacher_cos.tolist(),
        training_generated_edit_pair_cosine=generated_cos.tolist(),
        training_generated_vs_target_relative_error_matrix=cross_error.tolist(),
        training_critic_condition_probes=conditioning,
        teacher_edit_l2_norms=teacher_edit.norm(dim=1).tolist(),
        neutral_l2_norms=x.norm(dim=1).tolist(),
        reference_noise_l2_mean=float(noise.norm(dim=1).mean()),
        final_canary_relative_rms=(canary_energy/heldout_energy)**.5,
        final_positive_identity_relative_mse=identity_error,
        checkpoint_outputs_match_saved_vectors_exactly=True,
        files_unchanged=True, optimizer_constructed=False, optimizer_updates=0,
        gpu_used=False, weights_loaded='existing native-head/critic checkpoint and frozen readout tensors only',
        heldout_tuning=False, elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('final_training', 'final_heldout', 'elapsed_seconds', 'optimizer_updates')}))


if __name__ == '__main__':
    main()
