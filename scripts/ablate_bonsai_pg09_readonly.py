"""Training-only saved-head copy ablations; zero updates and no autograd."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    code, run = args.code.resolve(), args.run.resolve()
    plan_path = code/'evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal-v2.json'
    plan = json.loads(plan_path.read_text())
    for path, expected in plan['source_sha256'].items():
        assert digest(path) == expected
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        assert os.environ[key] == '1'
    sys.path.insert(0, str(code))
    import torch
    import yaml
    from torch import nn
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)

    def forbidden(*a, **kw):
        raise AssertionError('optimizer construction is forbidden')
    torch.optim.Optimizer.__init__ = forbidden
    from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic

    opened = {}
    index = json.loads((code/'evidence/pg09-text-video/bonsai-1200-v2-actual/artifact-index.json').read_text())
    checkpoint = run/'bispf_optimized/checkpoint-001200.pt'
    expected = next(f['sha256'] for f in index['files'] if f['path'] == str(checkpoint.relative_to(run)))
    assert digest(checkpoint) == expected
    opened[str(checkpoint)] = expected
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['completed_steps'] == 1200
    assert state['config']['recipe']['loss'] == 'non_saturating'
    assert state['config']['response_scale'] == 1.0 and state['config']['sigma'] == 1.0
    head = BonsaiParticleHead().float().eval().requires_grad_(False)
    names = state['config']['backend']['adapter_names']
    assert names == list(dict(head.named_parameters()))
    head.load_state_dict({name: state['network'][f'weights.{i}'] for i, name in enumerate(names)}, strict=True)
    critic = ConditionalReadoutCritic(5120, 5122).float().eval().requires_grad_(False)
    critic.load_state_dict(state['critic'], strict=True)
    original = {k: v.clone() for k, v in head.state_dict().items()}

    identity = plan['comparison']['readout_identity']
    cache = Path('/mnt/nvme/sliders/qualification/models/bonsai-caption-readouts-20261017')
    training = yaml.safe_load((code/'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml').read_text())

    def readout(text):
        key = hashlib.sha256((identity+'\0'+text).encode()).hexdigest()
        metadata = cache/(key+'.json')
        assert digest(metadata) == plan['source_sha256'][str(metadata)]
        record = json.loads(metadata.read_text())
        assert record['identity'] == identity and record['text'] == text
        path = cache/(key+'.pt')
        assert digest(path) == record['sha256']
        opened[str(path)] = record['sha256']
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value.shape == (1, 5120) and value.dtype == torch.float32
        return value

    x = torch.cat([readout(r['neutral']) for r in training])
    target = torch.cat([readout(r['positive']) for r in training])
    edits = target-x
    conditions = torch.cat((x, torch.tensor([[1., 0.]]).repeat(3, 1)), dim=1)

    def trace(model, value, route_mode='native', fixed_weights=None, fixed_z=None):
        features = model.down(value)
        query = model.bridge.router(features)
        weights = (query @ model.particles.T / math.sqrt(model.particles.shape[1])).softmax(-1)
        if route_mode == 'uniform':
            weights = torch.full_like(weights, 1/len(model.particles))
        elif fixed_weights is not None:
            weights = fixed_weights
        z = weights @ model.particles
        if route_mode == 'zero_z':
            z = torch.zeros_like(z)
        elif fixed_z is not None:
            z = fixed_z
        latent = model.bridge.net(torch.cat((features, z), dim=-1))
        delta = model.up(latent)*model.scale
        return features, query, weights, z, latent, delta

    baseline = [trace(head, row[None]) for row in x]
    reference_delta = torch.cat([v[-1] for v in baseline])
    native_outputs = []
    with head.scaled(1):
        for row, delta in zip(x, reference_delta):
            actual = head(row[None])
            assert torch.equal(actual, row[None]+delta[None])
            native_outputs.append(actual)
    routes = torch.cat([v[2] for v in baseline])
    zs = torch.cat([v[3] for v in baseline])
    features = torch.cat([v[0] for v in baseline])
    average_route = routes.mean(0, keepdim=True)
    mean_delta = reference_delta.mean(0, keepdim=True)

    def pair_cos(value):
        normalized = nn.functional.normalize(value, dim=1)
        return (normalized @ normalized.T).tolist()

    def summary(raw_delta):
        effective = (x+raw_delta)-x
        records = []
        for d, t in zip(effective, edits):
            a, b = float(d.square().sum()), float(t.square().sum())
            records.append(dict(relative_edit_mse=float((d-t).square().sum())/b,
                                cosine=float((d*t).sum())/max((a*b)**.5, 1e-30),
                                edit_l2=a**.5))
        return dict(cases=records,
                    mean_training_relative_edit_mse=sum(v['relative_edit_mse'] for v in records)/3,
                    delta_l2=raw_delta.norm(dim=1).tolist(),
                    delta_pair_cosine=pair_cos(raw_delta),
                    relative_change_vs_native=((raw_delta-reference_delta).norm(dim=1)/reference_delta.norm(dim=1)).tolist(),
                    common_mean_energy_fraction=float(3*raw_delta.mean(0).square().sum()/raw_delta.square().sum().clamp_min(1e-30)))

    # Predeclared mechanistic arms, no heldout reads/selection and no updates.
    arms = ['native', 'zero_input', 'decoder_final_bias_zero', 'decoder_hidden_biases_zero',
            'decoder_all_biases_zero', 'router_all_biases_zero', 'uniform_routes',
            'fixed_average_training_route', 'cyclic_training_routes', 'zero_z',
            'zero_z_and_decoder_biases_zero']
    outputs, results = {}, {}
    for arm in arms:
        model = copy.deepcopy(head)
        if arm == 'decoder_final_bias_zero':
            model.bridge.net[-1].bias.zero_()
        elif arm in ('decoder_hidden_biases_zero', 'decoder_all_biases_zero', 'zero_z_and_decoder_biases_zero'):
            for module in model.bridge.net.modules():
                if isinstance(module, nn.Linear) and (arm != 'decoder_hidden_biases_zero' or module is not model.bridge.net[-1]):
                    module.bias.zero_()
        elif arm == 'router_all_biases_zero':
            for module in model.bridge.router.modules():
                if isinstance(module, nn.Linear):
                    module.bias.zero_()
        values = []
        for i, row in enumerate(x):
            value = torch.zeros_like(row[None]) if arm == 'zero_input' else row[None]
            kwargs = {}
            if arm == 'uniform_routes':
                kwargs['route_mode'] = 'uniform'
            elif arm == 'fixed_average_training_route':
                kwargs['fixed_weights'] = average_route
            elif arm == 'cyclic_training_routes':
                kwargs['fixed_weights'] = routes[(i+1)%3:((i+1)%3)+1]
            elif arm in ('zero_z', 'zero_z_and_decoder_biases_zero'):
                kwargs['route_mode'] = 'zero_z'
            values.append(trace(model, value, **kwargs)[-1])
        output = torch.cat(values)
        assert output.isfinite().all()
        outputs[arm] = output
        results[arm] = summary(output)

    final_bias_vector = head.up(head.bridge.net[-1].bias[None])*head.scale
    final_bias_identity_error = float((outputs['native']-outputs['decoder_final_bias_zero']-final_bias_vector).abs().max())
    up = head.up.weight.double()
    u, singular, _ = torch.linalg.svd(up, full_matrices=False)
    teacher = edits.double()
    projection = (teacher @ u) @ u.T
    subspace_error = ((teacher-projection).square().sum(1)/teacher.square().sum(1)).tolist()

    # Frozen critic preference at identical Gaussian draws; this diagnoses its
    # final snapshot only, not the historical cause of learning dynamics.
    noise = torch.randn(256, 5120, generator=torch.Generator().manual_seed(123))
    critic_probes = {}
    for arm, raw_delta in {**outputs, 'exact_teacher_oracle': edits,
                           'no_edit': torch.zeros_like(edits)}.items():
        rows = []
        for i in range(3):
            condition = conditions[i:i+1].expand(256, -1)
            real = critic(noise, condition)
            residual = (torch.zeros_like(target[i:i+1]) if arm == 'exact_teacher_oracle'
                        else (x[i:i+1]+raw_delta[i:i+1])-target[i:i+1])
            fake = critic(noise+residual, condition)
            if arm == 'exact_teacher_oracle':
                assert torch.equal(real, fake)
            paired = nn.functional.softplus(-fake)-nn.functional.softplus(-real)
            rows.append(dict(g_adv=float(nn.functional.softplus(-fake).mean()),
                             g_adv_minus_teacher_same_noise=float(paired.mean()),
                             paired_difference_sem=float(paired.std()/256**.5),
                             fake_minus_real_logit=float((fake-real).mean())))
        critic_probes[arm] = dict(rows=rows, mean_g_adv=sum(v['g_adv'] for v in rows)/3,
                                mean_g_adv_minus_teacher=sum(v['g_adv_minus_teacher_same_noise'] for v in rows)/3)

    assert all(torch.equal(v, head.state_dict()[k]) for k, v in original.items())
    assert not torch.is_grad_enabled() and not torch.cuda.is_initialized()
    for path, expected in opened.items():
        assert digest(path) == expected
    for path, expected in plan['source_sha256'].items():
        assert digest(path) == expected
    result = dict(script_sha256=digest(Path(__file__)), source_sha256=plan['source_sha256'],
        recipe_sha256=digest(plan_path), read_artifact_hashes=opened, arms=results,
        critic_loss=state['config']['recipe']['loss'],
        response_scale=state['config']['response_scale'], reference_sigma=state['config']['sigma'],
        base_state_unchanged=True, saved_artifacts_unchanged=True, optimizer_constructed=False,
        optimizer_updates=0, gpu_used=False, autograd_enabled=False,
        heldout_tensor_reads=0, heldout_selection=False,
        native_manual_trace_matches_forward_bitwise=True,
        route_weights=routes.tolist(), route_entropies=(-(routes*routes.log()).sum(1)).tolist(),
        route_pair_l1_distances=torch.cdist(routes, routes, p=1).tolist(),
        routed_z=zs.tolist(), features=features.tolist(),
        final_bias_vector_l2=float(final_bias_vector.norm()),
        final_bias_cosine_with_mean_edit=float(nn.functional.cosine_similarity(final_bias_vector, mean_delta)),
        final_bias_projection_coefficient=float((final_bias_vector*mean_delta).sum()/mean_delta.square().sum()),
        final_bias_additive_identity_max_error=final_bias_identity_error,
        up_singular_values=singular.tolist(),
        up_leading_squared_singular_fraction=float(singular[0].square()/singular.square().sum()),
        learned_up_subspace_teacher_error_lower_bounds=subspace_error,
        teacher_edit_matrix_rank=int(torch.linalg.matrix_rank(teacher)),
        nominal_output_rank=8, training_critic_snapshot_preferences=critic_probes,
        caveat='Nonlinear ablations interact; norm fractions are not additive attribution. Frozen critic preference is not training causality.',
        elapsed_seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],
                      'native':results['native'],'zero_input':results['zero_input'],
                      'decoder_final_bias_zero':results['decoder_final_bias_zero'],
                      'up_leading_squared_singular_fraction':result['up_leading_squared_singular_fraction']}))


if __name__ == '__main__':
    main()
