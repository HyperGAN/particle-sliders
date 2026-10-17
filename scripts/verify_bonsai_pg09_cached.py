"""Owner-admitted CPU canary using existing official cached readouts only.

Preflight is metadata-only. Produce performs four shared updates, checkpointing
after two; restore replays the final two in a separate process. No server,
GGUF/model loader, network, CUDA, legacy trainer or installed-package mutation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

CORE_SHA256 = '426abc25323aea96be91e92f1f4101063d5b3c4062f042e869c729027b4f9256'
ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = Path('/mnt/nvme/sliders/qualification')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(core):
    from prepare_text_video_pg09_verification import audit_manifest, ASSETS
    module = core / 'packages/particle-sliders-core/src/particle_sliders/pg09.py'
    if digest(module) != CORE_SHA256:
        raise ValueError('core source differs from the frozen owner/QA model-admission pin')
    assets = audit_manifest(QUALIFICATION / 'models/bonsai-b072e1d', ASSETS['bonsai'])
    preparation = ROOT / 'evidence/pg09-text-video/assets-prepared.json'
    selected = json.loads(preparation.read_text())['bonsai_cached_readout_metadata']
    recipe = QUALIFICATION / 'runs/bonsai-caption-native-1200-20261017/recipe.json'
    previous = json.loads(recipe.read_text())
    identity = previous['frozen_readout_identity']
    identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_digest != previous['base_identity']:
        raise ValueError('archived official readout identity differs')
    cache = QUALIFICATION / 'models/bonsai-caption-readouts-20261017'
    if not cache.is_dir():
        raise FileNotFoundError('existing immutable readout cache required; never populate it here')
    for item in selected:
        path = Path(item['metadata_path'])
        if path.parent != cache or digest(path) != item['metadata_sha256']:
            raise ValueError('readout metadata differs from frozen preparation')
        metadata = json.loads(path.read_text())
        if (metadata['identity'] != identity_digest or not metadata.get('independent_requests_exact')
                or path.with_suffix('.pt').stat().st_size != item['tensor_bytes']):
            raise ValueError('official readout cache provenance/size differs')
    return dict(worker_sha256=digest(Path(__file__)), core_sha256=CORE_SHA256,
                assets_manifest_sha256=assets['manifest_sha256'], readout_identity=identity_digest,
                frozen_readout_identity=identity, preparation_sha256=digest(preparation),
                archived_recipe_sha256=digest(recipe), cache=str(cache), cached_readouts=len(selected),
                model_loaded=False, tensor_payload_loaded=False, gpu_used=False,
                permitted_scope='pending owner admission: cached-real-readout head only, CPU one thread')


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def execute(args, receipt):
    # Process execution is admitted by the core owner for an exact command.
    # The script cannot grant itself model/process admission.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(args.overlay.resolve()),
                   str(args.core.resolve() / 'packages/particle-sliders-core/src'), str(ROOT), str(ROOT / 'scripts')]
    import torch
    from particle_sliders.pg09 import ConditionalPG09, validate_pg09_domain
    validate_pg09_domain((1, 5120), response_domain='continuous_vector',
                         pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    from conceptmod.textsliders.caption_recovery import FrozenReadouts
    from conceptmod.textsliders.bonsai_gguf_particle import BonsaiParticleHead
    from conceptmod.textsliders.pg09_text_video import BonsaiReadoutResponseAdapter, BonsaiResponseRow
    from conceptmod.textsliders.pg09_readout_critic import ConditionalReadoutCritic
    from caption_quality import evaluate
    import yaml
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cache = FrozenReadouts(None, receipt['cache'], receipt['readout_identity'])
    head = BonsaiParticleHead().float()
    identity = {key: value for key, value in receipt.items()
                if key not in ('permitted_scope', 'model_loaded', 'tensor_payload_loaded', 'gpu_used')}
    identity['evaluation_source_sha256'] = digest(ROOT / 'scripts/caption_quality.py')
    prompts = ROOT / 'conceptmod/textsliders/data/prompts-bonsai-gguf.yaml'
    identity['prompts_sha256'] = digest(prompts)
    adapter = BonsaiReadoutResponseAdapter(cache, head, model_identity=identity,
                    pg_source_profile='alz-continuous-v1', penalty_input_domain='continuous')
    data = yaml.safe_load(prompts.read_text())
    rows = [adapter.prepare(BonsaiResponseRow(r['neutral'], r['positive'])) for r in data]
    conditions = torch.cat([torch.cat((r.neutral.flatten(), torch.tensor([1., 0.])))[None] for r in rows])
    critic = ConditionalReadoutCritic(5120, conditions.shape[1])
    game = ConditionalPG09(adapter, rows, critic, conditions=conditions, particles=adapter.particles,
                           **adapter.domain_options, preset=args.preset, total_steps=4, batch_size=2, seed=7)
    initial = {k: v.detach().clone() for k, v in adapter.network.state_dict().items()}
    initial_d = {k: v.detach().clone() for k, v in critic.state_dict().items()}
    initial_bank = head.particles.detach().clone()
    frozen = [t.detach().clone() for r in rows for t in (r.target, r.neutral)]
    context = dict(backend=cache, network=head, fixed=data)

    def sample():
        before, rng = game.state_dict(), torch.get_rng_state().clone()
        result = evaluate(context, 'bonsai')
        result[0]['note'] = ('Shared PG09 neutral-conditioned student; frozen positive teacher. '
                             'Squared errors are held-out diagnostics only; no semantic quality claim.')
        equal(before, game.state_dict())
        assert torch.equal(rng, torch.get_rng_state())
        return result

    def step():
        result = game.step()
        assert abs(result['g_loss'] - result['g_adv'] - result['prior_regularization']) < 1e-5
        assert result['d_pen'] > 0
        assert game.penalty.last_stats['input_domain'] == 'continuous'
        return result

    def save(value, name):
        path = args.output / name
        pending = path.with_suffix('.pending')
        torch.save(value, pending)
        pending.replace(path)

    def forbidden(*a, **kw):
        raise AssertionError('optimized MSE forbidden in shared GAN canary')
    torch.nn.functional.mse_loss = forbidden
    if args.mode == 'produce':
        args.output.mkdir(parents=True, exist_ok=False)
        initial_sample = sample()
        first = [step(), step()]
        save(game.state_dict(), 'checkpoint.pt')
        later = [step(), step()]
        sample_result = sample()
        export = game.export_state_dict()
        save(export, 'inference-export.pt')
        save(dict(metrics=later, state=game.state_dict(), sample=sample_result), 'expected.pt')
        result = dict(receipt, mode='produce', preset=args.preset, completed_steps=4,
                      metrics=first + later, heldout_initial=initial_sample[0], heldout_final=sample_result[0],
                      adapter_changed=any(not torch.equal(initial[k], v) for k, v in adapter.network.state_dict().items()),
                      critic_changed=any(not torch.equal(initial_d[k], v) for k, v in critic.state_dict().items()),
                      bank_changed=not torch.equal(initial_bank, head.particles))
        assert result['adapter_changed'] and result['critic_changed'] and result['bank_changed']
    else:
        if (args.output / 'restore-receipt.json').exists():
            raise FileExistsError('restore receipt exists; no automatic retry')
        game.load_state_dict(torch.load(args.output / 'checkpoint.pt', weights_only=True))
        later = [step(), step()]
        expected = torch.load(args.output / 'expected.pt', weights_only=True)
        assert later == expected['metrics']
        equal(game.state_dict(), expected['state'])
        equal(sample(), expected['sample'])
        game.load_export_state_dict(torch.load(args.output / 'inference-export.pt', weights_only=True))
        rng = torch.get_rng_state().clone()
        equal(evaluate(context, 'bonsai')[1:], expected['sample'][1:])
        assert torch.equal(rng, torch.get_rng_state())
        result = dict(receipt, mode='restore', preset=args.preset, completed_steps=4,
                      exact_full_state=True, exact_losses=True, exact_heldout_vectors=True,
                      exact_export_reload=True)
    assert all(torch.equal(saved, t) for saved, t in zip(frozen, [t for r in rows for t in (r.target, r.neutral)]))
    assert not torch.cuda.is_initialized()
    result.update(scope='real official frozen readouts; native torch-side head only',
                  frozen_readouts_unchanged=True, tensor_payload_loaded=True,
                  gguf_weights_loaded=False, server_started=False, gpu_used=False,
                  quality_qualified=False, installed_package_modified=False)
    (args.output / f'{args.mode}-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('mode', 'preset', 'completed_steps', 'scope', 'gpu_used', 'quality_qualified')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'produce', 'restore'], required=True)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=['bispf', 'bispf_optimized'], default='bispf_optimized')
    args = parser.parse_args()
    receipt = preflight(args.core.resolve())
    if args.mode == 'preflight':
        print(json.dumps(receipt, indent=2))
        return
    execute(args, receipt)


if __name__ == '__main__':
    main()
