"""Shared bounded CPU AR qualification and inference probes for PG0.9.1.

Only ConditionalPG09 performs adversarial updates. Fixtures are real native
random-weight AR models, not pretrained music or acoustic-quality evidence.
Native model loading, token sampling, acoustic decoding and GPU dispatch are
not CLI capabilities. An admitted caller may supply its own native components
to build_audio_run; this module never admits or launches a device/process.
"""
from dataclasses import dataclass
from copy import deepcopy
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from .pg09 import ConditionalPG09, validate_pg09_domain
from .pg09_audio import (AudioRow, AudioResponseCritic, Music3ARResponseAdapter,
                         YuE2ARResponseAdapter, audio_conditions)

QUALIFICATION_SCHEMA = 'audio-pg091-cpu-qualification-v1'
EXPORT_SCHEMA = 'audio-pg091-cpu-inference-v1'
FACTORY_SCHEMA = 'audio-native-random-v1'


def _require_legacy_pg091():
    from .pg09 import PG_VERSION
    if PG_VERSION != '0.9.1' or importlib.metadata.version('particlegan') != '0.9.1':
        raise RuntimeError('legacy audio-pg091 entrypoints require the exact 0.9.1 core line; use response sessions on other lines')


@dataclass
class AudioRun:
    adapter: object
    model: object
    rows: tuple
    heldout: tuple
    game: ConditionalPG09
    factory: dict | None = None


def capabilities(family):
    _require_legacy_pg091()
    if family not in ('music3', 'yue2'):
        raise ValueError('audio family must be music3 or yue2')
    return dict(schema='audio-pg091-capabilities-v1', family=family,
                engine='particle_sliders.pg09.ConditionalPG09',
                particlegan_required='0.9.1', cpu_random_native_ar=(family == 'music3'),
                trained_scales=[1.0], probe_scales=[0.0, 1.0],
                native_components_factory='build_audio_run',
                native_pretrained_cli=False, native_audio_render=False,
                encoder=False, flow=False, gpu_dispatch=False,
                old_checkpoint_migration=False, quality_qualified=False)


def plan_native(*, family, model_dir, manifest, train_prompts, heldout_prompts,
                out, until=3, total_steps=20, pg_source_profile='release',
                penalty_input_domain='auto', preset='bispf_optimized', seed=7,
                resume=None):
    """Verify local bytes and describe pending native work; no dispatch/output.

    Hashing reads bytes without deserializing tensors. Prompt-byte validation
    does not assert native tokenizer/history or acoustic qualification.
    """
    _require_legacy_pg091()
    if family != 'music3':
        raise ValueError('native YuE2 planning requires its external admitted producer; use capabilities')
    if preset not in ('bispf', 'bispf_optimized') or type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('supported public preset and nonnegative integer seed required')
    if resume is not None:
        raise ValueError('native checkpoint migration/resume needs a separately qualified producer; no automatic retarget')
    if type(until) is not int or type(total_steps) is not int or not 1 <= until <= total_steps <= 32:
        raise ValueError('bounded absolute until/horizon required')
    root, manifest, out = Path(model_dir).resolve(), Path(manifest), Path(out)
    if out.exists():
        raise FileExistsError('native plan requires an absent future output')
    document = json.loads(manifest.read_text())
    entries = [entry for entry in document['files'] if entry['filename'].startswith(('language_model/', 'tokenizer/'))]
    if not entries or not any(e['filename'] == 'language_model/config.json' for e in entries):
        raise ValueError('native LM/tokenizer manifest required')
    identities = {}
    for entry in entries:
        name = Path(entry['filename'])
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('manifest filename must be relative without traversal')
        value = hashlib.sha256()
        with (root / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(block)
        if value.hexdigest() != entry['sha256']:
            raise ValueError('native staged file hash differs: ' + str(name))
        identities[str(name)] = value.hexdigest()
    config = json.loads((root / 'language_model/config.json').read_text())
    gate = validate_pg09_domain((1, config['hidden_size']), response_domain='continuous_vector',
        pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
    prompt_pins = {}
    for name, path, count in (('train', train_prompts, 4), ('heldout', heldout_prompts, 2)):
        path = Path(path)
        data = path.read_bytes()
        if path.suffix == '.json':
            contents = json.loads(data)
        else:
            import yaml
            contents = yaml.safe_load(data)
        rows = contents['rows']
        if len(rows) != count or any(not isinstance(r, dict) or
                any(not isinstance(r.get(k), str) or not r[k].strip() for k in ('neutral', 'positive', 'lyrics')) or
                r['neutral'] == r['positive'] for r in rows):
            raise ValueError('fixed distinct native prompt rows required')
        prompt_pins[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), count=count)
    return dict(schema='audio-pg091-native-plan-v1', status='PLAN_ONLY_NOT_ADMITTED',
                family=family, dispatch_allowed=False, output_directory=str(out.resolve()),
                output_absent=True, model_directory=str(root), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                source_domain_gate=gate, model_files=identities, prompts=prompt_pins,
                until=until, total_steps=total_steps, preset=preset, seed=seed,
                resume=None, native_audio_rendered=False,
                pending=['root guardian/card and fresh admission', 'native token/history producer qualification',
                         'native pretrained updates/resume/export', 'native acoustic implementation/parity/listening'])


def build_audio_run(family, backend, slider, rows, heldout, *, model_identity,
                    preset='bispf_optimized', seed=7, total_steps=4,
                    checkpointing=True, pg_source_profile='release',
                    penalty_input_domain='auto'):
    """Public factory for explicit caller-supplied, already admitted components.

    It owns no loading, optimizer, custom objective, device admission or loop.
    Model/token/asset verification belongs to that caller and is checkpoint-bound
    through model_identity and the exact rows. Automatic CLI reload is fixture-only.
    """
    _require_legacy_pg091()
    if family not in ('music3', 'yue2'):
        raise ValueError('unknown audio family')
    model = backend.model
    if getattr(model.config, 'model_type', None) != ('qwen3' if family == 'music3' else 'yue2'):
        raise ValueError('native model configuration differs from declared audio family')
    options = dict(pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
    validate_pg09_domain((1, int(model.config.hidden_size)), response_domain='continuous_vector', **options)
    cls = Music3ARResponseAdapter if family == 'music3' else YuE2ARResponseAdapter
    adapter = cls(model if family == 'music3' else backend, slider,
                  model_identity=deepcopy(model_identity),
                  checkpoint_modules=list(model.model.layers) if checkpointing else (), **options)
    fixed = tuple(adapter.prepare(row) for row in rows)
    probes = tuple(adapter.prepare(row) for row in heldout)
    if len(fixed) < 2 or not probes:
        raise ValueError('at least two fixed training rows and a heldout row required')
    conditions = audio_conditions(fixed)
    critic = AudioResponseCritic(model.config.hidden_size, conditions.shape[1]).to(next(model.parameters()).device)
    game = ConditionalPG09(adapter, fixed, critic, conditions=conditions,
                          particles=adapter.particles, response_domain='continuous_vector',
                          preset=preset, batch_size=2, seed=seed, total_steps=total_steps, **options)
    return AudioRun(adapter, model, fixed, probes, game)


@torch.random.fork_rng(devices=[])
def make_audio_game(family='music3', *, preset='bispf_optimized', seed=7,
                    total_steps=4, width=32, pg_source_profile='release',
                    penalty_input_domain='auto', checkpointing=True):
    """Legacy random Qwen3 AR; YuE2 construction remains product-owned.

    Default Music3 fixture preserves native audio-start ID151669/vocab151670.
    Nondefault-width fixtures use vocab256 and are explicitly geometry-only.
    YuE2 callers supply native objects to build_audio_run or use response sessions.
    No replacement model is substituted. Seed controls streams; base/adapter
    seeds are fixed and included in source-bound metadata for fresh reload.
    Ambient CPU RNG and caller thread policy are preserved; CUDA is not seeded.
    """
    _require_legacy_pg091()
    if family not in ('music3', 'yue2') or type(width) is not int or width < 16 or width % 2:
        raise ValueError('known audio family and positive even native width >=16 required')
    if type(total_steps) is not int or not 1 <= total_steps <= 32:
        raise ValueError('bounded CPU fixture horizon must be in [1,32]')
    options = dict(pg_source_profile=pg_source_profile, penalty_input_domain=penalty_input_domain)
    validate_pg09_domain((1, width), response_domain='continuous_vector', **options)
    torch.random.default_generator.manual_seed(23)
    if family == 'music3':
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from .music3_native import ParticleNetwork
        vocab, end = (151670, 151669) if width == 32 else (256, 251)
        model = Qwen3ForCausalLM(Qwen3Config(hidden_size=width, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, vocab_size=vocab, attention_dropout=0., use_cache=False,
            attn_implementation='eager')).eval().requires_grad_(False)
        # Only the genuine model class is reused, never its legacy trainer.
        from types import SimpleNamespace
        backend = SimpleNamespace(model=model)
        rows = (AudioRow((11, 13, end), (29, 31, end)),
                AudioRow((23, 37, 41, end), (53, 59, 61, end)))
        heldout = (AudioRow((67, 71, end), (73, 79, end)),)
        torch.random.default_generator.manual_seed(77)
        slider = ParticleNetwork(model)
        token_policy = ('native_audio_start_151669' if width == 32 else
                        'geometry_only_vocab256_no_native_audio_token_claim')
    else:
        raise RuntimeError('YuE2 native construction belongs to its product; supply the native backend/slider to build_audio_run or register a ResponseBackend')
    factory = dict(family=family, preset=preset, seed=seed, total_steps=total_steps,
                   width=width, checkpointing=checkpointing, **options)
    run = build_audio_run(family, backend, slider, rows, heldout,
        model_identity=dict(fixture=FACTORY_SCHEMA, base_seed=23, adapter_seed=77,
                            token_policy=token_policy, factory=factory,
                            worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        preset=preset, seed=seed, total_steps=total_steps, checkpointing=checkpointing, **options)
    run.factory = factory
    return run


def frozen_base_digest(run):
    _require_legacy_pg091()
    owned = {id(p) for p in run.adapter.network.parameters()}
    digest = hashlib.sha256()
    for name, parameter in run.model.named_parameters():
        if id(parameter) in owned:
            continue
        if parameter.requires_grad:
            raise ValueError('unowned trainable base parameter')
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load(path):
    return torch.load(Path(path), map_location='cpu', weights_only=True)


def _exact(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_exact(x, y) for x, y in zip(a, b))
    return a == b


@torch.random.fork_rng(devices=[])
def probe_export(path, *, scales=(0., .5, 1.), expected_family=None):
    """Safe public export reload with actual heldout hidden responses, no audio."""
    _require_legacy_pg091()
    scales = tuple(float(s) for s in scales)
    if not scales or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scales):
        raise ValueError('finite unipolar probe strengths in [0,1] required')
    packet = _load(path)
    if packet.get('schema') != EXPORT_SCHEMA:
        raise ValueError('inference packet required; full checkpoint cannot be substituted')
    if expected_family is not None and packet['factory']['family'] != expected_family:
        raise ValueError('export audio family differs from requested inference entry point')
    run = make_audio_game(**packet['factory'])
    run.game.load_export_state_dict(packet['export'])
    samples = {str(s): [run.adapter.sample(row, scale=s) for row in run.heldout] for s in scales}
    off = [run.adapter.sample(row, scale=0) for row in run.heldout]
    if not all(torch.equal(a, row.neutral) for a, row in zip(off, run.heldout)):
        raise ValueError('scale zero did not replay the frozen native base')
    return dict(schema='audio-pg091-hidden-probes-v1', samples=samples,
                off_exact=True, audio_rendered=False, quality_qualified=False)


def _validate_metric_history(metrics, completed_steps):
    """Validate the complete GAN receipt before a saved game can be restored."""
    error = 'public GAN-only metric decomposition/history/positive penalty gate failed'
    if (not isinstance(metrics, list) or type(completed_steps) is not int
            or completed_steps < 0 or len(metrics) != completed_steps):
        raise ValueError(error)
    keys = ('g_loss', 'g_adv', 'prior_regularization', 'd_loss', 'd_adv', 'd_pen')
    for index, metric in enumerate(metrics):
        if (not isinstance(metric, dict) or type(metric.get('step')) is not int
                or metric['step'] != index + 1
                or any(type(metric.get(key)) not in (int, float)
                       or not math.isfinite(metric[key]) for key in keys)
                or metric['d_pen'] <= 0
                or not math.isclose(metric['g_loss'], metric['g_adv'] + metric['prior_regularization'], rel_tol=1e-5, abs_tol=1e-7)
                or not math.isclose(metric['d_loss'], metric['d_adv'] + metric['d_pen'], rel_tol=1e-5, abs_tol=1e-7)):
            raise ValueError(error)


def run_training(run, *, until, out, resume=None):
    """Bounded CPU orchestration; the sole update call is shared game.step().

    Output: checkpoint.pt (complete resume packet), qualification.pt (same packet),
    adapter-export.pt (public inference export + exact fixture factory), receipt.json.
    Neither inference export nor historical checkpoints can resume this new cohort.
    """
    _require_legacy_pg091()
    if run.factory is None or next(run.model.parameters()).device.type != 'cpu':
        raise ValueError('CLI training requires the bounded random native CPU factory')
    if type(until) is not int or not 1 <= until <= run.factory['total_steps']:
        raise ValueError('until is an absolute completed-step count within the fixed horizon')
    out = Path(out)
    if out.exists():
        raise FileExistsError('fresh output directory required')
    base = frozen_base_digest(run)
    metrics = []
    if resume is not None:
        previous = _load(resume)
        if previous.get('schema') != QUALIFICATION_SCHEMA or previous.get('factory') != run.factory:
            raise ValueError('complete same-cohort checkpoint/factory required to resume')
        if previous.get('frozen_base_sha256') != base or not isinstance(previous.get('game'), dict):
            raise ValueError('checkpoint base or complete metric history differs')
        _validate_metric_history(previous.get('metrics'), previous['game'].get('completed_steps'))
        if until <= previous['game']['completed_steps']:
            raise ValueError('until must advance beyond the restored completed step')
        run.game.load_state_dict(previous['game'])
        metrics = deepcopy(previous['metrics'])
    if until <= run.game.completed_steps:
        raise ValueError('until must advance beyond the restored completed step')
    bank_before = run.adapter.particles.detach().clone()
    parameters_before = [p.detach().clone() for p in run.adapter.network.parameters()]
    critic_before = [p.detach().clone() for p in run.game.critic.parameters()]
    while run.game.completed_steps < until:
        metrics.append(run.game.step())
    _validate_metric_history(metrics, run.game.completed_steps)
    if frozen_base_digest(run) != base:
        raise ValueError('frozen native base changed during adversarial updates')
    delta = (run.adapter.particles.detach() - bank_before).square().mean().sqrt().item()
    response = run.adapter.generated(run.rows[0])
    direction = torch.linspace(-1, 1, response.shape[-1], device=response.device)
    gradient = torch.autograd.grad((response * direction).sum(), run.adapter.particles)[0].abs().sum().item()
    if not math.isfinite(delta) or delta <= 0 or not math.isfinite(gradient) or gradient <= 0:
        raise ValueError('actual routed bank must change and participate in the response')
    changed = sum(not torch.equal(p, q) for p, q in zip(parameters_before, run.adapter.network.parameters()))
    critic_changed = sum(not torch.equal(p, q) for p, q in zip(critic_before, run.game.critic.parameters()))
    if not changed or not critic_changed:
        raise ValueError('both generator and critic must change')
    probes = [run.adapter.sample(row) for row in run.heldout]
    if not all(torch.equal(run.adapter.sample(row, scale=0), row.neutral) for row in run.heldout):
        raise ValueError('off replay differs')
    exported = dict(schema=EXPORT_SCHEMA, factory=deepcopy(run.factory), export=run.game.export_state_dict())
    # Reconstruct and replay export without consuming the training RNG.
    with torch.random.fork_rng(devices=[]):
        fresh = make_audio_game(**run.factory)
        fresh.game.load_export_state_dict(exported['export'])
        if not _exact(probes, [fresh.adapter.sample(row) for row in fresh.heldout]):
            raise ValueError('fresh native public export replay differs')
    packet = dict(schema=QUALIFICATION_SCHEMA, factory=deepcopy(run.factory),
                  game=run.game.state_dict(), metrics=metrics, probes=probes,
                  export=exported['export'], frozen_base_sha256=base)
    receipt = dict(schema=QUALIFICATION_SCHEMA, completed_steps=until,
        bank_delta_rms=delta, response_bank_gradient_l1=gradient,
        changed_adapter_tensors=changed, changed_critic_tensors=critic_changed,
        off_exact=True, export_exact=True, frozen_base_sha256=base,
        cpu_random_native_only=True, native_audio_rendered=False, quality_qualified=False,
        particlegan=importlib.metadata.version('particlegan'), factory=run.factory)
    out.mkdir(parents=True)
    torch.save(packet, out / 'checkpoint.pt')
    torch.save(packet, out / 'qualification.pt')
    torch.save(exported, out / 'adapter-export.pt')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    (out / 'progress.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    return packet


def main(argv=None):
    _require_legacy_pg091()
    argv = list(argv) if argv is not None else __import__('sys').argv[1:]
    if argv and argv[0] == 'plan':
        parser = argparse.ArgumentParser(description='Offline native Music3 plan; no process/device dispatch', allow_abbrev=False)
        parser.add_argument('--family', choices=('music3',), required=True)
        for name in ('model-dir', 'manifest', 'train-prompts', 'heldout-prompts', 'out'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--until', type=int, default=3)
        parser.add_argument('--total-steps', type=int, default=20)
        parser.add_argument('--preset', choices=('bispf', 'bispf_optimized'), default='bispf_optimized')
        parser.add_argument('--seed', type=int, default=7)
        parser.add_argument('--resume', type=Path)
        parser.add_argument('--pg-source-profile', default='release')
        parser.add_argument('--penalty-input-domain', default='auto')
        result = plan_native(**vars(parser.parse_args(argv[1:])))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return result
    probe = bool(argv and argv[0] == 'probe')
    if probe:
        parser = argparse.ArgumentParser(description='Public export hidden probes; no native audio', allow_abbrev=False)
        parser.add_argument('--export', type=Path, required=True)
        parser.add_argument('--family', choices=('music3', 'yue2'))
        parser.add_argument('--scales', default='0,0.5,1')
        parser.add_argument('--out', type=Path, required=True)
        args = parser.parse_args(argv[1:])
        if args.out.exists():
            raise FileExistsError('fresh probe output file required')
        result = probe_export(args.export, scales=args.scales.split(','), expected_family=args.family)
        args.out.mkdir(parents=True)
        torch.save(result, args.out / 'hidden-probes.pt')
        (args.out / 'probe-card.json').write_text(json.dumps(dict(schema=result['schema'],
            off_exact=result['off_exact'], scales=list(result['samples']), audio_rendered=False,
            quality_qualified=False), indent=2) + '\n')
        return result
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--family', choices=('music3', 'yue2'), required=True)
    parser.add_argument('--capabilities', action='store_true')
    parser.add_argument('--native-plan', action='store_true')
    parser.add_argument('--fixture', '--dummy', action='store_true')
    parser.add_argument('--preset', choices=('bispf', 'bispf_optimized'), default='bispf_optimized')
    parser.add_argument('--total-steps', type=int, default=4)
    parser.add_argument('--until', '--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--resume', '--resume_state', type=Path)
    parser.add_argument('--out', '--save_dir', type=Path)
    parser.add_argument('--pg-source-profile', default='release')
    parser.add_argument('--penalty-input-domain', default='auto')
    args = parser.parse_args(argv)
    if args.capabilities or args.native_plan:
        print(json.dumps(capabilities(args.family), sort_keys=True))
        return 0
    if not args.fixture or args.out is None:
        parser.error('only explicit --fixture --out CPU qualification is executable; native loading/render/GPU dispatch are unsupported; use --capabilities')
    if not 1 <= args.until <= args.total_steps:
        parser.error('--until/--steps must be within the unchanged --total-steps horizon')
    run = make_audio_game(args.family, preset=args.preset, seed=args.seed, total_steps=args.total_steps,
        width=args.width, pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
    run_training(run, until=args.until, out=args.out, resume=args.resume)
    return 0


def legacy_main(family, action='train', argv=None):
    _require_legacy_pg091()
    argv = list(argv) if argv is not None else __import__('sys').argv[1:]
    if family not in ('music3', 'yue2'):
        raise ValueError('unknown legacy audio entry point family')
    for index, option in enumerate(argv):
        if option == '--family':
            selected = argv[index + 1] if index + 1 < len(argv) else None
        elif option.startswith('--family='):
            selected = option.split('=', 1)[1]
        else:
            continue
        if selected != family:
            raise ValueError('legacy audio entry point family cannot be overridden')
    return main(['probe', '--family', family, *argv] if action == 'probe' else ['--family', family, *argv])


if __name__ == '__main__':
    torch.set_num_threads(1)
    result = main()
    raise SystemExit(result if isinstance(result, int) else 0)
