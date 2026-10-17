"""Stdlib-only recipe/source manifest builder. Does not load weights or Torch."""
from importlib.metadata import distribution
import importlib.util
import json
from pathlib import Path

ROOT = Path('/mnt/nvme/sliders/text-video-adapters-pg09-20261017')
CORE = Path('/mnt/nvme/sliders/text-video-core-3e94b13-20261017')
OVERLAY = Path('/tmp/alz-pg09-domain-proposal')
MODEL = Path('/mnt/nvme/sliders/qualification/models/tiny-qwen-da87bfb')
PYTHON = '/mnt/nvme/sliders/qualification/envs/pg090/bin/python'
PLAN = ROOT / 'evidence/pg09-text-video/tiny-native-cpu-smoke-proposal.json'
WORKER = ROOT / 'scripts/verify_tiny_pg09_native.py'
spec = importlib.util.spec_from_file_location('tiny_worker_metadata', WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
source_root = CORE / 'packages/particle-sliders-core/src/particle_sliders'
profile = json.loads((source_root / 'pg09_continuous_profile.json').read_text())
previous = json.loads((ROOT / 'evidence/pg09-text-video/bonsai-1200-cpu-admission-proposal-v2.json').read_text())
files = [ROOT / p for p in (
    'scripts/verify_tiny_pg09_native.py', 'tests/test_tiny_pg09_native_worker.py',
    'scripts/prepare_tiny_native_proposal.py',
    'scripts/verify_bonsai_pg09_cached.py', 'scripts/prepare_text_video_pg09_verification.py',
    'conceptmod/textsliders/tiny_llm_backend.py', 'conceptmod/textsliders/tiny_llm_particle.py',
    'conceptmod/textsliders/particle_bridge_gan.py', 'conceptmod/textsliders/pg09_text_video.py',
    'conceptmod/textsliders/pg09_backends.py', 'conceptmod/textsliders/pg09_readout_critic.py',
    'conceptmod/textsliders/data/prompts-tiny-llm.yaml',
    'evidence/pg09-text-video/tiny-native-smoke-dataset.json')]
files += [source_root / name for name in ('pg09.py', 'pg09_continuous_profile.json')]
files += [OVERLAY / 'particlegan' / name for name in profile['python_sources']]
files += [p for p in MODEL.iterdir() if p.is_file() and p.name != 'model.safetensors']
for package, names in {
    'transformers': ['models/qwen3/modeling_qwen3.py', 'models/qwen3/configuration_qwen3.py',
        'generation/utils.py', 'cache_utils.py', 'modeling_utils.py', 'configuration_utils.py',
        'tokenization_utils_base.py', 'tokenization_utils_tokenizers.py',
        'models/auto/modeling_auto.py', 'models/auto/tokenization_auto.py'],
    'torch': ['nn/modules/linear.py', 'nn/modules/activation.py', 'nn/functional.py'],
}.items():
    dist = distribution(package)
    files += [Path(dist.locate_file(package + '/' + name)) for name in names]
sources = {str(p): worker.digest(p) for p in sorted(set(files))}
env = dict(previous['environment'], TOKENIZERS_PARALLELISM='false', RAYON_NUM_THREADS='1',
           PYTHONPATH=f'{OVERLAY}:{source_root.parent}:{ROOT}:{ROOT}/scripts')
common = ['env'] + [k + '=' + v for k, v in env.items()]
commands = [common + ['timeout', str(worker.LIMITS[mode + '_seconds']), PYTHON, str(WORKER),
                     '--plan', str(PLAN), '--mode', mode] for mode in ('produce', 'restore')]
plan = dict(schema='tiny-qwen-native-pg09-cpu-smoke-v1',
    status='PENDING independent QA and explicit coordinator process admission',
    checkout=str(ROOT), source_base_commit='7f018780c073790e1d706c31716d720302750a08',
    core=str(CORE), core_commit='3e94b13eb5d6cd1ceee3f018f019a68603f35696', overlay=str(OVERLAY),
    source_profile=profile, domain=worker.DOMAIN, training=worker.TRAINING,
    resolved_public_recipe=previous['resolved_public_recipe'], critic=dict(hidden=16,
        condition='frozen native neutral last-token vector + fixed sign + no-time coordinate',
        objective='raw logits; public shared adversarial+continuous penalty'),
    generation=worker.GENERATION, signed_strengths=[0, -1, 0.5, 1],
    negative_strength_interpretation='extrapolation only; no negative teacher or antonym claim',
    model=str(MODEL), model_id='Qwen/Qwen3-0.6B-Base',
    model_revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
    model_weight_sha256='cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba',
    model_weight_bytes=1192135096, model_license='Apache-2.0; existing local assets only',
    model_placement='CPU FP32 frozen base and FP32 actual native attention adapters',
    interpreter=PYTHON, runtime_versions=worker.runtime_versions(),
    runtime_controls=dict(deterministic_algorithms=True, cpu_threads=1, interop_threads=1,
                          cuda_visible_devices='', attention='native Transformers resolved configuration'),
    environment=env, source_sha256=sources, limits=worker.LIMITS,
    output='/mnt/nvme/sliders/qualification/runs/tiny-qwen-native-pg09-cpu-smoke-20261017-v1/bispf_optimized',
    output_filesystem=dict(device='/dev/nvme0n1p2', mountpoint='/mnt/nvme'),
    dataset=str(ROOT / 'evidence/pg09-text-video/tiny-native-smoke-dataset.json'),
    split_policy=dict(train='three original native training pairs; neutral student, frozen positive target',
        validation='three historically inspected heldouts; validation only, no unseen claim',
        final_holdout='three newly declared pairs; no tokenization/forward/generation in smoke',
        canaries='three original unconditional native prompts; diagnostics only',
        fluency='one fixed prose completion; real text, no automatic semantic-quality certification'),
    evaluation=dict(hidden_vectors='all train/validation/canaries at0/-1/.5/+1, initial and final',
        text='first training, first validation, first canary and fluency at0/-1/.5/+1, initial and final',
        losses='Gadv+required public prior only; Dadv+public continuous penalty only',
        off='exact unattached base hidden states and greedy text before/after updates',
        resume='step2->4 separate process: exact full packet/loss/vectors/text; persist restored packet',
        export='zero all adapter leaves before export reload and repeat exact vectors/text'),
    stop_policy=['any command/resource/source/domain/parity/nonfinite/connectivity failure stops sequence',
        'no retry, extension, parameter override, sweep or fallback',
        'smoke results inform a separately frozen longer-run proposal; no automatic admission'],
    commands=commands,
    metadata_only_preflight=common + [PYTHON, str(WORKER), '--plan', str(PLAN), '--mode', 'preflight'],
    gpu_requested=False, server_requested=False, downloads_requested=False, installed_package_changes=False,
    optimized_mse=False, automatic_retry=False, semantic_quality_qualified=False,
    pretrained_weights_loaded_during_preparation=False)
Path('/tmp/tiny-native-cpu-smoke-proposal.json').write_text(json.dumps(plan, indent=2) + '\n')
print(json.dumps(dict(source_files=len(sources), output=plan['output'], model_loaded=False)))
