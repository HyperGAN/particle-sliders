"""CPU-auditable proposals and admitted-component bridges for remaining failures.

No model loader, device admission, process control, custom loss or optimizer.
The resource owner supplies frozen native components and admits every live call.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch

from .pg09_backends import FlowSchedule, PreparedResponse, ResponseRow, SupraRuntimeResponseAdapter
from .pg09_moss_campaign import (
    CampaignFieldCritic, build_campaign_run, canonical_digest, file_digest, _seed_initialization_rng,
)

FINAL_FORM_SCALE = 0.2522578997899252
SUPRA_SCALE = 0.32844100447280516
SUPRA_MODEL_REVISION = '10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b'
SUPRA_TEXT_REVISION = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
SUPRA_CONFIG_SHA256 = 'd32639ad9084f60d13faea1ef0e4d92b4bfc2e0420bc0fd15f4992c6270edd5b'


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


def required_supra_prompts(config):
    prompts = {''} | set(config['preservation'])
    for row in config['rows']:
        prompts.update((row['neutral'], row['positive']))
    for row in config['verification']:
        prompts.update((row['prompt'], row.get('teacher', row['prompt'])))
    return sorted(prompts)


def export_admitted_supra_contexts(runtime, config, destination, *, runtime_identity):
    """Explicit preparation phase, called only after native runtime admission.

    Reuses native encode (including stored unconditional); never loads components.
    No implicit re-encoding is allowed by the training bridge below.
    """
    from safetensors.torch import save_file
    prompts = required_supra_prompts(config)
    if len(prompts) > 19:
        raise ValueError('Supra preparation exceeds the19 unique-context encode budget')
    if not runtime_identity.get('model_revision') or not runtime_identity.get('text_revision'):
        raise ValueError('verified model and native T5 revision identity required')
    runtime_source = inspect.getsourcefile(type(runtime))
    if runtime_source is None:
        raise ValueError('native runtime must have an inspectable pinned source')
    destination = Path(destination)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    tensors, records = {}, []
    for index, prompt in enumerate(prompts):
        context, mask = runtime.encode(prompt)
        context, mask = context.detach().cpu().contiguous(), mask.detach().cpu().contiguous()
        if (context.ndim != 3 or context.shape[0] != 1 or mask.shape != context.shape[:2] or
                not torch.isfinite(context).all() or not torch.isfinite(mask).all() or not mask.any()):
            raise ValueError('invalid frozen native context')
        tensors[f'e{index}'], tensors[f'm{index}'] = context, mask
        records.append(dict(prompt=prompt, embedding=tensor_identity(context), mask=tensor_identity(mask)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination))
    receipt = dict(schema='supra-frozen-contexts-v1', sha256=file_digest(destination), records=records,
                   config_sha256=canonical_digest(config), unique_contexts=len(prompts), max_encode_calls=19,
                   stored_unconditional=runtime.stored_unconditional, runtime_identity=deepcopy(runtime_identity),
                   runtime_source=str(Path(runtime_source).resolve()),
                   runtime_source_sha256=file_digest(runtime_source), torch_version=str(torch.__version__),
                   policy='native pinned T5 encoding; original stored unconditional retained')
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt



def validate_live_supra_context_preparation(config, runtime_identity):
    """Strict reviewed live contract; no encoding, output or runtime creation."""
    if len(required_supra_prompts(config)) != 19 or canonical_digest(config) != SUPRA_CONFIG_SHA256:
        raise ValueError('live Supra preparation requires the exact reviewed19-context config')
    if (runtime_identity.get('model_revision') != SUPRA_MODEL_REVISION or
            runtime_identity.get('text_revision') != SUPRA_TEXT_REVISION):
        raise ValueError('live Supra model/T5 revisions differ from reviewed preparation')
    return dict(config_sha256=SUPRA_CONFIG_SHA256, unique_contexts=19,
                model_revision=SUPRA_MODEL_REVISION, text_revision=SUPRA_TEXT_REVISION)


def export_live_supra_contexts(runtime, config, destination, *, runtime_identity):
    """Owner-admitted preparation only: exact config/revisions checked first."""
    validate_live_supra_context_preparation(config, runtime_identity)
    return export_admitted_supra_contexts(runtime, config, destination, runtime_identity=runtime_identity)


def load_supra_contexts(path, expected_sha256, config):
    from safetensors.torch import load_file
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    if (receipt.get('schema') != 'supra-frozen-contexts-v1' or
            receipt.get('sha256') != expected_sha256 or file_digest(path) != expected_sha256 or
            [r['prompt'] for r in receipt['records']] != required_supra_prompts(config)):
        raise ValueError('frozen context checksum/prompt identity differs')
    tensors = load_file(str(path), device='cpu')
    expected = {f'{kind}{i}' for i in range(len(receipt['records'])) for kind in ('e', 'm')}
    if set(tensors) != expected:
        raise ValueError('frozen context keys differ')
    result = {}
    for i, record in enumerate(receipt['records']):
        context, mask = tensors[f'e{i}'], tensors[f'm{i}']
        if tensor_identity(context) != record['embedding'] or tensor_identity(mask) != record['mask']:
            raise ValueError('frozen context tensor identity differs')
        result[record['prompt']] = (context, mask)
    return result


class SupraCampaignRows:
    """Small CPU teacher cache; selected native responses only move to the model.

    Two copies per edited state and four per preservation state retain the old
    4:1 update-frequency mixture under balanced sampling. The old optimized MSE
    coefficient0.1 is deliberately absent from this GAN-only proposal.
    """
    policy = dict(version='supra-native-cfg-context-time-v1',
                  suffix=['native_time', 'scale', 'strength', 'guidance/3', 'positive_path', 'preservation'],
                  pooling='masked neutral T5 mean / pooled RMS', row_id_feature=False,
                  edit_repetitions=2, preservation_repetitions=4,
                  validation_path_without_archived_label=-1,
                  cfg_adapter_branch='conditional_only', optimized_mse=False)

    def __init__(self, cache_path, *, cache_sha256, config, contexts, split='train', device='cpu'):
        if split not in ('train', 'validation'):
            raise ValueError('explicit train or validation split required')
        if file_digest(cache_path) != cache_sha256:
            raise ValueError('native teacher cache checksum differs')
        self.split, self.device = split, torch.device(device)
        self.config = deepcopy(config)
        config = self.config
        self.cache_path, self.cache_sha256 = str(Path(cache_path).resolve()), cache_sha256
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        if not data.get('records') or not data.get('holds'):
            raise ValueError('edited and preservation native pools required')
        self.records, self.holds = data['records'], data['holds']
        self.contexts = {p: tuple(t.detach().cpu().clone() for t in contexts[p])
                         for p in required_supra_prompts(config)}
        self.context_width = self.contexts[''][0].shape[-1]
        self.condition_width = self.context_width + 6
        self.response_shape = tuple(self.records[0]['z'].shape)
        if self.response_shape != (1, 4, 32, 32):
            raise ValueError('Supra requires native [1,4,32,32] fields')
        self._pooled = {}
        for prompt, (embedding, mask) in self.contexts.items():
            if (embedding.ndim != 3 or embedding.shape[0] != 1 or
                    embedding.shape[-1] != self.context_width or mask.shape != embedding.shape[:2] or
                    not torch.isfinite(embedding).all() or not torch.isfinite(mask).all() or
                    not torch.all((mask == 0) | (mask == 1)) or not mask.any()):
                raise ValueError('frozen T5 context/mask schema differs')
            pooled = (embedding.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            self._pooled[prompt] = pooled / pooled.square().mean().sqrt().clamp_min(1e-12)
        row_by_name = {r['name']: r for r in config['rows']}
        for hold, pool in ((False, self.records), (True, self.holds)):
            for row in pool:
                if (not math.isfinite(row['t']) or not 0 <= row['t'] < 1 or
                        any(not isinstance(row[k], torch.Tensor) or tuple(row[k].shape) != self.response_shape or
                            row[k].dtype != torch.float32 or not torch.isfinite(row[k]).all()
                            for k in ('z', 'neutral', 'target'))):
                    raise ValueError('cached native velocity/time schema differs')
                if hold:
                    if row['prompt'] not in config['preservation'] or not torch.equal(row['neutral'], row['target']):
                        raise ValueError('preservation row/target differs')
                elif (row.get('row') not in row_by_name or row['prompt'] != row_by_name[row['row']]['neutral'] or
                      (split == 'train' and row.get('context') not in ('neutral', 'positive'))):
                    raise ValueError('teacher prompt/row pairing differs')
        self.row_by_name = row_by_name
        self.routes = ([(False, i) for _ in range(2) for i in range(len(self.records))] +
                       [(True, i) for _ in range(4) for i in range(len(self.holds))])
        self._identity = dict(schema='supra-campaign-provider-v1', split=split,
                              cache_path=self.cache_path, cache_sha256=cache_sha256,
                              config=deepcopy(config), policy=deepcopy(self.policy), response_shape=list(self.response_shape),
                              rows=len(self.routes), edited_states=len(self.records), preservation_states=len(self.holds),
                              contexts={p: [tensor_identity(t) for t in pair] for p, pair in self.contexts.items()})

    def __len__(self):
        return len(self.routes)

    def identity(self):
        return deepcopy(self._identity)

    def _source(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        hold, offset = self.routes[index]
        return hold, (self.holds if hold else self.records)[offset]

    def __getitem__(self, index):
        hold, source = self._source(index)
        positive = source['prompt'] if hold else self.row_by_name[source['row']]['positive']
        row = ResponseRow(source['prompt'], positive, '', source['z'].to(self.device).clone(),
                          timestep=torch.tensor(source['t'], device=self.device), scale=1., guidance=3.)
        return PreparedResponse(row, source['target'].clone(), source['neutral'].clone())

    def conditions(self, indices):
        rows = []
        for index in indices:
            hold, source = self._source(int(index))
            role = source.get('context')
            path = -1. if hold or role is None else float(role == 'positive')
            suffix = torch.tensor([[source['t'], 1., 1., 1., path, float(hold)]])
            rows.append(torch.cat((self._pooled[source['prompt']], suffix), dim=1))
        return torch.cat(rows).detach() if rows else torch.empty((0, self.condition_width))

    def response_scale(self):
        if self.split != 'train':
            raise ValueError('training split required for response scale')
        energy = sum(float((r['target'] - r['neutral']).double().square().sum()) for r in self.records)
        count = sum(r['target'].numel() for r in self.records)
        value = math.sqrt(energy / count)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('nonzero finite training contrast scale required')
        return value


@dataclass
class SupraCampaignRun:
    runtime: object
    rows: SupraCampaignRows
    backend: object
    game: object

    def step(self):
        if self.game.completed_steps >= self.game.recipe.total_steps:
            raise RuntimeError('fixed GAN campaign budget exhausted')
        return self.game.step()

    def render_cases(self):
        lookup = {r['neutral']: r['positive'] for r in self.rows.config['rows']}
        for row in self.rows.config['verification']:
            for seed in (42, 1234):
                for label, prompt, scale in (
                    ('off', row['prompt'], 0.), ('half', row['prompt'], .5),
                    ('plus', row['prompt'], 1.),
                    ('teacher', row.get('teacher', lookup.get(row['prompt'], row['prompt'])), 0.),
                ):
                    yield dict(row_id=row['name'], seed=seed, label=label, prompt=prompt, scale=scale,
                               steps=50, cfg=3., width=256, height=256,
                               original_cfg_adapter='both branches', corrected_cfg_adapter='conditional_only',
                               noise_generator='native CUDA SupraRuntime.noise', heldout_training=False)

    @torch.no_grad()
    def render(self, case):
        # Same native CUDA RNG and positive Euler convention as archived images.
        from PIL import Image
        noise = self.runtime.noise(case['seed'])
        row = ResponseRow(case['prompt'], case['prompt'], '', noise,
                          schedule=FlowSchedule.supra(case['steps']), guidance=case['cfg'])
        image = self.backend.render(row, scale=case['scale'])
        if not torch.isfinite(image).all():
            raise FloatingPointError('nonfinite Supra VAE float output before image conversion')
        pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
        return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())


def build_supra_campaign(runtime, rows, *, model_identity, total_steps=28800, batch_size=4,
                         seed=7, critic_width=32, response_scale=None, recipe_overrides=None):
    """Bind already-admitted native runtime; no model loads or implicit text encode."""
    from .pg09 import ConditionalPG09
    if rows.split != 'train':
        raise ValueError('training requires train split; validation is evaluation-only')
    if total_steps < 1 or total_steps > 28800 or batch_size != 4:
        raise ValueError('bounded Supra campaign requires batch4 and at most28800 steps')
    if not model_identity.get('frozen_parameter_sha256'):
        raise ValueError('resource owner must supply verified frozen base SHA')
    for prompt, expected in rows.contexts.items():
        if prompt not in runtime.cache:
            raise ValueError('admitted native runtime lacks frozen context: ' + prompt)
        if any(tensor_identity(a) != tensor_identity(b) for a, b in zip(runtime.cache[prompt], expected)):
            raise ValueError('admitted native text context differs from frozen export')
    if not runtime.loras or any(not torch.isfinite(p).all() for p in runtime.parameters()):
        raise ValueError('finite native LoRA parameters required')
    if any(torch.count_nonzero(module.up.weight) for module in runtime.loras):
        raise ValueError('fresh zero-up native LoRA required; restore only reviewed full GAN checkpoints after build')
    calibrated_scale = rows.response_scale()
    if response_scale is not None and not math.isclose(calibrated_scale, response_scale, rel_tol=0, abs_tol=1e-12):
        raise ValueError('reviewed training response scale differs')
    backend = SupraRuntimeResponseAdapter(runtime, model_identity=model_identity)
    native_decode = backend.decoder
    def checked_decode(latent):
        if not torch.isfinite(latent).all():
            raise FloatingPointError('nonfinite Supra latent before VAE decode')
        image = native_decode(latent)
        if not torch.isfinite(image).all():
            raise FloatingPointError('nonfinite Supra VAE float output')
        return image
    backend.decoder = checked_decode
    rows.device = torch.device(runtime.device)
    device = rows.device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        _seed_initialization_rng(seed, devices)
        critic = CampaignFieldCritic(4, rows.condition_width, critic_width).to(device)
    game = ConditionalPG09(backend, rows, critic, conditions=None, particles=None,
                          response_domain='continuous_field', response_scale=calibrated_scale,
                          preset='bispf_optimized', batch_size=batch_size, seed=seed,
                          total_steps=total_steps, sampling='balanced', recipe_overrides=recipe_overrides)
    return SupraCampaignRun(runtime, rows, backend, game)


def build_final_form_campaign(transformer, rows, *, model_identity, total_steps=3840,
                              batch_size=8, checkpointing=True, recipe_overrides=None):
    """FinalForm uses the existing reusable MOSS bridge without a copied loop."""
    if (rows.split != 'train' or rows.recipe['concept'] != 'final-form' or len(rows) != 15360 or
            total_steps != 3840 or batch_size != 8):
        raise ValueError('FinalForm requires full training cache and fixed3840/b8 proposal')
    if not model_identity.get('frozen_parameter_sha256'):
        raise ValueError('resource owner must supply verified frozen base SHA')
    return build_campaign_run(transformer, rows, model_identity=model_identity,
                              response_scale=FINAL_FORM_SCALE, total_steps=total_steps, batch_size=batch_size,
                              seed=1000, checkpointing=checkpointing, recipe_overrides=recipe_overrides)


def teacher_parity(run, indices, *, relative_rms_limit=.001, absolute_rms_limit=1e-6):
    """Read-only teacher/cache gate, never a fitted loss or optimizer.

    Edited rows retain the frozen edit RMS as their comparison scale.
    Preservation rows have no edit; compare against their frozen neutral
    velocity RMS, with an absolute RMS floor for zero or tiny velocities.
    """
    if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) for value in (relative_rms_limit, absolute_rms_limit)) or
            relative_rms_limit <= 0 or absolute_rms_limit < 0):
        raise ValueError('finite positive relative and nonnegative absolute RMS limits required')
    results = []
    with torch.no_grad():
        for index in indices:
            cached = run.rows[index]
            actual = run.backend.prepare(cached.row)
            edit = float((cached.target.float() - cached.neutral.float()).square().mean().sqrt())
            preservation = edit == 0.
            reference = float(cached.neutral.float().square().mean().sqrt()) if preservation else edit
            tolerance = relative_rms_limit * reference
            if preservation:
                tolerance = max(tolerance, absolute_rms_limit)
                reference = max(reference, absolute_rms_limit / relative_rms_limit)
            reference = max(reference, 1e-20)
            target_error = float((actual.target.float() - cached.target.to(
                device=actual.target.device, dtype=torch.float32)).square().mean().sqrt())
            base_error = float((actual.neutral.float() - cached.neutral.to(
                device=actual.neutral.device, dtype=torch.float32)).square().mean().sqrt())
            results.append(dict(index=index,
                                target_relative_rms=target_error / reference,
                                base_relative_rms=base_error / reference,
                                normalization='frozen_neutral_velocity_rms' if preservation else 'frozen_edit_rms',
                                normalization_rms=reference, target_rms_error=target_error,
                                base_rms_error=base_error, rms_tolerance=tolerance,
                                relative_rms_limit=relative_rms_limit,
                                absolute_rms_limit=absolute_rms_limit if preservation else 0.))
    if any(not all(math.isfinite(r[key]) for key in (
                'target_rms_error', 'base_rms_error', 'normalization_rms', 'rms_tolerance')) or
           max(r['target_rms_error'], r['base_rms_error']) > r['rms_tolerance'] for r in results):
        error = ValueError('native teacher/cache parity failed: ' + json.dumps(results))
        error.diagnostics = results
        raise error
    return results


def proposal_identity(proposal):
    return canonical_digest(proposal)
