"""Source admission and native SD3 joint-attention residuals; no game mathematics."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).parent
TARGETS = ('to_q', 'to_k', 'to_v', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_out.0', 'to_add_out')
PROTOCOL = 'sd3-joint-attention-spatial-velocity-hg010-v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admitted_builder():
    """Reject another core, runtime, graph or framework build before native mutation."""
    import particle_sliders
    own = json.loads((ROOT / 'source_lock.json').read_text())
    actual_own = {p.name: digest(p) for p in ROOT.iterdir()
                  if p.suffix in ('.py', '.hndl', '.json') and p.name != 'source_lock.json'}
    if actual_own != own:
        raise ValueError('SD3 package/graph source cohort differs before native mutation')
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError('SD3 shared requires Python >=3.11,<3.15')
    expected = json.loads((ROOT / 'core_lock.json').read_text())
    root = Path(particle_sliders.__file__).parent
    actual = {str(p.relative_to(root)): digest(p) for p in root.rglob('*')
              if p.suffix in ('.py', '.json', '.hndl')}
    if version('particle-sliders-core') != '0.12.0' or actual != expected['sources']:
        raise ValueError('SD3 requires its exact core lifecycle source cohort; retain original pins')
    for name, expected_version in {'diffusers': '0.38.0', 'torch': '2.14.1',
            'transformers': '4.57.6', 'safetensors': '0.8.0'}.items():
        if version(name).split('+', 1)[0] != expected_version:
            raise ValueError(f'SD3 requires {name}=={expected_version}')
    from particle_sliders.pg094 import validate_pg09_domain
    validate_pg09_domain((1, 1, 2, 2), response_domain='continuous_field')
    from particle_sliders.fibo_hndl import verified_builder
    builder, identity = verified_builder()
    identity = dict(identity, sd3_projection_sha256=digest(ROOT / 'projection.hndl'),
                    core_lock_sha256=digest(ROOT / 'core_lock.json'),
                    core_commit=expected['commit'])
    return builder, identity


@contextmanager
def initialization_rng(generator):
    """Adapt builders without a generator argument to an owned CPU RNG stream.

    HNDL/native constructors use default Torch initializers. Their temporary
    default state is scoped and restored on success/failure; the seed and
    advancing stream belong to the supplied local generator. Construction is
    single-threaded, not an isolation guarantee for concurrent default-RNG users.
    """
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        try:
            yield
        finally:
            generator.set_state(torch.get_rng_state())


def model_identity(model):
    """Actual staged base tensors/configuration, independent of path and row cache."""
    from diffusers import SD3Transformer2DModel
    if type(model) is not SD3Transformer2DModel:
        raise ValueError('the actual native SD3Transformer2DModel is required')
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('this bounded cohort admits staged CPU float32 SD3 only')
    from diffusers.models.attention_processor import JointAttnProcessor2_0
    if any(type(p) is not JointAttnProcessor2_0 for p in model.attn_processors.values()):
        raise ValueError('only the native SD3 JointAttnProcessor2_0 cohort is qualified')
    owners = [type(m) for m in model.modules() if type(m).__module__.startswith('diffusers.')]
    owners.append(JointAttnProcessor2_0)
    sources = {owner.__module__: digest(inspect.getsourcefile(owner)) for owner in owners}
    config = {k: v for k, v in dict(model.config).items() if not k.startswith('_')}
    sha = hashlib.sha256(json.dumps(dict(config=config, native_sources=sources), sort_keys=True).encode())
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        sha.update(name.encode())
        sha.update(str((tuple(value.shape), str(value.dtype))).encode())
        sha.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(protocol=PROTOCOL, base_sha256=sha.hexdigest(), config=config, native_sources=sources,
                diffusers_version=version('diffusers'), qualification='CPU software cohort; pretrained/GPU/model quality unqualified')


def targets(model):
    selected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and '.attn.' in name:
            suffix = name.split('.attn.', 1)[1]
            if suffix in TARGETS:
                if module._forward_hooks or module._forward_pre_hooks:
                    raise ValueError('SD3 target has existing hooks; an unadapted staged base is required')
                selected.append((name, module))
    if not selected or not any('.attn.add_q_proj' in n for n, _ in selected):
        raise ValueError('SD3 joint image/text attention projections are missing')
    return selected


def restore_all(actions):
    """Attempt every product cleanup and keep the active failure visible."""
    active = sys.exception()
    failures = []
    for action in actions:
        try:
            action()
        except BaseException as exc:
            failures.append(exc)
    if failures:
        if active is not None:
            for failure in failures:
                active.add_note('SD3 cleanup failure: ' + repr(failure))
        else:
            for failure in failures[1:]:
                failures[0].add_note('Additional SD3 cleanup failure: ' + repr(failure))
            raise failures[0]


class Projection(nn.Module):
    """Native last-axis LoRA, executing both matrices through verified HyperGAN/HNDL."""
    def __init__(self, module, rank, alpha, builder, identity):
        super().__init__()
        self.input_width, self.output_width = module.in_features, module.out_features
        self.rank, self.alpha = rank, alpha
        self.multiplier = 0.
        source = (ROOT / 'projection.hndl').read_text()
        self.graph = builder(source, input_shape={'value': ['B', self.input_width]},
            output_shape=['B', self.output_width], parameters={'rank': rank, 'output_width': self.output_width})
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=5 ** .5)
        nn.init.zeros_(self.graph['up'].weight)
        self.identity = dict(deepcopy(identity),
            sd3_plan_sha256=hashlib.sha256(self.graph.plan.to_json().encode()).hexdigest())

    def extra_repr(self):
        return json.dumps(dict(self.identity, rank=self.rank, alpha=self.alpha,
            input_width=self.input_width, output_width=self.output_width), sort_keys=True)

    def forward(self, value):
        shape = value.shape
        delta = self.graph(value=value.reshape(-1, self.input_width))
        return delta.reshape(*shape[:-1], self.output_width) * (self.multiplier * self.alpha / self.rank)

    def hook(self, module, arguments, output):
        if self.multiplier == 0:
            return output
        return output + self(arguments[0]).to(output)


class Adapter(nn.Module):
    def __init__(self, selected, rank, alpha, builder, identity):
        super().__init__()
        self.names = tuple(name for name, _ in selected)
        self.layers = nn.ModuleList([Projection(module, rank, alpha, builder, identity) for _, module in selected])
        self.handles = []

    def attach(self, selected):
        for (_, module), layer in zip(selected, self.layers):
            self.handles.append(module.register_forward_hook(layer.hook))

    def close(self):
        actions = [h.remove for h in reversed(self.handles)]
        self.handles = []
        restore_all(actions)

    @contextmanager
    def scaled(self, scale):
        previous = [layer.multiplier for layer in self.layers]
        try:
            for layer in self.layers:
                layer.multiplier = float(scale)
            yield
        finally:
            for layer, value in zip(self.layers, previous):
                layer.multiplier = value
