"""One installable implementation serves canonical and compatibility imports."""
import ast
from contextlib import contextmanager
import importlib
from pathlib import Path
import pickle

import pytest
import torch


SURFACES = {
    "pg09_backends": (
        "AdapterParameters", "FlowSchedule", "ResponseRow", "PreparedResponse",
        "FrozenResponseAdapter", "AnimaResponseAdapter", "SupraResponseAdapter",
        "MossResponseAdapter", "AnimaRuntimeResponseAdapter", "SupraRuntimeResponseAdapter",
        "checkpoint_with_adapter_scale", "bind_native_gradient_checkpointing",
        "response_signature", "save_adapter", "load_adapter", "moss_pipeline_decoder",
        "lora_scale_context", "_lora_scale_context", "_checkpoint_contexts",
    ),
    "pg09_audio": (
        "AudioRow", "PreparedAudioResponse", "AudioResponseCritic",
        "ARLastHiddenResponseAdapter", "YuE2ARResponseAdapter", "Music3ARResponseAdapter",
        "require_audio_width", "audio_conditions",
    ),
    "pg09_images": (
        "ImageResponseAdapter", "IrisResponseAdapter", "QwenImage21ResponseAdapter",
        "image_lora_parameters", "image_condition_features", "CONDITION_FEATURE_POLICY",
        "_condition_projection", "_condition_identity", "_snapshot",
    ),
    "pg09_readout_critic": ("ConditionalReadoutCritic",),
}


@pytest.mark.parametrize("module_name,names", SURFACES.items())
def test_repository_import_is_alias_of_installable_backend(module_name, names):
    canonical = importlib.import_module(f"particle_sliders.{module_name}")
    # The optional repository facade is tested only when the source checkout is
    # available. It is intentionally absent from the wheel and extracted sdist.
    if importlib.util.find_spec("conceptmod") is None:
        pytest.skip("legacy compatibility alias belongs to the repository")
    legacy = importlib.import_module(f"conceptmod.textsliders.{module_name}")
    assert legacy is canonical
    for name in names:
        assert getattr(legacy, name) is getattr(canonical, name)


@pytest.mark.parametrize("module_name", SURFACES)
def test_shared_backends_have_no_conceptmod_imports(module_name):
    module = importlib.import_module(f"particle_sliders.{module_name}")
    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("conceptmod") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("conceptmod")


def test_serialized_backend_objects_use_installable_class_identity():
    from particle_sliders.pg09_audio import AudioRow, AudioResponseCritic
    from particle_sliders.pg09_backends import FlowSchedule

    row = AudioRow((1, 2), (1, 3))
    schedule = FlowSchedule.supra(2)
    assert type(pickle.loads(pickle.dumps(row))) is AudioRow
    assert type(pickle.loads(pickle.dumps(schedule))) is FlowSchedule
    assert AudioRow.__module__ == "particle_sliders.pg09_audio"
    assert AudioResponseCritic.__module__ == "particle_sliders.pg09_audio"
    assert FlowSchedule.__module__ == "particle_sliders.pg09_backends"


def test_legacy_serialized_row_resolves_to_canonical_class():
    if importlib.util.find_spec("conceptmod") is None:
        pytest.skip("legacy compatibility alias belongs to the repository")
    from particle_sliders.pg09_audio import AudioRow

    row = AudioRow((1, 2), (1, 3))
    # Protocol 0 keeps the global class reference explicit. Model an existing
    # pickle written before the implementation moved into the shared package.
    encoded = pickle.dumps(row, protocol=0)
    reference = b"cparticle_sliders.pg09_audio\nAudioRow\n"
    assert reference in encoded
    legacy = encoded.replace(reference, b"cconceptmod.textsliders.pg09_audio\nAudioRow\n")
    restored = pickle.loads(legacy)
    assert type(restored) is AudioRow
    assert restored == row


def test_prepared_response_owns_nested_conditioning_tensor_snapshots():
    from particle_sliders.pg09_backends import AdapterParameters, FrozenResponseAdapter, ResponseRow

    @contextmanager
    def scaled(value):
        yield

    weight = torch.nn.Parameter(torch.tensor(0.25))

    class NestedAdapter(FrozenResponseAdapter):
        def _velocity(self, condition, latent, timestep):
            return latent + condition['nested']['values'][0][0] + weight * self.network.scale

    adapter = NestedAdapter(AdapterParameters([('weight', weight)], scaled))
    conditions = [dict(nested=dict(values=[(torch.full((1, 2), value),)])) for value in (1., 2., -1.)]
    row = ResponseRow(*conditions, latent=torch.zeros(1, 2), timestep=torch.tensor(500.), guidance=2.)
    prepared = adapter.prepare(row)
    expected_generated = adapter.generated(prepared).detach().clone()
    expected_target = prepared.target.clone()
    for condition in conditions:
        condition['nested']['values'][0][0].add_(100)
        condition['nested']['values'].append(torch.full((1, 2), -999.))
    assert torch.equal(adapter.generated(prepared), expected_generated)
    assert torch.equal(prepared.target, expected_target)
    for frozen, source, value in zip(
            (prepared.row.neutral, prepared.row.positive, prepared.row.negative), conditions, (1., 2., -1.)):
        assert len(frozen['nested']['values']) == 1
        assert frozen['nested']['values'][0][0] is not source['nested']['values'][0][0]
        assert torch.equal(frozen['nested']['values'][0][0], torch.full((1, 2), value))


@pytest.mark.parametrize('api', ['diffusers', 'transformers'])
def test_native_checkpoint_public_hooks_capture_signed_scales_and_replay(api):
    from functools import partial
    from torch.utils.checkpoint import checkpoint
    from particle_sliders.pg09_backends import AdapterParameters, bind_native_gradient_checkpointing

    class NativeModel(torch.nn.Module):
        is_gradient_checkpointing = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.7))
            self.scale = 1.
            self.calls = 0
            if api == 'diffusers':
                self.enable_gradient_checkpointing = self._enable_diffusers
            else:
                self.gradient_checkpointing_enable = self._enable_transformers

        def _enable_diffusers(self, gradient_checkpointing_func=None):
            self.calls += 1
            self._gradient_checkpointing_func = gradient_checkpointing_func

        def _enable_transformers(self, gradient_checkpointing_kwargs=None):
            self.calls += 1
            self.kwargs = gradient_checkpointing_kwargs
            self._gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

        def plain(self, value):
            return torch.sin(value * self.weight * self.scale)

        def forward(self, value):
            return self._gradient_checkpointing_func(self.plain, value)

    model = NativeModel()

    @contextmanager
    def scaled(scale):
        previous, model.scale = model.scale, scale
        try:
            yield
        finally:
            model.scale = previous

    network = AdapterParameters([('weight', model.weight)], scaled)
    bind_native_gradient_checkpointing(model, network)
    assert model.calls == 1
    if api == 'transformers':
        assert model.kwargs['use_reentrant'] is False
        assert callable(model.kwargs['context_fn'])
    expected, generated = [], []
    for scale in (1., -1.):
        with network.scaled(scale):
            value = torch.tensor([[0.3, -0.8]])
            expected.append(model.plain(value))
            generated.append(model(value))
    expected_loss = expected[0].sum() + 0.3 * expected[1].square().sum()
    generated_loss = generated[0].sum() + 0.3 * generated[1].square().sum()
    expected_grad, = torch.autograd.grad(expected_loss, model.weight)
    # The same non-reentrant graph is replayed repeatedly by the PG penalties.
    for _ in range(2):
        actual_grad, = torch.autograd.grad(generated_loss, model.weight, retain_graph=True)
        torch.testing.assert_close(actual_grad, expected_grad)
        assert model.scale == network.scale == 1.


def test_native_checkpoint_hook_reports_unsupported_public_api():
    from particle_sliders.pg09_backends import bind_native_gradient_checkpointing

    class Unsupported:
        is_gradient_checkpointing = True

        def gradient_checkpointing_enable(self):
            raise AssertionError('unsupported API must be rejected before calling it')

    with pytest.raises(ValueError, match='gradient_checkpointing_kwargs'):
        bind_native_gradient_checkpointing(Unsupported(), None)

    class Legacy(Unsupported):
        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            raise AssertionError('legacy API silently ignores kwargs and must not be called')

        def _set_gradient_checkpointing(self, module, value=False):
            pass

    with pytest.raises(ValueError, match='legacy Transformers checkpointing ignores'):
        bind_native_gradient_checkpointing(Legacy(), None)


def test_qwen3_public_checkpoint_hook_matches_uncheckpointed_signed_gradients():
    pytest.importorskip('transformers')
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from particle_sliders.pg09_backends import AdapterParameters, bind_native_gradient_checkpointing

    torch.manual_seed(81)
    model = Qwen3ForCausalLM(Qwen3Config(
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=32, attention_dropout=0., use_cache=False,
        attn_implementation='eager')).requires_grad_(False).train()
    scale = [1.]

    class SignedLinear(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.delta = torch.nn.Parameter(torch.randn_like(base.weight) * .01)
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return self.base(value) + scale[0] * torch.nn.functional.linear(value, self.delta)

    attention = model.model.layers[0].self_attn
    attention.q_proj = SignedLinear(attention.q_proj)

    @contextmanager
    def scaled(value):
        previous, scale[0] = scale[0], value
        try:
            yield
        finally:
            scale[0] = previous

    delta = attention.q_proj.delta
    network = AdapterParameters([('q_proj.delta', delta)], scaled)
    tokens = torch.tensor([[1, 3, 5, 7]])

    def signed_loss():
        outputs = []
        for sign in (1., -1.):
            with network.scaled(sign):
                outputs.append(model(input_ids=tokens, use_cache=False).logits[:, -1, :4])
        return outputs[0].sum() + .3 * outputs[1].square().sum()

    expected, = torch.autograd.grad(signed_loss(), delta)
    model.gradient_checkpointing_enable()
    bind_native_gradient_checkpointing(model, network)
    loss = signed_loss()
    for _ in range(2):
        calls = attention.q_proj.calls
        actual, = torch.autograd.grad(loss, delta, retain_graph=True)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        assert attention.q_proj.calls > calls
        assert network.scale == scale[0] == 1.
