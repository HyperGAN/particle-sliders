"""Product-owned targeting and compatibility on CPU fixtures, not native models."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particle_sliders import native_lora as native


@pytest.fixture(autouse=True)
def bounded_cpu():
    before = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(20261108)
        yield
        assert not torch.cuda.is_initialized()
    torch.set_num_threads(before)


class Projection(nn.Linear):
    """A supported operator whose product class name reveals no operator type."""


class Filter1(nn.Conv1d):
    pass


class Filter2(nn.Conv2d):
    pass


class LinearProjection(nn.Conv1d):
    """Operator dispatch must use its type, even with an ambiguous class name."""


def record_host(host):
    return (
        [(module, dict(vars(module))) for module in host.modules()],
        deepcopy(host.state_dict()),
        torch.get_rng_state().clone(),
    )


def assert_unchanged(host, before):
    attributes, state, rng = before
    for module, previous in attributes:
        assert vars(module).keys() == previous.keys()
        assert all(vars(module)[key] is value for key, value in previous.items())
    assert state.keys() == host.state_dict().keys()
    assert all(torch.equal(value, host.state_dict()[key]) for key, value in state.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_explicit_plan_bypasses_discovery_and_preserves_product_order_and_names(monkeypatch):
    host = nn.Module()
    host.time_embed = Projection(3, 3)
    host.add_projection = Projection(3, 3)
    host.unselected = Projection(3, 3)
    originals = [host.time_embed.forward, host.add_projection.forward]

    def forbidden(*args, **kwargs):
        pytest.fail("explicit targets invoked legacy discovery")

    monkeypatch.setattr(native, 'lora_target_plan', forbidden)
    monkeypatch.setattr(native, '_legacy_module_kind', forbidden)
    plan = ((name, module) for name, module in [
        ('product_second', host.add_projection), ('product_first', host.time_embed),
    ])
    network = native.LoRANetwork(host, rank=2, alpha=2., multiplier=0., target_plan=plan)
    assert [layer.lora_name for layer in network.unet_loras] == ['product_second', 'product_first']
    assert list(network.state_dict()) == [
        f'{name}.{key}' for name in ['product_second', 'product_first']
        for key in ['alpha', 'lora_down.weight', 'lora_up.weight']
    ]
    assert 'forward' not in vars(host.unselected)
    x = torch.randn(2, 3)
    with torch.no_grad():
        for layer in network.unet_loras:
            layer.lora_up.weight.fill_(.2)
    before = [original(x) for original in originals]
    network.set_lora_slider(.75)
    with network:
        assert not torch.equal(host.time_embed(x), before[0])
        assert not torch.equal(host.add_projection(x), before[1])
    assert torch.equal(host.time_embed(x), before[0])
    assert torch.equal(host.add_projection(x), before[1])


@pytest.mark.parametrize('layer,shape', [
    (lambda: Projection(3, 4), (2, 3)),
    (lambda: Filter1(3, 4, 3, padding=1), (2, 3, 5)),
    (lambda: Filter2(3, 4, 3, padding=1), (2, 3, 5, 5)),
    (lambda: LinearProjection(3, 4, 3, padding=1), (2, 3, 5)),
])
def test_explicit_operator_subclasses_preserve_existing_delta_math(layer, shape):
    host = layer()
    original = host.forward
    network = native.LoRANetwork(host, rank=2, alpha=2., target_plan=[('chosen', host)])
    branch = network.unet_loras[0]
    with torch.no_grad():
        branch.lora_down.weight.fill_(.2)
        branch.lora_up.weight.fill_(.3)
    x = torch.randn(shape)
    expected_delta = branch.lora_up(branch.lora_down(x))
    base = original(x)
    network.set_lora_slider(-.5)
    with network:
        torch.testing.assert_close(host(x), base - .5 * expected_delta)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(3, 3)
        self.to_out = nn.Linear(3, 3)
        self.add_k = nn.Linear(3, 3)
        self.conv2 = nn.Conv2d(3, 3, 1)


def legacy_host():
    host = nn.Module()
    host.attn1 = Attention()
    host.attn2 = Attention()
    host.time_embed = Attention()
    host.mid_block = nn.ModuleList([Attention(), Attention()])
    return host


@pytest.mark.parametrize('method,paths,leaves', [
    ('full', ['attn1', 'attn2', 'time_embed', 'mid_block-0', 'mid_block-1'], ['to_q', 'to_out', 'conv2']),
    ('noxattn', ['attn1', 'mid_block-0', 'mid_block-1'], ['to_q', 'to_out', 'conv2']),
    ('innoxattn', ['attn1', 'time_embed', 'mid_block-0', 'mid_block-1'], ['to_q', 'to_out', 'conv2']),
    ('selfattn', ['attn1'], ['to_q', 'to_out', 'conv2']),
    ('xattn', ['attn2'], ['to_q', 'to_out', 'conv2']),
    ('xattn-strict', ['attn2'], ['to_q', 'conv2']),
    ('noxattn-hspace', ['mid_block-0', 'mid_block-1'], ['to_q', 'to_out', 'conv2']),
    ('noxattn-hspace-last', ['mid_block-1'], ['conv2']),
])
def test_legacy_filters_keep_names_initialization_keys_and_rng(method, paths, leaves):
    host = legacy_host()
    explicit_host = deepcopy(host)
    before = record_host(host)
    options = dict(prefix='historical', delimiter='-', train_method=method)
    plan = native.lora_target_plan(host, ['Attention'], **options)
    expected = [f'historical-{path}-{leaf}' for path in paths for leaf in leaves]
    assert [name for name, _ in plan] == expected
    assert_unchanged(host, before)
    network = native.LoRANetwork(host, rank=2, alpha=2., target_replace=['Attention'], **options)
    after_legacy_rng = torch.get_rng_state().clone()
    torch.set_rng_state(before[-1])
    explicit_plan = native.lora_target_plan(explicit_host, ['Attention'], **options)
    explicit = native.LoRANetwork(explicit_host, rank=2, alpha=2., target_plan=explicit_plan)
    assert torch.equal(after_legacy_rng, torch.get_rng_state())
    assert list(network.state_dict()) == list(explicit.state_dict())
    for key, value in network.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


@pytest.mark.parametrize('plan,error,match', [
    (lambda host: [], ValueError, 'at least one'),
    (lambda host: [('valid', host[0]), ('valid', host[1])], ValueError, 'duplicated'),
    (lambda host: [('valid', host[0]), ('other', host[0])], ValueError, 'repeated'),
    (lambda host: [('valid', host[0]), ('foreign', nn.Linear(3, 3))], ValueError, 'does not belong'),
    (lambda host: [('valid', host[0]), ('bad.name', host[1])], ValueError, 'without'),
    (lambda host: [('', host[0])], ValueError, 'nonempty'),
    (lambda host: [(4, host[0])], ValueError, 'nonempty'),
    (lambda host: [('forward', host[0])], ValueError, 'reserved'),
    (lambda host: [('unet_loras', host[0])], ValueError, 'reserved'),
    (lambda host: [('_activation_stack', host[0])], ValueError, 'reserved'),
    (lambda host: [('valid', host[0]), ('unsupported', host[2])], TypeError, 'must be'),
    (lambda host: [('valid', host[0]), ('broken', host[1], 2)], ValueError, 'pairs'),
])
def test_invalid_plan_is_rejected_before_allocating_or_attaching(plan, error, match, monkeypatch):
    host = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Identity())
    entries = plan(host)
    before = record_host(host)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid plan allocated an adapter")

    monkeypatch.setattr(native, 'LoRAModule', forbidden)
    with pytest.raises(error, match=match):
        native.LoRANetwork(host, target_plan=entries)
    assert_unchanged(host, before)


@pytest.mark.parametrize('options', [dict(groups=3), dict(dilation=2), dict(padding_mode='reflect')])
def test_explicit_plan_refuses_unsupported_convolution_geometry(options):
    host = nn.Conv2d(3, 3, 3, **options)
    before = record_host(host)
    with pytest.raises(ValueError, match='requires groups=1'):
        native.LoRANetwork(host, target_plan=[('chosen', host)])
    assert_unchanged(host, before)


@pytest.mark.parametrize('options', [
    dict(target_replace=['Attention']), dict(train_method='noxattn'),
    dict(prefix='product'), dict(delimiter='-'),
])
def test_explicit_plan_rejects_ambiguous_legacy_arguments(options):
    host = nn.Linear(3, 3)
    before = record_host(host)
    with pytest.raises(ValueError, match='cannot be combined'):
        native.LoRANetwork(host, target_plan=[('chosen', host)], **options)
    assert_unchanged(host, before)


@pytest.mark.parametrize('error', [RuntimeError, KeyboardInterrupt])
def test_explicit_plan_attachment_failure_restores_forwards_and_rng(error, monkeypatch):
    host = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    before = record_host(host)

    def fail_after_attachment(self, name, module):
        nn.Module.add_module(self, name, module)
        raise error('after first host patch')

    monkeypatch.setattr(native.LoRANetwork, 'add_module', fail_after_attachment)
    with pytest.raises(error, match='after first host patch'):
        native.LoRANetwork(host, target_plan=[('first', host[0]), ('second', host[1])])
    assert_unchanged(host, before)
