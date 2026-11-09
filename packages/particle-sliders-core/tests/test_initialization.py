"""Synthetic CPU coverage of explicit fresh initialization (no model payloads)."""
from copy import deepcopy
import hashlib
import math

import pytest
import torch
from torch import nn
from particlegan import init as pg_init

from particle_sliders.initialization import ComponentInitialization as Policy
from particle_sliders.initialization import initialize_fresh_components


def initialize(components, **context):
    return initialize_fresh_components(components, cohort_id="synthetic-fresh-v1",
        source_identity={"host": "test-fixture-v1", "policy": "explicit-per-leaf"},
        **({"completed_steps": 0, "restored_from": None, "optimizers": ()} | context))


def policy(module, method="deterministic_orthogonal", **options):
    return Policy(module, method, {n: i + 17 for i, (n, p) in enumerate(module.named_parameters())
                                  if p.requires_grad and p.numel()}, **options)


def equal_state(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_qr_official_result_host_scale_and_global_rng():
    layer = nn.Linear(12, 7)
    reference = deepcopy(layer)
    seeds = {"weight": 17, "bias": 18}
    pg_init.deterministic_orthogonal_(reference, parameter_seeds=seeds)
    before_rng = torch.get_rng_state().clone()
    ids = [id(p) for p in layer.parameters()]
    receipt = initialize({"adapter.down": policy(layer)})
    equal_state(layer.state_dict(), reference.state_dict())
    assert [id(p) for p in layer.parameters()] == ids
    assert torch.equal(before_rng, torch.get_rng_state())
    # Default Linear U(-1/sqrt(fan_in), +...) => RMS 1/sqrt(3*fan_in).
    torch.testing.assert_close(layer.weight.square().mean(), torch.tensor(1 / 36), rtol=1e-6, atol=1e-8)
    document = receipt.as_dict()
    assert document["particlegan"]["version"] == "0.9.5"
    assert document["particlegan"]["initializer_source"]["sha256"] == "3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b"
    assert document["components"]["adapter.down"]["parameters"]["weight"]["host_declaration"] == {
        "kind": "Uniform", "low": -1 / math.sqrt(12), "high": 1 / math.sqrt(12)}
    assert receipt.sha256 == hashlib.sha256(receipt.canonical_json.encode()).hexdigest()
    document["components"].clear()
    assert receipt.as_dict()["components"]


def test_zero_up_nonzero_down_frozen_backbone_buffers_and_no_deepcopy():
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(9, 9), requires_grad=False)
            self.register_buffer("running", torch.randn(4))
        def __deepcopy__(self, memo):
            pytest.fail("frozen backbone must not be deepcopied")
    backbone, down, up = Backbone(), nn.Linear(9, 3, bias=False), nn.Linear(3, 9, bias=False)
    nn.init.zeros_(up.weight)
    before = {n: p.clone() for n, p in backbone.state_dict().items()}
    up_before = up.weight.clone()
    initialize({"backbone": Policy(backbone, "keep"), "down": policy(down), "up": policy(up, "xavier_uniform_zero_bias_v1")})
    equal_state(before, backbone.state_dict())
    assert torch.count_nonzero(down.weight) > 0
    assert torch.equal(up_before, up.weight)
    assert torch.count_nonzero(up(down(torch.ones(2, 9)))) == 0


def test_literal_xavier_matches_official_owned_stream_and_zero_bias():
    layer = nn.Linear(11, 5)
    reference = deepcopy(layer)
    pg_init.initialize_(reference, method="xavier_uniform_zero_bias_v1", gain=0.7,
        parameter_generators={"weight": torch.Generator(device="cpu").manual_seed(17)})
    before_rng = torch.get_rng_state().clone()
    receipt = initialize({"critic": policy(layer, "xavier_uniform_zero_bias_v1", gain=0.7)})
    equal_state(reference.state_dict(), layer.state_dict())
    assert torch.equal(before_rng, torch.get_rng_state())
    assert torch.count_nonzero(layer.bias) == 0
    detail = receipt.as_dict()["components"]["critic"]["parameters"]
    assert detail["weight"]["generator_before_sha256"] != detail["weight"]["generator_after_sha256"]
    assert "generator_before_sha256" not in detail["bias"]


def test_declared_bank_r2_and_explicit_keep_without_inserting_a_bank():
    bank = nn.Module()
    bank.register_parameter("particles", nn.Parameter(torch.randn(32, 4)))
    bank.register_parameter("temperature", nn.Parameter(torch.tensor(0.4)))
    bank.register_buffer("counter", torch.tensor(0))
    original_ids = [id(p) for p in bank.parameters()]
    before_temp = bank.temperature.clone()
    before_counter = bank.counter.clone()
    expected = deepcopy(bank)
    class Expected(nn.Module):
        pass
    expected_holder = Expected()
    expected_holder.register_parameter("particles", nn.Parameter(expected.particles.clone()))
    expected_holder.register_parameter("temperature", nn.Parameter(expected.temperature.clone()))
    pg_init.register(Expected, {"particles": pg_init.R2Normal(0, 0.25), "temperature": pg_init.KEEP})
    pg_init.deterministic_orthogonal_(expected_holder, parameter_seeds={"particles": 17, "temperature": 18})
    initialize({"existing-bank": policy(bank, distributions={"particles": pg_init.R2Normal(0, 0.25), "temperature": pg_init.KEEP})})
    assert torch.equal(bank.particles, expected_holder.particles)
    assert torch.equal(bank.temperature, before_temp)
    assert torch.equal(bank.counter, before_counter)
    assert [id(p) for p in bank.parameters()] == original_ids
    assert set(bank.state_dict()) == {"particles", "temperature", "counter"}


def test_sampled_host_normal_scale_matches_official_and_name_isolation():
    table = nn.Module()
    table.register_parameter("value", nn.Parameter(torch.randn(17, 3)))
    spec = pg_init.Normal(0.2, 0.03)
    expected = nn.Module()
    expected.register_parameter("value", nn.Parameter(table.value.clone()))
    pg_init.initialize_(expected, method="sample_distributions_v1", distributions={"value": spec},
        parameter_generators={"value": torch.Generator(device="cpu").manual_seed(17)})
    unrelated = nn.Linear(2, 2)
    before_rng = torch.get_rng_state().clone()
    initialize({"unrelated": policy(unrelated), "host-table": policy(table, "sample_distributions_v1", distributions={"value": spec})})
    assert torch.equal(table.value, expected.value)
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize("context", [{"completed_steps": 1}, {"completed_steps": True}, {"restored_from": "saved-FULL"}, {"optimizers": (object(),)}])
def test_restore_optimizer_and_completed_context_refused_without_mutation(context):
    layer = nn.Linear(4, 2)
    before, rng = deepcopy(layer.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        initialize({"layer": policy(layer)}, **context)
    equal_state(before, layer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(layer, "_particle_sliders_fresh_initialization_sha256")


@pytest.mark.parametrize("bad_policy", [
    lambda layer: Policy(layer, "deterministic_orthogonal", {"weight": True, "bias": 2}),
    lambda layer: Policy(layer, "deterministic_orthogonal", {"weight": 2}),
    lambda layer: Policy(layer, "deterministic_orthogonal", {"weight": 2**63, "bias": 2}),
    lambda layer: policy(layer, "sample_distributions_v1", distributions={"weight": pg_init.Normal(0, 0)}),
    lambda layer: policy(layer, "unknown"),
])
def test_later_invalid_scope_is_atomic_across_all_components(bad_policy):
    first, later = nn.Linear(4, 4), nn.Linear(5, 3)
    before = [deepcopy(m.state_dict()) for m in (first, later)]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        initialize({"first": policy(first), "later": bad_policy(later)})
    for module, state in zip((first, later), before):
        equal_state(state, module.state_dict())
        assert not hasattr(module, "_particle_sliders_fresh_initialization_sha256")
    assert torch.equal(rng, torch.get_rng_state())


def test_aliases_gradients_and_repeated_initialization_refused():
    layer = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="overlapping|aliased"):
        initialize({"one": policy(layer), "two": policy(layer)})
    alias = nn.Module()
    alias.register_parameter("a", layer.weight)
    alias.register_parameter("b", nn.Parameter(layer.weight.detach()))
    with pytest.raises(ValueError, match="storage"):
        initialize({"alias": policy(alias, distributions={"a": pg_init.Normal(), "b": pg_init.Normal()})})
    layer.weight.grad = torch.zeros_like(layer.weight)
    with pytest.raises(ValueError, match="gradients"):
        initialize({"layer": policy(layer)})
    layer.weight.grad = None
    initialize({"layer": policy(layer)})
    with pytest.raises(ValueError, match="already initialized"):
        initialize({"layer": policy(layer)})


def test_special_finalizers_composites_and_undeclared_parameters_refused():
    embedding = nn.Embedding(8, 4, padding_idx=0)
    with pytest.raises(ValueError, match="leaves"):
        initialize({"embedding": policy(embedding)})
    network = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
    with pytest.raises(ValueError, match="leaves"):
        initialize({"network": policy(network)})
    bank = nn.Module()
    bank.register_parameter("z", nn.Parameter(torch.randn(8, 4)))
    with pytest.raises(ValueError, match="requires Uniform"):
        initialize({"bank": policy(bank)})


def test_identity_and_existing_default_constructors_are_unchanged():
    layer = nn.Linear(3, 3)
    initialize({"routing": policy(layer, "identity_linear_v1")})
    assert torch.equal(layer.weight, torch.eye(3))
    assert torch.count_nonzero(layer.bias) == 0
    torch.manual_seed(37)
    expected = nn.Linear(6, 4).state_dict()
    torch.manual_seed(37)
    actual = nn.Linear(6, 4)
    equal_state(expected, actual.state_dict())
    assert not hasattr(actual, "_particle_sliders_fresh_initialization_sha256")
    assert not torch.cuda.is_initialized()


def test_partial_frozen_qr_preserves_frozen_values_and_seed_identity():
    layer = nn.Linear(7, 4)
    layer.bias.requires_grad_(False)
    frozen = layer.bias.clone()
    reference = deepcopy(layer)
    pg_init.deterministic_orthogonal_(reference, parameter_seeds={"weight": 17})
    initialize({"leaf": policy(layer)})
    equal_state(reference.state_dict(), layer.state_dict())
    assert torch.equal(frozen, layer.bias)
    other = nn.Linear(7, 4)
    other.bias.requires_grad_(False)
    before = deepcopy(other.state_dict())
    with pytest.raises(ValueError, match="wholly owned"):
        initialize({"leaf": policy(other, "xavier_uniform_zero_bias_v1")})
    equal_state(before, other.state_dict())


def test_public_exports_and_plain_json_policy_receipt():
    import particle_sliders
    assert particle_sliders.ComponentInitialization is Policy
    assert particle_sliders.initialize_fresh_components is initialize_fresh_components
    layer = nn.Linear(4, 2)
    receipt = initialize({"leaf": policy(layer, "xavier_uniform_zero_bias_v1", gain=0.3)})
    effective = receipt.as_dict()["components"]["leaf"]["parameters"]["weight"]["effective_operation"]
    bound = 0.3 * math.sqrt(6 / 6)
    assert effective == {"kind": "Uniform", "low": -bound, "high": bound}


def test_mismatched_initializer_source_rejected_before_values_or_markers(monkeypatch):
    import particle_sliders.initialization as implementation
    layer = nn.Linear(3, 2)
    before, rng = deepcopy(layer.state_dict()), torch.get_rng_state().clone()
    monkeypatch.setattr(implementation, "_ADMITTED_INIT_SHA256", "different-source")
    with pytest.raises(ValueError, match="initializer source"):
        initialize({"leaf": policy(layer)})
    equal_state(before, layer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(layer, "_particle_sliders_fresh_initialization_sha256")


def test_repeat_guard_follows_original_parameters_across_holders_and_keep_parents():
    leaf = nn.Linear(4, 3, bias=False)
    initialize({"original": policy(leaf)})
    rewrapped = nn.Module()
    rewrapped.register_parameter("weight", leaf.weight)
    before, rng = deepcopy(leaf.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="already initialized parameter"):
        initialize({"rewrapped": policy(rewrapped, distributions={"weight": pg_init.Normal()})})
    equal_state(before, leaf.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(rewrapped, "_particle_sliders_fresh_initialization_sha256")

    parent = nn.Sequential(nn.Linear(4, 3))
    initialize({"preserved-parent": Policy(parent, "keep")})
    before, rng = deepcopy(parent.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="already initialized parameter"):
        initialize({"later-child": policy(parent[0])})
    equal_state(before, parent.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(parent[0], "_particle_sliders_fresh_initialization_sha256")
