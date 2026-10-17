"""Optional upstream reuse, native geometry, grouping and training isolation."""
import importlib
import json
import random
from types import SimpleNamespace

import pytest
import torch

from particle_sliders import hypergan_diagnostics as diagnostics


@pytest.fixture
def upstream():
    return pytest.importorskip("hypergan.diversity_metrics")


def test_module_import_does_not_load_optional_hypergan(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("optional package loaded during import")
    monkeypatch.setattr(diagnostics.importlib, "import_module", forbidden)
    importlib.reload(diagnostics)


def test_missing_optional_package_has_exact_setup_guidance(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(diagnostics.importlib, "import_module", missing)
    with pytest.raises(diagnostics.HyperGANDiagnosticsUnavailable, match="separate evaluation environment") as exc:
        diagnostics.response_diversity({"condition": []})
    assert diagnostics.HYPERGAN_REVIEWED_COMMIT in str(exc.value)
    assert "hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@" in str(exc.value)


@pytest.mark.parametrize("version", [(3, 10), (3, 15)])
def test_optional_python_range_does_not_change_base_package(version, monkeypatch):
    monkeypatch.setattr(diagnostics.sys, "version_info", version)
    with pytest.raises(diagnostics.HyperGANDiagnosticsUnavailable, match="Python >=3.11,<3.15"):
        diagnostics.response_diversity({"condition": []})


def test_unreviewed_source_fails_closed(tmp_path, monkeypatch):
    changed = tmp_path / "diversity_metrics.py"
    changed.write_text("# same version, different implementation\n")
    monkeypatch.setattr(diagnostics.importlib, "import_module", lambda _: SimpleNamespace(__file__=str(changed)))
    with pytest.raises(diagnostics.HyperGANDiagnosticsUnavailable, match="differs from the reviewed"):
        diagnostics.response_diversity({"condition": []})


def test_unreadable_source_has_setup_guidance(monkeypatch):
    monkeypatch.setattr(diagnostics.importlib, "import_module", lambda _: SimpleNamespace(__file__=None))
    with pytest.raises(diagnostics.HyperGANDiagnosticsUnavailable, match="Cannot verify"):
        diagnostics.response_diversity({"condition": []})


@pytest.mark.parametrize("groups", [{}, [], {"": []}, {" ": []}, {1: []}])
def test_groups_require_explicit_condition_ids(groups):
    with pytest.raises(ValueError, match="condition"):
        diagnostics.response_diversity(groups)


@pytest.mark.parametrize("shape", [(), (7,), (2, 5), (2, 3, 4), (2, 3, 4, 5)])
def test_native_response_shapes_stream_singletons_with_upstream_parity(shape, upstream):
    width = 1
    for dimension in shape:
        width *= dimension
    reference = torch.arange(7 * width, dtype=torch.float64).reshape(7, *shape).sin() + 1e5
    generated = reference * .25 - 31
    groups = {"fixed-condition": (
        {"generated": g, "reference": r}
        for g, r in zip(generated.split(1), reference.split(1))) }
    result = diagnostics.response_diversity(groups)
    report = result["groups"]["fixed-condition"]
    assert report["metrics"] == pytest.approx(upstream.batch_diversity(generated, reference)["metrics"])
    coordinates = reference.reshape(7, -1)
    oracle = (torch.pdist(coordinates).square().mean() / width).sqrt().item()
    assert report["metrics"]["reference_rms"] == pytest.approx(oracle, rel=1e-11)
    assert report["metrics"]["ratio"] == pytest.approx(.25, rel=1e-10)
    assert report["generated_shape"] == report["reference_shape"] == shape
    assert report["generated_count"] == report["reference_count"] == 7
    assert not report["unavailable"]
    assert result["implementation"]["reviewed_commit"] == diagnostics.HYPERGAN_REVIEWED_COMMIT
    assert result["implementation"]["source_sha256"] == diagnostics.HYPERGAN_DIVERSITY_SHA256
    json.dumps(result, allow_nan=False)


def test_single_pass_unequal_sample_counts_are_never_repeated_or_truncated(upstream):
    generated = torch.tensor([[0.], [1.], [2.], [7.], [13.]])
    reference = torch.tensor([[0.], [4.], [17.]])
    batches = iter([{"generated": generated[:1], "reference": reference[:2]},
                    {"generated": generated[1:], "reference": reference[2:]}])
    result = diagnostics.response_diversity({"one-condition": batches})["groups"]["one-condition"]
    assert result["metrics"] == pytest.approx(upstream.batch_diversity(generated, reference)["metrics"])
    assert result["generated_count"] == 5 and result["reference_count"] == 3
    assert list(batches) == []


def test_condition_offsets_cannot_hide_within_condition_collapse(upstream):
    reference = torch.tensor([[0.], [1.], [4.]])
    result = diagnostics.response_diversity({
        "condition-A": [{"generated": torch.zeros(3, 1), "reference": reference}],
        "condition-B": [{"generated": torch.ones(3, 1) * 100, "reference": reference + 100}],
    })
    assert set(result["groups"]) == {"condition-A", "condition-B"}
    for group in result["groups"].values():
        assert group["metrics"]["generated_rms"] == group["metrics"]["ratio"] == 0
    assert "metrics" not in result


@pytest.mark.parametrize("count", [0, 1])
def test_insufficient_group_samples_report_unavailable(count, upstream):
    batches = [{"generated": torch.ones(1, 3), "reference": torch.ones(1, 3)}] * count
    result = diagnostics.response_diversity({"fixed": batches})["groups"]["fixed"]
    assert result["metrics"] == {}
    assert set(result["unavailable"]) == {"generated_rms", "reference_rms", "ratio"}
    json.dumps(result, allow_nan=False)


def test_constant_reference_does_not_create_infinite_ratio(upstream):
    result = diagnostics.response_diversity({"fixed": [{
        "generated": torch.tensor([[0.], [1.], [4.]]), "reference": torch.ones(2, 1)
    }]})["groups"]["fixed"]
    assert result["metrics"]["reference_rms"] == 0
    assert "ratio" not in result["metrics"]
    assert "nonzero reference" in result["unavailable"]["ratio"]


def test_native_geometry_mismatch_stays_undefined_after_pooling(upstream):
    reference = torch.arange(128.).reshape(2, 1, 8, 8)
    result = diagnostics.response_diversity({"fixed": [{
        "generated": reference, "reference": reference[:, :, :4, :4]
    }]}, pool_size=2)["groups"]["fixed"]
    assert set(result["metrics"]) == {"generated_rms", "reference_rms"}
    assert "matching" in result["unavailable"]["ratio"]


def test_opt_in_image_pooling_matches_upstream(upstream):
    reference = torch.arange(256.).reshape(4, 1, 8, 8).cos()
    result = diagnostics.response_diversity({"image": [{
        "generated": reference * 3, "reference": reference
    }]}, pool_size=4)["groups"]["image"]
    assert result["metrics"] == pytest.approx(upstream.batch_diversity(reference * 3, reference, pool_size=4)["metrics"])


@pytest.mark.parametrize("bad,reason", [
    (torch.tensor([[float("nan")]]), "finite"),
    (torch.ones(2, 1, dtype=torch.int64), "floating"),
    (torch.empty(0, 1), "empty"),
])
def test_invalid_batch_after_valid_data_never_emits_partial_metrics(bad, reason, upstream):
    batches = [{"generated": torch.ones(2, 1), "reference": torch.ones(2, 1)},
               {"generated": bad, "reference": torch.ones(2, 1)}]
    with pytest.raises(ValueError, match=reason):
        diagnostics.response_diversity({"fixed": batches})


def test_changed_stream_shape_rejected_and_vectors_are_not_image_pooled(upstream):
    with pytest.raises(ValueError, match="shape changed"):
        diagnostics.response_diversity({"fixed": [
            {"generated": torch.ones(1, 2), "reference": torch.ones(1, 2)},
            {"generated": torch.ones(1, 3), "reference": torch.ones(1, 2)}]})
    with pytest.raises(ValueError, match="NCHW"):
        diagnostics.response_diversity({"fixed": [
            {"generated": torch.ones(2, 4), "reference": torch.ones(2, 4)}]}, pool_size=2)


def test_batch_fields_are_unambiguous(upstream):
    with pytest.raises(ValueError, match="exactly generated and reference"):
        diagnostics.response_diversity({"fixed": [{"generated": torch.ones(2, 4)}]})


def test_observation_preserves_inputs_gradients_rng_and_backend_policy(upstream):
    generated = torch.arange(12., requires_grad=True).reshape(4, 3)
    generated.retain_grad()
    generated.square().sum().backward()
    reference = generated.detach() + 1
    values, gradients = generated.detach().clone(), generated.grad.clone()
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    flags = (torch.is_grad_enabled(), torch.backends.cudnn.benchmark,
             torch.are_deterministic_algorithms_enabled(), torch.cuda.is_initialized())
    diagnostics.response_diversity({"fixed": [{"generated": generated, "reference": reference}]})
    assert torch.equal(values, generated) and torch.equal(gradients, generated.grad)
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert flags == (torch.is_grad_enabled(), torch.backends.cudnn.benchmark,
                     torch.are_deterministic_algorithms_enabled(), torch.cuda.is_initialized())


def test_observing_evaluate_preserves_exact_game_resume_and_next_update(upstream):
    from test_conditional_pg09 import fixture, equal
    game = fixture()
    game.step()
    before = game.state_dict()
    diagnostics.response_diversity({"one-frozen-row": (
        {"generated": game.evaluate(0, seed=seed), "reference": game.rows[0].target}
        for seed in [3, 5, 7])})
    equal(before, game.state_dict())
    restored = fixture()
    restored.load_state_dict(before)
    assert game.step() == restored.step()
    equal(game.state_dict(), restored.state_dict())
    assert not torch.cuda.is_initialized()
