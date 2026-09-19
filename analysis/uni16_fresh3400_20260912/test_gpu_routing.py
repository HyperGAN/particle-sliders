from copy import deepcopy
import pytest

from gpu_routing import route_catalog


def test_routes_remaining_jobs_once_without_changing_recipe_or_catalog():
    catalog = [dict(id="finished", gpu=1, seed=7),
               dict(id="resume", gpu=0, warmup_signature={"recipe": "original"}),
               dict(id="queued", gpu=0, train_prompts="original.yaml")]
    before = deepcopy(catalog)
    routed = route_catalog(catalog, dict(version=1, training_manifest_sha256="same",
        overrides={"resume": 1, "queued": 1}), "same")
    lanes = {gpu: [r["id"] for r in routed if r["gpu"] == gpu] for gpu in (0, 1)}
    assert lanes == {0: [], 1: ["finished", "resume", "queued"]}
    assert catalog == before
    assert [dict(r, gpu=o["gpu"]) for r,o in zip(routed,catalog)] == catalog


@pytest.mark.parametrize("overrides", [{"unknown": 1}, {"known": 2}, {"known": True}])
def test_invalid_routes_are_rejected(overrides):
    with pytest.raises(ValueError, match="known sliders"):
        route_catalog([dict(id="known",gpu=0)], dict(version=1,
            training_manifest_sha256="same",overrides=overrides), "same")


def test_routes_cannot_be_applied_to_another_training_manifest():
    with pytest.raises(ValueError, match="different campaign"):
        route_catalog([], dict(version=1,training_manifest_sha256="old"), "new")
