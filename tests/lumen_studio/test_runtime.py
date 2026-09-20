import json
from types import SimpleNamespace

import pytest
import torch

from lumen_studio.backends.anima import TurboRuntime
from lumen_studio.provenance import cache_runtime_compatibility


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_anima_runtime_supplies_singleton_mask_to_broadcasting_transformer(batch):
    runtime = TurboRuntime.__new__(TurboRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.bfloat16
    runtime.scale = 8
    runtime.pipe = SimpleNamespace(scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))
    def transformer(hidden_states, padding_mask, timestep, encoder_hidden_states, return_dict):
        # Match the pinned Cosmos operation that failed on the real model.
        mask = torch.nn.functional.interpolate(padding_mask, size=hidden_states.shape[-2:])
        joined = torch.cat([hidden_states, mask.unsqueeze(2).repeat(batch, 1, 1, 1, 1)], dim=1)
        assert joined.shape == (batch, 5, 1, 2, 2)
        assert timestep.shape == (batch,) and encoder_hidden_states.shape[0] == batch
        return (hidden_states,)
    runtime.transformer = transformer
    latent = torch.randn(batch, 4, 1, 2, 2)
    value = runtime.predict(latent, torch.linspace(1000, 300, batch), torch.ones(batch, 3, 4))
    assert torch.equal(value, latent.to(torch.bfloat16).float())


def test_cache_compatibility_requires_bound_exact_teacher_evidence(tmp_path):
    before = dict(model="test", sha256="weights", runtime_sha256="old", environment_sha256="env")
    after = dict(before, runtime_sha256="new")
    assert cache_runtime_compatibility(None, before, before, "cache") is None
    with pytest.raises(ValueError, match="no compatibility"):
        cache_runtime_compatibility(tmp_path, before, after, "cache")
    proof = dict(before=before, after=after, passed=True, purpose="single-image-teacher-cache-reuse", single_image_max_abs=0,
                 positions=list(range(10)), batch_sizes=[1,2,4], cache_fingerprints=["cache"])
    path = tmp_path / 'runtime-compatibility.json'
    path.write_text(json.dumps(proof))
    assert len(cache_runtime_compatibility(tmp_path, before, after, "cache")) == 64
    with pytest.raises(ValueError, match="does not cover"):
        cache_runtime_compatibility(tmp_path, before, after, "other-cache")
    with pytest.raises(ValueError, match="does not cover"):
        cache_runtime_compatibility(tmp_path, before, dict(after, sha256="changed"), "cache")
    proof['single_image_max_abs'] = .001
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="does not cover"):
        cache_runtime_compatibility(tmp_path, before, after, "cache")
