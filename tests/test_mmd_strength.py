import copy

import pytest
import torch

from conceptmod.textsliders.lora import LoRAModule
from analysis.gan_bcap.mmd_game_20260905.calibrate_strength import calibrated_tensors


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_baked_alpha_exactly_matches_parent_multiplier(dtype):
    torch.manual_seed(12)
    base = torch.nn.Linear(6, 4, bias=False).to(dtype)
    clone = copy.deepcopy(base)
    parent = LoRAModule('test', base, multiplier=1.25, lora_dim=2, alpha=2.)
    parent.lora_up.weight.data.normal_()
    parent.apply_to()
    source = parent.state_dict()
    modified, sidecar, count = calibrated_tensors(
        {'layer.'+key: value for key, value in source.items()}, dict(alpha=2., unit_scale=1.), 1.25)
    candidate = LoRAModule('test', clone, multiplier=1., lora_dim=2, alpha=sidecar['alpha'])
    candidate.apply_to()
    candidate.load_state_dict({key.removeprefix('layer.'): value for key, value in modified.items()})
    x = torch.randn(2, 5, 6).to(dtype)
    assert torch.equal(base(x), clone(x))
    assert count == 1
    for key, value in source.items():
        if key != 'alpha':
            assert torch.equal(value, candidate.state_dict()[key])


def test_strength_rejects_inconsistent_alpha():
    with pytest.raises(ValueError, match='Alpha tensor'):
        calibrated_tensors({'layer.alpha': torch.tensor(4.)}, dict(alpha=8.), 1.25)
