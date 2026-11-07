"""CPU-only canonical PEFT scaling across dynamic adapter keys."""
import torch
from torch import nn

from particle_sliders.native_lora import apply_continuous_lora_scale


class PeftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Linear(2, 2, bias=False)
        self.scaling = {'first': 4., 'second': 2.}


def test_new_adapter_does_not_recapture_already_scaled_peer_baselines():
    layer = PeftLayer()
    apply_continuous_lora_scale(layer, .25)
    assert layer.scaling == {'first': 1., 'second': .5}
    baseline = layer._krea_base_scaling
    layer.scaling['new'] = 6.
    apply_continuous_lora_scale(layer, .5)
    assert layer.scaling == {'first': 2., 'second': 1., 'new': 3.}
    assert layer._krea_base_scaling is baseline
    assert baseline == {'first': 4., 'second': 2., 'new': 6.}
    for scale in (0., 1., -.5, .5, .5, 1.):
        apply_continuous_lora_scale(layer, scale)
        assert layer.scaling == {key: value * scale for key, value in baseline.items()}
    assert not torch.cuda.is_initialized()


def test_adding_adapter_at_zero_keeps_old_baseline_and_removed_keys():
    layer = PeftLayer()
    apply_continuous_lora_scale(layer, 0.)
    del layer.scaling['second']
    layer.scaling['new'] = 3.
    apply_continuous_lora_scale(layer, 0.)
    layer.scaling['second'] = 99.  # A known key retains its original baseline.
    apply_continuous_lora_scale(layer, 1.)
    assert layer.scaling == {'first': 4., 'new': 3., 'second': 2.}
