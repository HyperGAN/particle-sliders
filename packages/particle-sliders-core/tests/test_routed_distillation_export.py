"""Routed down-projection fitting through the current native ordinary LoRA API."""

from copy import deepcopy

import torch
from safetensors.torch import load_file
from torch import nn

from particle_sliders import fit_routed_down
from particle_sliders.native_lora import LoRANetwork
from particle_sliders.routed_projection import TinyParticleProjection
from particle_sliders.weight_policy import require_raw_checkpoint


@torch.no_grad()
def test_captured_routed_features_roundtrip_as_a_raw_ordinary_adapter(tmp_path):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        frozen = nn.Linear(6, 5).eval().requires_grad_(False)
        teacher_host, student_host, reload_host = (deepcopy(frozen) for _ in range(3))
        bank = nn.Parameter(torch.randn(128, 4))
        teacher = TinyParticleProjection('projection', teacher_host, bank, rank=3, alpha=6.)
        teacher.lora_up.weight.normal_(std=.3)
        teacher.multiplier = 1.
        calibration = torch.randn(32, 6)
        heldout = torch.randn(8, 6) + .3

    # Capture the actual intermediates at the same cut points used by native
    # routed fits. The teacher's up factor carries the full edit directions.
    captured = []
    with teacher.lora_up.register_forward_pre_hook(lambda _module, args: captured.append(args[0].clone())):
        teacher_host(calibration)
    routed, = captured
    x, features = calibration.double(), routed.double()
    fit, ridge = fit_routed_down(x, features)
    covariance = x.T @ x + ridge * torch.eye(x.shape[1], dtype=x.dtype)
    torch.testing.assert_close(fit @ covariance, features.T @ x)
    fit = fit.to(dtype=calibration.dtype)

    def ordinary(host):
        return LoRANetwork(host, rank=3, alpha=6., multiplier=0.,
                           target_plan=[('projection', host)])

    student = ordinary(student_host)
    branch = student.unet_loras[0]
    branch.lora_down.weight.copy_(fit)
    branch.lora_up.weight.copy_(teacher.lora_up.weight)
    path = tmp_path / 'fitted.safetensors'
    student.save_weights(str(path), metadata={'weights_kind': 'raw', 'ema_decay': 'null'})

    require_raw_checkpoint(path)
    reloaded = ordinary(reload_host)
    reloaded.load_state_dict(load_file(str(path)), strict=True)
    base = frozen(heldout)
    assert torch.equal(reload_host(heldout), base)
    expected = base + 2. * teacher.lora_up(heldout @ fit.T)
    with reloaded:
        torch.testing.assert_close(reload_host(heldout), expected)
    assert torch.equal(reload_host(heldout), base)
    assert not torch.equal(expected, base)
    assert torch.equal(reloaded.unet_loras[0].lora_up.weight, teacher.lora_up.weight)
