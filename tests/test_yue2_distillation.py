"""Regression guard for the data-dependent projection fit."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location('distillation',
    Path(__file__).resolve().parents[1] / 'scripts/distill_yue2_particles.py')
distill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(distill)


class LinearBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, bias=False)

    def forward(self, features, particles):
        return self.linear(features)


def test_linear_teacher_is_recovered_and_zero_is_exact():
    from conceptmod.textsliders.yue2_backend import YuE2Slider, attention_targets
    from conceptmod.textsliders.yue2_particle_bridge import ParticleSlider
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
    torch.manual_seed(9)
    torch.set_num_threads(1)
    model = YuE2ForCausalLM(YuE2Config(hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        intermediate_size=64, max_latent_frames=64)).eval()
    teacher = ParticleSlider(model)
    distill.detach(model, teacher)
    for adapter in teacher.adapters.values():
        adapter.bridge = LinearBridge()
        nn.init.normal_(adapter.lora_up.weight, std=.1)
    student = YuE2Slider(model)
    distill.detach(model, student)
    with torch.inference_mode():
        calibration = distill.Calibration(model, teacher)
        # Distinct training inputs; q/k/v share their real input as in YuE2.
        x = torch.randn(1024, 32)
        for module in attention_targets(model).values():
            module(x)
        calibration.close()
        calibration.solve(student, teacher, 1e-6)
        unseen = torch.randn(257, 32)
        for key, target in teacher.adapters.items():
            actual = student.adapters[key]
            desired = target.lora_up(target.bridge(target.lora_down(unseen), teacher.particles))
            prediction = actual.lora_up(actual.lora_down(unseen))
            torch.testing.assert_close(prediction, desired, atol=2e-6, rtol=2e-5)
        ids = [151643, 151851, 151853, 151854]
        base = distill.hidden(model, ids)
        with distill.attached(model, student, 0):
            assert torch.equal(distill.hidden(model, ids), base)
    assert not any('bridge' in k or 'particles' in k for k in student.state_dict())
