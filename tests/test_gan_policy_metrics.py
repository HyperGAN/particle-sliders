"""Check KL direction, positional normalization and distribution controls."""
import importlib.util
import math
from pathlib import Path
import sys

import torch

DIRECTORY = Path(__file__).resolve().parents[1] / "analysis/gan_bcap"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("gan_policy_evaluate", DIRECTORY / "policy_evaluate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_policy_identity_and_teacher_direction():
    teacher = torch.tensor([[.8, .2], [.3, .7]], dtype=torch.float64).log()
    neutral = torch.tensor([[.5, .5], [.5, .5]], dtype=torch.float64).log()
    exact = module.policy_distances(teacher, teacher, neutral)
    assert torch.allclose(exact["teacher_to_student_kl"], torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(exact["teacher_student_js"], torch.zeros(2, dtype=torch.float64), atol=1e-12)
    unchanged = module.policy_distances(neutral, teacher, neutral)
    expected = .8*math.log(.8/.5)+.2*math.log(.2/.5)
    assert abs(unchanged["teacher_to_student_kl"][0].item()-expected) < 1e-12
    assert torch.equal(unchanged["teacher_kl_improvement"], torch.zeros(2, dtype=torch.float64))
    assert not torch.allclose(unchanged["teacher_to_student_kl"], unchanged["student_to_teacher_kl"])


def test_js_symmetry_logit_shift_and_per_position_summary():
    a = torch.tensor([[4., -3.], [1., 2.], [-5., 6.]], dtype=torch.float64)
    b = -a
    ab = module.policy_distances(a, b, b)
    ba = module.policy_distances(b, a, a)
    shifted = module.policy_distances(a+40, b-20, b+5)
    assert torch.allclose(ab["teacher_student_js"], ba["teacher_student_js"])
    assert torch.all(ab["teacher_student_js"] <= math.log(2)+1e-12)
    for key in ab:
        assert torch.allclose(ab[key], shifted[key], atol=1e-12)
    summary = module.summarize_positions(ab)
    assert summary["audio_start"]["positions"] == 1
    assert summary["continuation"]["positions"] == 2
    assert summary["continuation"]["teacher_to_student_kl"] == float(ab["teacher_to_student_kl"][1:].mean())
