import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from concept_slider_core import SliderRecipe, EndpointGame, teacher_poles, require_same_critic
from concept_slider_core import teachers
from concept_slider_core.grad_regularizers import GradRegularizer

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[3]


def test_reference_sources_exact():
    package = Path(teachers.__file__).parent
    meta = json.loads((package / "locked_provenance.json").read_text())
    assert hashlib.sha256((package / "grad_regularizers.py").read_bytes()).hexdigest() == meta["grad_sha256"]
    upstream = (ROOT / meta["teacher_source"]).read_text()
    extracted = Path(teachers.__file__).read_text()
    def functions(text):
        return {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    a, b = functions(upstream), functions(extracted)
    assert all(a[name] == b[name] for name in meta["teacher_functions"])


@pytest.mark.parametrize("step", [1, 4])
@pytest.mark.parametrize("kappa", [.1, 1.7])
def test_true_cap_loss_and_parameter_gradients(step, kappa):
    torch.manual_seed(7)
    critic = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)).double()
    real, fake = torch.randn(4, 3, dtype=torch.double), torch.randn(4, 3, dtype=torch.double)
    actual, _ = GradRegularizer(coeff=1.3, kappa=kappa, lazy_k=4).penalty(critic, real, fake, step=step)
    if step % 4:
        assert actual == 0 and not actual.requires_grad
        return
    terms = []
    for x in (real, fake):
        x = x.detach().requires_grad_(True)
        g = torch.autograd.grad(critic(x).sum(), x, create_graph=True)[0]
        terms.append(torch.relu((g.square().sum(1) + 1e-12).sqrt() - kappa).square().mean())
    expected = 1.3 * 4 / 2 * sum(terms)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ga = torch.autograd.grad(actual, tuple(critic.parameters()), allow_unused=True)
    gb = torch.autograd.grad(expected, tuple(critic.parameters()), allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_teacher_guard_admits_and_rejects():
    cfg = SliderRecipe()
    n = torch.zeros(2)
    for p, expected_mode in [(torch.tensor([3., .1]), "guard_admitted"),
                             (torch.tensor([.1, 3.]), "guard_fallback_raw_poles")]:
        plus, minus, mode = teacher_poles(cfg, p, -p, n,
            leak_dir=torch.tensor([0., 1.]), slider_dir=torch.tensor([1., 0.]))
        assert mode == expected_mode
        assert torch.equal(plus, p) if "fallback" in mode else plus[1] == 0
        assert torch.equal(minus, -plus)
    assert teacher_poles(cfg, p, -p, n)[2] == "no_declared_leak_raw_poles"
    with pytest.raises(ValueError, match="explicit slider"):
        teacher_poles(cfg, p, -p, n, leak_dir=p)


def setup(microbatch=1, **changes):
    torch.manual_seed(4)
    adapter = torch.nn.Linear(3, 3, bias=False).double()
    critic = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()
    config = SliderRecipe.from_dict(dict(SliderRecipe().to_dict(), **changes))
    game = EndpointGame(adapter, critic, config, 12, microbatch=microbatch)
    x = torch.randn(12, 3, dtype=torch.double)
    y = torch.randn(12, 3, dtype=torch.double)
    @contextmanager
    def predict(ids, sign):
        yield adapter(x[ids]) * sign
    def target(ids):
        return y[ids], -y[ids]
    return game, predict, target


def test_full_microbatch_and_resume():
    a, pa, ta = setup(1)
    b, pb, tb = setup(8)
    a.update(pa, ta, 1)
    b.update(pb, tb, 1)
    for module_a, module_b in ((a.adapter, b.adapter), (a.critic, b.critic)):
        for x, y in zip(module_a.parameters(), module_b.parameters()):
            torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-12)
    saved = copy.deepcopy(a.state_dict()), copy.deepcopy(a.adapter.state_dict())
    result = a.update(pa, ta, 2)
    expected = copy.deepcopy(a.adapter.state_dict())
    a.load_state_dict(saved[0]); a.adapter.load_state_dict(saved[1])
    assert a.update(pa, ta, 2) == result
    assert all(torch.equal(value, a.adapter.state_dict()[key]) for key, value in expected.items())


def test_cover_and_pole_are_operative_and_sum_endpoints():
    a, pa, ta = setup(cover_weight=0., pole_weight=0.)
    b, pb, tb = setup(cover_weight=1., pole_weight=1.)
    x, y = a.update(pa, ta, 1), b.update(pb, tb, 1)
    assert y["cover"] == y["pole"] > 0
    assert x["d_rows"] == y["d_rows"] and x["g_rows"] == y["g_rows"]
    assert not torch.equal(a.adapter.weight.grad, b.adapter.weight.grad)
    assert x["d_rows"] != x["g_rows"]


def test_config_rejects_drift_and_unknown_settings():
    SliderRecipe().require_locked_shared()
    with pytest.raises(ValueError, match="drift"):
        SliderRecipe(cover_weight=1.5).require_locked_shared()
    for setting in ({"cover_wieght": 1}, {"noise_hold": 1}, {"particles": 128}, {"grad_every": 1.5}):
        with pytest.raises(ValueError):
            SliderRecipe.from_dict(setting)
    with pytest.raises(ValueError, match="Critic changed"):
        require_same_critic({"arch": "gmix"}, {"arch": "mlp"})
