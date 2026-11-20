"""CPU gate for the ASH augment claim (arXiv:2611.23124): a steering signal
trained WITH a margin term -- penalizing insufficient separation between
steered-plain and plain behavior relative to hinted behavior -- is a better
steering signal than the same signal trained by plain imitation alone.

Paper: "Get Hint-Augmented GRPO Done Right Through Activation Steering"
(Le, Hua & Chawla, 2026). ASH trains a steering vector so the behavior a
hint induces in context survives hint removal. Read through the Particle
Sliders augment lens (NOT as an adapter replacement): the salvageable piece
is the margin objective as an auxiliary training signal -- does adding a
margin term to steering-signal training improve the signal itself, over the
same parameterization trained without it?

Toy operationalization: hidden states h in R^16 with a linear behavior
readout: the correct behavior fires when r . h clears a bias threshold
(the plain policy is biased/diffuse and rarely fires). A hint shifts the
state by delta = A * r + B * n, where r is the behavior direction and n is
an orthogonal hint-context artifact direction (the hint's own tokens leave
a large activation trace that does nothing for behavior). A = 1.8 is the
behavior-relevant shift; B = 3.0 is the artifact -- imitation error is
dominated by the artifact. Both arms learn one steering vector s deployed
hint-free at the SAME fixed norm budget ||s|| = R = 1.5 (steering is
deployed at a calibrated strength; an unbounded vector is not the regime
either arm lives in):

  (a) imitation arm: minimize E ||h + s - h_hinted||^2  (mean-matching);
  (b) margin arm:    the same imitation loss PLUS a hinge term,
      relu(separation_hinted - separation_steered), where separation is
      the correct-behavior logit gap vs the plain state -- the steered
      state must separate from plain the way the hinted state does.

Under the norm budget the imitation optimum is the projected mean shift,
which spends most of its budget on the artifact (r-fraction A/sqrt(A^2+B^2)
~ 0.51); the margin term reallocates budget onto r. Deployment is hint-free
held-out accuracy of the correct behavior, against no-steer, shuffled-
direction and reversed-direction controls. Skips cleanly when the
installed particlegan does not provide the EMPF recipe.
"""
import pytest
import torch

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 16
A_BEHAV, B_ART = 1.8, 3.0     # hint shift: behavior component / artifact
R_BUDGET = 1.5                # fixed deployment norm for both arms
BIAS = 1.0                    # plain policy bias against correct behavior
SIGMA = 0.5                   # plain-state scale
LAMBDA_MARGIN = 8.0
STEPS = 400


def _world(seed):
    g = torch.Generator().manual_seed(7000 + seed)
    r = torch.randn(D, generator=g)
    r = r / r.norm()
    n = torch.randn(D, generator=g)
    n = n - (n @ r) * r
    n = n / n.norm()
    return g, r, n


def _data(g, r, n, count):
    h = torch.randn(count, D, generator=g) * SIGMA
    a = A_BEHAV + 0.2 * torch.randn(count, generator=g)
    b = B_ART + 0.4 * torch.randn(count, generator=g)
    hinted = h + a[:, None] * r[None, :] + b[:, None] * n[None, :]
    return h, hinted


def _train_arm(margin, seed, steps=STEPS, count=2048):
    g, r, n = _world(seed)
    h, hinted = _data(g, r, n, count)
    torch.manual_seed(900 + seed)
    w = torch.randn(D)
    w = torch.nn.Parameter(w / w.norm())
    opt = torch.optim.Adam([w], lr=0.05)
    sep_hint = (hinted - h) @ r  # hinted separation in behavior logit
    for _ in range(steps):
        s = R_BUDGET * w / w.norm()
        loss = ((h + s - hinted) ** 2).sum(1).mean()
        if margin:
            sep_steer = s @ r  # steered separation vs plain (same h)
            hinge = torch.relu(sep_hint - sep_steer).mean()
            loss = loss + LAMBDA_MARGIN * hinge
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return R_BUDGET * w / w.norm()


def _accuracy(s, seed, count=2048):
    g, r, n = _world(seed)
    ge = torch.Generator().manual_seed(31337 + seed)
    h, _ = _data(ge, r, n, count)
    with torch.no_grad():
        score = h @ r - BIAS
        if s is not None:
            score = score + s @ r
        return (score > 0).float().mean().item()


def run_seed(seed):
    g, r, n = _world(seed)
    s_imit = _train_arm(False, seed)
    s_marg = _train_arm(True, seed)
    gp = torch.Generator().manual_seed(4242 + seed)
    s_shuf = s_marg[torch.randperm(D, generator=gp)]
    delta_mean = A_BEHAV * r + B_ART * n
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))  # noqa: E731
    return dict(
        plain=_accuracy(None, seed),
        imit=_accuracy(s_imit, seed),
        marg=_accuracy(s_marg, seed),
        shuf=_accuracy(s_shuf, seed),
        rev=_accuracy(-s_marg, seed),
        cos_imit_delta=cos(s_imit, delta_mean),
        cos_imit_r=cos(s_imit, r),
        cos_marg_r=cos(s_marg, r),
    )


def test_margin_objective_improves_steering_signal():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Training sanity: the imitation arm reaches its own optimum --
        # it recovers the mean hint shift direction almost exactly, so
        # its shortfall is the objective's, not the optimizer's.
        assert r["cos_imit_delta"] > 0.95, (
            f"seed {seed}: imitation cos to mean shift {r['cos_imit_delta']:.3f}")
        # Steering at all must beat doing nothing, for both arms.
        assert r["imit"] > r["plain"] + 0.10, (
            f"seed {seed}: imitation {r['imit']:.3f} vs plain {r['plain']:.3f}")
        assert r["marg"] > r["plain"] + 0.10
        # The margin term reallocates the norm budget onto the behavior
        # direction.
        assert r["cos_marg_r"] > r["cos_imit_r"] + 0.25, (
            f"seed {seed}: cos to behavior dir margin {r['cos_marg_r']:.3f} "
            f"vs imitation {r['cos_imit_r']:.3f}")
        # The paper's augment claim, deployed hint-free: the margin-trained
        # signal beats the imitation-trained signal by a clear margin,
        # and beats the shuffled / reversed specificity controls.
        assert r["marg"] > r["imit"] + 0.25, (
            f"seed {seed}: margin acc {r['marg']:.3f} vs imitation {r['imit']:.3f}")
        assert r["marg"] > r["shuf"] + 0.25, (
            f"seed {seed}: margin acc {r['marg']:.3f} vs shuffled {r['shuf']:.3f}")
        assert r["marg"] > r["rev"] + 0.25, (
            f"seed {seed}: margin acc {r['marg']:.3f} vs reversed {r['rev']:.3f}")
