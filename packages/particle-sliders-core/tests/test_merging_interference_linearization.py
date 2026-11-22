"""CPU gate for the linearization-point claim of Hidekel &
Raviv, "Watch Where You Linearize: Predicting and Correcting
Model-Merging Interference" (arXiv:2611.25000): a first-order
directional derivative of task A's outputs along task B's
task vector, J_A(theta) tau_B, predicts realized merging
interference ONLY when evaluated at the receiver model
theta_A = theta_0 + tau_A. At the receiver it "forecasts
nearby merges per task and per example, halves the accuracy
error of a no-change baseline, and identifies most examples
whose predictions flip"; at the shared pretrained backbone
theta_0 the same derivative "is nearly orthogonal to the
realized changes in our vision experiments and only partly
predictive in LoRA language-model pilots"; controls link
the backbone failure "to displacement along the receiver's
own task vector, not simply parameter distance", appearing
"within the first few dozen fine-tuning steps". Provenance:
cycle 16 of the Particle Sliders paper watch, surfaced from
the model-merging interference literature; sibling of PR
#242's first-order translation test (same first-order
machinery, opposite anchor question: #242 asks whether a
weight edit can be translated at all, this asks WHERE the
Jacobian must be read for the translation to mean anything).

Toy operationalization: a small two-hidden-layer tanh MLP
(the #242-style toy family, deepened -- depth is
load-bearing, see below) with a shared trunk:

    z(x) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3,
    input dim 16, hidden 32, output dim 4.

Each task t is a random linear teacher y = V_t x read on a
task-specific input blob x ~ N(m_t, 0.5^2 I); a copy of the
shared base theta_0 is fine-tuned on each task separately
(full fine-tuning, Adam, fixed dataset of 256 examples), so
tau_t = theta_t - theta_0 is a genuine fine-tuning task
vector, not a synthetic perturbation. Task A is the
receiver; four B variants (independent teachers / blobs /
fine-tune seeds) supply the spread of merges; task C is an
unrelated third task trained for exactly as many steps as
A, so theta_C is displaced from theta_0 by a comparable
parameter distance in an unrelated direction -- the paper's
displacement control.

Readouts on 128 held-out A probes, for merge coefficients
lambda in {0.25, 0.5, 1.0} and each B variant:
  realized drift  Dz = z(theta_A + lambda tau_B) - z(theta_A)
  forecasts       lambda * J_A(theta) tau_B  (one JVP each)
                  at theta in {theta_A (receiver),
                  theta_0 (base), theta_C (third model)},
                  plus a random-direction control: the
                  receiver JVP along a per-tensor
                  norm-matched random vector in place of
                  tau_B.
  drift cosine    cos(forecast, Dz) on the flattened
                  probe x output vector (the paper's drift
                  cosine);
  relative error  ||forecast - Dz|| / ||Dz|| -- the
                  no-change forecast (predict zero drift)
                  scores exactly 1.0, so the paper's
                  "halves the error" is the 0.5 line here.

Measured on pop-os (particlegan 0.10.1), 3 seeds, means
over the 4 B variants (per-lambda detail in the PR body):
  receiver, lambda <= 0.5: cos 0.92 / 0.97 / 0.94,
      rel err at lambda = 0.25: 0.29 / 0.17 / 0.28
  base, lambda <= 0.5:     cos 0.48 / 0.16 / 0.09,
      rel err 1.21-1.44 at every lambda (worse than the
      no-change baseline, as the paper's pretrained-anchor
      LR-Merge also degrades its base merge)
  third model:             cos 0.37 / 0.28 / -0.09,
      rel err 1.19-1.93 -- fails like the base despite
      ||tau_C|| / ||tau_A|| being O(1) (measured per seed
      in the test), so the failure tracks WHICH task vector
      displaced the anchor, not how far it moved.
  random direction:        cos ~0, rel err ~1.3-1.6.
  lambda = 1.0: the single receiver JVP degrades (rel err
      0.49-1.00, cos 0.58-0.87) while remaining the best
      forecast -- the paper's own caveat that a full merge
      needs two evaluations along the path, reproduced.

The honest narrowing, stated plainly: the base anchor is
nearly orthogonal on seeds 1-2 (cos 0.09-0.16) but only
PARTLY predictive-failing on seed 0 (cos 0.48) -- the toy
sits between the paper's vision regime (near-orthogonal)
and its LoRA-pilot regime (partly predictive), and seed 0
lands on the LoRA side. Assertions therefore encode the
pooled near-orthogonality (seed-mean base |cos| <= 0.30),
a per-seed receiver-over-base gap, and per-seed base
error at/above the no-change baseline -- not a per-seed
orthogonality the toy does not demonstrate. A one-hidden-
layer version of this toy (the literal #242 shape) was
tried first and REJECTED by the loop: its base cosine sat
at 0.43-0.60 with receiver error ~0.6 -- the anchor gap
needs the second layer's compounded Jacobian rotation.

What is NOT asserted: the paper's flip-identification and
accuracy-MAE readouts (this toy's outputs are regression
targets, not class logits with a meaningful argmax-flip
semantics); LR-Merge itself (the correction built on the
forecast is a method, not this claim); the "first few
dozen steps" onset timing (fine-tuning here runs to
convergence; onset is not resolved); and anything about
real CLIP / LoRA checkpoints. Two modeling gifts, stated
plainly: teachers are linear and blobs are Gaussian, so
each task is exactly learnable and tau_t is uncontaminated
by underfitting noise; and probes come from A's own blob,
the paper's per-task probe protocol. Pure torch -- no GAN
machinery, because the claim is about where a Jacobian is
evaluated, not training dynamics. Skips cleanly when the
installed particlegan does not provide the EMPF recipe
(the house guard for this paper-test family; PR #211 is
what enables this test in CI).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
    # (any exception type: an ImportError from a failed inner
    # import, or a heavier raise -- both must skip, not error)
    pytest.skip("particlegan unavailable", allow_module_level=True)
try:
    from particlegan import get_recipe  # noqa: E402
except Exception:  # a particlegan without the recipe API at all
    get_recipe = None


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    if get_recipe is None:
        pytest.skip("installed particlegan has no recipe API")
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


DIN, HID, COUT = 16, 32, 4   # toy MLP shape
N_TRAIN = 256                # fixed fine-tuning set per task
N_PROBE = 128                # held-out A probes
SIG = 0.5                    # input blob width
A_STEPS = 400                # receiver / third-task fine-tune steps
B_STEPS = 150                # B-variant fine-tune steps
LR = 0.03                    # Adam step
LAMBDAS = (0.25, 0.5, 1.0)   # merge coefficients
NEAR = (0.25, 0.5)           # the paper's "nearby merges"
N_B = 4                      # B variants per seed


def _base(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return {
        "W1": torch.randn(HID, DIN, generator=g) / np.sqrt(DIN),
        "b1": torch.zeros(HID),
        "W2": torch.randn(HID, HID, generator=g) / np.sqrt(HID),
        "b2": torch.zeros(HID),
        "W3": torch.randn(COUT, HID, generator=g) / np.sqrt(HID),
        "b3": torch.zeros(COUT),
    }


def _fwd(p, x):
    h1 = torch.tanh(x @ p["W1"].T + p["b1"])
    h2 = torch.tanh(h1 @ p["W2"].T + p["b2"])
    return h2 @ p["W3"].T + p["b3"]


def _task(seed):
    """Random linear teacher V and input-blob mean m."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(COUT, DIN, generator=g) * (3.0 / np.sqrt(DIN))
    m = torch.randn(DIN, generator=g) * (3.0 / np.sqrt(DIN))
    return v, m


def _data(v, m, n, seed):
    g = torch.Generator().manual_seed(seed)
    x = m + SIG * torch.randn(n, DIN, generator=g)
    return x, x @ v.T


def _finetune(base, v, m, steps, seed):
    params = {k: t.clone().requires_grad_(True)
              for k, t in base.items()}
    opt = torch.optim.Adam(list(params.values()), lr=LR)
    x, y = _data(v, m, N_TRAIN, seed)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((_fwd(params, x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return {k: t.detach().clone() for k, t in params.items()}


def _jvp(anchor, probes, delta):
    """J(anchor) @ delta on the probe outputs, one JVP."""
    keys = list(anchor.keys())

    def f(*ps):
        return _fwd(dict(zip(keys, ps)), probes)

    _, tang = torch.func.jvp(
        f, tuple(anchor[k] for k in keys),
        tuple(delta[k] for k in keys))
    return tang


def _norm(params):
    return float(torch.sqrt(sum((t ** 2).sum() for t in params.values())))


def run_seed(seed):
    """All forecasts and realized drifts for one seed.

    Returns per-(variant, lambda) cosine / relative error for
    the receiver, base, third-model and random-direction
    forecasts, plus the pooled (variant, lambda <= 0.5,
    probe-entry) vectors for correlation readouts.
    """
    base = _base(seed)
    va, ma = _task(5000 + seed * 10)
    th_a = _finetune(base, va, ma, A_STEPS, 7000 + seed)
    vc, mc = _task(6000 + seed * 10)
    th_c = _finetune(base, vc, mc, A_STEPS, 7100 + seed)
    tau_a = {k: th_a[k] - base[k] for k in base}
    tau_c = {k: th_c[k] - base[k] for k in base}
    xp, _ = _data(va, ma, N_PROBE, 9000 + seed)
    z_a = _fwd(th_a, xp)

    rows = []
    pooled = {n: ([], []) for n in ("recv", "base", "third", "rand")}
    for j in range(N_B):
        vb, mb = _task(8000 + seed * 100 + j * 7)
        th_b = _finetune(base, vb, mb, B_STEPS, 7200 + seed * 10 + j)
        tau_b = {k: th_b[k] - base[k] for k in base}
        g = torch.Generator().manual_seed(1234 + seed * 10 + j)
        tau_r = {}
        for k in base:  # per-tensor norm-matched random direction
            r = torch.randn(tau_b[k].shape, generator=g)
            tau_r[k] = r * (tau_b[k].norm() / (r.norm() + 1e-12))
        preds = {
            "recv": _jvp(th_a, xp, tau_b),
            "base": _jvp(base, xp, tau_b),
            "third": _jvp(th_c, xp, tau_b),
            "rand": _jvp(th_a, xp, tau_r),
        }
        for lam in LAMBDAS:
            merged = {k: th_a[k] + lam * tau_b[k] for k in base}
            realized = _fwd(merged, xp) - z_a
            rn = realized.norm()
            row = {"lam": lam}
            for name, pr in preds.items():
                p = lam * pr
                row[name + "_cos"] = float(torch.nn.functional.cosine_similarity(
                    p.flatten(), realized.flatten(), dim=0))
                row[name + "_err"] = float((p - realized).norm() / rn)
                if lam in NEAR:
                    pooled[name][0].append(p.flatten().numpy())
                    pooled[name][1].append(realized.flatten().numpy())
            rows.append(row)
    corr = {}
    for name, (ps, rs) in pooled.items():
        p = np.concatenate(ps)
        r = np.concatenate(rs)
        corr[name] = float(np.corrcoef(p, r)[0, 1])
    return dict(rows=rows, corr=corr,
                tau_ratio=_norm(tau_c) / _norm(tau_a))


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def _mean(rows, key, lams):
    return float(np.mean([r[key] for r in rows if r["lam"] in lams]))


def test_receiver_local_forecast_tracks_interference():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = _seed(seed)
        rows = d["rows"]
        # Nearby merges: the receiver JVP points along the
        # realized drift (demonstrated mean cos 0.88-0.99
        # per lambda) and its pooled per-entry correlation
        # with the realized drift is high.
        assert _mean(rows, "recv_cos", NEAR) >= 0.80, (
            f"seed {seed}: receiver drift cosine at nearby "
            f"merges {_mean(rows, 'recv_cos', NEAR):.3f}")
        assert d["corr"]["recv"] >= 0.75, (
            f"seed {seed}: receiver pooled correlation "
            f"{d['corr']['recv']:.3f}")
        # At lambda = 0.25 the receiver forecast's relative
        # error is at most 0.40 -- clearly below half the
        # no-change baseline's 1.0 (demonstrated 0.17-0.29;
        # the paper's halving, in this toy's drift metric).
        assert _mean(rows, "recv_err", (0.25,)) <= 0.40, (
            f"seed {seed}: receiver rel err at lambda=0.25 "
            f"{_mean(rows, 'recv_err', (0.25,)):.3f}")
        assert _mean(rows, "recv_err", NEAR) <= 0.55, (
            f"seed {seed}: receiver rel err at nearby merges "
            f"{_mean(rows, 'recv_err', NEAR):.3f}")


def test_base_point_forecast_is_near_orthogonal():
    _empf_recipe()
    base_cos = []
    for seed in (0, 1, 2):
        d = _seed(seed)
        rows = d["rows"]
        bc = _mean(rows, "base_cos", NEAR)
        base_cos.append(abs(bc))
        # The base forecast is never good: its error stays
        # at/above the no-change baseline at nearby merges
        # (demonstrated 1.21-1.44) -- reading the Jacobian
        # at theta_0 certifies nothing about the merge.
        assert _mean(rows, "base_err", NEAR) >= 0.90, (
            f"seed {seed}: base rel err at nearby merges "
            f"{_mean(rows, 'base_err', NEAR):.3f}")
        # And the receiver beats it by a wide per-seed gap
        # (demonstrated 0.44-0.85).
        gap = _mean(rows, "recv_cos", NEAR) - bc
        assert gap >= 0.35, (
            f"seed {seed}: receiver-base cosine gap {gap:.3f}")
        assert d["corr"]["base"] <= d["corr"]["recv"] - 0.25, (
            f"seed {seed}: base pooled correlation "
            f"{d['corr']['base']:.3f} vs receiver "
            f"{d['corr']['recv']:.3f}")
    # Pooled across seeds the base forecast is nearly
    # orthogonal (demonstrated seed-mean |cos| ~0.24;
    # per-seed orthogonality is NOT asserted -- seed 0 is
    # partly predictive, the paper's LoRA-pilot regime).
    assert float(np.mean(base_cos)) <= 0.30, (
        f"seed-mean base |cos| {np.mean(base_cos):.3f}")


def test_controls_fail_like_the_base():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = _seed(seed)
        rows = d["rows"]
        # Apparatus: the third model is displaced from the
        # base by a distance comparable to the receiver's
        # (demonstrated ratio O(1)); its forecast failure
        # below therefore tracks the displacement's
        # DIRECTION (an unrelated task vector), not distance.
        assert 0.4 <= d["tau_ratio"] <= 2.5, (
            f"seed {seed}: ||tau_C|| / ||tau_A|| "
            f"{d['tau_ratio']:.2f}")
        # Third-model anchor: fails like the base -- error
        # at/above baseline, cosine far below the receiver's
        # (demonstrated cos -0.09..0.37, err 1.19-1.93).
        assert _mean(rows, "third_err", NEAR) >= 0.90, (
            f"seed {seed}: third-model rel err "
            f"{_mean(rows, 'third_err', NEAR):.3f}")
        assert _mean(rows, "third_cos", NEAR) <= 0.45, (
            f"seed {seed}: third-model drift cosine "
            f"{_mean(rows, 'third_cos', NEAR):.3f}")
        assert (_mean(rows, "recv_cos", NEAR)
                - _mean(rows, "third_cos", NEAR)) >= 0.35, (
            f"seed {seed}: receiver-third cosine gap")
        # Random direction at the receiver: no signal
        # (demonstrated |cos| <= 0.22, err >= 1.28), so the
        # receiver forecast's power is the direction tau_B
        # read at theta_A, not the anchor alone.
        assert abs(_mean(rows, "rand_cos", NEAR)) <= 0.30, (
            f"seed {seed}: random-direction drift cosine "
            f"{_mean(rows, 'rand_cos', NEAR):.3f}")
        assert _mean(rows, "rand_err", NEAR) >= 0.90, (
            f"seed {seed}: random-direction rel err "
            f"{_mean(rows, 'rand_err', NEAR):.3f}")
        # Full-merge degradation (the paper's two-evaluation
        # caveat): the single receiver JVP's error grows
        # from lambda = 0.25 to lambda = 1.0 on every seed
        # (demonstrated 0.17-0.29 -> 0.49-1.00), while
        # remaining the best of the four forecasts.
        assert (_mean(rows, "recv_err", (1.0,))
                > _mean(rows, "recv_err", (0.25,))), (
            f"seed {seed}: receiver rel err did not grow "
            f"toward the full merge")
        assert (_mean(rows, "recv_err", (1.0,))
                <= _mean(rows, "base_err", (1.0,))), (
            f"seed {seed}: receiver no longer best at "
            f"lambda=1.0")
