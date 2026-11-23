"""CPU gate for the composition-coefficient (scale) claim
that PR #307 and PR #308 left open in the harvested-delta
receiver class of PR #282.

Chain: PR #282 (arXiv:2611.25000) proved merging
interference is predictable only anchored at the RECEIVER
(adapted checkpoint), so every readout here is anchored
at the receiver. PR #307 (TIES, arXiv:2306.01708) found,
in this class, that plain summation is catastrophic
(lossret -0.16/-2.44/-0.87), plain AVERAGING beats TIES
on every seed (0.59/0.19/0.45 vs 0.47/-0.35/0.04), and
random trim matches TIES -- the operative lever was
composed-delta SCALE, not sign arbitration. PR #308
(DARE, arXiv:2311.03099) arbitrated the scale lever from
the sparsification side: the 1/(1-p) rescale is an
unbiased scale correction in drift (proj ~ 1.00) but
loss-unstable at p=0.9, DARE-merge never beats averaging,
and drop-WITHOUT-rescale repairs the full sum --
shrinkage again. This probe prices the scale lever
DIRECTLY: composed delta = gamma * mean(tau_t) for
gamma in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0} (gamma=1
is plain averaging; gamma=3 is the full sum at k=3
incoming deltas), in #307's exact receiver class.

Toy: identical construction to PR #307 / #308 (the
calibration arms below reproduce #307's reference
numbers exactly). Two-hidden-layer tanh MLP, input 16,
hidden 96, output 4. A shared base is PRETRAINED on a
general linear teacher V0 over a broad Gaussian
(sigma = 1.0, 600 steps); the receiver A is a fine-tune
of the base on V0 + dv_A on blob A; each incoming
behavior is an independent fine-tune of the same base
on V0 + dv_t on its own blob (blob sigma = 0.5);
tau_t = theta_t - theta_0. Per seed: 3 groups x 3
incoming deltas, giving 9 (seed x group) points.

Readouts, all anchored at the receiver theta_A:
  lossret  per behavior, the loss gain a composed delta
           realizes at the receiver divided by the gain
           the intact solo delta realizes at the receiver
  A_ret    receiver-behavior retention (the interference
           price of the composed delta)
Pre-composition statistics per point: sign-conflict
rate, mean pairwise cos(tau_i, tau_j), and the norm
ratio ||mean tau|| / mean||tau||.

Measured on pop-os (particlegan 0.10.1), 3 seeds:
  CALIBRATION: plain / avg / ties / rand_trim lossret
  reproduce PR #307 exactly (plain -0.16/-2.44/-0.87;
  avg 0.59/0.19/0.45; ties 0.47/-0.35/0.04; rand_trim
  0.47/0.10/0.45; sign-conflict 0.70-0.76 per group).
  (i) CURVE SHAPE / OPTIMUM: lossret(gamma) has an
  interior optimum in 8 of 9 points (rises to a peak,
  then falls; at gamma=3 every point is below its peak,
  and the seed-mean lossret at gamma=3 is -1.16). One
  point (seed 1, group 1) is monotone DECREASING over
  the whole grid (0.10 / 0.00 / -0.27 / -0.71 / -1.96 /
  -3.58 / -7.19). The optimum gamma* per point:
    seed 0: 1.5, 2.0, 1.0
    seed 1: 2.0, 0.25, 1.0
    seed 2: 0.75, 1.0, 1.0
  gamma* spans 0.25-2.0 with median exactly 1.0: plain
  averaging (gamma=1) is the MEDIAN optimum, not the
  universal one -- 3 of 9 points peak strictly above 1
  (up to 2.0), 2 of 9 peak strictly below 1, and only
  4 of 9 peak exactly at 1.
  Seed-mean lossret(gamma): 0.22 / 0.36 / 0.42 / 0.41 /
  0.22 / -0.15 / -1.16 -- the mean curve peaks at
  gamma=0.75-1.0 and the full sum (gamma=3) is
  catastrophic in the mean, per #307.
  (ii) PREDICTABILITY: gamma* is NOT predictable from
  any of the three pre-composition statistics over the
  9 points. Spearman correlations with gamma*:
  sign-conflict +0.39, mean pairwise cos -0.25, norm
  ratio -0.32 -- all weak (|rho| <= 0.4). A per-statistic
  linear predictor's mean |gamma_hat - gamma*| (0.41-
  0.43) does NOT beat the constant predictor gamma=1
  (MAE 0.39). The measured negative is the result: in
  this class the right scale must be measured at the
  receiver, not read off the deltas -- consistent with
  #282's receiver-anchoring verdict.
  (iii) A_ret: strictly the interference price of
  scale. A_ret(gamma) is monotone nonincreasing in
  gamma in 9 of 9 points; the mean falls 0.99 / 0.95 /
  0.90 / 0.83 / 0.64 / 0.40 / -0.18 across the grid.
  Every step up in gamma buys incoming-behavior gain
  (up to the point's peak) with receiver retention.

Verdict: the scale lever is real and priced -- an
interior optimum exists per point and gamma=1 is its
median -- but gamma* itself is not predictable from
pre-composition statistics in this class, and scale is
paid for in receiver retention, monotonically.
Assertions encode exactly these margins. Pure torch --
no GAN machinery. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family).
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except Exception:  # particlegan present but its import raises
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


DIN, HID, COUT = 16, 96, 4    # toy MLP shape (PR #307)
SIG = 0.5                     # per-blob input width
MEAN_SCALE = 2.5              # blob-mean draw scale
DV_SCALE = 1.5                # per-task teacher deviation
FT_STEPS = 40                 # per-task fine-tune steps
FT_LR = 0.005                 # fine-tune Adam step
PRE_STEPS = 600               # base pretraining steps
KEEP = 0.2                    # TIES trim keep-fraction (#307)
N_IN = 3                      # incoming deltas per group
N_GROUPS = 3                  # independent groups per seed
GAMMAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
CALIB_ARMS = ("plain", "avg", "ties", "rand_trim")
# PR #307's published table (seed order 0, 1, 2).
PR307_TABLE = {
    "plain": (-0.16, -2.44, -0.87),
    "avg": (0.59, 0.19, 0.45),
    "ties": (0.47, -0.35, 0.04),
    "rand_trim": (0.47, 0.10, 0.45),
}
# Measured gamma* per (seed, group) on pop-os.
GSTAR_TABLE = {(0, 0): 1.5, (0, 1): 2.0, (0, 2): 1.0,
               (1, 0): 2.0, (1, 1): 0.25, (1, 2): 1.0,
               (2, 0): 0.75, (2, 1): 1.0, (2, 2): 1.0}


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


def _data(v, m, n, seed, sig=SIG):
    g = torch.Generator().manual_seed(seed)
    x = m + sig * torch.randn(n, DIN, generator=g)
    return x, x @ v.T


def _train(base, v, m, steps, seed, lr, sig=SIG, n=256):
    params = {k: t.clone().requires_grad_(True)
              for k, t in base.items()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    x, y = _data(v, m, n, seed, sig)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((_fwd(params, x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    return {k: t.detach().clone() for k, t in params.items()}


def _loss(p, v, m, seed):
    x, y = _data(v, m, 128, seed)
    return float(((_fwd(p, x) - y) ** 2).mean())


def _trim(tau, keep_frac, rng=None, random=False):
    out = {}
    for k, t in tau.items():
        flat = t.flatten()
        keep = max(1, int(round(keep_frac * flat.numel())))
        if random:
            idx = torch.randperm(flat.numel(), generator=rng)[:keep]
        else:
            idx = torch.topk(flat.abs(), keep).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1
        out[k] = (flat * mask).reshape(t.shape)
    return out


def _elect(taus):
    out = {}
    for k in taus[0]:
        stack = torch.stack([t[k] for t in taus])
        sign = torch.sign(stack.sum(0))
        aligned = torch.where(stack * sign > 0, stack,
                              torch.zeros_like(stack))
        cnt = (aligned != 0).sum(0).clamp(min=1)
        out[k] = aligned.sum(0) / cnt
    return out


def _flat(tau):
    return np.concatenate([tau[k].numpy().flatten()
                           for k in sorted(tau)])


def _spearman(x, y):
    def ranks(v):
        order = np.argsort(v)
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_seed(seed):
    """Per-group gamma curves, predictors and calibration
    arms for one seed. Returns a list of per-group dicts
    plus the seed-mean calibration lossret per arm."""
    g = torch.Generator().manual_seed(3000 + seed)
    v0 = torch.randn(COUT, DIN, generator=g) * (3.0 / np.sqrt(DIN))
    base = _train(_base(seed), v0, torch.zeros(DIN),
                  PRE_STEPS, 2000 + seed, lr=0.03, sig=1.0, n=512)

    def make_task(s):
        gg = torch.Generator().manual_seed(s)
        dv = torch.randn(COUT, DIN, generator=gg) * (DV_SCALE / np.sqrt(DIN))
        m = torch.randn(DIN, generator=gg) * (MEAN_SCALE / np.sqrt(DIN))
        return v0 + dv, m

    va, ma = make_task(5000 + seed * 100)
    th_a = _train(base, va, ma, FT_STEPS, 7000 + seed, lr=FT_LR)
    a_gain = _loss(base, va, ma, 9000 + seed) - _loss(th_a, va, ma, 9000 + seed)

    calib = {name: [] for name in CALIB_ARMS}
    groups = []
    for grp in range(N_GROUPS):
        tasks = [make_task(6000 + seed * 1000 + grp * 100 + i * 17)
                 for i in range(N_IN)]
        thetas = [_train(base, v, m, FT_STEPS,
                         7100 + seed * 10 + grp * 3 + i, lr=FT_LR)
                  for i, (v, m) in enumerate(tasks)]
        taus = [{k: th[k] - base[k] for k in base} for th in thetas]
        flats = [_flat(t) for t in taus]
        stack = np.stack(flats)
        conflict = float(np.mean((stack > 0).any(0)
                                 & (stack < 0).any(0)))
        cos_pairs = []
        for i in range(N_IN):
            for j in range(i + 1, N_IN):
                cos_pairs.append(float(
                    flats[i] @ flats[j]
                    / (np.linalg.norm(flats[i])
                       * np.linalg.norm(flats[j]))))
        mean_cos = float(np.mean(cos_pairs))
        norm_ratio = float(
            np.linalg.norm(np.mean(stack, axis=0))
            / np.mean([np.linalg.norm(f) for f in flats]))

        rng = torch.Generator().manual_seed(999 + seed * 10 + grp)
        trimmed = [_trim(t, KEEP) for t in taus]
        rtrim = [_trim(t, KEEP, rng=rng, random=True) for t in taus]
        mean_tau = {k: sum(t[k] for t in taus) / len(taus)
                    for k in taus[0]}
        arms = {
            "plain": {k: sum(t[k] for t in taus) for k in taus[0]},
            "avg": mean_tau,
            "ties": _elect(trimmed),
            "rand_trim": {k: sum(t[k] for t in rtrim)
                          for k in taus[0]},
        }
        for gam in GAMMAS:
            arms[f"g{gam}"] = {k: gam * mean_tau[k] for k in taus[0]}

        def readouts(delta):
            comp = {k: th_a[k] + delta[k] for k in base}
            lrets = []
            for i, (v, m) in enumerate(tasks):
                solo = {k: th_a[k] + taus[i][k] for k in base}
                sg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(solo, v, m, 9100 + seed))
                cg = (_loss(th_a, v, m, 9100 + seed)
                      - _loss(comp, v, m, 9100 + seed))
                if sg > 0.02:
                    lrets.append(cg / sg)
            a_ret = ((_loss(base, va, ma, 9000 + seed)
                      - _loss(comp, va, ma, 9000 + seed)) / a_gain)
            return float(np.mean(lrets)), float(a_ret)

        rec = {"seed": seed, "grp": grp, "conflict": conflict,
               "mean_cos": mean_cos, "norm_ratio": norm_ratio,
               "gamma": {}, "n_valid": 0}
        for i, (v, m) in enumerate(tasks):
            solo = {k: th_a[k] + taus[i][k] for k in base}
            sg = (_loss(th_a, v, m, 9100 + seed)
                  - _loss(solo, v, m, 9100 + seed))
            if sg > 0.02:
                rec["n_valid"] += 1
        for gam in GAMMAS:
            rec["gamma"][gam] = readouts(arms[f"g{gam}"])
        for name in CALIB_ARMS:
            lr, _ = readouts(arms[name])
            calib[name].append(lr)
            rec.setdefault("calib", {})[name] = lr
        rec["gstar"] = max(GAMMAS, key=lambda gm: rec["gamma"][gm][0])
        groups.append(rec)
    return {"groups": groups,
            "calib_mean": {k: float(np.mean(v))
                           for k, v in calib.items()}}


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def _points():
    pts = []
    for seed in (0, 1, 2):
        pts.extend(_seed(seed)["groups"])
    return pts


def test_regime_and_calibration_vs_pr307():
    _empf_recipe()
    for seed in (0, 1, 2):
        d = _seed(seed)
        # #307's table reproduces within 0.06 on every arm
        # (probe reproduced it to the printed 2 decimals).
        for name in CALIB_ARMS:
            assert abs(d["calib_mean"][name]
                       - PR307_TABLE[name][seed]) <= 0.06, (
                f"seed {seed}: {name} lossret "
                f"{d['calib_mean'][name]:.3f} vs #307 "
                f"{PR307_TABLE[name][seed]:.2f}")
        for rec in d["groups"]:
            # The #307 interference regime per group: high
            # sign conflict, every solo gain valid.
            assert 0.65 <= rec["conflict"] <= 0.80, (
                f"seed {seed} grp {rec['grp']}: sign-conflict "
                f"{rec['conflict']:.3f}")
            assert rec["n_valid"] == N_IN, (
                f"seed {seed} grp {rec['grp']}: valid solo "
                f"gains {rec['n_valid']}")
        # Averaging is not beaten by TIES in this class.
        assert d["calib_mean"]["avg"] >= d["calib_mean"]["ties"] - 0.05
        # gamma=1 IS plain averaging and gamma=3 IS the
        # full sum: the sweep endpoints anchor to #307's
        # arms (seed-mean over groups, tolerance 0.06).
        g1 = float(np.mean([r["gamma"][1.0][0] for r in d["groups"]]))
        g3 = float(np.mean([r["gamma"][3.0][0] for r in d["groups"]]))
        assert abs(g1 - d["calib_mean"]["avg"]) <= 0.06
        assert abs(g3 - d["calib_mean"]["plain"]) <= 0.06


def test_gamma_curve_shape_and_optimum():
    _empf_recipe()
    pts = _points()
    gstars = []
    for rec in pts:
        curve = [rec["gamma"][gm][0] for gm in GAMMAS]
        gstar = rec["gstar"]
        gstars.append(gstar)
        # The full sum (gamma=3) never wins: every point's
        # peak beats its gamma=3 value by a wide margin
        # (demonstrated gaps >= 0.24, most >= 0.9).
        assert curve[-1] <= max(curve) - 0.20, (
            f"seed {rec['seed']} grp {rec['grp']}: lossret(3) "
            f"{curve[-1]:.3f} vs peak {max(curve):.3f}")
        # gamma* never sits at the top of the grid, and
        # the measured per-point table reproduces.
        assert gstar <= 2.0, (
            f"seed {rec['seed']} grp {rec['grp']}: gamma* {gstar}")
        assert gstar == GSTAR_TABLE[(rec["seed"], rec["grp"])], (
            f"seed {rec['seed']} grp {rec['grp']}: gamma* {gstar} "
            f"vs table {GSTAR_TABLE[(rec['seed'], rec['grp'])]}")
    # The measured gamma* distribution: median exactly
    # 1.0 (plain averaging is the median optimum), with
    # real spread on BOTH sides -- at least 3 points peak
    # strictly above 1 and at least 2 strictly below 1
    # (demonstrated 3 above: 1.5/2.0/2.0; 2 below:
    # 0.25/0.75).
    assert float(np.median(gstars)) == 1.0, f"gamma* {gstars}"
    assert sum(1 for x in gstars if x > 1.0) >= 3, f"gamma* {gstars}"
    assert sum(1 for x in gstars if x < 1.0) >= 2, f"gamma* {gstars}"
    assert min(gstars) <= 0.25 and max(gstars) >= 2.0
    # The one monotone point: seed 1 group 1 decreases
    # strictly across the whole grid and ends deep in
    # catastrophe (demonstrated -7.19 at gamma=3).
    mono = next(r for r in pts if r["seed"] == 1 and r["grp"] == 1)
    mcurve = [mono["gamma"][gm][0] for gm in GAMMAS]
    assert mono["gstar"] == 0.25
    assert all(mcurve[i + 1] < mcurve[i]
               for i in range(len(mcurve) - 1)), f"{mcurve}"
    assert mcurve[-1] <= -5.0, f"{mcurve}"
    # Seed-mean curve: peaks at gamma in {0.75, 1.0} and
    # the full sum is catastrophic in the mean
    # (demonstrated mean lossret(3) = -1.16).
    for gm in GAMMAS:
        mean_lr = float(np.mean([r["gamma"][gm][0] for r in pts]))
        if gm == 3.0:
            assert mean_lr <= -0.80, f"mean lossret(3) {mean_lr:.3f}"
    mean_curve = {gm: float(np.mean([r["gamma"][gm][0] for r in pts]))
                  for gm in GAMMAS}
    assert max(mean_curve, key=mean_curve.get) in (0.75, 1.0)


def test_gamma_star_not_predictable():
    _empf_recipe()
    pts = _points()
    gstars = np.array([r["gstar"] for r in pts], dtype=float)
    # No pre-composition statistic tracks gamma*: every
    # Spearman correlation is weak (demonstrated
    # conflict +0.39, mean_cos -0.25, norm_ratio -0.32).
    for name in ("conflict", "mean_cos", "norm_ratio"):
        rho = _spearman([r[name] for r in pts], gstars)
        assert abs(rho) <= 0.50, f"Spearman({name}, gamma*) {rho:+.3f}"
    # And no per-statistic linear predictor beats the
    # constant predictor gamma=1 (demonstrated MAE:
    # constant 0.39; conflict 0.43, cos 0.42, norm_ratio
    # 0.41). The negative is the assertion.
    const_mae = float(np.mean(np.abs(gstars - 1.0)))
    assert const_mae <= 0.50, f"constant-predictor MAE {const_mae:.3f}"
    for name in ("conflict", "mean_cos", "norm_ratio"):
        x = np.array([r[name] for r in pts], dtype=float)
        coef = np.polyfit(x, gstars, 1)
        mae = float(np.mean(np.abs(np.polyval(coef, x) - gstars)))
        assert mae >= const_mae - 0.05, (
            f"linear predictor on {name}: MAE {mae:.3f} vs "
            f"constant {const_mae:.3f}")


def test_a_ret_decreases_with_gamma():
    _empf_recipe()
    pts = _points()
    for rec in pts:
        aret = [rec["gamma"][gm][1] for gm in GAMMAS]
        # Receiver retention is monotone nonincreasing in
        # gamma in every point (demonstrated 9/9; small
        # epsilon for float noise at the flat top).
        assert all(aret[i + 1] <= aret[i] + 0.02
                   for i in range(len(aret) - 1)), (
            f"seed {rec['seed']} grp {rec['grp']}: A_ret {aret}")
        # The price is large: retention drops by at least
        # 0.30 from gamma=0.25 to gamma=3 in every point
        # (demonstrated drops 0.93-2.37).
        assert aret[0] - aret[-1] >= 0.30, (
            f"seed {rec['seed']} grp {rec['grp']}: A_ret drop "
            f"{aret[0]:.3f} -> {aret[-1]:.3f}")
    # Mean profile: near-intact at gamma=0.25, destroyed
    # at gamma=3 (demonstrated 0.99 -> -0.18).
    mean_aret = {gm: float(np.mean([r["gamma"][gm][1] for r in pts]))
                 for gm in GAMMAS}
    assert mean_aret[0.25] >= 0.95, f"{mean_aret}"
    assert mean_aret[3.0] <= 0.0, f"{mean_aret}"
    assert mean_aret[1.0] >= mean_aret[2.0] + 0.20, f"{mean_aret}"
