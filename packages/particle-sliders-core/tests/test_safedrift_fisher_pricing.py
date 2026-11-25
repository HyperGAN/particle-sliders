"""CPU gate for the SAFE-DRIFT claim (arXiv:2611.28560): SFT data
selection as constrained optimization -- maximize target-gradient
alignment subject to a Fisher-measured off-target drift budget.
The paper's update-side optimum is a damped natural-gradient step
downweighting reference-sensitive directions; the salvage tested
here is the SELECTION-side use of the same geometry: the Fisher
structure that prices an update also prices which candidate
EXAMPLES to admit, so priced admission should dominate the
gain/drift frontier at a matched example budget.

Toy operationalization (diagonal-free but spectrum-exact): a
linear student w in R^32 trained by gradient descent on admitted
examples. The reference Fisher is F = Q diag(f) Q^T with a known
spectrum -- 8 reference-sensitive directions (f = 25) and 24 flat
ones (f = 0.5) in a random eigenbasis Q (the toy stand-in for a
Fisher estimated from a reference set; the spectrum is the
diagonal approximation in the eigenbasis, stated as such).
Candidate pool of 120 examples; each candidate's gradient at
w = 0 is

  g_i = a_i * t + b_i * n_i + 0.15 * (flat noise),

with t the target direction, n_i a random reference-sensitive
direction. The pool mixes three subpopulations: clean (40%:
a ~ U(0.55, 1.0), b ~ U(0, 0.25)), toxic (40%: the HIGHEST
alignments a ~ U(0.75, 1.15) but heavy nuisance loading
b ~ U(0.7, 1.3)) and filler (20%: a ~ U(-0.1, 0.45), mixed b).
Each candidate therefore carries exactly the two numbers the
paper's rule needs: target alignment a_i = g_i . t and drift
cost c_i = g_i^T F g_i.

Selection arms at a MATCHED budget K = 24:
  align      -- top-K by alignment alone (naive curation default).
  priced     -- top-K by the Lagrangian score a_i - lam * c_i /
                median(c), lam = 0.3: the toy form of "maximize
                alignment subject to a drift budget", with lam
                the budget's shadow price (median normalization
                makes lam dimensionless across seeds).
  random     -- K drawn uniformly (control).
  drift_only -- K lowest drift cost regardless of alignment
                (mere caution: should buy safety at the price
                of gain, discriminating pricing from aversion).

Training: 30 steps of gradient descent (lr 0.15) on the admitted
examples' quadratic losses 0.5 * ||w - 2.5 * g_i||^2, identical
for every arm. Readouts per seed (5 seeds): target gain =
reduction in 0.5 * ||w - w*||^2, w* = 2t; drift = the Fisher
energy 0.5 * w^T F w of the trained student.

Entanglement condition: t is rotated so a fraction rho of its
norm lies in the reference-sensitive subspace (rho = 0 clean,
rho = 0.8 entangled). The paper's entanglement analysis predicts
the frontier itself degrades: when achieving the target REQUIRES
moving in Fisher-heavy directions, no selection rule can buy the
same absolute safety -- the priced arm's drift at matched gain
must inflate, and its drift advantage over alignment-only must
shrink.

Measured on pop-os (particlegan 0.10.1, torch 2.15), clean
condition (rho = 0): priced gain 1.963-1.986 -- the highest of
any arm on every seed -- at drift 1.19-1.39. Alignment-only:
gain 1.585-1.782 at drift 3.08-7.67 (2.6-5.7x the priced drift,
for LESS gain: the toxic subpopulation's nuisance displacement
also costs target distance, so naive curation is dominated on
both axes here). Random: gain 1.834-1.942, drift 1.42-3.71.
Drift-only: the drift floor 0.49-0.71, bought with gain
1.805-1.938 -- pricing beats mere caution on gain on every
seed (margin +0.03..+0.18, mean +0.12) while staying far under
random's drift. Entangled (rho = 0.8): priced gain holds at
1.960-1.989 but its drift inflates ~30x to 35.4-41.8, and its
drift advantage over alignment-only (drift 77.5-96.5, gain
1.099-1.521) shrinks from ~3.5x (clean, mean ratio) to ~2.3x:
entanglement degrades the frontier itself, as the paper's
analysis suggests. Assertions encode roughly half the thinnest
observed margins (the per-seed priced-vs-drift-only gain margin
is asserted as a sign, not a magnitude, for that reason).
Skips cleanly when the installed particlegan does not provide
the EMPF recipe. Pure torch geometry -- no GAN machinery,
because the claim is about selection geometry.
"""
import pytest
import torch

try:
    import particlegan  # noqa: F401
    from particlegan import get_recipe

    _recipe = get_recipe("empf", loss="relativistic")
    if _recipe is None:
        raise RuntimeError("empf recipe unavailable")
except Exception:
    pytest.skip("particlegan EMPF recipe unavailable", allow_module_level=True)


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


D = 32
N_NUIS = 8
N_POOL = 120
K = 24
F_HI = 25.0
F_LO = 0.5
LAM = 0.3
SEEDS = (100, 101, 102, 103, 104)


def _problem(gen, rho):
    f = torch.full((D,), F_LO)
    f[:N_NUIS] = F_HI
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=gen))
    t_eig = torch.zeros(D)
    flat = torch.randn(D - N_NUIS, generator=gen)
    t_eig[N_NUIS:] = (1.0 - rho**2) ** 0.5 * flat / flat.norm()
    nuis = torch.randn(N_NUIS, generator=gen)
    t_eig[:N_NUIS] = rho * nuis / nuis.norm()
    return q, f, q @ t_eig, t_eig


def _pool(gen, q, t_eig):
    g_eig = torch.zeros(N_POOL, D)
    kinds = torch.multinomial(
        torch.tensor([0.4, 0.4, 0.2]), N_POOL, replacement=True, generator=gen
    )
    for i in range(N_POOL):
        if kinds[i] == 0:
            a = 0.55 + 0.45 * torch.rand((), generator=gen)
            b = 0.25 * torch.rand((), generator=gen)
        elif kinds[i] == 1:
            a = 0.75 + 0.40 * torch.rand((), generator=gen)
            b = 0.7 + 0.6 * torch.rand((), generator=gen)
        else:
            a = -0.1 + 0.55 * torch.rand((), generator=gen)
            b = 0.9 * torch.rand((), generator=gen)
        n_dir = torch.zeros(D)
        nd = torch.randn(N_NUIS, generator=gen)
        n_dir[:N_NUIS] = nd / nd.norm()
        flat_noise = torch.zeros(D)
        fn = torch.randn(D - N_NUIS, generator=gen)
        flat_noise[N_NUIS:] = fn / fn.norm()
        g_eig[i] = a * t_eig + b * n_dir + 0.15 * flat_noise
    return g_eig @ q.T, g_eig


def _train_and_measure(g, q, f, t_world, idx):
    targets = 2.5 * g[idx]
    w = torch.zeros(D)
    for _ in range(30):
        grad = (w - targets).mean(dim=0)
        w = w - 0.15 * grad
    w_star = 2.0 * t_world
    gain = 0.5 * (w_star**2).sum() - 0.5 * ((w - w_star) ** 2).sum()
    w_eig = q.T @ w
    drift = 0.5 * (f * w_eig**2).sum()
    return float(gain), float(drift)


def _run_seed(seed, rho):
    gen = torch.Generator().manual_seed(seed)
    q, f, t_world, t_eig = _problem(gen, rho)
    g, g_eig = _pool(gen, q, t_eig)
    align = g @ t_world
    drift_cost = (g_eig**2) @ f
    arms = {
        "align": torch.argsort(align, descending=True)[:K],
        "drift_only": torch.argsort(drift_cost)[:K],
        "random": torch.randperm(N_POOL, generator=gen)[:K],
        "priced": torch.argsort(
            align - LAM * drift_cost / drift_cost.median(), descending=True
        )[:K],
    }
    return {k: _train_and_measure(g, q, f, t_world, v) for k, v in arms.items()}


def test_priced_admission_dominates_frontier():
    for seed in SEEDS:
        r = _run_seed(seed, rho=0.0)
        pg, pd_ = r["priced"]
        ag, ad = r["align"]
        rg, rd = r["random"]
        dg, dd = r["drift_only"]
        # priced: strictly more gain than alignment-only at far
        # less drift (observed gain margin +0.20..+0.38 per seed)
        assert pg >= ag + 0.10, (seed, pg, ag)
        assert pd_ <= 0.55 * ad, (seed, pd_, ad)
        # pricing is not mere caution: more gain than drift-only
        # aversion on every seed (observed margin +0.03..+0.18,
        # mean +0.12) and more than random, at drift still well
        # under random's; drift-only remains the drift floor
        assert pg >= dg, (seed, pg, dg)
        assert pg >= rg, (seed, pg, rg)
        assert dd < pd_ < rd, (seed, dd, pd_, rd)


def test_entanglement_degrades_the_frontier():
    clean_ratios, ent_ratios, clean_pd, ent_pd = [], [], [], []
    for seed in SEEDS:
        rc = _run_seed(seed, rho=0.0)
        re_ = _run_seed(seed, rho=0.8)
        clean_ratios.append(rc["align"][1] / rc["priced"][1])
        ent_ratios.append(re_["align"][1] / re_["priced"][1])
        clean_pd.append(rc["priced"][1])
        ent_pd.append(re_["priced"][1])
        # pricing still helps when entangled, just less
        assert re_["priced"][0] >= re_["align"][0] - 0.02
        assert re_["priced"][1] <= 0.7 * re_["align"][1]
    # the priced arm's absolute drift inflates steeply under
    # entanglement at (near-)unchanged gain: safety gets expensive
    assert sum(ent_pd) / len(ent_pd) >= 5.0 * (sum(clean_pd) / len(clean_pd))
    # and its relative drift advantage over alignment-only shrinks
    assert sum(ent_ratios) / len(ent_ratios) <= sum(clean_ratios) / len(clean_ratios)
