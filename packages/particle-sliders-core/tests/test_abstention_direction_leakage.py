"""CPU gate for the Abstention Direction claim (arXiv:2611.20441):
abstention directions extracted for different causes share a common
component, so steering along one cause's direction shifts the other
causes' abstention behaviour in proportion to that shared component.

Paper: "Abstention Direction in Large Language Models" (Itai, Hirako &
Yamagiwa). Its central qualitative finding: mean-difference directions
extracted from counterfactual pairs for three abstention causes --
query ambiguity, knowledge limits, and safety -- are not independent;
they share a common directional component, and steering along one
cause's direction moves abstention for the others. The paper reports
this qualitatively, with no headline metric. This toy makes the claim
quantitative and validates the *measurement*: extract directions,
estimate the shared component, and check that the estimate PREDICTS
the cross-cause leakage.

Toy operationalization: hidden states in R^32 are built from KNOWN
components. Each cause c has a true direction

    t_c = sqrt(rho) * s + sqrt(1 - rho) * e_c,

where s is a shared unit vector, the e_c are orthonormal and orthogonal
to s, and rho in [0, 1] is the KNOWN shared fraction, varied across
worlds (rho in {0.0, 0.2, 0.5, 0.8}). By construction the ground-truth
pairwise cosine is exactly cos(t_c, t_c') = rho for c != c'. A scripted
policy's abstention score for cause c is the state's projection onto
t_c. Counterfactual pairs for cause c differ only in that cause's
factor: (h0, h0 + A * t_c) with shared base h0 and per-state noise.
Directions d_c are extracted by the paper's recipe -- the normalized
mean difference over the pairs.

Measurements per world (seed x rho):
  (i)  recovered structure: pairwise cos(d_c, d_c') vs ground truth rho;
  (ii) leakage: steer states along d_1 by alpha and measure the change
       in abstention score for causes 2 and 3 relative to cause 1 --
       the cross/own ratio should equal rho;
  (iii) control: steering along a coordinate-shuffled copy of d_1
       (same norm, destroyed alignment) must produce ~no cross-cause
       shift in absolute terms.

Assertions: recovered cosines and leakage ratios track rho within
stated tolerances across 3 seeds x 4 rho worlds; seed-averaged leakage
is monotone in rho; the shuffled control's absolute cross-cause shift
stays small where the real direction's is large. Skips cleanly when
the installed particlegan does not provide the EMPF recipe.
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


D = 32            # hidden-state dimension
N_CAUSES = 3      # ambiguity, knowledge limits, safety
N_PAIRS = 256     # counterfactual pairs per cause
AMP = 2.0         # counterfactual factor amplitude along t_c
NOISE = 1.0       # per-state activation noise std
ALPHA = 1.5       # steering strength
RHOS = (0.0, 0.2, 0.5, 0.8)
SEEDS = (0, 1, 2)


def _orthonormal_basis(g, k):
    """k orthonormal vectors in R^D via Gram-Schmidt on Gaussian draws."""
    vecs = []
    for _ in range(k):
        v = torch.randn(D, generator=g, dtype=torch.float64)
        for u in vecs:
            v = v - (v @ u) * u
        vecs.append(v / v.norm())
    return vecs


def _world(seed, rho):
    """True directions t_c with pairwise cosine exactly rho."""
    g = torch.Generator().manual_seed(7000 + 131 * seed + int(100 * rho))
    basis = _orthonormal_basis(g, N_CAUSES + 1)
    s, es = basis[0], basis[1:]
    ts = [(rho ** 0.5) * s + ((1.0 - rho) ** 0.5) * es[c]
          for c in range(N_CAUSES)]
    return g, ts


def _extract_directions(g, ts):
    """Paper's recipe: normalized mean difference over counterfactual
    pairs that differ only in the cause's own factor."""
    ds = []
    for c in range(N_CAUSES):
        base = torch.randn(N_PAIRS, D, generator=g, dtype=torch.float64) * NOISE
        noise = torch.randn(N_PAIRS, D, generator=g, dtype=torch.float64) * NOISE
        h_minus = base
        h_plus = base + AMP * ts[c][None, :] + noise
        diff = (h_plus - h_minus).mean(0)
        ds.append(diff / diff.norm())
    return ds


def _run_world(seed, rho):
    g, ts = _world(seed, rho)
    ds = _extract_directions(g, ts)
    # (i) recovered pairwise cosines
    pair_cos = [float(ds[i] @ ds[j])
                for i in range(N_CAUSES) for j in range(i + 1, N_CAUSES)]
    # (ii) leakage: steer along d_1; abstention score for cause c is the
    # projection onto t_c, so the shift is alpha * (d_1 . t_c) exactly,
    # independent of the base state. Ratio is cross-cause / own-cause.
    own = float(ds[0] @ ts[0])
    cross = [float(ds[0] @ ts[c]) for c in (1, 2)]
    leak_ratio = sum(cross) / len(cross) / own
    # (iii) shuffled control: permute d_1's coordinates (norm preserved,
    # alignment destroyed); report absolute cross-cause projection.
    perm = torch.randperm(D, generator=g)
    d_shuf = ds[0][perm]
    ctrl_cross = sum(abs(float(d_shuf @ ts[c])) for c in (1, 2)) / 2.0
    return dict(pair_cos=pair_cos, leak_ratio=leak_ratio,
                cross_proj=sum(cross) / len(cross), ctrl_cross=ctrl_cross)


def test_cross_cause_leakage_tracks_shared_component():
    _empf_recipe()
    results = {(seed, rho): _run_world(seed, rho)
               for seed in SEEDS for rho in RHOS}
    # (i) + (ii): per-world tracking of the known shared fraction.
    for (seed, rho), r in results.items():
        for pc in r["pair_cos"]:
            assert abs(pc - rho) <= 0.12, (
                f"seed {seed} rho {rho}: recovered cos {pc:.3f}")
        assert abs(r["leak_ratio"] - rho) <= 0.08, (
            f"seed {seed} rho {rho}: leakage ratio {r['leak_ratio']:.3f}")
    # Leakage is monotone in the shared fraction (seed-averaged).
    mean_leak = {rho: sum(results[(s, rho)]["leak_ratio"] for s in SEEDS) / len(SEEDS)
                 for rho in RHOS}
    ordered = [mean_leak[rho] for rho in RHOS]
    assert ordered == sorted(ordered), f"leakage not monotone in rho: {ordered}"
    # (iii) shuffled control: ~no absolute cross-cause shift, and far
    # below the real direction's shift where the shared component is
    # large (rho = 0.8 -> true cross projection 0.8).
    for seed in SEEDS:
        ctrl = results[(seed, 0.8)]["ctrl_cross"]
        real = results[(seed, 0.8)]["cross_proj"]
        assert ctrl <= 0.25, f"seed {seed}: control cross shift {ctrl:.3f}"
        assert ctrl <= 0.5 * real, (
            f"seed {seed}: control {ctrl:.3f} vs real {real:.3f}")
