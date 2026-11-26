"""CPU gate for the BRR claim (arXiv:2611.33308): "Change Partners
at Every Layer: Balanced Random Regrouping for Model Merging" --
merging N experts into K<N groups per layer (a task uses the
module merged from its expert's group), randomly RE-PARTITIONING
the experts into equal-sized groups independently at every layer
(BRR-layer) or weight matrix (BRR-matrix) beats both (a) keeping
the same partners across layers and (b) similarity-based
grouping, in all 24 CLIP settings (+1.0-6.8 points). First-order
mechanism: at each layer, merging a task's expert with its
partners shifts the task's output features; with the SAME
partners at every layer the shifts point in similar directions
and ADD COHERENTLY (~L growth), while fresh partners make the
shifts vary independently so they add up less (random-walk
growth ~sqrt(L)).

Provenance and anchors inside this watch:

  * PR #282 (composition checks must anchor at the receiver):
    merging interference is first-order predictable at the
    receiver. This toy lives in #282's receiver class -- the
    per-layer shift a merge induces at a task's own stream is
    the quantity tracked, and it is computed exactly from known
    planted deltas.
  * PR #307 (plain averaging beats TIES in the receiver class):
    every group merge here is plain averaging, #307's default;
    BRR changes WHO is averaged together per layer, never the
    merge operator.
  * PR #324 (merging recovery is ~1/K-diluted for region-local
    deltas while consolidation holds ~0.99): dilution is ONE
    merge cost -- the merged module carries only 1/g of the
    task's own delta, identically in every arm here. BRR names
    a SECOND, independent merge cost: per-layer shift
    COHERENCE, which regrouping removes at zero training cost.
    The mechanism readout below isolates exactly the
    partner-induced component of the shift (the dilution
    component is common to all arms and is not what BRR acts
    on), so the two costs are priced separately.

Toy operationalization. A residual stack of L=48 layers, each
layer holding TWO matrices (an up/down pair, m=0,1), width
d=16: h <- h + Delta_l,m h per matrix. N=12 specialists are
PLANTED (not trained -- a modeling gift, stated plainly) from a
shared identity base with known per-layer deltas

  Delta[n, l, m] = scale * (outer(u_n, w_m) + noise * E[n, l, m]),

where u_n is specialist n's signature direction, w_m a fixed
per-matrix readout direction, and E per-(n, l, m) noise. The
signatures carry a LOOSE latent cluster structure: u_n is a
cluster centre (3 centres) plus a jitter perturbation of norm
1.2, so within-cluster delta cosine is ~0.4 and cross-cluster
~0 -- similarity between experts is real but weak, the regime
the paper's CLIP experts live in (fine-tunes of one base whose
task vectors are far from near-copies). At jitter 0.45 (tight
clusters, near-copy partners) similarity grouping wins outright
in this toy and the claim does NOT reproduce; at jitter 0.7
with the cluster construction broken the clusters are absent
and the similarity arm is vacuous. Both boundary regimes were
measured during development and are part of the evidence: the
claim is a statement about the weak-similarity regime, and the
toy says so.

Grouped merge: a partition of the 12 specialists into K=3
balanced groups of 4; the module serving task n at (l, m) is
the plain average of its group members' Delta[:, l, m]. Arms:

  (1) FIXED-SIMILARITY: one partition computed from the deltas
      themselves (cosine of each specialist's mean-over-layers
      delta, farthest-point centres + capacity-constrained
      greedy assignment -- a FlexMerge-style stand-in, not
      FlexMerge itself), reused at every layer and matrix.
  (2) FIXED-RANDOM control: one uniformly random balanced
      partition, reused everywhere. Separates "random partners"
      from "FRESH partners": it has BRR's partner distribution
      with fixed grouping's coherence, and it prices like fixed
      grouping, not like BRR.
  (3) BRR-layer: a fresh random balanced partition per layer
      (shared by the layer's two matrices).
  (4) BRR-matrix: a fresh random balanced partition per
      (layer, matrix) -- 72 independent partitions.
  (5) K=1 FULL-MERGE control: one group of all 12 everywhere
      (the worst case; every task served by the global average).
  (6) SPECIALIST ceiling: each task served by its own deltas
      (retained gain 1.0 by construction).

Readouts per seed (5 seeds), per task:
  * RETAINED GAIN: 1 - (squared output error of the grouped
    network vs the specialist's own outputs) / (squared output
    error of the base vs the specialist's outputs), averaged
    over 8 seeded probe inputs per task; reported as the mean
    over tasks and the worst task.
  * MECHANISM, partner-shift cosine: at each layer, the
    partner-induced shift s_l = (sum of partners' Delta / g)
    applied to the task's current stream (the two matrices'
    contributions summed), evaluated on the task's signature
    probe; the readout is the mean pairwise cosine between
    s_l across layers. Predicted: clearly positive (~0.8)
    under both fixed arms, much lower under BRR. It does NOT
    land at exactly 0 under BRR (mean +0.41, per-seed range
    +0.31..+0.59 measured): random
    partner means fluctuate around a nonzero population-mean
    direction, a residual common component the paper's
    idealized independence assumption omits -- stated plainly.
  * MECHANISM, composed shift norm: ||sum_l s_l|| at full
    depth (and at half depth, reported in the PR table):
    fixed grouping's coherent sum should exceed BRR's
    random-walk sum by a clear factor.

Modeling gifts, stated plainly: (i) the paper selects merge
scales on validation; every arm here uses the same FIXED
scale (plain average, coefficient 1/g) -- BRR's win cannot be
a scale-tuning artifact, but the paper's exact margins are
not being reproduced. (ii) Routing is per-task group
membership, exactly the paper's serving rule. (iii) Deltas
are planted with known ground truth in #282/#324's receiver
class, where real experts are trained; nothing here speaks
to trained-delta geometry beyond the planted signature +
noise structure. (iv) The similarity arm is a greedy
stand-in for FlexMerge, not FlexMerge.

Demonstrated shape asserted below (verdict PARTIAL, stated
exactly): the paper's claim (a) is PROVEN in this toy --
BRR-layer and BRR-matrix beat FIXED-RANDOM on every seed by
a wide margin (+0.15..+0.23), so freshness, not partner
randomness, is the active ingredient; the MECHANISM is
PROVEN -- shift cosine separates as predicted (fixed ~+0.81..
+0.85, BRR ~+0.41, mean gap > 0.3) and the composed
partner-shift norm under fixed grouping exceeds BRR's by
1.4-1.6x; BRR's worst task beats both fixed arms' worst task
by a wide margin (+0.29 vs similarity). Claim (b) against
similarity grouping reproduces only as PARITY on mean
retained gain (BRR-layer 0.448 vs similarity 0.444 over the
5 seeds; BRR ahead on 3 of 5 seeds), NOT the paper's clear
margin over FlexMerge -- the assertions therefore encode
parity-plus-worst-task-dominance against similarity, not a
mean win. In an independent set of draws (the numpy
prototype of this same construction) the mean gap was
+0.06..+0.10 in BRR's favour, so the similarity margin is
draw-sensitive at toy scale in a way the fixed-grouping
margin and the mechanism are not. The assertions encode
the demonstrated margins only, with headroom, and the
measured table in the PR body is the primary evidence
either way.

Measured on pop-os (particlegan 0.10.1), 5 seeds, L=48
(arm: retained mean / worst task / shift cosine / composed
partner-shift norm at full depth):
  sim:       0.444 / -0.113 / +0.851 / 0.523
  rand:      0.259 / -0.214 / +0.806 / 0.446
  brr_layer: 0.448 / +0.175 / +0.414 / 0.319
  brr_mat:   0.444 / +0.172 / +0.420 / 0.322
  full:      0.162 / -0.206 / +0.859 / 0.377
  own:       1.000 / +1.000 /  --    / 0.000

What is NOT asserted: the paper's exact numbers, anything
about CLIP or trained experts, the grouping-search extension
the paper builds on top of BRR, and any claim outside the
weak-similarity regime named above (in the tight-cluster
regime this same toy prefers similarity grouping).
Pure torch toy on CPU, single-threaded, seeded. Skips cleanly
when the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family, PR #273's
guarded-import pattern).
"""
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


# ------------------------------------------------------------------ world
N = 12          # specialists
K_GROUPS = 3    # groups per layer (balanced: 4 specialists each)
L = 48          # layers
M_PER_LAYER = 2  # matrices per layer (up/down pair)
D = 16          # width
N_CLUST = 3     # latent signature clusters
JITTER = 1.2    # signature jitter norm (weak-similarity regime)
SCALE = 0.06    # delta scale (fixed for every arm; see gifts)
NOISE = 0.05    # per-(n, l, m) delta noise, relative to SCALE
N_PROBES = 8    # probe inputs per task for the retained-gain readout
SEEDS = (0, 1, 2, 3, 4)


def _build_world(seed):
    """Planted specialists: returns (U, deltas, probes).

    U: (N, D) signature directions; deltas: (N, L, M, D, D);
    probes: (N, N_PROBES, D) unit probe inputs per task.
    """
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(N_CLUST, D, generator=g, dtype=torch.float64)
    centres = centres / centres.norm(dim=1, keepdim=True)
    w = torch.randn(M_PER_LAYER, D, generator=g, dtype=torch.float64)
    w = w / w.norm(dim=1, keepdim=True)
    U = torch.zeros(N, D, dtype=torch.float64)
    for n in range(N):
        e = torch.randn(D, generator=g, dtype=torch.float64)
        e = e / e.norm()
        u = centres[n % N_CLUST] + JITTER * e
        U[n] = u / u.norm()
    E = torch.randn(N, L, M_PER_LAYER, D, D, generator=g,
                    dtype=torch.float64) / D ** 0.5
    deltas = torch.zeros(N, L, M_PER_LAYER, D, D, dtype=torch.float64)
    for n in range(N):
        for l in range(L):
            for m in range(M_PER_LAYER):
                deltas[n, l, m] = SCALE * (
                    torch.outer(U[n], w[m]) + NOISE * E[n, l, m])
    probes = torch.randn(N, N_PROBES, D, generator=g, dtype=torch.float64)
    probes = probes / probes.norm(dim=2, keepdim=True)
    return U, deltas, probes


def _similarity_partition(deltas):
    """FlexMerge-style stand-in: group by cosine similarity of
    each specialist's mean-over-layers delta (farthest-point
    centres, capacity-constrained greedy assignment). Balanced
    by construction."""
    cap = N // K_GROUPS
    M = deltas.mean(dim=(1, 2)).reshape(N, -1)
    M = M / M.norm(dim=1, keepdim=True)
    S = M @ M.T
    centres = [0]
    while len(centres) < K_GROUPS:
        dmin = S[:, centres].min(dim=1).values.clone()
        for c in centres:
            dmin[c] = float("inf")
        centres.append(int(dmin.argmin()))
    groups = {c: [c] for c in centres}
    assigned = set(centres)
    pairs = sorted(
        ((float(S[n, c]), n, c) for n in range(N) for c in centres
         if n != c),
        reverse=True)
    for _, n, c in pairs:
        if n in assigned:
            continue
        if len(groups[c]) < cap:
            groups[c].append(n)
            assigned.add(n)
    for n in range(N):
        if n not in assigned:
            room = [c for c in centres if len(groups[c]) < cap]
            c = max(room, key=lambda c: float(S[n, c]))
            groups[c].append(n)
            assigned.add(n)
    return [sorted(v) for v in groups.values()]


def _random_partition(g):
    perm = torch.randperm(N, generator=g).reshape(K_GROUPS, N // K_GROUPS)
    return [sorted(r) for r in perm.tolist()]


def _group_of(partition, n):
    for grp in partition:
        if n in grp:
            return grp
    raise AssertionError("specialist missing from partition")


def _run_seed(seed):
    """Returns {arm: (retained_mean, retained_worst, shift_cos,
    composed_norm_full, composed_norm_half)}."""
    U, deltas, probes = _build_world(seed)
    g = torch.Generator().manual_seed(5000 + seed)
    sim_part = _similarity_partition(deltas)
    rand_part = _random_partition(g)
    layer_parts = [_random_partition(g) for _ in range(L)]
    mat_parts = [[_random_partition(g) for _ in range(M_PER_LAYER)]
                 for _ in range(L)]

    def group_fn(arm):
        if arm == "sim":
            return lambda n, l, m: _group_of(sim_part, n)
        if arm == "rand":
            return lambda n, l, m: _group_of(rand_part, n)
        if arm == "brr_layer":
            return lambda n, l, m: _group_of(layer_parts[l], n)
        if arm == "brr_mat":
            return lambda n, l, m: _group_of(mat_parts[l][m], n)
        if arm == "full":
            return lambda n, l, m: list(range(N))
        if arm == "own":
            return lambda n, l, m: [n]
        raise ValueError(arm)

    def forward(fn, n, x0, collect=False):
        x = x0.clone()
        shifts = []
        for l in range(L):
            layer_shift = torch.zeros(D, dtype=torch.float64)
            for m in range(M_PER_LAYER):
                grp = fn(n, l, m)
                merged = deltas[grp, l, m].mean(dim=0)
                if collect:
                    partners = [i for i in grp if i != n]
                    if partners:
                        layer_shift = layer_shift + (
                            deltas[partners, l, m].sum(dim=0)
                            / len(grp)) @ x
                x = x + merged @ x
            if collect:
                shifts.append(layer_shift)
        if collect:
            return x, torch.stack(shifts)
        return x

    arms = ("sim", "rand", "brr_layer", "brr_mat", "full", "own")
    fns = {a: group_fn(a) for a in arms}
    spec = torch.stack([
        torch.stack([forward(fns["own"], n, probes[n, p])
                     for p in range(N_PROBES)])
        for n in range(N)])

    out = {}
    for arm in arms:
        fn = fns[arm]
        retained = []
        cosines = []
        comp_full = []
        comp_half = []
        for n in range(N):
            errs = 0.0
            dens = 0.0
            for p in range(N_PROBES):
                y = forward(fn, n, probes[n, p])
                errs += float(((y - spec[n, p]) ** 2).sum())
                dens += float(((probes[n, p] - spec[n, p]) ** 2).sum())
            retained.append(1.0 - errs / dens)
            _, shifts = forward(fn, n, U[n], collect=True)
            norms = shifts.norm(dim=1)
            if float(norms.min()) > 1e-9:
                S = shifts / norms[:, None]
                gram = S @ S.T
                iu = torch.triu_indices(L, L, 1)
                cosines.append(float(gram[iu[0], iu[1]].mean()))
            comp_full.append(float(shifts.sum(dim=0).norm()))
            comp_half.append(float(shifts[: L // 2].sum(dim=0).norm()))
        out[arm] = (
            sum(retained) / N,
            min(retained),
            sum(cosines) / len(cosines) if cosines else 0.0,
            sum(comp_full) / N,
            sum(comp_half) / N,
        )
    return out


def test_brr_regrouping_merging():
    _empf_recipe()
    per_seed = [_run_seed(seed) for seed in SEEDS]

    def mean(arm, idx):
        return sum(r[arm][idx] for r in per_seed) / len(per_seed)

    # Ceiling and floor sanity: the specialist serves itself
    # exactly, and the K=1 full merge is far below BRR.
    for r in per_seed:
        assert abs(r["own"][0] - 1.0) < 1e-9
    assert mean("full", 0) < mean("brr_layer", 0) - 0.15

    # Claim (a): fresh partners beat the SAME random partners
    # kept across layers -- on every seed, by a clear margin.
    # (Randomness of partners is not the active ingredient;
    # freshness is.)
    for r in per_seed:
        assert r["brr_layer"][0] > r["rand"][0] + 0.04
        assert r["brr_mat"][0] > r["rand"][0] + 0.04

    # Claim (b), as demonstrated: against similarity grouping
    # BRR reproduces as PARITY on mean retained gain (not the
    # paper's clear margin -- see the docstring), ahead on at
    # least 3 of 5 seeds, with worst-task dominance asserted
    # below. The assertions encode exactly that.
    assert mean("brr_layer", 0) > mean("sim", 0) - 0.01
    assert mean("brr_mat", 0) > mean("sim", 0) - 0.01
    wins = sum(1 for r in per_seed
               if r["brr_layer"][0] > r["sim"][0])
    assert wins >= 3

    # Worst task: regrouping protects the worst-served task
    # against coherent accumulation.
    assert mean("brr_layer", 1) > mean("sim", 1) + 0.15
    assert mean("brr_layer", 1) > mean("rand", 1) + 0.15

    # Mechanism: per-layer partner shifts are strongly aligned
    # under both fixed groupings and much less aligned under
    # BRR (not exactly 0 -- the population-mean component in
    # the docstring -- but separated by a wide gap).
    for r in per_seed:
        assert r["sim"][2] > 0.65
        assert r["rand"][2] > 0.65
        assert r["brr_layer"][2] < 0.65
        assert r["brr_mat"][2] < 0.65
    assert mean("sim", 2) - mean("brr_layer", 2) > 0.30
    assert mean("rand", 2) - mean("brr_layer", 2) > 0.30

    # Mechanism: the composed partner shift (the coherent sum
    # over all layers) is much larger under fixed grouping
    # than under BRR's random-walk sum.
    assert mean("sim", 3) > 1.35 * mean("brr_layer", 3)
    assert mean("rand", 3) > 1.20 * mean("brr_layer", 3)
