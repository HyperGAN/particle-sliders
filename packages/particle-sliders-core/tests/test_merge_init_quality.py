"""CPU gate for the chain claim joining PR #341 (BRR,
arXiv:2611.33308) and PR #342 (Uni-MOPD merge-init,
arXiv:2611.34397) through PR #324's consolidation
apparatus: "merge quality prices init value".

Chain question. #342 proved, in #324's harvested-delta
receiver class (MLP 16->96->4 specialists from a shared
checkpoint), that consolidation from a UNIFORM merge
init beats base-init at equal budget (+0.005..+0.011 on
10/10 seed/level combos) -- but it priced ONE fixed
merge. #341 proved, in its planted residual-stack class,
that merge QUALITY varies sharply by grouping rule:
worst-task retained gain +0.175 for per-layer random
regrouping (BRR-layer) vs -0.113 for fixed similarity
grouping vs -0.214 for fixed random grouping. If the
init's value is the merge's quality transferred, the
consolidation advantage should ORDER with the init
merge's quality -- in particular with its WORST-task
retention, not just its mean.

Claim tested: consolidation from a BRR-regrouped
merge-init beats consolidation from a fixed-similarity
merge-init beats base-init at equal budget, with the
ordering tracking the inits' worst-task retention.

Apparatus reconciliation (the two anchors live in
different worlds; one toy must host both):

  * Consolidation, specialists, budgets and the
    recovery readout are #342's, VERBATIM where
    calibration is claimed: same MLP, same pretrain,
    same fine-tune recipes, same on-policy balanced
    distillation (fresh per-domain batches, specialist
    outputs as targets, MSE), B = 100 with snapshots
    at 5/10/25/50 from the same RNG stream, shuffled-
    teacher control. The fidelity check below reruns
    #342's exact K = 4 configuration and replicates
    its merge-only table on #324's seeds.
  * Grouping enters as the merge construction, the
    only place it can act on a single-model init:
    a grouped merge here averages, PER PARAMETER
    BLOCK (W1, b1, W2, b2, W3, b3 -- the MLP's
    "layers"), a SUBSET of K specialists' deltas,
    scaled 1/K (non-members contribute zero), so
    every merge init carries the uniform merge's
    total scale and arms differ only in WHO is
    averaged where:
      - BRR: a fresh uniformly random subset of
        size G = K/2 per block (the single-model
        analogue of #341's per-layer regrouping);
      - FIXED-SIMILARITY: one subset -- the medoid
        specialist (max mean tau cosine) plus its
        G-1 nearest neighbours by flattened-tau
        cosine, #341's data-driven stand-in moved
        from partitions to subsets -- reused at
        every block;
      - FIXED-RANDOM control: one random subset
        reused at every block (separates "random
        partners" from "fresh partners", as #341);
      - UNIFORM: #342's merge (all K, every block),
        the reference merge init;
      - BASE: the shared checkpoint, no merge.
    Documented compromise: #341's grouped serving
    keeps one merged module PER GROUP and routes
    each task to its group; a consolidation init
    must be ONE model, so grouping here selects
    who enters the single average per block. The
    per-task serving protection #341 measured is
    therefore NOT preserved by construction -- the
    inits' own worst-domain retention is measured
    (below) rather than assumed, and it is that
    measured quality whose transfer is priced.
  * Specialists: K = 8 (not #342's 4) so subsets
    of size 4 and a worst-domain readout have
    room; ONE divergence level (#342's L1 recipe:
    40 steps, lr 0.005, DV_SCALE 1.5) -- stated
    deviation: #342 found the init effect at both
    levels with the same sign and size, and the
    chain question is the ORDERING across merges,
    priced at L1; the K = 4 calibration below
    still runs both L1 and L2.
  * Consolidation batch is 32 fresh states per
    domain per step (not #342's 64) -- stated
    deviation, cost only; arms share it.
  * RNG: ONE stack, torch only, everywhere (world,
    subsets, consolidation). #341 disclosed its
    similarity margin was numpy-vs-torch draw-
    sensitive; this toy does not shop draws --
    seeds 0..4 and subset draws are fixed by the
    seeded generators below, first draws taken.

Readouts: per-domain RECOVERY exactly as #324/#342
defined it, reported as mean + WORST domain, for
(a) each merge init BEFORE consolidation (the
predictor; merge-only arms), (b) consolidation
snapshots at 5/10/25/50/100 from each init, (c) the
shuffled-teacher control consolidated from the BRR
init (must fail, per #342). Ceiling: each specialist
recovers its own domain at exactly 1.000.

Measured on pop-os (particlegan 0.10.1), 5 seeds
(K = 8, L1 recipe; seed means, per-seed values in
the PR body):

  init (merge-only)     mean    worst
    uniform             0.312   +0.063
    brr                 0.196   +0.033   <- best worst
    sim                 0.186   -0.147   <- worst worst
    rand                0.187   -0.094
  consolidation (mean / worst):
    @5   uniform .549/.264  brr .500/.176
         sim .506/.172      rand .514/.209
         base .441/.073
    @50  uniform .869/.792  merges .859-.860/.776-.779
         base .852/.764
    @100 uniform .925/.881  merges .922/.874-.877
         base .916/.864
    shuffled-from-brr @100: -0.41 mean (<= -0.16/seed)
  Calibration: the K = 4 instance replicates #342's
  merge-only table on #324's seeds to <= 0.001
  (0.297/0.416/0.341 at L1, 0.167/0.349/0.307 at L2).

Verdict, stated plainly: the chain claim does NOT
reproduce. The init ordering the claim needs --
BRR's worst-task advantage transferred into a
consolidation advantage -- is absent at every
budget: at B = 100 (and B/2 = 50) the merge inits
are FLAT (generic warm start, all >= base by the
small #342 margin), and on the early curve, where
init differences are still visible, the advantage
tracks the inits' MEAN retention (uniform, the
best-mean merge, prices highest; BRR, the best-
worst merge, prices LOWEST among merge inits) --
the opposite of the claim's "worst-task, not just
mean" prediction. BRR's worst-task protection is
a property of grouped SERVING (#341's setting);
as a single-model init it is the first thing
balanced consolidation erases. Assertions encode
exactly these demonstrated margins: the #342
replication, the predictor ordering (BRR best
worst / uniform best mean / similarity worst
worst), the generic warm start, the flatness at
B and B/2, the early mean-tracking (uniform top,
BRR bottom among merge inits at step 5), and the
shuffled control. Pure torch -- no GAN machinery.
Skips cleanly when the installed particlegan does
not provide the EMPF recipe (the house guard for
this paper-test family).
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


DIN, HID, COUT = 16, 96, 4    # toy MLP shape (#307/#324/#342)
SIG = 0.5                     # per-blob input width (#282/#307)
MEAN_SCALE = 2.5              # blob-mean draw scale (#307)
PRE_STEPS = 600               # base pretraining steps (#307/#324)
CONSOL_LR = 0.01              # consolidation lr (#324/#342)
CONSOL_PER = 32               # fresh states per domain per step
B = 100                       # matched consolidation budget (#342)
CHECKPOINTS = (5, 10, 25, 50, 100)
BLOCKS = ("W1", "b1", "W2", "b2", "W3", "b3")
K_MAIN = 8                    # specialists in the chain sweep
G_MAIN = 4                    # subset size (K/2)
FT_L1 = (40, 0.005, 1.5)      # #342's L1 recipe (steps, lr, dv)
SEEDS = (0, 1, 2, 3, 4)
# #342's merge-only table on #324's seeds (its docstring):
CALIB = {  # seed -> (L1 merge-only, L2 merge-only)
    0: (0.297, 0.167),
    1: (0.416, 0.349),
    2: (0.341, 0.307),
}
CALIB_LEVELS = ((40, 0.005, 1.5), (120, 0.005, 3.0))


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
    x, y = _data(v, m, 256, seed)
    return float(((_fwd(p, x) - y) ** 2).mean())


def _consolidate_curve(init, teachers, means, checkpoints,
                       seed, K, shuffle=False):
    """#342's on-policy balanced distillation, snapshotted.

    The per-step RNG stream follows #342's scheme
    (seed * 100000 + step * K + domain), so a snapshot
    at step t is exactly a t-step run.
    """
    params = {k: t.clone().requires_grad_(True)
              for k, t in init.items()}
    opt = torch.optim.Adam(list(params.values()), lr=CONSOL_LR)
    snaps = {}
    wanted = set(checkpoints)
    if 0 in wanted:
        snaps[0] = {k: t.detach().clone() for k, t in params.items()}
    for s in range(max(checkpoints)):
        opt.zero_grad()
        tot = 0.0
        for i in range(K):
            ti = (i + 1) % K if shuffle else i
            g = torch.Generator().manual_seed(
                seed * 100000 + s * K + i)
            x = means[i] + SIG * torch.randn(
                CONSOL_PER, DIN, generator=g)
            with torch.no_grad():
                tgt = _fwd(teachers[ti], x)
            tot = tot + ((_fwd(params, x) - tgt) ** 2).mean()
        (tot / K).backward()
        opt.step()
        if (s + 1) in wanted:
            snaps[s + 1] = {k: t.detach().clone()
                            for k, t in params.items()}
    return snaps


def _world(seed, K, ft):
    """Shared base + K specialists at fine-tune recipe ft.

    RNG seeds follow #342 exactly (task/specialist/loss
    streams), so the K = 4 instance is #342's world.
    """
    steps, lr, dvs = ft
    g = torch.Generator().manual_seed(3000 + seed)
    v0 = torch.randn(COUT, DIN, generator=g) * (3.0 / np.sqrt(DIN))
    base = _train(_base(seed), v0, torch.zeros(DIN),
                  PRE_STEPS, 2000 + seed, lr=0.03, sig=1.0, n=512)

    def make_task(s):
        gg = torch.Generator().manual_seed(s)
        dv = torch.randn(COUT, DIN, generator=gg) * (dvs / np.sqrt(DIN))
        m = torch.randn(DIN, generator=gg) * (MEAN_SCALE / np.sqrt(DIN))
        return v0 + dv, m

    tasks = [make_task(6000 + seed * 1000 + i * 17) for i in range(K)]
    specs = [_train(base, v, m, steps, 7100 + seed * 10 + i, lr=lr)
             for i, (v, m) in enumerate(tasks)]
    taus = [{k: sp[k] - base[k] for k in base} for sp in specs]
    means = [m for _, m in tasks]
    return base, tasks, specs, taus, means


def _rec_vec(base, tasks, specs, model, seed):
    out = []
    for i, (v, m) in enumerate(tasks):
        l0 = _loss(base, v, m, 9100 + seed)
        ls = _loss(specs[i], v, m, 9100 + seed)
        lm = _loss(model, v, m, 9100 + seed)
        out.append((l0 - lm) / (l0 - ls))
    return out


def _summ(base, tasks, specs, model, seed):
    rv = _rec_vec(base, tasks, specs, model, seed)
    return float(np.mean(rv)), float(np.min(rv))


def run_calibration(seed):
    """#342's exact K = 4 world: merge-only per level."""
    rows = []
    for li, ft in enumerate(CALIB_LEVELS):
        base, tasks, specs, taus, means = _world(seed, 4, ft)
        merged = {k: base[k] + sum(t[k] for t in taus) / 4
                  for k in base}
        solo = float(np.mean([
            _loss(base, v, m, 9100 + seed)
            - _loss(specs[i], v, m, 9100 + seed)
            for i, (v, m) in enumerate(tasks)]))
        row = dict(merge_only=_summ(base, tasks, specs, merged, seed)[0],
                   solo_gain=solo)
        if li == 0:
            single = {k: base[k] + taus[0][k] for k in base}
            v, m = tasks[0]
            l0 = _loss(base, v, m, 9100 + seed)
            ls = _loss(specs[0], v, m, 9100 + seed)
            l1 = _loss(single, v, m, 9100 + seed)
            row["single_own"] = (l0 - l1) / (l0 - ls)
        rows.append(row)
    return rows


def run_seed(seed):
    """Chain sweep for one seed (K = 8, L1 recipe).

    Returns init (mean, worst) per merge arm, consolidation
    (mean, worst) per checkpoint per arm, shuffled@B, and
    the mean solo gain (validity floor).
    """
    K = K_MAIN
    base, tasks, specs, taus, means = _world(seed, K, FT_L1)
    solo_gain = float(np.mean([
        _loss(base, v, m, 9100 + seed)
        - _loss(specs[i], v, m, 9100 + seed)
        for i, (v, m) in enumerate(tasks)]))

    # Grouping draws: one torch generator, first draws taken.
    grng = torch.Generator().manual_seed(5000 + seed)
    flat = torch.stack([
        torch.cat([taus[i][k].flatten() for k in BLOCKS])
        for i in range(K)])
    fn = flat / flat.norm(dim=1, keepdim=True)
    S = fn @ fn.T
    medoid = int(S.sum(dim=1).argmax())
    sim_subset = sorted(
        S[medoid].argsort(descending=True)[:G_MAIN].tolist())
    rand_subset = sorted(
        torch.randperm(K, generator=grng)[:G_MAIN].tolist())
    brr_subsets = [
        sorted(torch.randperm(K, generator=grng)[:G_MAIN].tolist())
        for _ in BLOCKS]

    def subset_init(subsets_per_block):
        return {k: base[k] + sum(taus[i][k]
                                 for i in subsets_per_block[bi]) / K
                for bi, k in enumerate(BLOCKS)}

    inits = {
        "uniform": {k: base[k] + sum(t[k] for t in taus) / K
                    for k in base},
        "brr": subset_init(brr_subsets),
        "sim": subset_init([sim_subset] * len(BLOCKS)),
        "rand": subset_init([rand_subset] * len(BLOCKS)),
        "base": base,
    }
    out = {"solo_gain": solo_gain, "init": {}, "curve": {}}
    for name, init in inits.items():
        out["init"][name] = _summ(base, tasks, specs, init, seed)
    for name in ("uniform", "brr", "sim", "rand", "base"):
        snaps = _consolidate_curve(
            inits[name], specs, means, CHECKPOINTS, seed * 10, K)
        out["curve"][name] = {
            cp: _summ(base, tasks, specs, snaps[cp], seed)
            for cp in CHECKPOINTS}
    shuf = _consolidate_curve(
        inits["brr"], specs, means, (B,), seed * 10 + 555, K,
        shuffle=True)
    out["shuffled_B"] = _summ(base, tasks, specs, shuf[B], seed)
    return out


_CACHE = {}
_CAL_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def _cal(seed):
    if seed not in _CAL_CACHE:
        _CAL_CACHE[seed] = run_calibration(seed)
    return _CAL_CACHE[seed]


def test_fidelity_replicates_342_merge_only():
    _empf_recipe()
    for seed, (l1_want, l2_want) in CALIB.items():
        rows = _cal(seed)
        for li, want in enumerate((l1_want, l2_want)):
            got = rows[li]["merge_only"]
            # Bit-faithful apparatus: #342's merge-only
            # numbers on #324's seeds, replicated to
            # float tolerance.
            assert abs(got - want) <= 0.005, (
                f"seed {seed} L{li + 1}: merge-only {got:.3f} "
                f"vs #342's {want:.3f}")
            assert rows[li]["solo_gain"] >= 0.5, (
                f"seed {seed} L{li + 1}: solo gain "
                f"{rows[li]['solo_gain']:.3f}")
        assert rows[0]["single_own"] >= 0.99, (
            f"seed {seed}: single-specialist own-domain "
            f"{rows[0]['single_own']:.3f}")


def test_predictor_init_quality_ordering():
    _empf_recipe()
    bw, rw = [], []
    for seed in SEEDS:
        r = _seed(seed)
        assert r["solo_gain"] >= 0.5, (
            f"seed {seed}: solo gain {r['solo_gain']:.3f}")
        ini = r["init"]
        # Predictor, worst domain: the BRR-regrouped
        # merge protects the worst domain against the
        # fixed similarity subset on every seed
        # (demonstrated gaps +0.075..+0.293), and the
        # similarity subset's worst domain is never
        # positive (demonstrated -0.214..-0.045).
        assert ini["brr"][1] >= ini["sim"][1] + 0.03, (
            f"seed {seed}: init worst brr vs sim "
            f"{ini['brr'][1]:.3f} vs {ini['sim'][1]:.3f}")
        assert ini["sim"][1] <= 0.0, (
            f"seed {seed}: init worst sim {ini['sim'][1]:.3f}")
        bw.append(ini["brr"][1])
        rw.append(ini["rand"][1])
        # Predictor, mean: the uniform merge is the
        # best-mean merge init (demonstrated >= +0.065
        # over every subset merge on every seed).
        for arm in ("brr", "sim", "rand"):
            assert ini["uniform"][0] >= ini[arm][0] + 0.04, (
                f"seed {seed}: init mean uniform vs {arm} "
                f"{ini['uniform'][0]:.3f} vs {ini[arm][0]:.3f}")
    # Against the fixed-random control, BRR's worst-
    # domain edge is a seed-mean effect (demonstrated
    # mean gap +0.127; BRR ahead on 4 of 5 seeds --
    # seed 3's fixed subset happens to cover the
    # worst domain, stated plainly).
    assert float(np.mean(bw)) >= float(np.mean(rw)) + 0.08, (
        f"init worst means brr vs rand "
        f"{np.mean(bw):.3f} vs {np.mean(rw):.3f}")
    wins = sum(1 for a, b in zip(bw, rw) if a > b)
    assert wins >= 4, f"brr worst beats rand worst on {wins}/5 seeds"


def test_generic_warm_start_at_B_and_half():
    _empf_recipe()
    for seed in SEEDS:
        r = _seed(seed)
        for cp in (B // 2, B):
            base_m = r["curve"]["base"][cp][0]
            for arm in ("uniform", "brr", "sim", "rand"):
                # #342's generic effect, replicated in
                # the chain sweep: every merge init is
                # at or above base-init at equal budget
                # (demonstrated margins +0.003..+0.022).
                assert r["curve"][arm][cp][0] >= base_m - 0.002, (
                    f"seed {seed} @{cp}: {arm} vs base "
                    f"{r['curve'][arm][cp][0]:.3f} vs {base_m:.3f}")


def test_claim_fails_flat_at_B_and_ordering_inverts_early():
    _empf_recipe()
    arms = ("uniform", "brr", "sim", "rand")
    for seed in SEEDS:
        r = _seed(seed)
        # At B and B/2 the merge inits are FLAT: the
        # spread across merge quality is within noise
        # (demonstrated <= 0.012 on mean at both
        # budgets on probed seeds) -- no ordering by
        # init quality survives consolidation.
        for cp in (B // 2, B):
            means = [r["curve"][a][cp][0] for a in arms]
            worsts = [r["curve"][a][cp][1] for a in arms]
            assert max(means) - min(means) <= 0.018, (
                f"seed {seed} @{cp}: merge-init mean spread "
                f"{max(means) - min(means):.3f}")
            assert max(worsts) - min(worsts) <= 0.030, (
                f"seed {seed} @{cp}: merge-init worst spread "
                f"{max(worsts) - min(worsts):.3f}")
        # Early, where init differences ARE visible,
        # they track init MEAN, not worst: uniform
        # (best init mean) tops the merge inits at
        # step 5 and BRR (best init worst, worst init
        # mean) sits in the bottom two of them -- on
        # the worst-domain readout the inversion is
        # largest (demonstrated uniform - brr >= +0.032
        # at step 5 on every seed).
        c5 = r["curve"]
        for arm in ("brr", "sim", "rand"):
            assert c5["uniform"][5][0] >= c5[arm][5][0] - 0.002, (
                f"seed {seed} @5 mean: uniform vs {arm} "
                f"{c5['uniform'][5][0]:.3f} vs {c5[arm][5][0]:.3f}")
        assert c5["uniform"][5][0] >= c5["brr"][5][0] + 0.015, (
            f"seed {seed} @5 mean: uniform vs brr "
            f"{c5['uniform'][5][0]:.3f} vs {c5['brr'][5][0]:.3f}")
        brr_rank = sum(1 for arm in ("uniform", "sim", "rand")
                       if c5[arm][5][0] < c5["brr"][5][0])
        assert brr_rank <= 1, (
            f"seed {seed} @5 mean: brr has {brr_rank} merge "
            f"inits below it (bottom two required)")
        assert c5["uniform"][5][1] >= c5["brr"][5][1] + 0.02, (
            f"seed {seed} @5 worst: uniform vs brr "
            f"{c5['uniform'][5][1]:.3f} vs {c5['brr'][5][1]:.3f}")
        # And every merge init still clears base early
        # by a wide margin (the warm start itself is
        # real; demonstrated >= +0.038 on mean @5).
        for arm in arms:
            assert c5[arm][5][0] >= c5["base"][5][0] + 0.03, (
                f"seed {seed} @5 mean: {arm} vs base "
                f"{c5[arm][5][0]:.3f} vs {c5['base'][5][0]:.3f}")


def test_shuffled_teacher_control_fails():
    _empf_recipe()
    for seed in SEEDS:
        r = _seed(seed)
        # Consolidating from the BRR init against
        # rotated teachers destroys recovery at B
        # (demonstrated <= -0.30 on probed seeds).
        assert r["shuffled_B"][0] <= 0.0, (
            f"seed {seed}: shuffled recovery "
            f"{r['shuffled_B'][0]:.3f}")
        assert r["curve"]["base"][B][0] >= r["shuffled_B"][0] + 0.2, (
            f"seed {seed}: base@B vs shuffled "
            f"{r['curve']['base'][B][0]:.3f} vs "
            f"{r['shuffled_B'][0]:.3f}")
