"""CPU gate for the strong-start claim of "Strong Starts,
Simple Recipes: Unified Multi-Teacher On-Policy
Distillation from Multimodal Specialists" (Uni-MOPD,
arXiv:2611.34397), probed as a CHAIN test inside the
composition class of PR #324 (Branch and Consolidate,
arXiv:2611.32515).

Paper claim (from the abstract): multi-teacher on-policy
distillation (MOPD) consolidates specialists into one
model, and Uni-MOPD treats the STARTING POINT as the
design variable: (1) a training-free UNIFORM MERGE of
the specialists initializes the student -- before any
training, the merge already matches the score the BASE
model reaches after 200 MOPD updates; (2) SHORT MOPD
from the merge init then matches higher-scoring
search-based starts; (3) from a strong start, uniform
domain sampling + full-vocabulary reverse KL match or
exceed every fancier sampling strategy / divergence --
no schedules, per-token weighting, or truncation
hyperparameters needed.

Chain provenance -- what PR #324 measured, quoted
exactly (its docstring table, 3 seeds, pop-os,
particlegan 0.10.1):

  level | |tau| | merge rec       | cons rec        | shuffled
  L1    | 4.04  | 0.297/0.416/0.341 | 0.996/0.992/0.990 | -0.83/-0.46/-0.23
  L2    | 6.81  | 0.167/0.349/0.307 | 0.997/0.997/0.993 | -1.04/-0.80/-0.50
  L3    | 12.72 | 0.186/0.336/0.222 | 0.993/0.994/0.991 | -1.34/-1.25/-0.93

#324's verdict: merging recovery 0.25-0.35 (seed means
0.35 -> 0.27 -> 0.25 across L1-L3) vs on-policy
consolidation ~0.99 at EVERY divergence level -- merging
was never close, because region-local deltas dilute
~1/K under averaging. But #324 compared merge-ONLY
against consolidate-from-base; it never ran the paper's
actual composition -- merge as INITIALIZATION, then
consolidate. The open question this toy decides: does
merge-init change consolidation's recovery or its price
(updates to a fixed recovery), and does the paper's
claim (1) hold in this class (merge-only ~= base-init
after substantial consolidation)?

Toy operationalization (the #282/#307/#324 MLP family,
bit-faithful where calibration is claimed):

    z(x) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3,
    input dim 16, hidden 96, output dim 4.

Shared base PRETRAINED on a general linear teacher V0
over a broad Gaussian (sigma = 1.0, 600 steps, lr 0.03)
-- #324's exact pretrain. K = 4 specialists are
independent fine-tunes of the SAME base, each on its own
domain: teacher V0 + dv_i read on blob i (blob means at
the #282 scale, sigma = 0.5). Divergence levels are
#324's L1/L2 constructions verbatim:

  L1  40 steps, lr 0.005, DV_SCALE 1.5 -- #324/#307's
      exact fine-tune recipe, the calibration level
  L2  120 steps, lr 0.005, DV_SCALE 3.0

(#324's L3 is dropped -- stated deviation: L1/L2 already
bracket the merge-only range 0.17-0.49 and the question
here is the init, not the boundary; dropping L3 keeps
the 5-seed sweep cheap.)

Consolidation is #324's on-policy distillation verbatim:
every step draws a fresh, equal-size batch from EACH
domain blob and the target on a domain's states is that
domain's SPECIALIST output -- never ground truth. In
this stateless regression class "on-policy" collapses
to fresh balanced domain states labeled by the teachers,
exactly as #324 documented. Matched consolidation
budget B = 100 steps (lr 0.01, 64 fresh states per
domain per step, as #324). B is chosen mid-curve --
#324's B = 600 saturates both inits at ~0.99, where no
init effect could be priced; at B = 100 base-init sits
at ~0.95-0.97, unsaturated.

Arms (per seed, per level):

  merge-only   theta_0 + mean_i tau_i, no training --
               #324's merge arm, must replicate ~0.25-0.35
  base@B       consolidate from the shared checkpoint
               theta_0 for B steps -- #324's ~0.99 arm,
               here truncated to the unsaturated B
  merge@B      consolidate from the UNIFORM MERGE for
               B steps -- the paper's composition
  merge@B/2    consolidate from the merge for 50 steps --
               the "short MOPD" efficiency form
  best@B       consolidate from the BEST-SINGLE
               specialist (largest solo gain) for B
               steps -- a search-start proxy for claim (2)
  shuffled@B   consolidate from theta_0 with the
               teacher<->domain assignment rotated by
               one -- #324's control, must fail

Snapshots at intermediate budgets (5/10/25/50) come from
the SAME training run as the B snapshot: the per-step
RNG stream is #324's, so a snapshot at step t is exactly
the model a t-step consolidation run would produce.
Claim checks:

  (i)  efficiency: merge@B >= base@B at equal budget, OR
       merge@B/2 reaches base@B (half the budget);
  (ii) claim (1)'s local form: merge-only vs base-init
       at SMALL budgets (5/10/25 steps) -- does base-init
       need many updates to pass the merge?
  (iii) claim (2)'s local form: merge@B vs best@B (does
       the merge start match the search-start proxy?).

Readout: RECOVERY on domain i = (loss(theta_0) -
loss(model)) / (loss(theta_0) - loss(specialist_i)) on
that domain's probes (256 probes), averaged over
domains, per seed -- exactly #324's definition.
Single-specialist ceiling: merging ONE specialist
recovers its own domain at exactly 1.000 (L1
calibration, as #324).

Modeling gifts / deviations, stated plainly:

  * The consolidation loss is #324's MSE-to-teacher-
    outputs, NOT the paper's full-vocabulary reverse KL:
    this class has no vocabulary and no token
    distribution, and keeping #324's loss is what makes
    the numbers comparable to #324. Claim (3)'s
    DIVERGENCE half is therefore OUT OF SCOPE. Its
    SAMPLING half is the toy's default: consolidation
    here already uses uniform domain sampling and
    nothing fancier, as in #324.
  * best@B is a PROXY for a search-based start (pick the
    strongest specialist), not a searched initialization
    procedure; claim (2) is tested only in that proxy
    form.
  * The paper's specialists are multimodal; this class
    is a small regression MLP family. Nothing is asserted
    about large pretrained checkpoints.

Measured on pop-os (particlegan 0.10.1), 5 seeds
(recoveries; base@25 shown for claim (1)):

  L1: merge-only 0.297/0.416/0.341/0.492/0.315 (mean 0.372)
      base@25    0.825/0.862/0.817/0.814/0.850 (mean 0.834)
      base@100   0.963/0.967/0.947/0.970/0.963 (mean 0.962)
      merge@50   0.939/0.952/0.923/0.937/0.942 (mean 0.939)
      merge@100  0.974/0.976/0.957/0.975/0.971 (mean 0.971)
      best@100   0.966/0.970/0.946/0.969/0.962 (mean 0.963)
  L2: merge-only 0.167/0.349/0.307/0.473/0.216 (mean 0.302)
      base@25    0.747/0.806/0.776/0.789/0.791 (mean 0.782)
      base@100   0.952/0.963/0.953/0.960/0.957 (mean 0.957)
      merge@50   0.915/0.942/0.931/0.938/0.928 (mean 0.931)
      merge@100  0.961/0.970/0.964/0.967/0.964 (mean 0.965)
      best@100   0.954/0.968/0.948/0.958/0.950 (mean 0.956)

  Replication: on the three seeds #324 ran, merge-only
  reproduces #324's table EXACTLY at L1 (0.297/0.416/
  0.341) and L2 (0.167/0.349/0.307) -- the apparatus is
  bit-faithful, so the new arms price the init, not a
  re-implementation.

Verdict, stated plainly: PARTIAL.

  * Claim (1) does NOT reproduce in this class.
    Merge-only (0.17-0.49) does not match a trained base
    at any budget that matters: base-init passes the
    merge within 10 consolidation steps on every
    seed/level (base@10 >= merge-only + 0.05), and by
    25 steps leads it by >= 0.31. In the paper the merge
    is worth 200 updates; here it is worth fewer than
    10. This is #324's dilution mechanism seen from the
    other side: a start diluted ~1/K per domain has
    little head start to give.
  * Claim (i), equal-budget form, HOLDS WEAKLY:
    merge@B beats base@B on 10/10 seed/level combos, but
    by +0.005 to +0.011 (seed means +0.009 L1, +0.008
    L2) -- a real, consistent, SMALL head start, not
    the paper's strong-start effect.
  * Claim (i), half-budget form, FAILS: merge@B/2
    trails base@B on 10/10 combos by 0.015-0.037.
    Merge-init does not halve consolidation's price in
    this class; interpolating the curves, it saves
    roughly a fifth to a third of the budget to a fixed
    recovery, not half.
  * Claim (2)'s proxy form HOLDS: merge@B >= best@B on
    10/10 combos (+0.002 to +0.014) -- the training-free
    merge is at least as good a start as the strongest
    single specialist, whose own average recovery at
    step 0 is NEGATIVE (-0.12 to -1.54) yet washes out
    to ~base@B by B steps. No search is needed to match
    the search-start proxy -- but the prize is small,
    because at B = 100 every init sits within ~0.015.

Assertions encode exactly these measured margins --
the replication, the claim-(1) failure, the weak
equal-budget win, the half-budget failure, the proxy
win, and the shuffled control. Pure torch -- no GAN
machinery, because the claim is about how specialist
knowledge is consolidated, not training dynamics.
Skips cleanly when the installed particlegan does not
provide the EMPF recipe (the house guard for this
paper-test family).
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


DIN, HID, COUT = 16, 96, 4    # toy MLP shape (#307/#324)
SIG = 0.5                     # per-blob input width (#282/#307)
MEAN_SCALE = 2.5              # blob-mean draw scale (#307)
PRE_STEPS = 600               # base pretraining steps (#307/#324)
K = 4                         # specialists
CONSOL_LR = 0.01              # consolidation lr (#324)
CONSOL_PER = 64               # fresh states per domain per step (#324)
B = 100                       # matched consolidation budget
CHECKPOINTS = (0, 5, 10, 25, 50, 100)
# (fine-tune steps, lr, teacher-deviation scale) per level;
# both are #324's exact constructions (its L1 and L2).
LEVELS = ((40, 0.005, 1.5), (120, 0.005, 3.0))
SEEDS = (0, 1, 2, 3, 4)


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
                       seed, shuffle=False):
    """On-policy balanced distillation-consolidation,
    snapshotted at each checkpoint step.

    Student initialized at `init`; each step visits fresh
    equal-size batches from every domain and is supervised
    by that domain's specialist (rotated by one when
    shuffle=True). Targets are teacher outputs only -- no
    ground truth anywhere. The per-step RNG stream is
    #324's _consolidate stream, so the snapshot at step t
    is exactly the model a t-step run would produce.
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


def run_seed(seed):
    """All arm recoveries for one seed, over L1/L2.

    Returns per level: mean solo gain, merge-only
    recovery, per-checkpoint recoveries for the base-init
    and merge-init curves, best-single-init recovery at
    0 and at B, shuffled recovery at B, and (L1 only) the
    single-specialist own-domain ceiling.
    """
    g = torch.Generator().manual_seed(3000 + seed)
    v0 = torch.randn(COUT, DIN, generator=g) * (3.0 / np.sqrt(DIN))
    base = _train(_base(seed), v0, torch.zeros(DIN),
                  PRE_STEPS, 2000 + seed, lr=0.03, sig=1.0, n=512)

    def make_task(s, dv_scale):
        gg = torch.Generator().manual_seed(s)
        dv = torch.randn(COUT, DIN, generator=gg) * (dv_scale / np.sqrt(DIN))
        m = torch.randn(DIN, generator=gg) * (MEAN_SCALE / np.sqrt(DIN))
        return v0 + dv, m

    levels = []
    for li, (steps, lr, dvs) in enumerate(LEVELS):
        tasks = [make_task(6000 + seed * 1000 + i * 17, dvs)
                 for i in range(K)]
        specs = [_train(base, v, m, steps,
                        7100 + seed * 10 + i, lr=lr)
                 for i, (v, m) in enumerate(tasks)]
        taus = [{k: sp[k] - base[k] for k in base} for sp in specs]
        avg = {k: sum(t[k] for t in taus) / K for k in base}
        merged = {k: base[k] + avg[k] for k in base}
        means = [m for _, m in tasks]

        def rec(model):
            out = []
            for i, (v, m) in enumerate(tasks):
                l0 = _loss(base, v, m, 9100 + seed)
                ls = _loss(specs[i], v, m, 9100 + seed)
                lm = _loss(model, v, m, 9100 + seed)
                out.append((l0 - lm) / (l0 - ls))
            return float(np.mean(out))

        solo_gains = [_loss(base, v, m, 9100 + seed)
                      - _loss(specs[i], v, m, 9100 + seed)
                      for i, (v, m) in enumerate(tasks)]
        best = int(np.argmax(solo_gains))

        base_snaps = _consolidate_curve(
            base, specs, means, CHECKPOINTS, seed * 10 + li)
        merge_snaps = _consolidate_curve(
            merged, specs, means, CHECKPOINTS, seed * 10 + li)
        best_snaps = _consolidate_curve(
            specs[best], specs, means, (0, B), seed * 10 + li)
        shuf_snaps = _consolidate_curve(
            base, specs, means, (B,), seed * 10 + li + 555,
            shuffle=True)

        row = dict(
            solo_gain=float(np.mean(solo_gains)),
            merge_only=rec(merged),
            base_curve={cp: rec(base_snaps[cp]) for cp in CHECKPOINTS},
            merge_curve={cp: rec(merge_snaps[cp]) for cp in CHECKPOINTS},
            best_0=rec(best_snaps[0]),
            best_B=rec(best_snaps[B]),
            shuffled_B=rec(shuf_snaps[B]),
        )
        if li == 0:
            single = {k: base[k] + taus[0][k] for k in base}
            v, m = tasks[0]
            l0 = _loss(base, v, m, 9100 + seed)
            ls = _loss(specs[0], v, m, 9100 + seed)
            l1 = _loss(single, v, m, 9100 + seed)
            row["single_own"] = (l0 - l1) / (l0 - ls)
        levels.append(row)
    return levels


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_replication_of_324_merge_only():
    _empf_recipe()
    for seed in SEEDS:
        lv = _seed(seed)
        for li, row in enumerate(lv):
            # Every specialist realizes a real solo gain at
            # every level (validity floor, as #324).
            assert row["solo_gain"] >= 0.5, (
                f"seed {seed} L{li + 1}: solo gain "
                f"{row['solo_gain']:.3f}")
            # Merge-only stays in #324's weak band
            # (demonstrated 0.167-0.492 over 5 seeds;
            # #324's three seeds replicate exactly).
            assert 0.05 <= row["merge_only"] <= 0.55, (
                f"seed {seed} L{li + 1}: merge-only recovery "
                f"{row['merge_only']:.3f}")
            # Consolidation from base at B = 100 already
            # towers over merge-only (demonstrated gaps
            # 0.48-0.79; at #324's B = 600 it reaches 0.99).
            assert row["base_curve"][B] >= row["merge_only"] + 0.40, (
                f"seed {seed} L{li + 1}: base@B vs merge-only "
                f"{row['base_curve'][B]:.3f} vs "
                f"{row['merge_only']:.3f}")
        # Sanity: merging a single specialist recovers
        # its own domain exactly (demonstrated 1.000).
        assert lv[0]["single_own"] >= 0.99, (
            f"seed {seed}: single-specialist recovery "
            f"{lv[0]['single_own']:.3f}")


def test_claim1_merge_does_not_match_trained_base():
    _empf_recipe()
    for seed in SEEDS:
        for li, row in enumerate(_seed(seed)):
            mo = row["merge_only"]
            # The paper's merge is worth 200 updates; here
            # base-init passes it within 10 steps on every
            # seed/level (demonstrated base@10 - merge
            # >= +0.18 on the probed seeds, asserted with
            # headroom).
            assert row["base_curve"][10] >= mo + 0.05, (
                f"seed {seed} L{li + 1}: base@10 vs merge-only "
                f"{row['base_curve'][10]:.3f} vs {mo:.3f}")
            # By 25 steps the lead is decisive
            # (demonstrated >= +0.31 everywhere).
            assert row["base_curve"][25] >= mo + 0.25, (
                f"seed {seed} L{li + 1}: base@25 vs merge-only "
                f"{row['base_curve'][25]:.3f} vs {mo:.3f}")


def test_merge_init_equal_budget_win_half_budget_loss():
    _empf_recipe()
    diffs_B, diffs_half_vs_B = [], []
    for seed in SEEDS:
        for li, row in enumerate(_seed(seed)):
            base_B = row["base_curve"][B]
            merge_B = row["merge_curve"][B]
            merge_half = row["merge_curve"][B // 2]
            # Equal budget: merge-init is never worse and
            # is ahead on every demonstrated combo
            # (+0.005 to +0.011); asserted per combo with
            # a small tolerance, and on the mean below.
            assert merge_B >= base_B - 0.002, (
                f"seed {seed} L{li + 1}: merge@B vs base@B "
                f"{merge_B:.3f} vs {base_B:.3f}")
            # The win is already present at B/2 against
            # base at B/2 (demonstrated +0.009 to +0.037).
            assert merge_half >= row["base_curve"][B // 2] - 0.002, (
                f"seed {seed} L{li + 1}: merge@B/2 vs base@B/2 "
                f"{merge_half:.3f} vs "
                f"{row['base_curve'][B // 2]:.3f}")
            # Half-budget form FAILS: merge@B/2 trails
            # base@B on every combo (demonstrated gaps
            # 0.015-0.037).
            assert merge_half <= base_B - 0.010, (
                f"seed {seed} L{li + 1}: merge@B/2 vs base@B "
                f"{merge_half:.3f} vs {base_B:.3f}")
            diffs_B.append(merge_B - base_B)
            diffs_half_vs_B.append(base_B - merge_half)
    # Seed-mean pricing: the equal-budget head start is
    # real but small (demonstrated mean +0.0085), and
    # the half-budget shortfall is systematic
    # (demonstrated mean 0.026).
    assert float(np.mean(diffs_B)) >= 0.004, (
        f"mean merge@B - base@B {np.mean(diffs_B):.3f}")
    assert float(np.mean(diffs_half_vs_B)) >= 0.015, (
        f"mean base@B - merge@B/2 {np.mean(diffs_half_vs_B):.3f}")


def test_merge_init_matches_best_single_start():
    _empf_recipe()
    for seed in SEEDS:
        for li, row in enumerate(_seed(seed)):
            # The strongest single specialist is a BAD
            # start on average recovery (demonstrated
            # -1.54 to -0.12 at step 0) ...
            assert row["best_0"] <= 0.0, (
                f"seed {seed} L{li + 1}: best-single start "
                f"recovery {row['best_0']:.3f}")
            # ... yet consolidation washes it out to
            # ~base@B by B steps (demonstrated within
            # 0.007 of base@B in either direction).
            assert row["best_B"] >= row["base_curve"][B] - 0.02, (
                f"seed {seed} L{li + 1}: best@B vs base@B "
                f"{row['best_B']:.3f} vs "
                f"{row['base_curve'][B]:.3f}")
            # And the training-free merge start matches or
            # beats that search-start proxy at B
            # (demonstrated +0.002 to +0.014).
            assert row["merge_curve"][B] >= row["best_B"] - 0.005, (
                f"seed {seed} L{li + 1}: merge@B vs best@B "
                f"{row['merge_curve'][B]:.3f} vs "
                f"{row['best_B']:.3f}")


def test_shuffled_teacher_control_fails():
    _empf_recipe()
    for seed in SEEDS:
        for li, row in enumerate(_seed(seed)):
            # Rotating the teacher<->domain assignment
            # destroys recovery at B (demonstrated
            # shuffled@B <= -0.21 on the probed seeds,
            # always far below base@B).
            assert row["shuffled_B"] <= 0.0, (
                f"seed {seed} L{li + 1}: shuffled recovery "
                f"{row['shuffled_B']:.3f}")
            assert row["base_curve"][B] >= row["shuffled_B"] + 0.8, (
                f"seed {seed} L{li + 1}: base@B vs shuffled "
                f"{row['base_curve'][B]:.3f} vs "
                f"{row['shuffled_B']:.3f}")
