"""CPU gate for the consolidation claim of "Branch and
Consolidate: A Unified Recipe for Multi-Teacher On-Policy
Distillation" (arXiv:2611.32515), probed as a CHAIN test
inside the composition class of PR #282 (receiver-anchored
merging, arXiv:2611.25000) and PR #307 (TIES vs averaging
in that class, arXiv:2306.01708).

Paper claim: specialists post-trained from a shared
checkpoint acquire complementary capabilities; existing
consolidation -- from WEIGHT MERGING to MOPD -- falls
short when specialists DIVERGE SUBSTANTIALLY from the
shared checkpoint (prior studies mostly examine
specialists that remain close). Their recipe
(initialization from the shared checkpoint, data
selection, balanced supervision across teachers,
capability preservation + targeted refinement) recovers
95.7-97.5% of specialists' average gain, vs 94.1% for
IM-MOPD, in the open setting.

Chain question: PR #307 proved plain AVERAGING beats
TIES and is the stronger default in #282's harvested-
delta receiver class -- but at what DIVERGENCE does
merging stop recovering specialists' gains, and does
on-policy distillation-consolidation keep recovering
past that boundary? #307's verdict may be a
low-divergence fact; this probe prices the boundary.

Toy operationalization (the #282/#307 MLP family,
bit-faithful where calibration is claimed):
    z(x) = W3 tanh(W2 tanh(W1 x + b1) + b2) + b3,
    input dim 16, hidden 96, output dim 4.
A shared base is PRETRAINED on a general linear teacher
V0 over a broad Gaussian (sigma = 1.0, 600 steps, lr
0.03) -- #307's exact pretrain. K = 4 specialists are
independent fine-tunes of the SAME base, each on its own
domain: teacher V0 + dv_i read on blob i (blob means at
the #282 scale, sigma = 0.5). Divergence is swept over
3 levels by the fine-tune length AND the teacher
deviation scale dv (the share of each specialist's gain
that is domain-SPECIFIC rather than the shared V0
residual):

  L1  40 steps, lr 0.005, DV_SCALE 1.5 -- #307's EXACT
      fine-tune recipe (its FT_STEPS / FT_LR / DV_SCALE),
      the calibration level
  L2  120 steps, lr 0.005, DV_SCALE 3.0
  L3  240 steps, lr 0.01,  DV_SCALE 6.0

Measured delta norms grow 4.0 -> 6.9 -> 12.7 across
levels (the divergence axis is real in weight space);
pairwise delta cosines stay ~0 at every level, so in
this class divergence is carried by delta SCALE and
gain specificity, not by angle -- stated plainly, not
smoothed.

Arms:
  merge        plain AVERAGE of the specialist deltas
               applied at the shared checkpoint
               (theta_0 + mean_i tau_i) -- the #307
               default, transplanted from its receiver
               to the paper's shared checkpoint
  ties         (L1 only, calibration) TIES at #307's
               KEEP = 0.2, applied the same way
  consolidate  a student INITIALIZED AT THE SHARED
               CHECKPOINT, trained ON-POLICY against
               the specialists as teachers with BALANCED
               supervision: every step draws a fresh,
               equal-size batch from EACH domain blob
               (the states the student is trained on are
               freshly visited each step, never a fixed
               harvested set), and the target on a
               domain's states is that domain's SPECIALIST
               output -- never the ground-truth teacher.
               The student sees no V_i and no ground
               truth at all. This is the paper's core
               recipe minus its capability-preservation /
               targeted-refinement stage (documented
               omission: the toy has no general-capability
               probe distinct from the domains, so that
               stage has nothing to act on). In this
               stateless regression class the input
               distribution is exogenous, so "on-policy"
               collapses to fresh balanced domain states
               labeled by the teachers -- the MOPD
               structure, without a trajectory to roll out.
  shuffled     consolidate with the teacher<->domain
               assignment rotated by one (domain i's
               states labeled by teacher i+1) -- must fail
  single       (sanity) the merge of ONE specialist:
               its own-domain recovery must be ~1.0

Readout: RECOVERY on domain i = (loss(theta_0) -
loss(model)) / (loss(theta_0) - loss(specialist_i)) on
that domain's probes (256 probes; #307 used 128 --
deviation noted, chosen for stabler ratios), averaged
over domains, per seed. Solo gains are positive and
large at every level (validity floor asserted).

Measured on pop-os (particlegan 0.10.1), 3 seeds:

  level | |tau| | merge rec | cons rec | shuffled
  L1    | 4.04  | 0.297/0.416/0.341 | 0.996/0.992/0.990 | -0.83/-0.46/-0.23
  L2    | 6.81  | 0.167/0.349/0.307 | 0.997/0.997/0.993 | -1.04/-0.80/-0.50
  L3    | 12.72 | 0.186/0.336/0.222 | 0.993/0.994/0.991 | -1.34/-1.25/-0.93

  Calibration at L1 (#307's regime): plain averaging
  beats TIES on every seed (merge - ties = +0.37 /
  +0.13 / +0.71; ties recovery -0.07 / 0.29 / -0.37) --
  #307's ordering is intact in this transplant.
  Single-specialist merge: own-domain recovery exactly
  1.000 on every seed (the average of one delta is the
  delta).

Verdict, stated plainly: the paper's DIRECTION
reproduces and its implied boundary does NOT sit where
the paper's framing suggests for this class. Merging
DOES degrade as specialists diverge (seed-mean recovery
0.35 -> 0.27 -> 0.25; L1->L3 drop on every seed,
0.08-0.12) while consolidation stays at 0.99 at every
level -- so past the boundary, consolidation is the
only arm still recovering, exactly as the paper claims.
But there is NO low-divergence plateau in this class
where merge ~= consolidate: at the lowest level --
#307's exact regime, specialists as close to the shared
checkpoint as this family gets -- merging already
recovers only 0.30-0.42 against consolidation's 0.99.
The mechanism is visible in the construction: an MLP
fine-tune's delta is REGION-LOCAL (it repairs the base
mostly on its own blob), so averaging dilutes each
specialist's correction by ~1/K on its own domain at
ANY divergence; divergence then adds specificity loss
on top. For this class the merge/consolidate boundary
lies BELOW #307's regime, not above it: #307's
"averaging is the stronger default" is a verdict about
which MERGE is least bad, and it survives (calibration
above), but no merge in this class was ever close to
consolidation. Assertions encode exactly these measured
margins -- merge degrading per seed, consolidation flat
and high, the gap wide at every level, the calibration
ordering, and both controls.

What is NOT asserted: anything about large pretrained
checkpoints, classification accuracy, the paper's
preservation/refinement stage, or its exact 95.7-97.5%
figures (consolidation here recovers MORE, ~99%, in a
far smaller class). Pure torch -- no GAN machinery,
because the claim is about how specialist knowledge is
consolidated, not training dynamics. Skips cleanly when
the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family).
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


DIN, HID, COUT = 16, 96, 4    # toy MLP shape (#307)
SIG = 0.5                     # per-blob input width (#282/#307)
MEAN_SCALE = 2.5              # blob-mean draw scale (#307)
PRE_STEPS = 600               # base pretraining steps (#307)
K = 4                         # specialists
KEEP = 0.2                    # TIES trim keep-fraction (#307)
CONSOL_STEPS = 600            # consolidation steps
CONSOL_LR = 0.01
CONSOL_PER = 64               # fresh states per domain per step
# (fine-tune steps, lr, teacher-deviation scale) per level;
# L1 is #307's exact fine-tune recipe (40 / 0.005 / 1.5).
LEVELS = ((40, 0.005, 1.5), (120, 0.005, 3.0), (240, 0.01, 6.0))


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


def _consolidate(base, teachers, means, steps, seed,
                 shuffle=False):
    """On-policy balanced distillation-consolidation.

    Student initialized at the shared checkpoint; each
    step visits fresh equal-size batches from every
    domain and is supervised by that domain's specialist
    (rotated by one when shuffle=True). Targets are
    teacher outputs only -- no ground truth anywhere.
    """
    params = {k: t.clone().requires_grad_(True)
              for k, t in base.items()}
    opt = torch.optim.Adam(list(params.values()), lr=CONSOL_LR)
    for s in range(steps):
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
    return {k: t.detach().clone() for k, t in params.items()}


def _trim(tau, keep_frac):
    out = {}
    for k, t in tau.items():
        flat = t.flatten()
        keep = max(1, int(round(keep_frac * flat.numel())))
        idx = torch.topk(flat.abs(), keep).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1
        out[k] = (flat * mask).reshape(t.shape)
    return out


def _elect(taus):
    """TIES disjoint mean under the elected sign (#307)."""
    out = {}
    for k in taus[0]:
        stack = torch.stack([t[k] for t in taus])
        sign = torch.sign(stack.sum(0))
        aligned = torch.where(stack * sign > 0, stack,
                              torch.zeros_like(stack))
        cnt = (aligned != 0).sum(0).clamp(min=1)
        out[k] = aligned.sum(0) / cnt
    return out


def run_seed(seed):
    """All arm recoveries for one seed, over all levels.

    Returns per level: mean delta norm, mean solo gain,
    and per-arm mean recovery (merge / consolidate /
    shuffled; ties and single only at L1, the
    calibration level).
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
        norms = [float(np.sqrt(sum(float((t[k] ** 2).sum())
                                   for k in t))) for t in taus]

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
        avg = {k: sum(t[k] for t in taus) / K for k in base}
        merged = {k: base[k] + avg[k] for k in base}
        cons = _consolidate(base, specs, [m for _, m in tasks],
                            CONSOL_STEPS, seed * 10 + li)
        shuf = _consolidate(base, specs, [m for _, m in tasks],
                            CONSOL_STEPS, seed * 10 + li + 555,
                            shuffle=True)
        row = dict(norm=float(np.mean(norms)),
                   solo_gain=float(np.mean(solo_gains)),
                   merge=rec(merged), cons=rec(cons),
                   shuffled=rec(shuf))
        if li == 0:
            tiesd = _elect([_trim(t, KEEP) for t in taus])
            tiesm = {k: base[k] + tiesd[k] for k in base}
            row["ties"] = rec(tiesm)
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


def test_divergence_axis_is_real():
    _empf_recipe()
    for seed in (0, 1, 2):
        lv = _seed(seed)
        # Delta norms grow strictly and substantially
        # across levels (demonstrated means 4.04 -> 6.81
        # -> 12.72; per-seed L3/L1 ratio >= 2.9).
        assert lv[0]["norm"] < lv[1]["norm"] < lv[2]["norm"], (
            f"seed {seed}: norms {[r['norm'] for r in lv]}")
        assert lv[2]["norm"] >= 2.5 * lv[0]["norm"], (
            f"seed {seed}: norm ratio "
            f"{lv[2]['norm'] / lv[0]['norm']:.2f}")
        # Every specialist realizes a real solo gain at
        # every level (validity floor; demonstrated mean
        # solo gains >= 1.2 at L1 and larger beyond).
        for li, row in enumerate(lv):
            assert row["solo_gain"] >= 0.5, (
                f"seed {seed} L{li + 1}: solo gain "
                f"{row['solo_gain']:.3f}")


def test_calibration_307_ordering_at_lowest_level():
    _empf_recipe()
    for seed in (0, 1, 2):
        row = _seed(seed)[0]
        # #307's ordering, transplanted to the shared
        # checkpoint at #307's exact fine-tune recipe:
        # plain averaging beats TIES on every seed
        # (demonstrated merge - ties = +0.37/+0.13/+0.71).
        assert row["merge"] >= row["ties"] + 0.08, (
            f"seed {seed}: merge vs ties recovery "
            f"{row['merge']:.3f} vs {row['ties']:.3f}")
        # And merging is already WEAK here -- no
        # low-divergence plateau in this class
        # (demonstrated merge recovery 0.30-0.42 while
        # consolidation sits at 0.99, next test).
        assert row["merge"] <= 0.50, (
            f"seed {seed}: L1 merge recovery {row['merge']:.3f}")
        # Sanity: merging a single specialist recovers
        # its own domain exactly (demonstrated 1.000).
        assert row["single_own"] >= 0.99, (
            f"seed {seed}: single-specialist recovery "
            f"{row['single_own']:.3f}")


def test_merge_degrades_consolidation_does_not():
    _empf_recipe()
    merge_means, cons_means = [], []
    for seed in (0, 1, 2):
        lv = _seed(seed)
        # Merge recovery drops from L1 to L3 on every
        # seed (demonstrated drops 0.11 / 0.08 / 0.12).
        assert lv[2]["merge"] <= lv[0]["merge"] - 0.05, (
            f"seed {seed}: merge L1->L3 "
            f"{lv[0]['merge']:.3f} -> {lv[2]['merge']:.3f}")
        for li, row in enumerate(lv):
            # Consolidation stays high at EVERY level
            # (demonstrated 0.990-0.997 everywhere).
            assert row["cons"] >= 0.95, (
                f"seed {seed} L{li + 1}: cons recovery "
                f"{row['cons']:.3f}")
            # The gap is wide at every level -- including
            # the lowest (demonstrated gaps 0.58-0.81).
            assert row["cons"] >= row["merge"] + 0.45, (
                f"seed {seed} L{li + 1}: cons vs merge "
                f"{row['cons']:.3f} vs {row['merge']:.3f}")
        merge_means.append([r["merge"] for r in lv])
        cons_means.append([r["cons"] for r in lv])
    # Seed-mean boundary pricing: merge 0.35 -> 0.25
    # across the sweep while consolidation holds 0.99.
    mm = np.mean(merge_means, axis=0)
    cm = np.mean(cons_means, axis=0)
    assert mm[0] - mm[2] >= 0.08, (
        f"seed-mean merge drop {mm[0]:.3f} -> {mm[2]:.3f}")
    assert float(np.min(cm)) >= 0.98, (
        f"seed-mean cons by level {cm}")


def test_shuffled_teacher_control_fails():
    _empf_recipe()
    for seed in (0, 1, 2):
        for li, row in enumerate(_seed(seed)):
            # Rotating the teacher<->domain assignment
            # destroys recovery at every level
            # (demonstrated shuffled recovery -1.34 to
            # -0.23, always <= 0 and >= 1.2 below cons).
            assert row["shuffled"] <= 0.0, (
                f"seed {seed} L{li + 1}: shuffled recovery "
                f"{row['shuffled']:.3f}")
            assert row["cons"] >= row["shuffled"] + 0.8, (
                f"seed {seed} L{li + 1}: cons vs shuffled "
                f"{row['cons']:.3f} vs {row['shuffled']:.3f}")
