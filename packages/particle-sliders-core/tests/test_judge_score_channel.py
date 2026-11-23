"""CPU gate for the judge-score subliminal-transfer claim of
Kim & Choi, "Subliminal Reinforcement Learning: Traits Travel
Through a Judge's Scores" (arXiv:2610.41301): in judge-scored
RL (GRPO) the student learns only from its own outputs plus
scores, yet a trait carried ONLY in the judge's score loading
transfers to the student (17 of 48 design cells establish the
trait at 5% FDR). Removing only the reward difference between
judges removes transfer. Their four-stage chain: the trait
must (1) show in the judge's scores, (2) favour strings that
move the student -- the loading must ride on strings this
student actually produces; judges whose loading sits on other
strings do NOT transfer, (3) accumulate, (4) reach the
answers. Provenance: cycle 15 of the Particle Sliders paper
watch (brief #2); the first subliminal result that lives at
our critic stage -- Jev-guided curation is selection, a
one-step hard version of the same advantage weighting.

Toy operationalization: a categorical student over a fixed
set of N = 128 candidate "strings". Each string has a known
quality q and a trait marker t in {0, 1}. Quality takes five
levels (-1, -0.5, 0, +0.5, +1); within each level exactly
half the strings carry the trait, so q and t are orthogonal
by construction (corr ~ -0.02) and the top quality level
contains equally good trait and non-trait strings -- the
matched-quality probe is built into the design. The student
is a softmax over logits (small init noise, broad support);
16 strings are UNREACHABLE (logits pinned at -20, never
sampled, gradient frozen) -- the support the student's
distribution never visits. The student never sees q or t,
only scores of its own samples.

Judge: score(string) = q + lambda * t with lambda = 1.0
(trait-loaded) vs score = q (counterfactual judge --
identical quality scoring, zero trait loading).

Training: GRPO/REINFORCE form. Each step samples a group of
32 strings from the student, advantage = (score - group
mean) / group std, logits updated by the advantage-weighted
softmax gradient, 300 steps.

Arms (all arms of a seed share init logits and the training
RNG stream; they differ only in the reward):
  (a) loaded:      reward = score from the trait-loaded
                   judge (loading on reachable strings);
  (b) counterfactual: reward = score from the q-only judge;
  (c) differenced: reward computed by the audit procedure --
                   score the same sampled strings with BOTH
                   judge configurations, estimate the trait
                   component as their difference, subtract
                   it: reward = s_loaded - (s_loaded - s_cf).
                   With this toy's noiseless judge the
                   difference is exact, so (c) must track (b)
                   to numerical precision -- the arm tests
                   the differencing PROCEDURE (the audit's
                   control), not a noisy estimate of it;
  (d) misplaced:   the loaded judge's trait loading sits
                   ONLY on the unreachable strings (stage 2
                   of the paper's chain fails by
                   construction) -- no transfer expected.

Readouts:
  (1) score loading (stage 1): OLS of judge scores on
      [1, q, t] over the REACHABLE strings -- the loaded
      judge's t coefficient is lambda by construction; the
      counterfactual's is 0; the misplaced judge's is 0 on
      the reachable support (its loading exists, but not
      where the student lives);
  (2) transfer: the student's probability mass on trait
      strings within the reachable support at the end of
      training (trait share; init ~0.49-0.52);
  (3) accumulation (stage 3): trait share recorded at steps
      0 / 100 / 200 / 299 of arm (a) -- it must grow;
  (4) quality sanity: mean q under the final policy rises
      in every arm (the training itself works; arm (a)'s
      trait install is not bought by sacrificing quality --
      its mean q is the highest of the arms, because the
      trait strings at the top level are exactly as good).

Measured on pop-os (particlegan 0.10.1), 5 seeds -- trait
share at end of training, arms (a)/(b)/(c)/(d):
  seed 0: 0.984 / 0.448 / 0.448 / 0.448
  seed 1: 0.996 / 0.463 / 0.463 / 0.463
  seed 2: 0.992 / 0.559 / 0.559 / 0.559
  seed 3: 0.995 / 0.470 / 0.470 / 0.470
  seed 4: 0.995 / 0.373 / 0.373 / 0.373
Arm (a) accumulation (steps 0/100/200/299): seed 0
0.489/0.582/0.715/0.984; seed 4 0.463/0.556/0.715/0.995 --
monotone growth on every seed. Mean q at end: arm (a)
0.97-0.99, arms (b/c/d) 0.66-0.80. Score loading on the
reachable support: exactly lambda = 1.0 for the loaded
judge, exactly 0.0 for the misplaced judge. Arms (c) and
(d) track (b) EXACTLY (difference 0.000 on every seed):
differencing removes the trait component completely, and
loading placed outside the student's support transfers
nothing -- both are the paper's claims in their strongest
form, and both are exact here because the judge is
noiseless and the unreachable strings are never sampled.
The counterfactual arm's own trait share wanders around
its init (0.37-0.56 vs init 0.46-0.53): within a quality
level, q-only advantages are identical, but sampling
frequency compounds (rich-get-richer), so the q-only
policy concentrates on a random within-level subset. That
wander is why the assertions are stated as per-seed
MARGINS of (a) over each control (demonstrated 0.43-0.62),
never as an absolute level for the controls. Assertions
below encode exactly these margins with modest headroom.

What is NOT asserted: anything about noisy judges (with
score noise the differencing control in arm (c) removes the
trait only in expectation, and the residual transfer rate
becomes an empirical question this toy does not arbitrate);
transfer through a student that must DISCOVER strings (the
support is fixed and given); the paper's 17/48 cell rate or
its FDR machinery (the toy tests the mechanism -- score
loading alone suffices, differencing removes it, misplaced
loading does not transfer -- not the paper's statistics);
and any claim about real LM judges, whose trait loadings
are neither binary nor known. One modeling gift, stated
plainly: q and t are orthogonal BY CONSTRUCTION here --
in a real pool the audit's first job is exactly to measure
the loading the toy assumes known. Pure numpy -- no GAN
machinery, because the claim is about score-mediated
training dynamics, not the adapter. Skips cleanly when the
installed particlegan does not provide the EMPF recipe (the
house guard for this paper-test family; PR #211 is what
enables this test in CI).
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


N = 128              # candidate strings
N_UNREACH = 16       # strings outside the student's support
LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # quality levels
LAM = 1.0            # judge trait loading
GROUP = 32           # GRPO group size per step
STEPS = 300          # training steps per arm
LR = 0.35            # REINFORCE step on logits
SEEDS = (0, 1, 2, 3, 4)
CHECKPOINTS = (0, 100, 200, STEPS - 1)


def build(seed):
    """Strings for one seed: q levels, t balanced per level.

    Level assignment and the within-level trait split are
    shuffled per seed; the unreachable tail is fixed as the
    last N_UNREACH indices.
    """
    rng = np.random.default_rng(41000 + seed)
    lvl = np.tile(np.arange(len(LEVELS)), N // len(LEVELS) + 1)[:N]
    rng.shuffle(lvl)
    q = LEVELS[lvl]
    t = np.zeros(N)
    for lv in range(len(LEVELS)):
        idx = np.where(lvl == lv)[0].copy()
        rng.shuffle(idx)
        t[idx[: len(idx) // 2]] = 1.0
    unreach = np.zeros(N, dtype=bool)
    unreach[N - N_UNREACH:] = True
    return dict(q=q, t=t, unreach=unreach, reach=~unreach)


def _softmax(logits):
    p = np.exp(logits - logits.max())
    return p / p.sum()


def _trait_share(p, d):
    r = d["reach"]
    return float(p[r & (d["t"] == 1)].sum() / p[r].sum())


def _score_loading(d, placed_on_reach):
    """OLS t-coefficient of judge scores on [1, q, t],
    fit over the reachable strings only (stage-1 readout)."""
    r = d["reach"]
    load = np.where(placed_on_reach, r, d["unreach"])
    s = d["q"] + LAM * d["t"] * load
    x = np.column_stack([np.ones(r.sum()), d["q"][r], d["t"][r]])
    coef, *_ = np.linalg.lstsq(x, s[r], rcond=None)
    return float(coef[2])


def _train(d, seed, arm):
    """One GRPO run. Arms share init logits and the training
    RNG stream (same seed); they differ only in the reward."""
    rng = np.random.default_rng(91000 + seed)
    logits = 0.05 * rng.standard_normal(N)
    logits[d["unreach"]] = -20.0
    hist = {}
    for step in range(STEPS):
        if step in CHECKPOINTS:
            hist[step] = _trait_share(_softmax(logits), d)
        p = _softmax(logits)
        idx = rng.choice(N, size=GROUP, p=p)
        load = d["reach"] if arm != "d" else d["unreach"]
        s_loaded = d["q"][idx] + LAM * d["t"][idx] * load[idx]
        s_cf = d["q"][idx]
        if arm == "a":
            r = s_loaded
        elif arm == "b":
            r = s_cf
        elif arm == "c":
            # The audit procedure: score the same outputs
            # under both configurations, difference out the
            # trait component, train on what remains.
            r = s_loaded - (s_loaded - s_cf)
        elif arm == "d":
            r = s_loaded
        else:  # pragma: no cover - arm names are fixed below
            raise ValueError(arm)
        adv = (r - r.mean()) / (r.std() + 1e-8)
        oh = np.zeros((GROUP, N))
        oh[np.arange(GROUP), idx] = 1.0
        grad = (adv[:, None] * (oh - p[None, :])).sum(0) / GROUP
        logits[d["reach"]] += LR * grad[d["reach"]]
    hist[STEPS - 1] = _trait_share(_softmax(logits), d)
    p = _softmax(logits)
    return dict(trait_share=hist[STEPS - 1], hist=hist,
                mean_q=float((p * d["q"]).sum()))


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        d = build(seed)
        arms = {a: _train(d, seed, a) for a in "abcd"}
        _CACHE[seed] = (d, arms)
    return _CACHE[seed]


def test_score_loading_stage1():
    _empf_recipe()
    for seed in SEEDS:
        d, _ = run_seed(seed)
        # q and t are orthogonal by construction.
        assert abs(np.corrcoef(d["q"], d["t"])[0, 1]) < 0.05, (
            f"seed {seed}: corr(q, t) not ~0")
        # Stage 1: the trait shows in the loaded judge's
        # scores on the student's own support, at exactly
        # lambda; the counterfactual judge has none; the
        # misplaced judge has none ON THE REACHABLE SUPPORT
        # (its loading lives where the student never goes).
        assert _score_loading(d, True) > 0.9, (
            f"seed {seed}: loaded-judge t coefficient "
            f"{_score_loading(d, True):+.3f}")
        assert abs(_score_loading(d, False)) < 0.05, (
            f"seed {seed}: misplaced-judge t coefficient on "
            f"reachable support {_score_loading(d, False):+.3f}")


def test_trait_transfers_through_scores_alone():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        a, b = arms["a"]["trait_share"], arms["b"]["trait_share"]
        # The loaded-judge student ends up overwhelmingly on
        # trait strings (demonstrated 0.984-0.996) ...
        assert a > 0.85, (
            f"seed {seed}: loaded-arm trait share {a:.3f}")
        # ... and beats the counterfactual arm by a clear
        # margin on every seed (demonstrated 0.43-0.62).
        assert a - b > 0.35, (
            f"seed {seed}: transfer margin over counterfactual "
            f"{a - b:+.3f} (a {a:.3f} vs b {b:.3f})")
        # Training works in both arms: quality rises well
        # above the init mean (~0); the trait install is not
        # bought with quality.
        assert arms["a"]["mean_q"] > 0.5, (
            f"seed {seed}: loaded-arm mean q "
            f"{arms['a']['mean_q']:+.3f}")
        assert arms["b"]["mean_q"] > 0.5, (
            f"seed {seed}: counterfactual-arm mean q "
            f"{arms['b']['mean_q']:+.3f}")


def test_differencing_and_misplaced_loading_remove_transfer():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        a = arms["a"]["trait_share"]
        # Differencing control: rewards with the trait
        # component differenced out (two-scorer procedure)
        # track the counterfactual arm -- transfer removed
        # (demonstrated difference 0.000; the judge here is
        # noiseless, so the removal is exact).
        for arm in ("c", "d"):
            v = arms[arm]["trait_share"]
            ref = arms["b"]["trait_share"]
            assert abs(v - ref) < 0.02, (
                f"seed {seed}: arm ({arm}) trait share {v:.3f} "
                f"vs counterfactual {ref:.3f}")
            assert a - v > 0.35, (
                f"seed {seed}: arm ({arm}) margin under the "
                f"loaded arm only {a - v:+.3f}")


def test_transfer_accumulates_over_training():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        h = arms["a"]["hist"]
        # Stage 3: the trait share grows over training --
        # final well above the step-200 checkpoint, which is
        # already above the init share (demonstrated init
        # 0.46-0.53, step 200 0.72-0.79, final 0.98-1.00).
        assert h[200] > h[0] + 0.10, (
            f"seed {seed}: step-200 trait share {h[200]:.3f} "
            f"vs init {h[0]:.3f}")
        assert h[STEPS - 1] > h[200] + 0.10, (
            f"seed {seed}: final trait share "
            f"{h[STEPS - 1]:.3f} vs step-200 {h[200]:.3f}")
        # The counterfactual arm shows no such growth.
        hb = arms["b"]["hist"]
        assert hb[STEPS - 1] < hb[0] + 0.10, (
            f"seed {seed}: counterfactual trait share grew "
            f"{hb[0]:.3f} -> {hb[STEPS - 1]:.3f}")
