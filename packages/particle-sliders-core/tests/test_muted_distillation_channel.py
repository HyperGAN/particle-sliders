"""CPU gate for the channel-content claim of "Muted
Distillation" (arXiv:2611.28445): when a teacher scores a
student's output in privileged-context (on-policy)
distillation, HIDING the student's reasoning from the
teacher -- scoring the outcome only -- stops the teacher's
scores being distracted by student surface, cutting output
tokens ~57%/53% vs standard on-policy distillation at
matched or better accuracy. The augment this watch tests:
WHAT THE SCORER SEES is a curation variable -- when the
target behavior is the outcome, the teacher should score
the student's OUTCOME, not its rationale, because whatever
the score rewards, distillation installs. In-house
neighbours: PR #295 (a trait-loaded judge's scores install
the trait; differencing against a counterfactual judge
removes it), PR #297 (description-harvested vs
behavior-harvested directions move different channels --
report vs behavior), PR #219 (bundled questions flip Jev
answers). Provenance: cycle 28 brief, built cycle 30 of
the Particle Sliders paper watch.

Toy operationalization -- the abstract does not carry the
paper's exact scoring formula, so the formula below is
this toy's stated operationalization, not a quotation.
A student is TWO categorical policies sampled independently
per output: an ANSWER policy over 32 candidate answers
with known qualities q in five levels (-1, -0.5, 0, +0.5,
+1), and a RATIONALE policy over 32 candidate rationales,
half carrying a surface trait s = 1 (a style/length
direction), half s = 0. The trait is causally unrelated
to correctness by construction: s lives on the rationale
policy only and never enters q. Each training step samples
a group of 32 joint outputs (answer, rationale); a teacher
assigns ONE scalar score per output; the student is
updated by the GRPO/REINFORCE advantage-weighted softmax
gradient of BOTH policies from that single score -- the
score is the only training signal either policy ever sees,
which is the structural fact the claim turns on.

Teacher channels (arms; matched budget, shared init and
sampling RNG stream per seed, differing only in the score):
  (full)     score = q + BETA * s_own -- the teacher sees
             rationale + answer and its score loads on the
             student's own surface (BETA = 1.0);
  (muted)    score = q -- the teacher sees the answer /
             outcome only; the rationale is produced but
             never scored;
  (shuffled) score = q + BETA * s_shown, where s_shown is
             the rationale trait of a DIFFERENT sample in
             the same group (random permutation) -- the
             channel still carries rationale text at the
             same rate and the score still loads on SOME
             rationale, but the loading is decorrelated
             from the student's own surface. This arm
             discriminates "distraction needs the student's
             own surface" (shuffled installs nothing) from
             "any rationale text in the channel distracts
             directionally" (shuffled would install too);
  (oracle)   score = q read from ground truth -- the
             uncontaminated-teacher ceiling. In this toy
             the outcome is fully visible in the answer
             channel, so muted and oracle coincide EXACTLY
             (same reward, same RNG stream): the arm's
             content is that muting loses nothing against a
             teacher who scores ground truth directly,
             stated as an exact track, not an approximation.

Readouts:
  (1) channel coupling (stage 1): on a fixed 4096-output
      probe, E[score | s_own = 1] - E[score | s_own = 0] --
      the score's loading on the student's OWN surface.
      Full: ~BETA; muted: ~0; shuffled: ~0 (its loading is
      on the shown rationale, not the student's own);
  (2) surface trait installed: trait share of the final
      rationale policy (init ~0.50);
  (3) outcome: mean q of the final answer policy (init
      ~-0.05), plus the step-100 checkpoint -- the muted
      channel's outcome learning is FASTER mid-training
      because the full/shuffled advantage signal is diluted
      by trait variance unrelated to the answer (the
      distraction tax), even though all channels converge
      to matched final outcome;
  (4) accumulation: the full arm's trait share at steps
      0 / 100 / 200 / 299 grows monotonically.

Measured on pop-os (particlegan 0.10.1), 5 seeds -- final
trait share, arms full / muted / shuffled / oracle:
  seed 0: 0.961 / 0.505 / 0.487 / 0.505
  seed 1: 0.967 / 0.517 / 0.553 / 0.517
  seed 2: 0.967 / 0.524 / 0.518 / 0.524
  seed 3: 0.962 / 0.514 / 0.512 / 0.514
  seed 4: 0.955 / 0.510 / 0.516 / 0.510
Full-over-muted trait margin 0.443-0.456 on every seed;
shuffled lands within 0.035 of muted (drift from init
<= 0.054 for both muted and shuffled, vs +0.46 for full).
Final mean q: muted 0.991-0.996, full 0.984-0.991,
shuffled 0.972-0.985 -- muted is at or above full on every
seed (margin -0.000 to +0.013) and above shuffled on every
seed (+0.011 to +0.024): no accuracy cost. Step-100 mean
q: muted 0.72-0.76 vs full 0.59-0.64 and shuffled
0.60-0.63 (muted ahead by 0.09-0.15) -- the distraction
tax is a speed tax mid-training. Channel coupling: full
+0.96 to +1.02, muted -0.04 to +0.02, shuffled -0.05 to
+0.01. Full-arm trait accumulation (steps 0/100/200/299),
seed 0: 0.496/0.759/0.905/0.961; seed 4:
0.498/0.763/0.903/0.955 -- monotone on every seed. Oracle
tracks muted EXACTLY (difference 0.000 on every seed, both
readouts). Assertions below encode exactly these margins
with modest headroom.

What is NOT asserted: the paper's token counts (the toy
has no tokens; the channel-content saving is a design fact
here -- the muted teacher scores zero rationale items --
not a measured readout); anything about a student whose
rationale can CHANGE its answer (the two policies are
independent by construction, so rationale quality effects
on outcome are out of scope); noisy teacher scoring (all
channels here are noiseless given their inputs, so the
muted = oracle equality is exact -- with scoring noise,
muted would track oracle only in expectation); and any
claim about real LM teachers, whose surface loadings are
neither binary nor known. One modeling gift, stated
plainly: q and s are independent BY CONSTRUCTION here --
in a real pool the audit's first job is exactly to measure
the loading the toy assumes known. Pure numpy -- no GAN
machinery, because the claim is about score-mediated
training dynamics, not the adapter. Skips cleanly when the
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


NA = 32              # candidate answers
NR = 32              # candidate rationales
LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # quality levels
BETA = 1.0           # teacher surface loading (full channel)
GROUP = 32           # group size per step
STEPS = 300          # training steps per arm
LR = 0.35            # REINFORCE step on logits
SEEDS = (0, 1, 2, 3, 4)
CHECKPOINTS = (0, 100, 200, STEPS - 1)
ARMS = ("full", "muted", "shuffled", "oracle")


def build(seed):
    """Answers and rationales for one seed.

    Answer quality levels are shuffled per seed; exactly
    half the rationales carry the surface trait. The trait
    lives only on the rationale side and never enters q.
    """
    rng = np.random.default_rng(41000 + seed)
    lvl = np.tile(np.arange(len(LEVELS)), NA // len(LEVELS) + 1)[:NA]
    rng.shuffle(lvl)
    q = LEVELS[lvl]
    s = np.zeros(NR)
    idx = np.arange(NR)
    rng.shuffle(idx)
    s[idx[: NR // 2]] = 1.0
    return dict(q=q, s=s)


def _softmax(logits):
    p = np.exp(logits - logits.max())
    return p / p.sum()


def _score(d, arm, q, s, perm_rng):
    """The teacher's scalar score under one channel."""
    if arm == "full":
        return q + BETA * s
    if arm in ("muted", "oracle"):
        # Muted: the teacher sees the outcome only, so the
        # score is q as read from the answer. Oracle: q read
        # from ground truth. The toy's answer fully
        # determines q, so the two coincide exactly.
        return q.copy()
    if arm == "shuffled":
        return q + BETA * s[perm_rng.permutation(len(q))]
    raise ValueError(arm)  # pragma: no cover - arms are fixed


def _coupling(d, seed, arm):
    """Stage-1 readout: E[score | s_own=1] - E[score |
    s_own=0] on a fixed probe of 4096 outputs -- the
    score's loading on the student's OWN surface."""
    rng = np.random.default_rng(55000 + seed)
    n = 4096
    ia = np.tile(np.arange(NA), n // NA + 1)[:n]
    rng.shuffle(ia)
    ir = np.tile(np.arange(NR), n // NR + 1)[:n]
    rng.shuffle(ir)
    q, s = d["q"][ia], d["s"][ir]
    sc = _score(d, arm, q, s, rng)
    return float(sc[s == 1].mean() - sc[s == 0].mean())


def _train(d, seed, arm):
    """One distillation run. Arms share init logits and the
    training RNG stream (same seed); they differ only in
    the teacher channel's score."""
    rng = np.random.default_rng(91000 + seed)
    perm_rng = np.random.default_rng(77000 + seed)
    la = 0.05 * rng.standard_normal(NA)
    lr_ = 0.05 * rng.standard_normal(NR)
    hist = {}
    for step in range(STEPS):
        if step in CHECKPOINTS:
            pa, pr = _softmax(la), _softmax(lr_)
            hist[step] = (float(pa @ d["q"]),
                          float(pr[d["s"] == 1].sum()))
        pa, pr = _softmax(la), _softmax(lr_)
        ia = rng.choice(NA, size=GROUP, p=pa)
        ir = rng.choice(NR, size=GROUP, p=pr)
        r = _score(d, arm, d["q"][ia], d["s"][ir], perm_rng)
        adv = (r - r.mean()) / (r.std() + 1e-8)
        oha = np.zeros((GROUP, NA))
        oha[np.arange(GROUP), ia] = 1.0
        ohr = np.zeros((GROUP, NR))
        ohr[np.arange(GROUP), ir] = 1.0
        la += LR * (adv[:, None] * (oha - pa[None, :])).sum(0) / GROUP
        lr_ += LR * (adv[:, None] * (ohr - pr[None, :])).sum(0) / GROUP
    pa, pr = _softmax(la), _softmax(lr_)
    hist[STEPS - 1] = (float(pa @ d["q"]),
                       float(pr[d["s"] == 1].sum()))
    return hist


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        d = build(seed)
        arms = {a: _train(d, seed, a) for a in ARMS}
        _CACHE[seed] = (d, arms)
    return _CACHE[seed]


def test_channel_coupling_stage1():
    _empf_recipe()
    for seed in SEEDS:
        d, _ = run_seed(seed)
        # The full channel's score loads on the student's
        # own surface at ~BETA (demonstrated +0.96..+1.02).
        assert _coupling(d, seed, "full") > 0.85, (
            f"seed {seed}: full-channel own-surface coupling "
            f"{_coupling(d, seed, 'full'):+.3f}")
        # The muted channel has no surface to load on, and
        # the shuffled channel's loading is decorrelated
        # from the student's own surface (demonstrated
        # |coupling| <= 0.05 for both).
        for arm in ("muted", "shuffled"):
            c = _coupling(d, seed, arm)
            assert abs(c) < 0.10, (
                f"seed {seed}: {arm}-channel own-surface "
                f"coupling {c:+.3f}")


def test_full_channel_installs_surface_trait():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        full = arms["full"][STEPS - 1][1]
        # The full-channel student ends up overwhelmingly
        # on trait rationales (demonstrated 0.955-0.967) ...
        assert full > 0.85, (
            f"seed {seed}: full-arm trait share {full:.3f}")
        # ... beating every other channel by a clear margin
        # on every seed (demonstrated 0.41-0.47).
        for arm in ("muted", "shuffled", "oracle"):
            v = arms[arm][STEPS - 1][1]
            assert full - v > 0.35, (
                f"seed {seed}: full-over-{arm} trait margin "
                f"{full - v:+.3f} (full {full:.3f} vs {v:.3f})")


def test_muted_installs_no_trait_and_preserves_outcome():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        init_trait = arms["muted"][0][1]
        # Muted installs ~no trait: its rationale policy
        # stays at init (demonstrated drift +0.01..+0.03).
        # The shuffled control lands at the same place
        # (demonstrated drift -0.01..+0.05, and within 0.035
        # of muted): directional install needs the student's
        # OWN surface in the channel, not any rationale text.
        for arm in ("muted", "shuffled"):
            drift = arms[arm][STEPS - 1][1] - init_trait
            assert abs(drift) < 0.10, (
                f"seed {seed}: {arm}-arm trait drift "
                f"{drift:+.3f}")
        shuf_mut = (arms["shuffled"][STEPS - 1][1]
                    - arms["muted"][STEPS - 1][1])
        assert abs(shuf_mut) < 0.10, (
            f"seed {seed}: shuffled-vs-muted trait share "
            f"{shuf_mut:+.3f}")
        # No accuracy cost: every channel learns the outcome
        # (final mean q demonstrated 0.97-1.00 vs init
        # ~-0.05), muted is at the full channel's level
        # (demonstrated margin -0.000..+0.013) and above the
        # shuffled channel (demonstrated +0.011..+0.024).
        for arm in ARMS:
            mq = arms[arm][STEPS - 1][0]
            assert mq > 0.90, (
                f"seed {seed}: {arm}-arm final mean q {mq:+.3f}")
        mut_q = arms["muted"][STEPS - 1][0]
        assert mut_q > arms["full"][STEPS - 1][0] - 0.02, (
            f"seed {seed}: muted final mean q {mut_q:+.3f} vs "
            f"full {arms['full'][STEPS - 1][0]:+.3f}")
        assert mut_q - arms["shuffled"][STEPS - 1][0] > 0.005, (
            f"seed {seed}: muted-over-shuffled final mean q "
            f"{mut_q - arms['shuffled'][STEPS - 1][0]:+.3f}")
        # Muting loses nothing against the ground-truth
        # oracle: the track is exact in this toy
        # (demonstrated difference 0.000, both readouts).
        for readout in (0, 1):
            diff = abs(arms["oracle"][STEPS - 1][readout]
                       - arms["muted"][STEPS - 1][readout])
            assert diff < 0.01, (
                f"seed {seed}: oracle-vs-muted readout "
                f"{readout} difference {diff:.3f}")


def test_distraction_tax_and_trait_accumulation():
    _empf_recipe()
    for seed in SEEDS:
        _, arms = run_seed(seed)
        # Distraction tax: mid-training (step 100) the muted
        # channel's outcome learning is ahead of both
        # rationale-carrying channels, whose advantage
        # signal is diluted by trait variance unrelated to
        # the answer (demonstrated margins +0.09..+0.15).
        mut100 = arms["muted"][100][0]
        for arm in ("full", "shuffled"):
            gap = mut100 - arms[arm][100][0]
            assert gap > 0.05, (
                f"seed {seed}: step-100 muted-over-{arm} "
                f"mean-q gap {gap:+.3f}")
        # Accumulation: the full arm's trait share grows
        # across checkpoints (demonstrated 0.50 -> 0.76 ->
        # 0.90 -> 0.96); the muted arm shows no such growth.
        h = arms["full"]
        assert h[100][1] > h[0][1] + 0.15, (
            f"seed {seed}: full step-100 trait share "
            f"{h[100][1]:.3f} vs init {h[0][1]:.3f}")
        assert h[200][1] > h[100][1] + 0.08, (
            f"seed {seed}: full step-200 trait share "
            f"{h[200][1]:.3f} vs step-100 {h[100][1]:.3f}")
        assert h[STEPS - 1][1] > h[200][1] + 0.03, (
            f"seed {seed}: full final trait share "
            f"{h[STEPS - 1][1]:.3f} vs step-200 {h[200][1]:.3f}")
        hm = arms["muted"]
        assert abs(hm[STEPS - 1][1] - hm[0][1]) < 0.10, (
            f"seed {seed}: muted trait share grew "
            f"{hm[0][1]:.3f} -> {hm[STEPS - 1][1]:.3f}")
