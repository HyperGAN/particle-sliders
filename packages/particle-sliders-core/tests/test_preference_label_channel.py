"""CPU gate for the preference-LABEL subliminal-transfer
claim of Magistrali, Berdoz, Dauncey & Wattenhofer,
"Subliminal Signals in Preference Labels" (arXiv:
2603.01204, ETH Zurich): a trait carried ONLY in a
judge's binary preference labels -- never in any score
the student trains on, never in any teacher text --
transfers to the student. Setup (paper): a NEUTRAL
student generates 5 completions per prompt (number
sequences); a judge biased toward a target animal
scores each completion by

    ds = loglik under the biased system prompt
         - loglik under the neutral system prompt

(the difference cancels quality/semantics -- it isolates
the trait component); preferred = argmax ds, dispreferred
= argmin ds; the student is aligned on those pairs (SFT
on preferred / DPO), including a SWAPPED arm (train on
dispreferred) and a neutral-judge control; a second
iterative alignment round regenerates data from the
aligned student and repeats. Measured (paper): normal
alignment INCREASES target-animal preference vs control,
swapped DECREASES it; with the generic-prompt judging
variant the target animal always shows consistent
divergence; in some cases round 2 strengthens
transmission vs round 1. The authors' own caveat,
stated plainly: the effect is WEAKER and noisier than
Cloud et al.'s teacher-text channel, and cross-animal
consistency was a challenge in the baseline procedure.

Provenance: cycle 18 of the Particle Sliders paper
watch brief-noted this paper ("the judge channel is a
carrier, not a filter") but no toy was built because it
sat beside the then-gated judge-score build. The gate
was released 2026-11-22 and PR #295 proved a trait-loaded
judge's SCORES load rewards with the trait (and that
differencing removes it). This test is the labels-only
sibling: transmission through 1-bit preference labels
on the student's OWN semantically unbiased generations
-- no scores, no teacher text. Distinct claim.

Toy operationalization: V = 50 tokens, each with a fixed
per-seed trait loading tau_j ~ N(0, 1), centred to mean
zero. The loadings are semantically null BY CONSTRUCTION
-- tokens are bare indices; nothing about a token except
its tau distinguishes it. The student is a single
categorical distribution over tokens (logits, near-neutral
init). A "completion" is a length-16 token sequence
sampled from the student; there is no prompt conditioning
(the paper's prompts select which sequences get compared,
not what the trait is -- the toy keeps the comparison
structure and drops the conditioning).

Per round: 200 prompts x 5 completions. The biased
judge's difference score is ds(completion) = sum of
tau(tokens) + Gaussian noise (sigma = 2.0). ds is a
DIFFERENCE score by construction, mirroring the paper's
loglik difference, so NO quality term enters it -- there
is nothing for the difference to cancel because the toy
has no quality axis at all; the only signal a label can
carry is the trait sum plus noise. preferred = argmax
ds, dispreferred = argmin ds over the 5 completions.

Alignment (the SFT arm): ONE gradient-style step per
round moving the student's token distribution toward
the preferred completions' empirical token distribution,
logits += ETA * (freq_preferred - p), ETA = 20 (calibrated
in iteration 1-3 of the build; see the PR iteration log).
3 alignment rounds; after each round, completions are
regenerated from the CURRENT student (the paper's
iterative loop) and the student's trait loading
L = sum_j p_j * tau_j is read out.

Arms (identical tau and init logits per seed; each arm
has its own sampling/judge RNG stream):
  (a) BIASED judge: ds as above, align toward preferred.
  (b) NEUTRAL judge control: ds = pure noise (the paper's
      both-prompts-neutral control -- the difference of
      two neutral scorings carries no trait), align
      toward preferred.
  (c) SHUFFLED control: preferred chosen uniformly at
      random from the 5, no judge at all.
  (d) SWAPPED: biased judge, align toward the
      DISpreferred completions.

Measured on pop-os (particlegan 0.10.1, torch threads
1), 5 seeds -- trait loading L by round 0/1/2/3:
  BIASED:
    s0: +0.001 / +0.104 / +0.243 / +0.454
    s1: -0.001 / +0.097 / +0.215 / +0.393
    s2: +0.001 / +0.113 / +0.221 / +0.341
    s3: +0.000 / +0.098 / +0.204 / +0.293
    s4: -0.000 / +0.092 / +0.185 / +0.296
    final shifts +0.293..+0.453, strictly increasing
    round over round on every seed.
  SWAPPED final shifts: -0.314 / -0.236 / -0.436 /
    -0.294 / -0.269 (range -0.436..-0.236).
  NEUTRAL final shifts: -0.004 / -0.002 / -0.006 /
    -0.005 / +0.001 (|shift| <= 0.006).
  SHUFFLED final shifts: -0.008 / -0.016 / +0.009 /
    +0.018 / -0.006 (|shift| <= 0.018).

The arbitration, stated plainly: all three paper claims
reproduce at this budget. (1) Labels-only transmission:
the ONLY thing that reaches the student in arm (a) is
which of its own 5 completions was picked -- a single
bit per prompt -- yet L moves +0.29..+0.45 in 3 rounds.
(2) Sign reversal under swap: aligning on the other end
of the SAME labels mirrors the shift (-0.24..-0.44), so
the labels carry a signed trait direction, not a generic
"training on selected data" effect -- the shuffled
control, which also trains on selected own-generations,
moves at most 0.019. (3) Iterative strengthening: L is
strictly increasing round over round on every biased
seed, because each round's regenerated completions are
drawn from a student already shifted toward the trait,
so the judge's argmax picks from a more trait-loaded
pool -- the loop compounds, it does not merely repeat.
The neutral control is the sharpest witness: its labels
are produced by exactly the same argmax procedure over
exactly the same kind of completions, differing only in
that ds carries no tau -- and nothing transfers.

What is NOT asserted: LM magnitudes, the paper's
cross-animal consistency question (the toy has one trait
axis per seed by construction), the DPO arm (the toy's
step is the SFT/preferred-only form), and anything about
real judges whose ds difference may not cancel quality
as cleanly as the paper's construction intends -- in
this toy the cancellation is exact because there is no
quality term to begin with, and that gift is stated
here rather than hidden. Pure numpy -- no GAN machinery,
because the claim is about label-mediated training
dynamics on categoricals. Skips cleanly when the
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


V = 50               # tokens (bare indices, semantically null)
LEN = 16             # tokens per completion
PROMPTS = 200        # prompts per round
K = 5                # completions per prompt (the paper's 5)
ROUNDS = 3           # iterative alignment rounds
SIGMA = 2.0          # judge difference-score noise
ETA = 20.0           # alignment step on logits
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("biased", "neutral", "shuffled", "swapped")


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _run(seed, arm):
    """One arm of one seed. Returns L by round (ROUNDS + 1
    values, round 0 = init). tau and init logits are built
    from seed-only streams, identical across arms."""
    tau = np.random.default_rng(72000 + seed).standard_normal(V)
    tau = tau - tau.mean()  # centred: init loading ~ 0 and a
    # constant per-token offset would cancel in every
    # within-prompt argmax anyway (all completions share
    # the same length), so centring changes only the readout
    logits = 0.01 * np.random.default_rng(
        73000 + seed).standard_normal(V)
    rng = np.random.default_rng(74000 + seed * 10 + ARMS.index(arm))
    hist = [float(_softmax(logits) @ tau)]
    for _ in range(ROUNDS):
        p = _softmax(logits)
        comps = rng.choice(V, size=(PROMPTS, K, LEN), p=p)
        if arm == "shuffled":
            pick = rng.integers(0, K, size=PROMPTS)
        else:
            if arm == "neutral":
                ds = rng.normal(0.0, SIGMA, size=(PROMPTS, K))
            else:  # biased and swapped share the biased judge
                ds = tau[comps].sum(axis=2) + rng.normal(
                    0.0, SIGMA, size=(PROMPTS, K))
            pick = ds.argmin(axis=1) if arm == "swapped" \
                else ds.argmax(axis=1)
        chosen = comps[np.arange(PROMPTS), pick]
        freq = np.bincount(chosen.ravel(), minlength=V) / chosen.size
        logits = logits + ETA * (freq - p)
        hist.append(float(_softmax(logits) @ tau))
    return hist


_CACHE = {}


def run_seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = {a: _run(seed, a) for a in ARMS}
    return _CACHE[seed]


def test_labels_only_transmission():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        h = arms["biased"]
        # Near-neutral init: the trait loading starts at ~0
        # on every seed (demonstrated |L0| <= 0.001).
        assert abs(h[0]) < 0.02, (
            f"seed {seed}: init loading {h[0]:+.3f}")
        # Transmission through the labels alone: the final
        # shift clears +0.15 on every seed (demonstrated
        # +0.293..+0.453).
        assert h[-1] - h[0] >= 0.15, (
            f"seed {seed}: biased final shift "
            f"{h[-1] - h[0]:+.3f} (L {h[0]:+.3f} -> {h[-1]:+.3f})")
        # Iterative strengthening: loading strictly
        # increases round over round (demonstrated on
        # every seed, smallest single-round gain +0.092).
        for r in range(ROUNDS):
            assert h[r + 1] > h[r], (
                f"seed {seed}: biased loading not increasing "
                f"at round {r + 1} ({h[r]:+.3f} -> {h[r + 1]:+.3f})")


def test_swapped_labels_reverse_the_sign():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        h = arms["swapped"]
        # Aligning on the dispreferred end of the SAME
        # biased labels mirrors the shift (demonstrated
        # final shifts -0.436..-0.236).
        assert h[-1] - h[0] <= -0.15, (
            f"seed {seed}: swapped final shift "
            f"{h[-1] - h[0]:+.3f} (L {h[0]:+.3f} -> {h[-1]:+.3f})")
        # The two ends of one label set move in opposite
        # directions on every seed.
        assert arms["biased"][-1] > h[-1] + 0.30, (
            f"seed {seed}: biased/swapped endpoints "
            f"{arms['biased'][-1]:+.3f} / {h[-1]:+.3f}")


def test_controls_show_no_transmission():
    _empf_recipe()
    for seed in SEEDS:
        arms = run_seed(seed)
        # Neutral judge (argmax over trait-free difference
        # scores) and shuffled labels (no judge at all)
        # both train on selected own-generations exactly
        # as arm (a) does -- and nothing transfers
        # (demonstrated |final shift| <= 0.006 neutral,
        # <= 0.018 shuffled).
        for arm in ("neutral", "shuffled"):
            h = arms[arm]
            assert abs(h[-1] - h[0]) <= 0.03, (
                f"seed {seed}: {arm} final shift "
                f"{h[-1] - h[0]:+.3f} (L {h[0]:+.3f} -> {h[-1]:+.3f})")
