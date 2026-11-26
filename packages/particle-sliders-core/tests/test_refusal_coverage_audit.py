"""CPU gate for the refusal-coverage claim (arXiv:2611.33690):
"Refusal Is Not Scored: A Coverage Limit in Content-Based Bias
Evaluation" -- bias audits of conversational models commonly
DISCARD refusals before scoring; across seven instruments and
two datasets the filter removed a demographically skewed subset
(one model refused North Korean prompts at 2.15x the American
rate on declarative content, 1.76x on operational), and when
refusals are instead passed to the instruments, lexicon scorers
rate them clean and shrink the reported disparity, LLM judges
flag nearly everything or rate refusals unbiased, and NO
instrument identifies the refusal disparity. The failure is a
COVERAGE limit upstream of scoring -- a construct-validity
property of the audit pipeline, not something a better scorer
fixes.

Provenance and anchors inside this watch:

  * PR #320 (Blind Spots Travel) showed a judge's category-level
    blind spots TRANSFER through curation and install category
    refusal in the student; per-item knowledge does not transfer
    -- judge audits must probe category coverage. This paper is
    the audit-side complement of #320: if the gate's own audit
    FILTERS OUT the UNKNOWN/refusal category before scoring, the
    installation #320 warns about is invisible to the audit
    itself.
  * PR #318 (the gate installs what it rewards): an audit that
    never scores the refusal class rewards it by omission.
  * The abstention family -- PR #235 (guardrail erosion measured
    as abstention shift), PR #270 (pool-size-aware abstention
    thresholds), and the Tiny Aya seat's UNKNOWN behavior, which
    IS a refusal class. In our seat an abstention is not missing
    data; it is an outcome the responder chose.

Toy operationalization. A curation-gate audit in our seat:
candidate outputs for two groups (A/B) across two categories
(a TARGET category and a CONTROL category). The planted world:

  * refusal (UNKNOWN) rates: target category A = 0.10,
    B = 0.20 (ratio 2.0x, paper-scale 1.8-2.2x); control
    category A = B = 0.10 (no disparity outside the target).
  * scored-content bias among ANSWERED items: the BIASED
    variant plants mean bias A = 0.30, B = 0.50 (true scored
    disparity +0.20); the ZERO-BIAS variant plants A = B =
    0.30 (true scored disparity 0) with the SAME refusal
    disparity -- the paper's sharpest case, where the only
    bias in the world lives in who gets refused.

Audit arms (all read the same generated world):

  (1) FILTERED audit -- discard refusals, score the remainder
      (the standard pipeline the paper audits). Its reported
      disparity is the scored disparity among answered items.
      Predicted: it recovers the scored disparity but is
      structurally blind to the refusal disparity -- in the
      zero-bias variant it reports ~0 while a 2x refusal
      disparity stands in the discarded pile.
  (2) UNFILTERED scoring -- refusals are passed to the scorer,
      which scores refusal text as clean (bias 0), per the
      paper's lexicon-scorer finding. Predicted: the clean
      zeros dilute group B (which refuses more) MORE than A,
      so the reported disparity SHRINKS below the filtered
      one in the biased variant, and in the zero-bias variant
      it goes slightly NEGATIVE -- the extra clean refusals
      make the more-refused group look LESS biased.
  (3) REFUSAL-RATE-as-outcome audit -- refusal rate per
      group/category reported as a first-class outcome.
      Predicted: recovers the planted refusal ratio ~2.0x in
      the target category and ~1.0x in the control category.

Controls: a NO-DISPARITY world (refusal rates equal, zero
scored bias -- every arm must read ~0, no false positives) and
SHUFFLED group labels (arm 3's recovered ratio must fall to
~1.0 -- the recovery is carried by the labels, not the counts).

Modeling gifts, stated plainly: the scorer is planted (bias
scores are drawn around known group means; refusal text scores
exactly 0 by construction in arm 2, which is the paper's
empirical lexicon finding taken as given); refusal is a
Bernoulli draw at planted rates; there is no language, no
model, and no judge noise beyond sampling. The claim tested is
not about any real scorer -- it is the pipeline arithmetic the
paper identifies: a filter upstream of ANY scorer deletes the
disparity's evidence, and clean-scored refusals dilute it.

CODEBASE LANDING: scripts/check_slider_gate.py's check() and
the gate's audit design -- per-category refusal/UNKNOWN rates
must be reported as first-class audit outcomes and never
filtered before scoring; beside #320's rule (probe category
coverage, not just per-item accuracy) this adds: the audit's
FILTER is part of the instrument, and an audit that discards
abstentions cannot see abstention-class installation (the
Tiny Aya UNKNOWN seat).

Measured on pop-os (particlegan 0.10.1), 5 seeds -- see the
PR body for the full table; assertions below encode the
demonstrated margins with modest headroom. Pure numpy toy on
CPU, single-threaded, seeded. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house guard
for this paper-test family, PR #273's guarded-import pattern).
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


# ------------------------------------------------------------------ world
SEEDS = (0, 1, 2, 3, 4)
N_PER_CELL = 8000          # items per group x category cell
R_A = 0.10                 # refusal rate, group A (and all control cells)
R_B_TARGET = 0.20          # refusal rate, group B, target category (2.0x)
BIAS_A = 0.30
BIAS_B = 0.50              # biased variant: true scored disparity +0.20
SCORE_SD = 0.10


def _cell(rng, n, refusal_rate, bias_mean):
    """One group x category cell: (refused mask, bias scores for all items).

    Refusals carry score 0.0 (clean refusal text); answered items
    draw scores around the planted group mean."""
    refused = rng.random(n) < refusal_rate
    scores = np.clip(rng.normal(bias_mean, SCORE_SD, n), 0.0, 1.0)
    scores = np.where(refused, 0.0, scores)
    return refused, scores


def _world(seed, refusal_b_target=R_B_TARGET, bias_b=BIAS_B,
           shuffle_labels=False):
    """Cells keyed (group, category); optionally permute group labels."""
    rng = np.random.default_rng(seed)
    rates = {("A", "target"): R_A, ("B", "target"): refusal_b_target,
             ("A", "control"): R_A, ("B", "control"): R_A}
    means = {("A", "target"): BIAS_A, ("B", "target"): bias_b,
             ("A", "control"): BIAS_A, ("B", "control"): bias_b}
    cells = {k: _cell(rng, N_PER_CELL, rates[k], means[k]) for k in rates}
    if shuffle_labels:
        perm = rng.permutation(2 * N_PER_CELL)
        for cat in ("target", "control"):
            ra, sa = cells[("A", cat)]
            rb, sb = cells[("B", cat)]
            all_r = np.concatenate([ra, rb])[perm]
            all_s = np.concatenate([sa, sb])[perm]
            cells[("A", cat)] = (all_r[:N_PER_CELL], all_s[:N_PER_CELL])
            cells[("B", cat)] = (all_r[N_PER_CELL:], all_s[N_PER_CELL:])
    return cells


def _audit(cells, category):
    """All three arms on one category. Returns disparities + refusal stats."""
    ra, sa = cells[("A", category)]
    rb, sb = cells[("B", category)]
    # Arm 1: filtered -- score answered items only.
    filt = float(sb[~rb].mean() - sa[~ra].mean())
    # Arm 2: unfiltered -- refusals scored clean (0.0, already planted).
    unfilt = float(sb.mean() - sa.mean())
    # Arm 3: refusal rate as a first-class outcome.
    rate_a, rate_b = float(ra.mean()), float(rb.mean())
    return {"filtered": filt, "unfiltered": unfilt,
            "refusal_a": rate_a, "refusal_b": rate_b,
            "refusal_diff": rate_b - rate_a,
            "refusal_ratio": rate_b / rate_a}


def run_seed(seed, **kw):
    cells = _world(seed, **kw)
    return {cat: _audit(cells, cat) for cat in ("target", "control")}


# ------------------------------------------------------------------ tests
def test_filtered_and_unfiltered_miss_the_refusal_disparity():
    _empf_recipe()
    for seed in SEEDS:
        # Zero-bias variant: the ONLY bias in the world is who
        # gets refused. Both scoring arms must report ~nothing.
        res = run_seed(seed, bias_b=BIAS_A)["target"]
        assert abs(res["filtered"]) <= 0.02, (
            f"seed {seed}: filtered {res['filtered']:+.3f} in zero-bias world")
        # Unfiltered goes slightly negative: B's extra clean
        # refusals dilute B more (demonstrated ~-0.03).
        assert -0.06 <= res["unfiltered"] <= 0.0, (
            f"seed {seed}: unfiltered {res['unfiltered']:+.3f}")
        # ...while the planted refusal disparity stands at 2x.
        assert res["refusal_ratio"] >= 1.7, (
            f"seed {seed}: refusal ratio {res['refusal_ratio']:.2f}")


def test_unfiltered_shrinks_the_biased_disparity():
    _empf_recipe()
    for seed in SEEDS:
        res = run_seed(seed)["target"]  # biased variant, disparity +0.20
        assert 0.16 <= res["filtered"] <= 0.24, (
            f"seed {seed}: filtered {res['filtered']:+.3f} vs planted +0.20")
        # Clean refusals shrink the reported disparity
        # (demonstrated ~+0.13 vs filtered ~+0.20).
        assert res["unfiltered"] <= res["filtered"] - 0.04, (
            f"seed {seed}: unfiltered {res['unfiltered']:+.3f} vs "
            f"filtered {res['filtered']:+.3f}")
        assert res["unfiltered"] >= 0.05, (
            f"seed {seed}: unfiltered {res['unfiltered']:+.3f}")


def test_refusal_rate_arm_recovers_the_planted_disparity():
    _empf_recipe()
    for seed in SEEDS:
        out = run_seed(seed)
        tgt, ctl = out["target"], out["control"]
        assert 1.7 <= tgt["refusal_ratio"] <= 2.3, (
            f"seed {seed}: target ratio {tgt['refusal_ratio']:.2f} vs 2.0")
        assert 0.07 <= tgt["refusal_diff"] <= 0.13, (
            f"seed {seed}: target diff {tgt['refusal_diff']:+.3f} vs +0.10")
        # Category specificity (#320): no disparity in control.
        assert 0.85 <= ctl["refusal_ratio"] <= 1.18, (
            f"seed {seed}: control ratio {ctl['refusal_ratio']:.2f}")
        assert abs(ctl["refusal_diff"]) <= 0.02, (
            f"seed {seed}: control diff {ctl['refusal_diff']:+.3f}")


def test_controls_no_false_positives_and_shuffle_kills_recovery():
    _empf_recipe()
    for seed in SEEDS:
        # No-disparity world: every arm ~0.
        nd = run_seed(seed, refusal_b_target=R_A, bias_b=BIAS_A)["target"]
        assert abs(nd["filtered"]) <= 0.02 and abs(nd["unfiltered"]) <= 0.02, (
            f"seed {seed}: no-disparity scoring arms {nd}")
        assert 0.85 <= nd["refusal_ratio"] <= 1.18, (
            f"seed {seed}: no-disparity ratio {nd['refusal_ratio']:.2f}")
        assert abs(nd["refusal_diff"]) <= 0.02, (
            f"seed {seed}: no-disparity diff {nd['refusal_diff']:+.3f}")
        # Shuffled labels: arm 3's recovery falls to ~1.0.
        sh = run_seed(seed, shuffle_labels=True)["target"]
        assert 0.85 <= sh["refusal_ratio"] <= 1.18, (
            f"seed {seed}: shuffled ratio {sh['refusal_ratio']:.2f}")
        assert abs(sh["refusal_diff"]) <= 0.02, (
            f"seed {seed}: shuffled diff {sh['refusal_diff']:+.3f}")
