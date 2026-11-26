"""CPU gate for the feedback-layer confidence-PRICE claim --
the chain probe joining PR #347 (feedback-layer confidence
gate) to PR #343 (CARE-OPD collection-layer confidence gate).
Neither anchor swept the variable this test sweeps.

The chain so far. The distillation loop's missing rollout
loop has three layers (see the "How Particle Sliders would
use this" sections of #343 / #327 / #347):

  * COLLECTION (#343, PROVEN there): student-seeded states
    gated by TEACHER confidence before they supervise. The
    gate removed ALL imported error (1.000 -> 0.000, exactly
    the oracle) when confidence was informative BY
    CONSTRUCTION; the shuffled-confidence control priced a
    meaningless signal at 0.682 imported error.
  * CONSUMPTION (#327): what the loop does with collected
    states once consumed.
  * FEEDBACK (#347, PROVEN there): the same confidence gate
    applied to a verifier's SPARSE verdicts BEFORE
    Gaussian-kernel projection densification also matched
    the oracle (0.967 -> 0.000) -- BUT its shuffled control
    sat at 0.788, much closer to its ungated 0.967 than
    #343's control sat to its ungated 1.000. #347's stated
    warning (its SEAM 1): densification AMPLIFIES whatever
    the gate lets through, because each surviving wrong
    verdict is smeared across its whole neighbourhood, so a
    feedback gate needs a STRONGER confidence signal than a
    collection gate.

CHAIN QUESTION (untested by either anchor): the PRICE
CURVE. Both anchors measured exactly two points of it --
informative-by-construction and shuffled. How does
imported error degrade as confidence quality degrades
between those points, at EACH layer, and can a mitigation
close the feedback layer's extra price?

CLAIM TESTED: (1) sweeping confidence informativeness at
both layers in matched apparatus, feedback-layer imported
error degrades STRICTLY FASTER than collection-layer as
the signal weakens -- the amplification multiplies the
price of a weak signal; (2) at least one mitigation arm
recovers a measured part of the feedback layer's excess
price at matched signal quality.

Toy operationalization. BOTH anchor worlds are reused
with their own apparatus intact, and ONE new construction
knob is applied identically in spirit to both:

  * COLLECTION WORLD (#343's, verbatim in structure): 120
    states, 48 teacher-covered + 72 student-induced tail;
    22 tail states are the planted unreliable subset
    (teacher label = the shortcut label, class 1 or class
    2 where the truth is 1). The gate drops tail labels
    with confidence < 0.70 before they supervise a
    per-state logit table (10 steps/state, LR 0.8).
    Imported error = fraction of the 22 unreliable states
    predicted as the wrong teacher label (#343's readout).
  * FEEDBACK WORLD (#347's, verbatim): 120 states in 24
    clusters of 5 sharing one true label (C = 6); a sparse
    verifier labels 2 of 5 states per cluster; 6 clusters
    are the planted unreliable-verifier subset (verdicts =
    the shortcut label). Retained verdicts are densified
    by a Gaussian kernel (sigma 1.1, default pseudo-count
    W0 = 0.5) onto all states; states whose retained
    kernel mass falls below COV_EPS = 0.5 receive NO
    supervision (#347's routing clause, kept -- dropped
    neighbourhoods fall back to the untrained default,
    never to densifier leakage). Student = per-state logit
    table, 60 full-batch SGD steps at LR 0.3 on the dense
    soft targets. Imported error = fraction of the 30
    unreliable-cluster states predicted as the wrong label
    (#343's definition, as in #347).

  * THE CONFIDENCE-QUALITY KNOB (the new construction,
    stated as gift (i)): every verdict's confidence is
    drawn from its reliability group's range -- reliable
    U(0.82, 0.98), unreliable U(0.25, 0.55), the anchors'
    ranges -- with probability p, and from the POOLED
    marginal (the same two ranges mixed at the world's
    reliable:unreliable verdict share, drawn blind to
    reliability) with probability 1 - p. Levels:
    perfect p = 1.00, high p = 0.75, medium p = 0.50,
    low p = 0.25, noise p = 0.00. By construction the
    pooled distribution of confidences -- and the gate's
    expected drop count -- is IDENTICAL at every level;
    only the correlation between confidence and actual
    reliability varies. (Round 1 of the iteration log: a
    naive informative-vs-flat-noise mixture confounded
    signal quality with drop rate and was replaced by
    this knob.) The SHUFFLED endpoint (permutation of the
    perfect level's confidences, the anchors' control) is
    run alongside as the empirical zero-signal price; the
    noise level is the same price by construction, and
    the two must agree -- that agreement is asserted as a
    knob cross-check, not assumed.

  * MITIGATION ARMS (feedback layer only, applied on top
    of the confidence gate at the SAME level, so signal
    quality is matched):
    (A) AGREEMENT -- multi-draw agreement confidence: the
        verifier is queried 3 times per labelled state.
        A reliable verifier repeats its verdict on every
        draw; the planted unreliable verifier is UNSTABLE
        (gift (v)): each draw is its shortcut label with
        probability 0.5, else a uniform other label. A
        verdict supervises only if all 3 draws unanimously
        agree AND it passes the confidence gate.
    (B) CORROBORATE -- post-densification re-gating: after
        densification, a state receives supervision only
        if at least 2 distinct retained sources contribute
        kernel weight >= 0.25 to it (neighbourhood
        consensus: no state is supervised by a lone smeared
        verdict). States failing the check are routed to
        NO supervision, per #347's routing clause.

Readouts per seed (8 seeds, both layers): imported-error
mass (hard) at every level; feedback layer also soft
imported mass, task fit on reliable states, coverage.
Mitigation recovery is reported as the fraction of the
gated arm's excess-over-perfect imported error that the
mitigation removes at the same level, with its fit cost.

Modeling gifts, stated plainly: (i) confidence quality is
a CONSTRUCTION knob (planted correlation between the
confidence draw and verdict reliability, via the mixture
above); real verifier confidence is not a mixture of two
known ranges, and the levels are ordered landmarks, not
measurements of any real signal. The shuffled/noise
endpoint is the honest zero-signal price, as in the
anchors. (ii) The two layers' worlds are the anchors' own
and are NOT the same world: collection has no geometry
(states share nothing), feedback's cluster-mates share a
true label (geometry-label alignment is the densifier's
own working assumption, #347 gift (ii)). The comparison
is therefore of the two PRICE CURVES (8-seed means of
imported error vs signal quality), not of paired worlds:
per-seed cross-layer gaps are not meaningful and are not
asserted (round 3 of the iteration log). (iii) The
unreliable subsets, shortcut labels and cluster geometry
are PLANTED with known ground truth in both worlds, as in
the anchors; the verifier/teacher is wrong in exactly one
way, never ambiguous. (iv) Students are per-state logit
tables in both layers: states share parameters ONLY
through the densifier's targets at the feedback layer,
which is exactly the channel under test; nothing here
speaks to smearing through a shared adapter's weights
(#347 gift (iv)). (v) Mitigation A's extra channel --
unreliable-verifier INSTABILITY under re-querying -- is
planted and independent of the confidence knob by
construction; a real unreliable verifier that is stably,
confidently wrong would defeat arm A exactly as it
defeats the confidence gate, and arm B (which uses no
second channel, only the densifier's geometry) is the
honest price of a geometry-only mitigation.

Demonstrated shape asserted below (verdict PROVEN): both
layers' price curves are monotone in signal quality, and
the feedback curve sits STRICTLY ABOVE the collection
curve at every degraded level, with the gap WIDENING as
the signal weakens -- imported error (hard, 8-seed means):

  level     collection   feedback   excess (fb - coll)
  ungated      1.000       0.967            --
  perfect      0.000       0.000           0.000
  high         0.148       0.225          +0.077
  medium       0.312       0.425          +0.113
  low          0.528       0.650          +0.122
  noise        0.631       0.796          +0.165
  shuffled     0.699       0.792   (zero-signal control)

The endpoints reproduce both anchors (collection ungated
1.000 / gated 0.000 / shuffled 0.699 vs #343's 0.682 on 6
seeds; feedback ungated 0.967 / gated 0.000 / shuffled
0.792 vs #347's 0.967 / 0.000 / 0.788), and the two
zero-signal prices agree within noise at each layer
(feedback 0.796 vs 0.792; collection 0.631 vs 0.699),
cross-checking the knob. Mechanism, in the toy's terms:
at collection a surviving wrong label fits exactly ONE
state; at feedback a surviving wrong verdict's kernel
mass covers its whole cluster, so the same per-verdict
gate leakage buys ~a cluster's worth of imported error --
the amplification #347's seam priced at the shuffled
endpoint alone is paid at EVERY intermediate signal
quality, and paid more the weaker the signal gets.

Mitigations (feedback layer, excess-over-perfect
recovered / fit-on-reliable cost vs the gated arm at the
same level):
  level     AGREEMENT recovery   CORROBORATE recovery
  high       91% (fit -0.003)      100% (fit -0.144)
  medium     78% (fit -0.007)       76% (fit -0.202)
  low        87% (fit -0.002)       50% (fit -0.206)
  noise      84% (fit -0.011)       41% (fit -0.218)
Arm A recovers most of the excess price at matched
signal quality at near-zero fit cost -- through its
planted second channel (gift (v)), not through the
confidence signal. Arm B recovers a real but smaller
part using geometry alone, and pays for it in reliable
fit: corroboration cannot tell a lone CORRECT verdict
from a lone smeared wrong one, so it routes both to no
supervision. (Two further re-gates were tried in round 1
and are NOT in the final arms: target-sharpness re-gating
recovered nothing at any level, and single-source
dominance re-gating was dominated by arm B.)

Iteration log (one refinement per round):
  1. Naive knob (informative draw vs flat U(0.25, 0.98)
     noise draw): endpoints reproduce #347 exactly
     (0.967 / 0.000 / 0.792), but the knob confounds
     signal quality with drop rate -- at p = 0 the gate
     drops ~62% of all verdicts vs the shuffled control's
     ~25%, so imported error at "noise" (0.400) lands far
     BELOW shuffled (0.792) for a purely mechanical
     reason (coverage collapse + fallback). Sharpness
     re-gate tried as mitigation: recovers nothing.
  2. Pooled-marginal knob (this test's construction):
     marginal and expected drop count identical at every
     level, only the reliability correlation varies.
     Clean monotone curves at both layers; feedback
     strictly above collection at every degraded level;
     noise ~= shuffled at both layers (the knob
     cross-check). Mitigations priced: agreement
     recovers 78-91% of excess at <= 0.011 fit cost;
     corroboration recovers 41-100% at 0.14-0.22 fit
     cost; dominance re-gate dominated by corroboration
     and dropped.
  3. Per-seed audit: mitigation recovery is uniform per
     seed (agreement removes >= +0.167 imported error on
     EVERY seed at medium and below), but cross-layer
     gaps are mean-curve statements only -- the two
     layers' worlds are independently planted per seed,
     so per-seed gaps range negative and are not
     asserted. Assertions set on 8-seed means with
     margins inside the demonstrated values.
  4. Torch port as this file (both anchor worlds keep
     their own training loops verbatim in structure) +
     clean skip verified against the broken particlegan
     stub. Claims decided at round 3; loop stops.

What is NOT asserted: the anchors' paper numbers, any
real verifier's position on these curves (where a real
signal lands between noise and perfect is exactly what
a planted knob cannot measure), anything about
sequence-level feedback, and any claim that arm A's
recovery survives a stably-wrong unreliable verifier
(gift (v)) -- arm B is the bound for that case.
Pure torch toy on CPU, single-threaded, seeded. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test
family, PR #273's guarded-import pattern).
"""
import random

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


# ------------------------------------------------------------------ knob
C = 6                # label classes
THRESH = 0.70        # confidence gate threshold (both anchors')
LEVELS = (("perfect", 1.00), ("high", 0.75), ("medium", 0.50),
          ("low", 0.25), ("noise", 0.00))
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def _informative(rng, hi):
    return rng.uniform(0.82, 0.98) if hi else rng.uniform(0.25, 0.55)


def _draw_conf(rng, rel, p, hi_share):
    """The confidence-quality knob. With probability p the
    confidence is drawn from the verdict's own reliability
    range; with probability 1 - p from the pooled marginal
    (reliability-blind). The pooled marginal -- and so the
    gate's expected drop count -- is identical at every
    level; only the correlation with reliability varies.
    hi_share is the world's reliable share of verdicts."""
    if rng.random() < p:
        return _informative(rng, rel)
    return _informative(rng, rng.random() < hi_share)


# ------------------------------------------------- collection world (#343)
N_COV = 48           # teacher-covered states
N_TAIL = 72          # student-induced-only states
N_UNREL = 22         # planted unreliable tail states
FULL_STEPS = 10      # training steps per collected state


def _build_collection(seed):
    """#343's planted world: per-state dicts with kind, true
    label y, teacher label yt, reliability rel. Covered
    states first, then tail states."""
    rng = random.Random(seed)

    def balanced(n):
        labs = [i % C for i in range(n)]
        rng.shuffle(labs)
        return labs

    cov_y = balanced(N_COV)
    tail_y = balanced(N_TAIL)
    order = list(range(N_TAIL))
    rng.shuffle(order)
    unrel = set(order[:N_UNREL])
    states = []
    for y in cov_y:
        states.append(dict(kind="cov", y=y, yt=y, rel=True))
    for i, y in enumerate(tail_y):
        rel = i not in unrel
        yt = y if rel else (1 if y != 1 else 2)
        states.append(dict(kind="tail", y=y, yt=yt, rel=rel))
    return states


def _train_collection(states, steps_per):
    """#343's per-state logit table: SGD on the teacher
    labels of the given states, round-robin by step round."""
    n = len(states)
    logits = torch.zeros(n, C, dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.SGD([logits], lr=0.8)
    max_steps = max(steps_per.values()) if steps_per else 0
    for r in range(max_steps):
        for idx in sorted(steps_per):
            if steps_per[idx] <= r:
                continue
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                logits[idx:idx + 1],
                torch.tensor([states[idx]["yt"]]))
            loss.backward()
            opt.step()
    with torch.no_grad():
        return logits.argmax(dim=1).tolist()


def _run_collection(seed):
    states = _build_collection(seed)
    cov = [i for i, s in enumerate(states) if s["kind"] == "cov"]
    tail = [i for i, s in enumerate(states) if s["kind"] == "tail"]
    unrel = [i for i in tail if not states[i]["rel"]]
    hi_share = (N_TAIL - N_UNREL) / N_TAIL

    def imported(pred):
        return float(np.mean(
            [pred[i] == states[i]["yt"] for i in unrel]))

    out = {}
    pred = _train_collection(states, {i: FULL_STEPS for i in cov + tail})
    out["ungated"] = imported(pred)
    for li, (name, p) in enumerate(LEVELS):
        rng = random.Random(20000 + seed * 131 + li)
        conf = {i: _draw_conf(rng, states[i]["rel"], p, hi_share)
                for i in tail}
        acc = cov + [i for i in tail if conf[i] >= THRESH]
        pred = _train_collection(states, {i: FULL_STEPS for i in acc})
        out[name] = imported(pred)
    # Shuffled endpoint: permutation of the perfect draws.
    rngp = random.Random(20000 + seed * 131 + 0)
    confs = [_draw_conf(rngp, states[i]["rel"], 1.0, hi_share)
             for i in tail]
    rng = random.Random(30000 + seed)
    sh = confs[:]
    rng.shuffle(sh)
    sh_conf = {idx: sh[k] for k, idx in enumerate(tail)}
    acc = cov + [i for i in tail if sh_conf[i] >= THRESH]
    pred = _train_collection(states, {i: FULL_STEPS for i in acc})
    out["shuffled"] = imported(pred)
    return out


# --------------------------------------------------- feedback world (#347)
N_CLUST = 24         # state clusters (neighbourhoods)
PER = 5              # states per cluster
N = N_CLUST * PER    # 120 states
DIM = 8              # embedding dimension
LAB_PER_CLUST = 2    # verifier-labelled states per cluster
N_UNREL_CLUST = 6    # planted unreliable-verifier clusters
SIGMA = 1.1          # densifier kernel width
NOISE_EMB = 0.35     # within-cluster embedding noise
CENT_SCALE = 1.6     # cluster centroid spread
W0 = 0.5             # default pseudo-count in the densifier
COV_EPS = 0.5        # coverage floor on retained kernel mass
FB_STEPS = 60        # full-batch SGD steps on the dense targets
FB_LR = 0.3
CORROB_W = 0.25      # arm B: per-source weight that counts as support


def _build_feedback(seed):
    """#347's planted world: cluster embeddings, true labels,
    the unreliable-cluster set, sparse verdicts (shortcut
    label in unreliable clusters)."""
    rng = np.random.default_rng(seed)
    cl_lab = np.array([i % C for i in range(N_CLUST)])
    rng.shuffle(cl_lab)
    cents = rng.normal(0, CENT_SCALE, size=(N_CLUST, DIM))
    emb = (np.repeat(cents, PER, axis=0)
           + rng.normal(0, NOISE_EMB, size=(N, DIM)))
    cl_of = np.repeat(np.arange(N_CLUST), PER)
    y = cl_lab[cl_of]
    unrel_cl = set(
        rng.choice(N_CLUST, size=N_UNREL_CLUST, replace=False).tolist())
    labeled = []
    for c in range(N_CLUST):
        idx = np.where(cl_of == c)[0]
        labeled.extend(
            rng.choice(idx, size=LAB_PER_CLUST, replace=False).tolist())
    labeled = np.array(sorted(labeled))
    verdict = {}
    for j in labeled:
        j = int(j)
        verdict[j] = ((1 if y[j] != 1 else 2)
                      if cl_of[j] in unrel_cl else int(y[j]))
    wrong = np.array([1 if t != 1 else 2 for t in y])
    return dict(emb=emb, y=y, cl_of=cl_of, unrel_cl=unrel_cl,
                labeled=labeled, verdict=verdict, wrong=wrong)


def _kernel(emb):
    d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2 * SIGMA ** 2))


def _densify(W, world, retained):
    """#347's densifier: retained verdicts projected onto all
    states by the Gaussian kernel; the W0 pseudo-count sends
    dropped mass toward the uniform default instead of
    renormalizing it onto the surviving sources."""
    src = [int(j) for j in world["labeled"] if int(j) in retained]
    if not src:
        return np.full((N, C), 1.0 / C), np.zeros(N), src
    Ws = W[:, src]
    tot = Ws.sum(axis=1)
    onehot = np.zeros((len(src), C))
    for k, j in enumerate(src):
        onehot[k, world["verdict"][j]] = 1.0
    tgt = (Ws @ onehot + W0 / C) / (tot + W0)[:, None]
    return tgt, tot, src


def _train_feedback(tgt, active_mask):
    """#347's student: per-state logit table, full-batch SGD
    on the dense soft targets of the active states (per-state
    losses summed). Inactive states keep the default."""
    logits = torch.zeros(N, C, dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.SGD([logits], lr=FB_LR)
    target = torch.tensor(tgt, dtype=torch.float64)
    active = torch.tensor(active_mask, dtype=torch.bool)
    for _ in range(FB_STEPS):
        opt.zero_grad()
        logp = torch.log_softmax(logits, dim=1)
        loss = -(target[active] * logp[active]).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    return pred, probs


def _run_feedback(seed):
    world = _build_feedback(seed)
    W = _kernel(world["emb"])
    y, wrong = world["y"], world["wrong"]
    unrel_mask = np.array(
        [world["cl_of"][i] in world["unrel_cl"] for i in range(N)])
    rel_mask = ~unrel_mask
    lab = [int(j) for j in world["labeled"]]
    rel_of = {j: world["cl_of"][j] not in world["unrel_cl"]
              for j in lab}
    hi_share = float(np.mean([rel_of[j] for j in lab]))

    def evaluate(retained, corroborate=False):
        tgt, tot, src = _densify(W, world, retained)
        active = tot >= COV_EPS
        if corroborate and src:
            Ws = W[:, src]
            active = active & ((Ws >= CORROB_W).sum(axis=1) >= 2)
        pred, probs = _train_feedback(tgt, active)
        return dict(
            imp_hard=float(np.mean(pred[unrel_mask] == wrong[unrel_mask])),
            imp_soft=float(np.mean(probs[unrel_mask, wrong[unrel_mask]])),
            fit_rel=float(np.mean(pred[rel_mask] == y[rel_mask])),
            cov=float(np.mean(active)),
        )

    out = {}
    out["ungated"] = evaluate(set(lab))
    for li, (name, p) in enumerate(LEVELS):
        rng = random.Random(40000 + seed * 131 + li)
        conf = {j: _draw_conf(rng, rel_of[j], p, hi_share) for j in lab}
        retained = {j for j in lab if conf[j] >= THRESH}
        out[name] = evaluate(retained)
        out[f"{name}+corroborate"] = evaluate(retained, corroborate=True)
    # Mitigation A: agreement gate stacked on the confidence
    # gate at the same level. Reliable verifier repeats its
    # verdict on all 3 draws; the planted unreliable
    # verifier is unstable (shortcut w.p. 0.5 per draw, else
    # a uniform other label) -- gift (v).
    for li, (name, p) in enumerate(LEVELS):
        rng = random.Random(40000 + seed * 131 + li)
        conf = {j: _draw_conf(rng, rel_of[j], p, hi_share) for j in lab}
        rngd = random.Random(60000 + seed * 131 + li)
        retained = set()
        for j in lab:
            if conf[j] < THRESH:
                continue
            if rel_of[j]:
                draws = [world["verdict"][j]] * 3
            else:
                draws = []
                for _ in range(3):
                    if rngd.random() < 0.5:
                        draws.append(world["verdict"][j])
                    else:
                        others = [c for c in range(C)
                                  if c != world["verdict"][j]]
                        draws.append(others[rngd.randrange(len(others))])
            if len(set(draws)) == 1:
                retained.add(j)
        out[f"{name}+agree"] = evaluate(retained)
    # Shuffled endpoint: permutation of the perfect draws.
    rngp = random.Random(40000 + seed * 131 + 0)
    confs = np.array([_draw_conf(rngp, rel_of[j], 1.0, hi_share)
                      for j in lab])
    rng = np.random.default_rng(50000 + seed)
    sh = confs.copy()
    rng.shuffle(sh)
    retained = {lab[k] for k in range(len(lab)) if sh[k] >= THRESH}
    out["shuffled"] = evaluate(retained)
    return out


@pytest.fixture(scope="module")
def results():
    _empf_recipe()
    return {seed: (_run_collection(seed), _run_feedback(seed))
            for seed in SEEDS}


def _coll(results, key):
    return float(np.mean([results[s][0][key] for s in SEEDS]))


def _fb(results, key, field="imp_hard"):
    return float(np.mean([results[s][1][key][field] for s in SEEDS]))


def test_anchor_endpoints_reproduced(results):
    # Both anchors' endpoints, in their own worlds: ungated
    # imports (nearly) everything, the informative gate
    # imports nothing. Collection: #343's 1.000 / 0.000.
    # Feedback: #347's 0.967 / 0.000 hard, soft at the 1/C
    # no-information floor.
    assert _coll(results, "ungated") > 0.99
    assert _coll(results, "perfect") < 0.02
    assert _fb(results, "ungated") > 0.90
    assert _fb(results, "ungated", "imp_soft") > 0.60
    assert _fb(results, "perfect") < 0.02
    assert _fb(results, "perfect", "imp_soft") < 0.19


def test_price_curve_monotone_in_signal_quality(results):
    # At BOTH layers, imported error rises strictly as the
    # planted signal weakens, level by level (8-seed means;
    # demonstrated steps are >= 0.09 at every level).
    for layer in (_coll, _fb):
        curve = [layer(results, name) for name, _ in LEVELS]
        for a, b in zip(curve, curve[1:]):
            assert b > a + 0.05


def test_feedback_degrades_strictly_faster(results):
    # Headline: the feedback curve sits strictly above the
    # collection curve at every degraded level (demonstrated
    # excess +0.077 / +0.113 / +0.122 / +0.165), and the
    # excess WIDENS as the signal weakens -- densification
    # multiplies the price of a weak signal.
    gaps = {}
    for name, _ in LEVELS[1:]:
        gaps[name] = _fb(results, name) - _coll(results, name)
        assert gaps[name] > 0.04
    assert gaps["noise"] > gaps["high"] + 0.04
    assert gaps["low"] > gaps["high"]


def test_zero_signal_endpoints_agree(results):
    # Knob cross-check: the noise level (zero correlation by
    # construction) and the shuffled control (the anchors'
    # empirical zero-signal price) land at the same price at
    # each layer -- feedback demonstrated 0.796 vs 0.792,
    # collection 0.631 vs 0.699 -- and both sit far above
    # the perfect gate and below/at the ungated arm.
    assert abs(_fb(results, "noise") - _fb(results, "shuffled")) < 0.10
    assert abs(_coll(results, "noise") - _coll(results, "shuffled")) < 0.15
    for layer in (_coll, _fb):
        assert layer(results, "shuffled") > layer(results, "perfect") + 0.50
        assert layer(results, "shuffled") < layer(results, "ungated")


def test_agreement_mitigation_recovers_excess(results):
    # Mitigation A (multi-draw agreement, stacked on the
    # same level's confidence gate) recovers most of the
    # feedback layer's excess price at matched signal
    # quality -- demonstrated 78-91% of the excess at
    # high/medium/low/noise -- at near-zero fit cost
    # (<= 0.011 on reliable states). Its channel is planted
    # (gift (v)); the recovery is asserted, the channel is
    # not hidden.
    for name, _ in LEVELS[1:]:
        base = _fb(results, name)
        mit = _fb(results, f"{name}+agree")
        assert mit < base - 0.10
        assert (base - mit) / base > 0.50
        assert (_fb(results, name, "fit_rel")
                - _fb(results, f"{name}+agree", "fit_rel")) < 0.05


def test_corroboration_mitigation_recovers_at_fit_cost(results):
    # Mitigation B (post-densification corroboration
    # re-gate, geometry only, no second channel) recovers a
    # real part of the excess at medium signal quality --
    # demonstrated 76% -- but pays for it in reliable fit
    # (demonstrated -0.202): a lone correct verdict is
    # indistinguishable from a lone smeared wrong one, so
    # both are routed to no supervision.
    base = _fb(results, "medium")
    mit = _fb(results, "medium+corroborate")
    assert (base - mit) / base > 0.40
    assert (_fb(results, "medium", "fit_rel")
            - _fb(results, "medium+corroborate", "fit_rel")) > 0.08
