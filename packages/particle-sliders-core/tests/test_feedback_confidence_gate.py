"""CPU gate for the feedback-layer confidence-gate claim --
the chain test joining PR #343 (CARE-OPD, arXiv:2611.34756)
to PRs #326/#333 (VOPD, arXiv:2611.30325). Neither anchor
tested this composition; this test decides it at toy scale.

The chain, layer by layer. The distillation loop's missing
rollout loop (see the "How Particle Sliders would use this"
section of PR #343 / #326 / #327) has three layers:

  * COLLECTION (#343, PROVEN there): student-seeded states
    gated by TEACHER confidence before they supervise. The
    gate removed ALL imported error (1.000 -> 0.000, exactly
    the oracle gate) because dropped states fall back to the
    default instead of being fitted to a wrong label; the
    shuffled-confidence control priced the signal (0.682
    imported error when confidence is meaningless).
  * CONSUMPTION (#327): what the loop does with collected
    states once consumed (provenance, weighting).
  * FEEDBACK (#326/#333, this test's layer): a verifier's
    SPARSE feedback on student rollouts is densified by
    projection onto the student's state geometry, so
    supervision reaches states the verifier never labelled.
    #326 showed the densification carries signal only along
    a representable direction and runs on a fallback where
    the per-group coefficient is undefined; #333 showed the
    carrier depends on where partial credit is placed.

CHAIN QUESTION: does a CONFIDENCE GATE on the verifier's
sparse feedback -- verdicts below a confidence threshold
dropped BEFORE densification, so their states receive NO
supervision rather than wrong supervision -- remove a
planted unreliable-verifier subset's imported error the
way #343's gate did at collection?

Toy operationalization. 120 states in 24 clusters of 5,
C = 6 classes. Each cluster shares ONE true label (4
clusters per class, balanced): state geometry and labels
are aligned, which is the assumption densification itself
rests on (stated as gift (ii) below). State embeddings are
cluster centroids (scale 1.6, 8-D) plus noise (0.35), so
geometric neighbours are overwhelmingly cluster-mates.

  * SPARSE VERIFIER: labels 2 of 5 states per cluster
    (48 of 120 states; 72 states are never labelled and can
    only be reached by densification).
  * PLANTED UNRELIABLE SUBSET: 6 of the 24 clusters are
    unreliable-verifier clusters -- every verdict sourced
    from them is systematically WRONG (the #343 shortcut
    label: class 1, or class 2 where the truth is 1) and
    carries LOW confidence (0.25-0.55). Reliable clusters'
    verdicts are the true label with HIGH confidence
    (0.82-0.98). Confidence is informative BY CONSTRUCTION
    (gift (i)): at threshold 0.70 the confidence gate
    coincides with the oracle gate, exactly as in #343;
    the shuffled control is what prices the signal.
  * DENSIFICATION (the #326 mechanism, state-geometry
    form): retained verdicts are projected onto ALL states
    by a Gaussian kernel over embeddings (sigma = 1.1):
    target_s = (sum_j w_sj onehot(verdict_j) + W0/C) /
    (sum_j w_sj + W0), with a default pseudo-count
    W0 = 0.5 -- dropped mass falls back toward the uniform
    default instead of renormalizing onto surviving
    sources (round 2 of the iteration log: without it, a
    gate that drops one of two wrong sources in a cluster
    barely moves the imported mass, because the survivor's
    relative weight doubles).
  * FALLBACK RULE (the #343 discipline, load-bearing --
    see the leaktrain arm): a state whose retained kernel
    mass falls below the coverage floor COV_EPS = 0.5
    receives NO training and keeps the untrained default
    prediction (class 0). Densifier leakage alone never
    supervises.
  * STUDENT: a per-state logit table (as in #343) trained
    by full-batch SGD on the dense soft targets of the
    covered states, 60 steps at LR 0.3.

Arms (identical worlds, verdicts and budgets per seed;
8 seeds):
  (a) ungated   -- all 48 verdicts densified.
  (b) gated     -- verdicts with confidence < 0.70 dropped
                   before densification.
  (c) shuffled  -- arm (b)'s gate applied to a per-seed
                   permutation of the confidences (same
                   number dropped, no signal).
  (d) oracle    -- drops exactly the planted subset.
  (e) nofb      -- no feedback: the untrained default.
  (f) leaktrain -- DIAGNOSTIC: arm (b)'s gate, but states
                   below the coverage floor ARE trained on
                   their leakage targets. Prices the
                   fallback rule: it isolates the smearing
                   channel (densifier leakage re-importing
                   the shortcut label into gated
                   neighbourhoods) that the fallback rule
                   exists to close.

Readouts per seed:
  * IMPORTED-ERROR MASS, hard: fraction of the 30 states
    in unreliable clusters whose final prediction equals
    the planted wrong label (#343's definition).
  * IMPORTED-ERROR MASS, soft: mean student probability
    on the wrong label over those states. The
    no-information floor is 1/C = 0.167 (the default's
    uniform), so a gated soft readout AT 0.167 is zero
    excess imported mass.
  * TASK FIT ON RELIABLE STATES: hard accuracy over the
    90 states in reliable clusters.
  * COVERAGE: fraction of all 120 states whose retained
    kernel mass clears the floor (i.e. that receive any
    supervision at all).

Modeling gifts, stated plainly: (i) confidence separates
the planted subset exactly by construction, so gated ==
oracle here; real verifier confidence is noisier and the
shuffled control (not the oracle gap, which is zero by
construction) is the honest price of the signal -- #343's
gift (i), inherited unchanged. (ii) Cluster-mates share a
true label: geometry-label alignment is perfect, which is
the densifier's own working assumption; a geometry that
mixed labels within neighbourhoods would tax EVERY arm's
fit, gated or not. (iii) The unreliable subset, its
shortcut label, and the cluster geometry are all PLANTED
with known ground truth; the verifier is wrong in exactly
one way, never ambiguous. (iv) The student is a per-state
logit table: states share parameters ONLY through the
densifier's targets, which is exactly the channel under
test -- nothing here speaks to smearing through a shared
adapter's weights, where gated neighbourhoods could be
reached by a second channel this toy removes by
construction.

Demonstrated shape asserted below (verdict PROVEN, with
two measured seams): the confidence gate at the FEEDBACK
layer removes the planted subset's imported error exactly
as at collection -- ungated densified feedback imports
0.967 hard / 0.705 soft; the gated arm sits at 0.000 hard
with soft at the 0.167 no-information floor, identical to
the oracle gate on every readout, while task fit on
reliable states is undamaged (0.978 vs ungated 0.979) and
overall fit IMPROVES (0.780 vs 0.735: the ungated arm
actively fits a quarter of the world to the wrong label).
The price is coverage: 0.736 vs 0.979, a -0.243 cost that
is, to within noise, exactly the planted unreliable share
(0.25) -- the gate's cost concentrates in exactly the
neighbourhoods whose verifier was unreliable, which fall
back to the default instead of being supervised. The
shuffled gate prices the signal: 0.788 hard / 0.590 soft,
strictly between gated and ungated on both readouts --
and SEAM 1: much closer to ungated than #343's
collection-layer control (0.682 against a 1.000 ungated
there; 0.788 against 0.967 here). Densification AMPLIFIES
each surviving wrong verdict across its whole
neighbourhood, so a same-size gate without the signal
fails harder at the feedback layer than at collection:
one retained wrong source flips its cluster. SEAM 2
(the task's named mechanism risk, measured): the gate's
removal is conditional on the fallback rule. The
leaktrain diagnostic -- same gate, but below-floor states
trained on densifier leakage -- re-imports 0.250 of the
error (hard), because the shortcut label is also the true
label of reliable clusters elsewhere, and their leakage
tips near-default states onto it. A feedback gate must
therefore do two things with dropped states, not one:
drop their verdicts before densification AND route their
neighbourhoods to no supervision, not to whatever the
densifier still leaks.

Measured locally (numpy prototype) and on pop-os (torch
port, particlegan 0.10.1), 8 seeds, means (arm: imported
hard / imported soft / fit on reliable / coverage):
  ungated:   0.967 / 0.705 / 0.979 / 0.979
  gated:     0.000 / 0.167 / 0.978 / 0.736
  shuffled:  0.788 / 0.590 / 0.806 / 0.782
  oracle:    0.000 / 0.167 / 0.978 / 0.736
  nofb:      0.000 / 0.167 / 0.160 / 0.000
  leaktrain: 0.250 / 0.176 / 0.996 / 0.736
(leaktrain coverage counts states above the floor, as for
every arm; it trains the below-floor states as well.)

Iteration log (one refinement per round):
  1. Baseline (cluster-planted unreliable subset, kernel
     vote renormalized over retained sources, sigma 1.5):
     ungated imports 0.979, gated == oracle, but gated
     still imports 0.296 hard / 0.205 soft and the shuffled
     gate is indistinguishable from ungated (0.975) --
     with a wide kernel, leakage mass rivals local mass,
     and renormalization doubles each surviving wrong
     source's relative weight.
  2. Default pseudo-count W0 = 0.5 in the densifier
     (dropped mass falls back to uniform instead of
     renormalizing onto survivors): shuffled soft moves
     0.585 -> 0.519; hard unmoved -- leakage still
     dominates at sigma 1.5.
  3. Kernel sigma 1.5 -> 1.1 (local sources dominate;
     leakage becomes a residue): fit on reliable states
     0.946 -> 0.983 gated, coverage starts to price the
     gate (0.853 vs ~0.99); gated imported error stuck at
     ~0.20 soft / 0.25 hard -- diagnosed as the uniform
     floor (1/C) inside the soft readout plus argmax flips
     of near-default states on leakage (round 4 check).
  4. Centroid scale 1.0 -> 1.6 (wider cluster separation):
     reliable fit saturates (0.996 gated) and the coverage
     cost cleans up to ~the planted share, but gated hard
     imported error is unchanged (0.250) -- confirming the
     residue is leakage-tipped argmax on trained
     below-floor states, not direct supervision.
  5. Fallback rule (below-floor states receive NO
     training) + soft readout moved to the student's
     post-training probabilities: gated imported error
     0.000 hard / floor-level soft on EVERY seed, shuffled
     strictly between on both readouts (0.788 / 0.590),
     reliable fit undamaged. The leaktrain diagnostic arm
     is added to keep round 4's smearing measured inside
     the final test. Claims decided; loop stops at 5
     design rounds.
  6. Torch port (float64, per-state losses summed --
     #326's round-7 lesson) + convergence check: the port
     reproduces the numpy prototype table EXACTLY (every
     cell to 3 decimals, pop-os); STEPS 60 -> 120 leaves
     every hard readout identical and moves soft readouts
     by <= 0.005, so 60 steps is at equilibrium for every
     asserted margin. Clean skip verified against the
     broken particlegan stub (module-level skip).

What is NOT asserted: the anchors' paper numbers,
anything about sequence-level verifier feedback or real
verifiers (whose confidence is noisy and whose errors are
not a single planted shortcut), anything about smearing
through shared adapter weights (gift (iv)), and any claim
that a gate whose threshold is mis-set relative to real
confidence would land at the oracle -- where it lands
between the shuffled control and the oracle is exactly
what this toy cannot measure.
Pure torch toy on CPU, single-threaded, seeded. Skips
cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test
family, PR #273's guarded-import pattern).
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
C = 6                # label classes
N_CLUST = 24         # state clusters (neighbourhoods)
PER = 5              # states per cluster
N = N_CLUST * PER    # 120 states
DIM = 8              # embedding dimension
LAB_PER_CLUST = 2    # verifier-labelled states per cluster (sparse)
N_UNREL_CLUST = 6    # planted unreliable-verifier clusters
THRESH = 0.70        # confidence gate threshold (#343's)
SIGMA = 1.1          # densifier kernel width
NOISE = 0.35         # within-cluster embedding noise
CENT_SCALE = 1.6     # cluster centroid spread
W0 = 0.5             # default pseudo-count in the densifier
COV_EPS = 0.5        # coverage floor on retained kernel mass
STEPS = 60           # full-batch SGD steps on the dense targets
LR = 0.3
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
ARMS = ("ungated", "gated", "shuffled", "oracle", "nofb", "leaktrain")


def _build_world(seed):
    """Planted world. Returns dict with embeddings, true
    labels, cluster ids, the unreliable-cluster set, the
    labelled-state list, and per-labelled-state verdict +
    confidence. Cluster labels are balanced (4 per class);
    unreliable clusters' verdicts are the #343 shortcut
    label with low confidence."""
    rng = np.random.default_rng(seed)
    cl_lab = np.array([i % C for i in range(N_CLUST)])
    rng.shuffle(cl_lab)
    cents = rng.normal(0, CENT_SCALE, size=(N_CLUST, DIM))
    emb = (np.repeat(cents, PER, axis=0)
           + rng.normal(0, NOISE, size=(N, DIM)))
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
    verdict, conf = {}, {}
    for j in labeled:
        if cl_of[j] in unrel_cl:
            verdict[int(j)] = 1 if y[j] != 1 else 2
            conf[int(j)] = float(rng.uniform(0.25, 0.55))
        else:
            verdict[int(j)] = int(y[j])
            conf[int(j)] = float(rng.uniform(0.82, 0.98))
    wrong = np.array([1 if t != 1 else 2 for t in y])
    return dict(emb=emb, y=y, cl_of=cl_of, unrel_cl=unrel_cl,
                labeled=labeled, verdict=verdict, conf=conf, wrong=wrong)


def _kernel(emb):
    d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2 * SIGMA ** 2))


def _densify(W, world, retained):
    """Project retained verdicts onto all states. Returns
    (targets (N, C), retained kernel mass per state). The
    default pseudo-count W0 sends dropped mass toward the
    uniform default instead of renormalizing it onto the
    surviving sources."""
    src = [int(j) for j in world["labeled"] if int(j) in retained]
    if not src:
        return np.full((N, C), 1.0 / C), np.zeros(N)
    Ws = W[:, src]
    tot = Ws.sum(axis=1)
    onehot = np.zeros((len(src), C))
    for k, j in enumerate(src):
        onehot[k, world["verdict"][j]] = 1.0
    tgt = (Ws @ onehot + W0 / C) / (tot + W0)[:, None]
    return tgt, tot


def _train(tgt, tot, train_below_floor=False):
    """Per-state logit table trained by full-batch SGD on
    the dense soft targets. States below the coverage floor
    are not trained (fallback: they keep the default)
    unless train_below_floor (the leaktrain diagnostic)."""
    logits = torch.zeros(N, C, dtype=torch.float64,
                         requires_grad=True)
    opt = torch.optim.SGD([logits], lr=LR)
    target = torch.tensor(tgt, dtype=torch.float64)
    active = torch.tensor(
        (tot >= COV_EPS) | train_below_floor, dtype=torch.bool)
    for _ in range(STEPS):
        opt.zero_grad()
        logp = torch.log_softmax(logits, dim=1)
        # Per-state losses SUMMED (#326 round-7 lesson: a
        # mean over states would shrink every independent
        # row's gradient by 1/N).
        loss = -(target[active] * logp[active]).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1).numpy()
        pred = logits.argmax(dim=1).numpy()
    return pred, probs


def _run_seed(seed):
    world = _build_world(seed)
    W = _kernel(world["emb"])
    y, wrong = world["y"], world["wrong"]
    unrel_mask = np.array(
        [world["cl_of"][i] in world["unrel_cl"] for i in range(N)])
    rel_mask = ~unrel_mask
    lab = [int(j) for j in world["labeled"]]
    conf = world["conf"]
    rng = np.random.default_rng(10000 + seed)

    retained = {
        "ungated": set(lab),
        "gated": {j for j in lab if conf[j] >= THRESH},
        "oracle": {j for j in lab
                   if world["cl_of"][j] not in world["unrel_cl"]},
        "nofb": set(),
    }
    confs = np.array([conf[j] for j in lab])
    sh = confs.copy()
    rng.shuffle(sh)
    retained["shuffled"] = {lab[k] for k in range(len(lab))
                            if sh[k] >= THRESH}
    retained["leaktrain"] = retained["gated"]

    out = {}
    for arm in ARMS:
        tgt, tot = _densify(W, world, retained[arm])
        pred, probs = _train(tgt, tot,
                             train_below_floor=(arm == "leaktrain"))
        out[arm] = dict(
            imp_hard=float(np.mean(
                pred[unrel_mask] == wrong[unrel_mask])),
            imp_soft=float(np.mean(
                probs[unrel_mask, wrong[unrel_mask]])),
            fit_rel=float(np.mean(pred[rel_mask] == y[rel_mask])),
            fit_all=float(np.mean(pred == y)),
            cov=float(np.mean(tot >= COV_EPS)),
            n_ret=len(retained[arm]),
        )
    return out


@pytest.fixture(scope="module")
def results():
    _empf_recipe()
    return {seed: _run_seed(seed) for seed in SEEDS}


def _mean(results, arm, key):
    return float(np.mean([results[s][arm][key] for s in SEEDS]))


def test_ungated_densified_feedback_imports_planted_error(results):
    # Arm (a): with no gate, projection-densified feedback
    # fits the unreliable neighbourhoods to the verifier's
    # wrong label -- demonstrated 0.967 hard / 0.705 soft.
    for s in SEEDS:
        assert results[s]["ungated"]["imp_hard"] > 0.85
    assert _mean(results, "ungated", "imp_hard") > 0.90
    assert _mean(results, "ungated", "imp_soft") > 0.60


def test_confidence_gate_removes_imported_error(results):
    # Headline: the gate, applied BEFORE densification,
    # removes the planted imported error to the oracle
    # gate's level -- demonstrated 0.000 hard on every seed
    # with soft at the 1/C no-information floor (0.167),
    # exactly the no-feedback control's soft level.
    for s in SEEDS:
        assert results[s]["gated"]["imp_hard"] < 0.05
        assert abs(results[s]["gated"]["imp_hard"]
                   - results[s]["oracle"]["imp_hard"]) < 0.05
    assert _mean(results, "gated", "imp_soft") < 0.19
    assert (_mean(results, "gated", "imp_soft")
            <= _mean(results, "nofb", "imp_soft") + 0.02)
    assert (_mean(results, "ungated", "imp_hard")
            - _mean(results, "gated", "imp_hard")) > 0.85


def test_gate_cost_is_coverage_not_reliable_fit(results):
    # The gate's price is coverage, concentrated in the
    # unreliable neighbourhoods (planted share 0.25):
    # coverage 0.736 vs ungated 0.979, while task fit on
    # reliable states is undamaged (0.978 vs 0.979) and
    # overall fit improves (the ungated arm fits a quarter
    # of the world to the wrong label).
    assert _mean(results, "ungated", "cov") > 0.95
    assert _mean(results, "gated", "cov") < 0.80
    assert (_mean(results, "ungated", "cov")
            - _mean(results, "gated", "cov")) > 0.15
    for s in SEEDS:
        assert (results[s]["gated"]["fit_rel"]
                > results[s]["ungated"]["fit_rel"] - 0.05)
    assert _mean(results, "gated", "fit_rel") > 0.95
    assert (_mean(results, "gated", "fit_all")
            > _mean(results, "ungated", "fit_all"))


def test_shuffled_confidence_prices_the_signal(results):
    # Control: dropping the same number of verdicts under a
    # permuted confidence keeps most of the imported error
    # (demonstrated 0.788 hard / 0.590 soft, strictly
    # between gated and ungated on both) -- and sits much
    # closer to ungated than #343's collection-layer
    # control did, because densification amplifies each
    # surviving wrong verdict across its neighbourhood.
    for s in SEEDS:
        assert results[s]["shuffled"]["imp_hard"] > 0.40
    sh_h = _mean(results, "shuffled", "imp_hard")
    assert sh_h > _mean(results, "gated", "imp_hard") + 0.50
    assert sh_h < _mean(results, "ungated", "imp_hard") - 0.05
    sh_s = _mean(results, "shuffled", "imp_soft")
    assert sh_s > _mean(results, "gated", "imp_soft") + 0.25
    assert sh_s < _mean(results, "ungated", "imp_soft") - 0.03
    # The shuffled gate also taxes reliable fit (it drops
    # reliable verdicts at random): demonstrated 0.806 vs
    # the true gate's 0.978.
    assert (_mean(results, "shuffled", "fit_rel")
            < _mean(results, "gated", "fit_rel") - 0.08)


def test_fallback_rule_is_load_bearing_leaktrain(results):
    # Seam 2, the measured smearing channel: the SAME gate,
    # but with below-floor states trained on densifier
    # leakage, re-imports a quarter of the planted error
    # (demonstrated 0.250 hard) -- leakage from reliable
    # clusters whose true label IS the shortcut label tips
    # near-default states onto it. Dropping verdicts is not
    # enough; dropped neighbourhoods must receive NO
    # supervision, not the densifier's residue.
    lt = _mean(results, "leaktrain", "imp_hard")
    assert lt > _mean(results, "gated", "imp_hard") + 0.10
    assert lt < _mean(results, "ungated", "imp_hard") - 0.40
    assert _mean(results, "leaktrain", "imp_soft") \
        > _mean(results, "gated", "imp_soft") + 0.005


def test_no_feedback_control_is_the_default(results):
    # Arm (e): no feedback at all -- no imported error, no
    # coverage, and reliable-state fit at the default's
    # level (the class-0 share, ~1/6).
    assert _mean(results, "nofb", "cov") == 0.0
    assert _mean(results, "nofb", "imp_hard") < 0.02
    assert _mean(results, "nofb", "fit_rel") < 0.25
    for arm in ARMS:
        if arm != "nofb":
            assert (_mean(results, arm, "fit_rel")
                    > _mean(results, "nofb", "fit_rel") + 0.40)
