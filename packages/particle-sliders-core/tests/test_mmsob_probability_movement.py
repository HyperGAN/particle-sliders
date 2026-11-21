"""CPU toy for the MM-SystemOneBias claim (arXiv:2611.26557):
typed decision models are MORE stable than their backbones under
presentation-only changes -- but the robustness is reduced
sensitivity in BOTH directions (they also gain less from correct
external judgments), and accuracy / flip-rate audits understate
the instability that remains, because answer probabilities keep
moving even when the semantic decision does not change.

Paper: "MM-SystemOneBias: Evaluating Bias in Multimodal
Jev-Style System One Decision Models" (Li et al.). The authors
evaluate 13 models on 5,323 instances across 9 datasets and 11
scenarios, under six paired presentation interventions (option
order, option label, external judgment, image-text description,
irrelevant information, rating scale). Across 7 matched
decision-model-backbone pairs: decision models are more stable
under option-label, irrelevant-information and rating-scale
changes in ALL 7 pairs; show lower option-order flip rates in
6/7; suffer smaller accuracy losses under INCORRECT external
judgments in 7/7 and under conflicting descriptions in 6/7; BUT
gain LESS accuracy from CORRECT external judgments in 6/7 --
robustness as reduced sensitivity both ways. Signature finding:
among pairs whose semantic decision is UNCHANGED, mean absolute
changes in answer probabilities still range 4.56-11.55
percentage points -- accuracy/flip audits miss probability
movement. (All numbers from the abstract; nothing beyond it is
claimed here.)

Toy operationalization: binary items with a latent quality gap
d (option A is truly better iff d > 0). Both arms score the same
observed semantic signal sem = d + noise through the feature
vector x = [sem, order, label, irr, ext, sem*scale], where the
last five channels are presentation-only:

  order  +1 if A is shown first (primacy channel)
  label  +1 if A carries the favoured option label
  irr    strength of an irrelevant detail favouring A
  ext    an external judgment endorsing A (+1) / B (-1)
  scale  rating-scale remap, acting as a slope change on sem

BACKBONE arm: fixed hand-set readout weights with substantial
presentation loadings (sem 1.00, order 0.55, label 0.50,
irr 0.45, ext 0.95, scale-slope 0.50) -- the prompted-backbone
shape, whose readout loads on how the item is presented.
DECISION arm: the same feature vector, but weights FITTED by
logistic regression on paired-augmented training data -- each
training item appears under several presentations (order/label
flipped, irrelevant detail resampled, external judgment
randomized independently of the label, scale flag on/off) with
the SAME label, the typed-decision-model training shape. The
fit learns presentation invariance up to finite-sample residue:
with 200 items x 4 presentations the fitted presentation
weights land at |w| <= ~0.1, not exactly zero -- which is what
leaves the decision arm a small, real probability movement.

Paired interventions at eval (same items, one channel changed):
option-order swap, option-label permutation, irrelevant-info
addition, rating-scale remap, external-judgment injection
(correct / incorrect). Five of the paper's six interventions;
the image-text description conflict is not operationalized (no
image channel in a text toy) -- its closest analogue here is
the incorrect external judgment. Controls: (i) identity
re-presentation -- both scorers are deterministic, so movement
must be EXACTLY zero; any nonzero movement in the intervention
arms is therefore presentation-driven, not readout noise;
(ii) shuffled external judgments -- the endorsement vector is
permuted across items, destroying truth-alignment while keeping
its marginal distribution: it must show no systematic accuracy
effect comparable to the matched judgments (a small residual
loss for the backbone is expected and asserted bounded: random
endorsements hurt a better-than-chance judge asymmetrically,
but by far less than matched incorrect judgments).

Measured on pop-os (particlegan 0.10.1), seeds 0-2 (values
below are the pop-os run of this file; see PR body):
  flips, backbone vs decision -- order 0.305-0.345 vs
  0.004-0.029; label 0.302-0.308 vs 0.040-0.049; irrelevant
  0.094-0.111 vs 0.006-0.033; scale 0.041-0.053 vs 0.001-0.003.
  Mean |dp| among UNCHANGED decisions, averaged over the four
  presentation interventions: backbone 0.114-0.116, decision
  0.014-0.018 -- the audit-blindness effect exists in BOTH arms
  (decision movement is materially > 0) and is ~7x smaller for
  the decision arm.
  External judgments, accuracy change vs base -- backbone:
  correct +0.151..+0.181, incorrect -0.319..-0.354; decision:
  |change| <= 0.021 both ways (loses less AND gains less).
  Shuffled control, backbone: -0.085..-0.101 -- under half the
  matched-incorrect loss on every seed. Base accuracy:
  backbone 0.782-0.809, decision 0.891-0.905 (the backbone's
  base read already pays for its presentation loadings).
Assertions below encode exactly these margins, no more.

Iteration summary: local calibration rounds found (1) that
deterministic-label training inflates the decision fit's
semantic weight until its probabilities saturate and its
probability movement vanishes, and (2) that readout noise
swamps the decision arm's presentation signal (its flips
become its noise floor, and the scale flip reverses) -- the
final design trains on probabilistic labels and uses
deterministic scorers with the identity control above. Pop-os
loop, iteration 1: 2 passed (4.55s), margins as measured
above; proven and refined, no further rounds needed.

What is NOT asserted: any real model's bias magnitude -- both
scorers are synthetic, and the claim tested is the STRUCTURE
(presentation-loaded readout vs paired-augmentation fit;
flip audits missing probability movement; symmetric
desensitization to external judgments), not the paper's
benchmark numbers. The decision arm's residual sensitivity is
finite-sample residue of a small fit, standing in for -- not
measuring -- a trained decision model's residual sensitivity.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe (the house guard for this paper-test family).
Pure CPU toy -- no GAN machinery, because the claim is about
the structure of a decision readout, not training dynamics.
"""
import numpy as np
import pytest
import torch

try:
    particlegan = pytest.importorskip("particlegan")
except ImportError:  # particlegan present but its import raises
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


# Backbone readout: fixed weights loading on presentation.
W_BACKBONE = np.array([1.00, 0.55, 0.50, 0.45, 0.95, 0.50])
N_EVAL = 1500          # eval items per seed
N_TRAIN = 200          # decision-fit items (x 4 presentations)
LABEL_SLOPE = 1.2      # training labels ~ Bernoulli(sigmoid(1.2 d))
SEM_NOISE = 0.35       # observation noise on the semantic signal
PRESENTATIONS = ("order", "label", "irr", "scale")


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _features(sem, order, label, irr, ext, scale):
    return np.stack([sem, order, label, irr, ext, sem * scale],
                    axis=1)


def _fit_decision(rng):
    """Logistic fit on paired-augmented data: every training
    item is seen under 4 presentations with the same label, and
    the presentation channels are randomized independently of
    the label -- so the fit can only earn by downweighting them,
    up to finite-sample residue."""
    d = rng.normal(0.0, 1.1, N_TRAIN)
    sem = d + rng.normal(0.0, SEM_NOISE, N_TRAIN)
    y = (rng.random(N_TRAIN) < _sigmoid(LABEL_SLOPE * d)
         ).astype(np.float64)
    xs, ys = [], []
    for _ in range(4):
        xs.append(_features(
            sem,
            rng.choice([-1.0, 1.0], N_TRAIN),
            rng.choice([-1.0, 1.0], N_TRAIN),
            rng.normal(0.0, 1.0, N_TRAIN),
            rng.choice([-1.0, 0.0, 1.0], N_TRAIN),
            rng.integers(0, 2, N_TRAIN).astype(np.float64)))
        ys.append(y)
    x = np.concatenate(xs)
    t = np.concatenate(ys)
    w = np.zeros(6)
    for _ in range(400):  # full-batch GD, deterministic
        grad = x.T @ (_sigmoid(x @ w) - t) / len(t) + 0.02 * w
        w -= 0.5 * grad
    return w


def run_pair(seed):
    """One matched backbone / decision pair under all paired
    interventions. Returns per-arm metrics dicts."""
    rng = np.random.default_rng(77000 + seed)
    w_dec = _fit_decision(rng)
    d = rng.normal(0.0, 1.1, N_EVAL)
    sem = d + rng.normal(0.0, SEM_NOISE, N_EVAL)
    truth = d > 0
    order0 = rng.choice([-1.0, 1.0], N_EVAL)
    label0 = rng.choice([-1.0, 1.0], N_EVAL)
    irr_new = rng.normal(0.0, 1.0, N_EVAL)
    sgn = np.sign(d)
    sgn[sgn == 0.0] = 1.0
    zero = np.zeros(N_EVAL)
    one = np.ones(N_EVAL)

    def probs(w, order, label, irr, ext, scale):
        return _sigmoid(_features(sem, order, label, irr, ext,
                                  scale) @ w)

    out = {}
    for arm, w in (("backbone", W_BACKBONE), ("decision", w_dec)):
        base = probs(w, order0, label0, zero, zero, zero)
        ident = probs(w, order0, label0, zero, zero, zero)
        variants = {
            "order": probs(w, -order0, label0, zero, zero, zero),
            "label": probs(w, order0, -label0, zero, zero, zero),
            "irr": probs(w, order0, label0, irr_new, zero, zero),
            "scale": probs(w, order0, label0, zero, zero, one),
            "ext_correct": probs(w, order0, label0, zero, sgn,
                                 zero),
            "ext_incorrect": probs(w, order0, label0, zero, -sgn,
                                   zero),
            "ext_shuffled": probs(w, order0, label0, zero,
                                  rng.permutation(sgn), zero),
        }
        dec0 = base > 0.5
        acc0 = float(np.mean(dec0 == truth))
        res = {"w": w, "acc0": acc0,
               "identity_dp": float(np.mean(np.abs(base - ident)))}
        for name, p in variants.items():
            dec = p > 0.5
            unch = dec == dec0
            res[name] = {
                "flip": float(np.mean(dec != dec0)),
                "dp_unchanged": float(
                    np.mean(np.abs(p - base)[unch])),
                "dacc": float(np.mean(dec == truth)) - acc0,
            }
        res["mean_dp_unchanged"] = float(np.mean(
            [res[k]["dp_unchanged"] for k in PRESENTATIONS]))
        out[arm] = res
    return out


def test_decision_arm_more_stable_under_presentation_changes():
    _empf_recipe()
    for seed in (0, 1, 2):
        pair = run_pair(seed)
        bb, dec = pair["backbone"], pair["decision"]
        # Mechanism check: the paired fit really did downweight
        # the presentation channels relative to the semantic one.
        assert dec["w"][0] > 0.5, (
            f"seed {seed}: decision semantic weight "
            f"{dec['w'][0]:.3f}")
        assert np.max(np.abs(dec["w"][1:5])) < 0.25, (
            f"seed {seed}: decision presentation weights "
            f"{np.round(dec['w'][1:5], 3)}")
        assert dec["w"][5] < 0.30, (
            f"seed {seed}: decision scale-slope {dec['w'][5]:.3f}")
        # Control (i): identity re-presentation moves NOTHING --
        # both scorers are deterministic, so all movement below
        # is presentation-driven.
        assert bb["identity_dp"] < 1e-12
        assert dec["identity_dp"] < 1e-12
        # Base accuracy: the decision arm is at least as accurate
        # (the backbone's base read pays for its loadings).
        assert 0.75 < bb["acc0"] < 0.85, (
            f"seed {seed}: backbone base acc {bb['acc0']:.3f}")
        assert dec["acc0"] > bb["acc0"] + 0.05, (
            f"seed {seed}: decision base acc {dec['acc0']:.3f} "
            f"vs backbone {bb['acc0']:.3f}")
        # Flip rates: decision < backbone on every presentation
        # intervention (the paper's all-7-pairs channels are
        # label / irrelevant / scale; order is its 6/7 channel).
        for name, margin in (("order", 0.15), ("label", 0.15),
                             ("irr", 0.04), ("scale", 0.025)):
            assert dec[name]["flip"] < bb[name]["flip"] - margin, (
                f"seed {seed} {name}: flip decision "
                f"{dec[name]['flip']:.3f} vs backbone "
                f"{bb[name]['flip']:.3f}")
        assert dec["order"]["flip"] < 0.06
        assert dec["label"]["flip"] < 0.06
        assert dec["irr"]["flip"] < 0.06
        assert dec["scale"]["flip"] < 0.01
        # Signature finding: among UNCHANGED decisions the
        # probabilities still move -- materially in BOTH arms,
        # and much less in the decision arm. A flip-only audit
        # would call the decision arm (near-)stable and miss
        # this entirely.
        assert bb["mean_dp_unchanged"] > 0.09, (
            f"seed {seed}: backbone unchanged |dp| "
            f"{bb['mean_dp_unchanged']:.3f}")
        assert 0.010 < dec["mean_dp_unchanged"] < 0.030, (
            f"seed {seed}: decision unchanged |dp| "
            f"{dec['mean_dp_unchanged']:.3f} -- audit-blindness "
            f"effect must be present but small")
        assert dec["mean_dp_unchanged"] < (
            bb["mean_dp_unchanged"] - 0.05), (
            f"seed {seed}: unchanged |dp| decision "
            f"{dec['mean_dp_unchanged']:.3f} vs backbone "
            f"{bb['mean_dp_unchanged']:.3f}")
        for name in PRESENTATIONS:
            assert dec[name]["dp_unchanged"] < (
                bb[name]["dp_unchanged"]), (
                f"seed {seed} {name}: unchanged |dp| decision "
                f"{dec[name]['dp_unchanged']:.3f} vs backbone "
                f"{bb[name]['dp_unchanged']:.3f}")


def test_external_judgment_asymmetry_and_shuffled_control():
    _empf_recipe()
    for seed in (0, 1, 2):
        pair = run_pair(seed)
        bb, dec = pair["backbone"], pair["decision"]
        gain_bb = bb["ext_correct"]["dacc"]
        loss_bb = -bb["ext_incorrect"]["dacc"]
        gain_dec = dec["ext_correct"]["dacc"]
        loss_dec = -dec["ext_incorrect"]["dacc"]
        # The backbone is highly sensitive both ways.
        assert gain_bb > 0.12, (
            f"seed {seed}: backbone correct-judgment gain "
            f"{gain_bb:+.3f}")
        assert loss_bb > 0.28, (
            f"seed {seed}: backbone incorrect-judgment loss "
            f"{loss_bb:.3f}")
        # The decision arm is desensitized BOTH ways: it loses
        # less under incorrect judgments AND gains less under
        # correct ones (robustness = reduced sensitivity, not
        # discernment -- its fitted ext weight is ~0, so it
        # cannot tell correct from incorrect at all).
        assert abs(gain_dec) < 0.06, (
            f"seed {seed}: decision correct-judgment gain "
            f"{gain_dec:+.3f}")
        assert abs(loss_dec) < 0.06, (
            f"seed {seed}: decision incorrect-judgment loss "
            f"{loss_dec:+.3f}")
        assert gain_bb - gain_dec > 0.10, (
            f"seed {seed}: gain gap backbone-decision "
            f"{gain_bb - gain_dec:+.3f}")
        assert loss_bb - loss_dec > 0.20, (
            f"seed {seed}: loss gap backbone-decision "
            f"{loss_bb - loss_dec:+.3f}")
        # Control (ii): shuffled (truth-unaligned) judgments show
        # no systematic accuracy effect comparable to matched
        # ones -- for the backbone their residual loss stays
        # under half the matched-incorrect loss, and for the
        # decision arm they are inert like the matched ones.
        shuf_bb = bb["ext_shuffled"]["dacc"]
        assert abs(shuf_bb) < 0.5 * loss_bb, (
            f"seed {seed}: backbone shuffled-judgment dacc "
            f"{shuf_bb:+.3f} vs matched-incorrect {-loss_bb:+.3f}")
        assert shuf_bb > bb["ext_incorrect"]["dacc"] + 0.15, (
            f"seed {seed}: backbone shuffled {shuf_bb:+.3f} not "
            f"clearly separated from matched-incorrect "
            f"{bb['ext_incorrect']['dacc']:+.3f}")
        assert abs(dec["ext_shuffled"]["dacc"]) < 0.03, (
            f"seed {seed}: decision shuffled-judgment dacc "
            f"{dec['ext_shuffled']['dacc']:+.3f}")
