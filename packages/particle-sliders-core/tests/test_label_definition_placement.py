"""CPU gate for the label-over-definition placement claim
(arXiv:2610.02586 + arXiv:2609.26758, reached via the reference list
of the TDM-Gauntlet paper, arXiv:2611.20191, gated in PR #252):
typed decision models follow the option NAME over the definition
bound to it -- so WHERE a convention is stated decides whether it
lands. A convention carried in the definition text is largely
ignored while the label states the plain rule; the same convention
carried in the label itself governs.

Papers. (1) Azizi, Baghaei Potraghloo & Pedram, "Labels Override
Definitions in Jev-Style Typed Decision Models" (arXiv:2610.02586):
deleting every definition leaves accuracy unchanged (0.8559 vs
0.8487) although definitions alone support 0.7971; renaming the
options to A/B RAISES accuracy by +0.1511. (2) Sun et al., "Type-
Safe Is Not Error-Free" (arXiv:2609.26758): swapping the name <->
definition mapping flips decisions up to 70.4pp more with yes/no
names than with a 0/1 control, and mean AUC falls 93.8% -> 23.2%
(below chance) under reassignment. Both were reached via the
reference list of Suppa & Kopcan's TDM-Gauntlet paper
(arXiv:2611.20191) -- the paper PR #252 gates. PR #252's standing
fix order says "state the convention in the question first"; these
two papers say the question text is the definition channel, the
weak carrier. This toy gates the PLACEMENT consequence: statement
in the definition vs statement in the label.

Toy operationalization: a FIXED typed-scorer stand-in with two
channels, exactly as #252's stand-in is fixed across its arms.
Items x in R^8, plain score s = w . x (s ~ N(0, 1)); the plain rule
R: label 1 iff s > TAU (TAU = 0.7). A marker (25% of items) defines
subset S, where a hidden convention C flips the label (as #252).

  Channels: the scorer reads the item through a LABEL channel
  z_L = s + N(0, SIG_L) and a DEFINITION channel z_D = s + N(0,
  SIG_D), and decides 1[E > 0] on the combined evidence
  E = W_L * z_L + W_D * z_D (default threshold 0, as #252).

  STATED PREMISE (open, fitted qualitatively from the papers --
  the toy gates the consequence structure GIVEN this premise; the
  papers supply the premise for real models): the scorer's
  training pool binds behaviour to names, so the label channel is
  the dominant, low-noise carrier (W_L = 0.75, SIG_L = 0.35) and
  the definition channel the weaker, noisier one (W_D = 0.25,
  SIG_D = 1.0). Content stated in a channel can only negate THAT
  channel's contribution: a convention stated in the definition
  flips the definition channel's sign on S (E = W_L z_L - W_D z_D
  on S), a convention carried in the label flips the label
  channel's sign on S (E = -W_L z_L + W_D z_D on S). One mechanism,
  every arm; arms differ only in channel signs / presence.

  Arms (same generated items, paired across arms):
    base     no convention stated anywhere.
    full     label + definition, convention in the DEFINITION.
    deleted  definitions deleted: labels only (E = z_L), no
             convention (it lived in the deleted text).
    defs-only  definitions only (E = z_D), convention in the
             definition: with no label channel competing, the
             definition's flip governs its own output outright.
    label    convention in the LABEL (the option is renamed so
             the label carries C on S); the definition states
             only the base rule.
    reassign name <-> definition mapping swapped (Sun): the
             definition channel's sign flips on EVERY item;
             scored against definition-truth (1 - y) and against
             name-truth (y).
    oracle   the same combined readout with the true rule (flip
             on S, threshold TAU) -- the ceiling for this scorer.

  Apparatus control: a definition-weighted scorer variant with
  the channel parameters SWAPPED (W_L = 0.25, SIG_L = 1.0,
  W_D = 0.75, SIG_D = 0.35), run through the identical arm code.
  It must show the OPPOSITE placement ordering, a real deletion
  cost, and no reassignment collapse. If it did not, the arms
  themselves -- not the stated channel premise -- would be
  producing the ordering, and the toy would be rigged.

Metrics per seed (4,000 items): accuracy per arm; on-S accuracy;
S-subset gap closure = (arm_S - base_S) / (oracle_S - base_S) for
the two placement arms; reassignment accuracies against both
truths.

Measured on pop-os (particlegan 0.10.1, 3 seeds), main scorer:
full 0.615-0.627 vs labels-only 0.611-0.620 (|difference| <=
0.007 -- Azizi's field difference was 0.0072); base on-S accuracy
0.247-0.272; convention in the definition: on-S 0.263-0.280, gap
closed 0.014-0.025; convention in the label: on-S 0.720-0.737,
gap closed 0.719-0.763; definitions-only on-S 0.661-0.714 (the
same definition text lands once the label stops competing);
reassignment: definition-truth 0.400-0.411 (below chance) vs
name-truth 0.589-0.600. Apparatus control (channels swapped):
definition-carried closure 0.650-0.751 vs label-carried
0.017-0.082; deletion cost 0.135-0.155; reassigned definition-
truth 0.594-0.603 (no collapse). Assertions below encode exactly
these margins, no more.
Skips cleanly when the installed particlegan does not provide
the EMPF recipe. Pure numpy -- no GAN machinery, because the
claim is about a fixed scorer's channels, not training.
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


D = 8             # feature dimension
TAU = 0.7         # the plain rule's true threshold
P_S = 0.25        # convention-subset fraction (as #252)
N_TEST = 4000     # held-out items per seed (as #252)

# The scorer's fixed plain-rule weights (unit norm, so s ~ N(0, 1)).
W = np.array([1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
W = W / np.linalg.norm(W)

# Stated premise: channel weights and readout noises. The main
# scorer is label-weighted; the apparatus control swaps them.
MAIN = dict(w_l=0.75, sig_l=0.35, w_d=0.25, sig_d=1.0)
CTRL = dict(w_l=0.25, sig_l=1.0, w_d=0.75, sig_d=0.35)


def _evidence(z_l, z_d, m, ch, sign_l, sign_d, on_s_only=True):
    """Combined evidence with per-channel signs.

    sign = -1 on a channel means the content bound to that channel
    reverses its reading (a convention flip carried by the channel,
    or -- with on_s_only=False -- a swapped name<->definition
    binding). Signs apply on S only unless on_s_only is False.
    """
    if on_s_only:
        a_l = np.where(m == 1, float(sign_l), 1.0)
        a_d = np.where(m == 1, float(sign_d), 1.0)
    else:
        a_l = np.full(len(m), float(sign_l))
        a_d = np.full(len(m), float(sign_d))
    return ch["w_l"] * a_l * z_l + ch["w_d"] * a_d * z_d


def run_seed(seed, ch):
    """All arms for one seed under channel parameters ``ch``."""
    rng = np.random.default_rng(11000 + seed)
    x = rng.standard_normal((N_TEST, D))
    s = x @ W
    m = (rng.random(N_TEST) < P_S).astype(int)
    y_plain = (s > TAU).astype(int)
    y = np.where(m == 1, 1 - y_plain, y_plain)
    z_l = s + rng.normal(0.0, ch["sig_l"], N_TEST)
    z_d = s + rng.normal(0.0, ch["sig_d"], N_TEST)

    def pred(e):
        return (e > 0).astype(int)

    p_base = pred(_evidence(z_l, z_d, m, ch, 1, 1))
    p_full = pred(_evidence(z_l, z_d, m, ch, 1, -1))
    p_deleted = (z_l > 0).astype(int)
    p_defs = np.where(m == 1, (-z_d > 0), (z_d > 0)).astype(int)
    p_label = pred(_evidence(z_l, z_d, m, ch, -1, 1))
    p_reassign = pred(_evidence(z_l, z_d, m, ch, 1, -1,
                                on_s_only=False))
    e_full = ch["w_l"] * z_l + ch["w_d"] * z_d
    p_oracle = np.where(m == 1, 1 - (e_full > TAU).astype(int),
                        (e_full > TAU).astype(int))

    acc = lambda p, yy=y: float((p == yy).mean())  # noqa: E731
    acc_s = lambda p: float((p[m == 1] == y[m == 1]).mean())  # noqa: E731
    r = dict(base=acc(p_base), full=acc(p_full),
             deleted=acc(p_deleted), defs_only=acc(p_defs),
             label=acc(p_label), oracle=acc(p_oracle),
             base_s=acc_s(p_base), full_s=acc_s(p_full),
             deleted_s=acc_s(p_deleted), defs_s=acc_s(p_defs),
             label_s=acc_s(p_label), oracle_s=acc_s(p_oracle),
             reassign_def=acc(p_reassign, 1 - y),
             reassign_name=acc(p_reassign, y))
    denom = r["oracle_s"] - r["base_s"]
    r["closed_def"] = (r["full_s"] - r["base_s"]) / denom
    r["closed_label"] = (r["label_s"] - r["base_s"]) / denom
    return r


def test_deleting_definitions_leaves_accuracy_unchanged():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed, MAIN)
        # Azizi's deletion result, in miniature: the full scorer
        # (convention stated in the definition) and the labels-only
        # scorer sit within a small band of each other.
        assert abs(r["full"] - r["deleted"]) < 0.03, (
            f"seed {seed}: full {r['full']:.3f} vs labels-only "
            f"{r['deleted']:.3f} -- deleting the definitions moved "
            f"accuracy, so the label channel is not the carrier "
            f"the premise states")
        assert r["deleted"] > 0.58, (
            f"seed {seed}: labels-only accuracy {r['deleted']:.3f} "
            f"should be a real scorer, not chance")
        # The definitions are not illiterate: alone, with no label
        # channel competing, the SAME definition text's convention
        # governs outright and the subset is largely repaired.
        # The override is competition, not incomprehension.
        assert r["defs_s"] > r["full_s"] + 0.30, (
            f"seed {seed}: definitions-only on-S {r['defs_s']:.3f} "
            f"vs full {r['full_s']:.3f} -- alone, the definition's "
            f"convention should land")


def test_convention_lands_in_label_not_in_definition():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed, MAIN)
        # The base scorer's deficit lives on S (as #252).
        assert r["base_s"] < 0.35, (
            f"seed {seed}: base on-S accuracy {r['base_s']:.3f}")
        # Convention in the definition: the subset gap barely
        # moves -- the weak channel's flip cannot overturn the
        # combined decision the label channel dominates.
        assert r["closed_def"] < 0.15, (
            f"seed {seed}: definition-carried closure "
            f"{r['closed_def']:.3f}")
        assert r["full_s"] < 0.35, (
            f"seed {seed}: full on-S accuracy {r['full_s']:.3f}")
        # The same convention carried in the label closes most
        # of the subset gap.
        assert r["closed_label"] > 0.60, (
            f"seed {seed}: label-carried closure "
            f"{r['closed_label']:.3f}")
        assert r["label_s"] > 0.65, (
            f"seed {seed}: label-carried on-S {r['label_s']:.3f}")
        assert r["closed_label"] > r["closed_def"] + 0.50, (
            f"seed {seed}: placement gap label {r['closed_label']:.3f} "
            f"vs definition {r['closed_def']:.3f}")


def test_reassignment_collapses_definition_truth_accuracy():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed, MAIN)
        # Sun's reassignment result, in miniature: with the
        # name<->definition mapping swapped, the scorer's decisions
        # still track the NAME -- accuracy against name-truth stays
        # up while accuracy against definition-truth falls below
        # chance. The scorer is not reading the definitions it is
        # nominally bound by.
        assert r["reassign_def"] < 0.45, (
            f"seed {seed}: reassigned definition-truth accuracy "
            f"{r['reassign_def']:.3f} should be below chance")
        assert r["reassign_name"] > 0.55, (
            f"seed {seed}: reassigned name-truth accuracy "
            f"{r['reassign_name']:.3f} should stay up")
        assert r["reassign_name"] - r["reassign_def"] > 0.15, (
            f"seed {seed}: name/def truth gap "
            f"{r['reassign_name'] - r['reassign_def']:.3f}")


def test_apparatus_control_reverses_every_ordering():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed, CTRL)
        # Same arm code, channel parameters swapped: now the
        # DEFINITION carries the convention and the label does
        # not. If the main result came from the arm construction
        # rather than the stated channel premise, this control
        # would show the same ordering -- it must show the
        # opposite.
        assert r["closed_def"] > 0.55, (
            f"seed {seed}: control definition-carried closure "
            f"{r['closed_def']:.3f}")
        assert r["closed_label"] < 0.20, (
            f"seed {seed}: control label-carried closure "
            f"{r['closed_label']:.3f}")
        assert r["closed_def"] > r["closed_label"] + 0.45, (
            f"seed {seed}: control placement gap not reversed: "
            f"def {r['closed_def']:.3f} vs label "
            f"{r['closed_label']:.3f}")
        # Deleting the definitions now COSTS real accuracy --
        # deletion invariance was a property of the label-weighted
        # scorer, not of the toy.
        assert r["full"] - r["deleted"] > 0.08, (
            f"seed {seed}: control deletion cost "
            f"{r['full'] - r['deleted']:.3f}")
        # And reassignment no longer collapses definition-truth:
        # this scorer follows its definitions.
        assert r["reassign_def"] > 0.55, (
            f"seed {seed}: control reassigned definition-truth "
            f"{r['reassign_def']:.3f} should stay up")
