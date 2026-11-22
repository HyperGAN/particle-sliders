"""CPU gate for the KL-free on-policy distillation claim of
Lin, Long, Zhao, Wang, Jiang, Luo, Lan, Ma, Wen, Wu, Chen,
Liu, Yan & Huang, "Do We Really Need KL Divergence for
On-Policy Distillation of Large Language Models?"
(arXiv:2609.33791). The paper's claim has three parts.
(a) KL divergence is NOT necessary for on-policy
distillation: assigning a reward of +1 to tokens where
the teacher's probability exceeds the student's and -1
where it is lower -- pure update DIRECTION toward the
teacher, no magnitude information at all -- reproduces
almost the same training mode as reverse-KL OPD.
(b) Only the update direction on a SMALL SUBSET of
tokens with large teacher--student disagreement is
critical: training still works when those tokens are
pushed toward the teacher even if ALL OTHER tokens are
pulled AWAY from the teacher. (c) Application (not
tested here, noted in the PR): Consensus Multi-Teacher
OPD, where every sample is supervised by all teachers,
beats routed multi-teacher OPD. Provenance: cycle 19 of
the Particle Sliders paper watch; the objective-level
sibling of the divergence-token screen (PR #275), where
the tokens that carry transfer are the high-divergence
ones.

Toy operationalization: the smallest exactly enumerable
on-policy distillation world. C = 8 contexts, V = 12
tokens, position-wise categorical distributions. The
teacher is fixed and peaked per context: one token at
0.80, a second at 0.18, the remaining ten sharing 0.02
(0.002 each). The student is parameterized by per-context
logits (softmax) and initialized with the teacher's top
two masses SWAPPED (0.18 on the teacher's top token,
0.80 on its second) and the tail exact -- so the initial
forward KL is exactly (0.80 - 0.18) * log(0.80 / 0.18)
= 0.925 on every seed, and the ONLY initial disagreement
lives on the top-two tokens of each context. Training is
ON-POLICY: at each step, one token is sampled from the
student per context, scored against the teacher, and the
logits take a policy-gradient (score-function) step
lr * r * (onehot(sample) - p_student) with the arm's
reward r. Same step budget (300 steps, lr = 0.02) and
the same sampling stream for every arm of a seed.

Arms and controls:
  (1) reverse-KL OPD (the standard objective): reward
      r = log p_teacher(sample) - log p_student(sample)
      -- magnitude-aware, the gradient signal of the
      reverse-KL objective on sampled tokens;
  (2) sign-reward OPD (claim (a)): r = sign of the same
      log-ratio, +1 / -1 / 0 -- direction only, every
      magnitude discarded;
  (3) disagreement-subset sign + anti-rest (claim (b)):
      the subset is the top-2 of 12 tokens (16.7%) per
      context by |log p_teacher - log p_student| AT
      INITIALIZATION, fixed for the run; sampled tokens
      in the subset get the toward-teacher sign reward,
      every other sampled token gets the REVERSED
      (anti-teacher) sign reward;
  (4) random-subset control: a same-size (2-token) subset
      drawn at random per context gets the toward-teacher
      sign and every other token the anti-teacher sign.
      It must fail where arm (3) succeeds -- the subset
      has to be the disagreement subset, not any subset.

Why the subset is fixed at initialization, stated
plainly: an earlier design round recomputed the subset
from the CURRENT student at every step; that variant
failed (final KL ~0.33-0.65, near or above half the
initial divergence), because once the top tokens
converge, the recomputed subset chases the tail tokens
whose disagreement was CREATED by the anti-teacher
updates, and the true top tokens fall into the anti set
and are pushed away. The fixed-at-initialization subset
is the operationalization that decides claim (b) in
this toy class; whether per-step reselection with a
better rule also works is not asserted. The toy's other
structural gift, also plain: the teacher's tail mass is
tiny (0.02 total), so anti-teacher damage on tail tokens
is bounded in forward KL by their teacher mass, while
the random control's damage is NOT bounded that way --
it puts a teacher-top token (sampled ~80% of the time)
in the anti set on most contexts and pushes it away
constantly. That asymmetry is the mechanism the paper's
claim (b) names: direction on the disagreement tokens
carries the distillation; direction elsewhere is, at
worst, a bounded disturbance.

Readouts: exact forward KL and total variation between
the student's and teacher's per-context distributions
(enumerable, no sampling error in the readout), and the
training trajectory (exact forward KL every 25 steps)
for the KL and sign arms -- the "same training mode"
comparison of claim (a).

Measured on pop-os (particlegan 0.10.1), 5 seeds
(initial KL / KL-arm final / sign final / subset final /
random final, all forward KL):
  seed 0: init 0.925, kl 0.0052, sign 0.0000,
          subset 0.0001, random 1.8844
  seed 1: init 0.925, kl 0.0050, sign 0.0000,
          subset 0.0001, random 1.8160
  seed 2: init 0.925, kl 0.0048, sign 0.0000,
          subset 0.0001, random 1.8257
  seed 3: init 0.925, kl 0.0051, sign 0.0000,
          subset 0.0001, random 1.4665
  seed 4: init 0.925, kl 0.0061, sign 0.0000,
          subset 0.0001, random 1.8828
(Sign finals print as 0.0000: all are below 0.0001.
Final TV tells the same story: KL arm 0.040-0.046, sign
0.001-0.003, subset 0.003, random 0.68-0.74.)

The arbitration, stated plainly: both tested parts of
the claim reproduce at toy scale. Claim (a): the KL arm
ends at 0.0048-0.0061 (about 0.6% of the initial 0.925)
and the sign arm ends BELOW it, under 0.0001 on every
seed -- discarding all magnitude information does not
cost the endpoint anything; if anything the sign reward
keeps pushing after the KL reward (proportional to the
shrinking log-ratio) has nearly vanished. The training
mode matches too: both trajectories cross 75% of the
initial divergence by step 50 (KL 0.41-0.44, sign
0.52-0.56), 25% by step 100 (KL 0.15-0.16, sign
0.17-0.19) and 5% by step 200 (KL 0.022-0.025, sign at
most 0.0001) -- the sign arm is a little slower early,
when the large initial gap is exactly the magnitude
information it throws away, and faster late. Claim (b):
the subset + anti-rest arm ends at 0.0001 on every seed
(about 0.01% of initial) -- toward-teacher direction on
2 of 12 tokens carries the whole distillation while the
other 10 are actively pulled away -- while the
random-subset control ends at 1.47-1.88, 1.6-2.0x the
initial divergence: it does not merely fail to learn,
it unlearns. Assertions below encode exactly these
margins, no more.

What is NOT asserted: the paper's exact numbers, claim
(c) (Consensus Multi-Teacher OPD -- no second teacher
exists in this toy), and anything about real LM
distillation, where the disagreement subset is not
known at initialization, the teacher is not a swapped
copy of the student, sampled sequences couple tokens
across positions, and a sign reward interacts with a
deep network's shared parameters rather than a
per-context logit table. The subset here is exactly the
initial top-two by construction (the student's tail
starts exact), which is the cleanest possible case for
claim (b). Pure numpy -- no GAN machinery, because the
claim is about the distillation objective, not training
dynamics of a GAN. Skips cleanly when the installed
particlegan does not provide the EMPF recipe (the house
guard for this paper-test family; PR #211 is what enables
this test in CI).
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


C = 8                # contexts
V = 12               # tokens per context
P_TOP = 0.80         # teacher mass on its top token
P_SECOND = 0.18      # teacher mass on its second token
P_TAIL = 0.002       # teacher mass on each of 10 tail tokens
SUBSET = 2           # disagreement-subset size (top-2 of 12)
STEPS = 300          # on-policy steps, same budget every arm
LR = 0.02            # step size, same every arm
TRAJ_EVERY = 25      # exact-KL readout interval (steps)
SEED_BASE = 41000    # RNG stream base
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("kl", "sign", "subset", "random")


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fkl(teacher, student):
    return float(np.mean(np.sum(teacher * np.log(teacher / student),
                                axis=1)))


def _tv(teacher, student):
    return float(np.mean(0.5 * np.abs(teacher - student).sum(axis=1)))


def _world(rng):
    """Teacher (top two swapped for the student init),
    the fixed disagreement subset and the random subset."""
    teacher = np.full((C, V), P_TAIL)
    init = np.full((C, V), P_TAIL)
    top_sets, rand_sets = [], []
    for c in range(C):
        a, b = rng.choice(V, size=2, replace=False)
        teacher[c, a] = P_TOP
        teacher[c, b] = P_SECOND
        init[c, a] = P_SECOND
        init[c, b] = P_TOP
        dis = np.abs(np.log(teacher[c]) - np.log(init[c]))
        top_sets.append(set(np.argsort(-dis)[:SUBSET].tolist()))
        rand_sets.append(
            set(rng.choice(V, size=SUBSET, replace=False).tolist()))
    return teacher, init, top_sets, rand_sets


def _train(teacher, init, top_sets, rand_sets, arm, rng):
    logits = np.log(init).copy()
    traj = []
    for step in range(STEPS + 1):
        student = _softmax(logits)
        if step % TRAJ_EVERY == 0:
            traj.append(_fkl(teacher, student))
        if step == STEPS:
            break
        for c in range(C):
            ps = student[c]
            a = int(rng.choice(V, p=ps))
            diff = np.log(teacher[c, a]) - np.log(ps[a])
            if arm == "kl":
                r = diff
            elif arm == "sign":
                r = np.sign(diff)
            elif arm == "subset":
                r = np.sign(diff) if a in top_sets[c] \
                    else -np.sign(diff)
            elif arm == "random":
                r = np.sign(diff) if a in rand_sets[c] \
                    else -np.sign(diff)
            else:
                raise ValueError(arm)
            grad = -ps.copy()
            grad[a] += 1.0
            logits[c] += LR * r * grad
            student[c] = _softmax(logits[c:c + 1])[0]
    student = _softmax(logits)
    return {
        "final_kl": _fkl(teacher, student),
        "final_tv": _tv(teacher, student),
        "traj": traj,
    }


def run_seed(seed):
    """One seed: one world, all four arms trained from the
    same initialization with the same sampling stream."""
    world_rng = np.random.default_rng(SEED_BASE + seed)
    teacher, init, top_sets, rand_sets = _world(world_rng)
    out = {"init_kl": _fkl(teacher, init)}
    for arm in ARMS:
        train_rng = np.random.default_rng(SEED_BASE + 100 + seed)
        out[arm] = _train(teacher, init, top_sets, rand_sets,
                          arm, train_rng)
    return out


_CACHE = {}


def _seed(seed):
    if seed not in _CACHE:
        _CACHE[seed] = run_seed(seed)
    return _CACHE[seed]


def test_sign_reward_reproduces_kl_training_mode():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        init = res["init_kl"]
        kl, sign = res["kl"], res["sign"]
        # Both arms distill: final forward KL is a small
        # fraction of the initial 0.925 (demonstrated
        # KL 0.0048-0.0061, sign below 0.0001).
        assert kl["final_kl"] <= 0.02, (
            f"seed {seed}: KL arm final {kl['final_kl']:.4f}")
        assert sign["final_kl"] <= 0.02, (
            f"seed {seed}: sign arm final {sign['final_kl']:.4f}")
        # Claim (a): the sign arm lands where the KL arm
        # lands. Both finals sit near the readout floor,
        # where a ratio is meaningless, so the margin is
        # absolute: the sign final is within 0.01 of the
        # KL final (demonstrated: sign lands BELOW the
        # KL final on every seed, by 0.005-0.006).
        assert sign["final_kl"] <= kl["final_kl"] + 0.01, (
            f"seed {seed}: sign {sign['final_kl']:.4f} vs "
            f"KL {kl['final_kl']:.4f}")
        # Same training mode: both trajectories cross the
        # same divergence fractions at the same checkpoints
        # (traj index 2 = step 50, 4 = step 100, 8 = step
        # 200; demonstrated step-50 KL 0.41-0.44 / sign
        # 0.52-0.56, step-100 KL 0.15-0.16 / sign
        # 0.17-0.19, step-200 KL 0.022-0.025 / sign at
        # most 0.0001, against thresholds 0.694 / 0.231 /
        # 0.046).
        for arm in (kl, sign):
            assert arm["traj"][2] <= 0.75 * init, (
                f"seed {seed}: traj[50] {arm['traj'][2]:.3f}")
            assert arm["traj"][4] <= 0.25 * init, (
                f"seed {seed}: traj[100] {arm['traj'][4]:.3f}")
            assert arm["traj"][8] <= 0.05 * init, (
                f"seed {seed}: traj[200] {arm['traj'][8]:.3f}")


def test_disagreement_subset_survives_anti_teacher_rest():
    _empf_recipe()
    for seed in SEEDS:
        res = _seed(seed)
        init = res["init_kl"]
        subset, random = res["subset"], res["random"]
        # Claim (b): toward-teacher sign on the top-2
        # disagreement tokens alone carries the
        # distillation even with every other token pulled
        # AWAY -- final KL a small fraction of initial
        # (demonstrated 0.0001 on every seed).
        assert subset["final_kl"] <= 0.02, (
            f"seed {seed}: subset final {subset['final_kl']:.4f}")
        assert subset["final_kl"] <= 0.05 * init, (
            f"seed {seed}: subset final {subset['final_kl']:.4f} "
            f"vs init {init:.3f}")
        # Random-subset control: the same budget with a
        # random subset does NOT learn -- it stays at or
        # above the initial divergence (demonstrated
        # 1.4665-1.8844, 1.6-2.0x the initial 0.925).
        assert random["final_kl"] >= init, (
            f"seed {seed}: random final {random['final_kl']:.4f} "
            f"vs init {init:.3f}")
        # ... and the disagreement subset beats the random
        # control by a wide margin on the same world.
        assert subset["final_kl"] <= 0.05 * random["final_kl"], (
            f"seed {seed}: subset {subset['final_kl']:.4f} vs "
            f"random {random['final_kl']:.4f}")
