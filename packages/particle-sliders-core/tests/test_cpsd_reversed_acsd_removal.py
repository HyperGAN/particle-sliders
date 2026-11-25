"""CPU gate for the CPSD claim (arXiv:2611.31932): "Contrastive
On-Policy Self-Distillation for Unlearning" -- the privileged-vs-plain
context shift is treated as a knowledge direction and REVERSED to
build a counterfactual target, distilled along the model's OWN
trajectories; the paper reports this beats NPO's forgetting-utility
trade-off (removal without the utility cost).

Provenance and anchors inside this watch:

  * PR #210 (ACSD) showed the forward construction at toy scale: a
    direction extracted by contrasting privileged against plain
    trajectories has cos 0.979-0.989 to the true latent axis, and
    distilling the privileged-context teacher INTO a student
    installs the trait the direction points at. CPSD is that
    construction run in REVERSE as a removal mechanism, so this
    test first installs a trait in a student by exactly the #210
    move (distill the privileged-context teacher), then tries to
    remove it.
  * PR #288 (the balloon effect) prices the removal claim: output
    bans RELOCATE a trait (banned-channel screen 0.000 while the
    unbanned screen rises +0.095..+0.269, cross-channel loading
    conserved 0.996-1.047x); only a writer-side cut removes. A
    reversed-target distill counts as a writer-side cut only if it
    rewrites the writer's distribution ON-POLICY -- which is
    exactly CPSD's design. This toy tests whether that pricing is
    right: the same reversed target distilled on FIXED
    pre-generated states (off-policy) is predicted to land
    measurably short of the on-policy distill.

Toy operationalization. Contexts: 216 discrete contexts, 24
"common" and 192 "tail". Each context c has a known task-correct
token k_c in a vocabulary of 6; tokens 0 and 1 are trait-preferred.
One scripted teacher under two context conditions (CPSD's setup:
the same model, privileged vs plain context):

  plain logits(c):      trait tokens +0.3, correct token +1.5,
                        other tokens 0.0 (both output channels).
                        The plain model still expresses the trait
                        (per-step trait rate ~0.26): the knowledge
                        is IN the plain model; privilege elicits
                        more of it. This floor is what separates
                        "distill back to plain" from removal.
  privileged logits(c): plain + alpha_c on the trait tokens,
                        alpha = 1.0 (common) / 2.8 (tail).

The knowledge direction is the contrast, per context,
delta_c = privileged(c) - plain(c) (exact here because the
teacher is scripted -- a modeling gift, stated plainly; in #210
the same contrast had to be recovered from sampled activations
and landed at cos 0.979-0.989). The CPSD counterfactual target
reverses it: cf(c) = plain(c) - delta_c, whose trait logits
(+0.3 - alpha_c) sit BELOW the plain floor -- the reversal
signature: expression after CPSD should fall under the plain
model's own expression, not merely back to it. The task
component of the target is plain's, untouched.

The STUDENT is factorized the way #288's writer/reader split
demands: logits(c, ch) = plain_task(c, ch) + tau_c * pattern,
where pattern is +1 on the trait tokens and tau_c is a single
per-context WRITER scalar shared by both channels; the two
channels (A: the trained channel; B: a held-out second format)
are readers of the same writer. The task component is the
student's pretrained plain capability and is FROZEN throughout:
everything learned in this toy -- installation and removal
alike -- is learned at the writer, which is the seat both
anchor papers identify. Installation (the #210 move): distill
the privileged teacher into the student over all contexts
(MSE on logits, full batch, tau the only trainable) -- tau_c
rises to ~alpha_c and the trait is installed on both channels.

Trajectories couple expression to state visitation, which is
what makes on-policy vs off-policy decidable at toy scale. A
rollout starts in a common context; each step samples both
channels' tokens, and the session state advances on channel
B's token (A and B are distributionally identical readers of
the same writer, so the choice is without loss before any
ban; under a ban on A it is exactly the balloon mechanism in
trajectory form -- the session continues from the format that
was actually emitted). A trait token enters the tail (a
uniform tail context) or, already in the tail, STAYS in the
same tail context (sticky tail: trait expression holds the
session in the context that elicits it); a correct token
advances within the common set or exits the tail; a neutral
token drifts within the region. A trait-carrying student
therefore spends most of its own trajectories inside single
tail contexts, while plain-teacher trajectories enter the
tail at the plain rate, leave it quickly (the correct token
dominates there), and -- spread over 192 tail contexts -- cover
any one of them thinly. The FIXED off-policy dataset is
pre-generated once from plain-teacher rollouts (32
trajectories, ~512 steps -- a small static corpus of the
kind offline distillation actually holds): a substantial
minority of tail contexts never appear in it at all, and
most appear only a handful of times.

Removal arms, matched budget (20 rounds x 32 trajectories x
16 steps of visited contexts, 20 SGD steps per round on the
round's visited-context batch, loss = forward KL from the
target's channel-A distribution to the student's, the #210
distillation loss; only channel A is ever trained during
removal -- channel B is the held-out balloon check, carried
only by the shared writer tau_c):

  (1) CPSD on-policy: counterfactual target cf(c); each round's
      states are FRESH rollouts of the current student.
  (2) CPSD off-policy: same target, same budget; states sampled
      with replacement from the FIXED plain-teacher pool.
  (3) BAN control: no weight change; at inference the trait
      tokens are masked on channel A only, and a masked-out
      step's expression falls through to channel B (an extra
      B emission for that step). Expect #288's signature: A
      screen 0, B trait emissions per step rise sharply (the
      fallthrough emissions concentrate in high-trait
      contexts; the per-emission B rate rises only slightly,
      +0.003..+0.016, because both channels read the same
      writer at the same rate), cross-channel loading
      conserved exactly (the writer was never touched).
  (4) SHUFFLED-direction control: target = plain(c) -
      ||delta_c|| * u_c, u_c an independent per-seed,
      per-context random unit vector in token space (a random
      direction reversed instead of the trait direction),
      distilled on-policy like arm (1). Its trait component is
      ~0 in expectation, so it can strip at most the privileged
      amplification; the base trait -- the plain floor -- is
      what it should leave standing.

Readouts per seed (5 seeds), on the student's OWN eval
rollouts (384 trajectories x 16 steps) unless stated:
  * trait expression per channel: fraction of sampled tokens
    that are trait-preferred, on A and on held-out B (for the
    ban, B also reports trait emissions per step, where the
    fallthrough inflation is largest);
  * cross-channel LOADING: per-channel displacement
    d_ch = mean over ALL contexts (fixed probe set) of
    (mean trait-token logit - mean non-trait-token logit);
    excess loading = (d_A + d_B) - (plain d_A + plain d_B)
    = 2 * mean(tau), the part of the loading the writer
    carries;
  * UTILITY: mean probability of the task-correct token under
    the student's channel-A distribution over all contexts
    (fixed probe set) -- the capability the paper says must
    survive removal;
  * PLAIN FLOOR: the plain teacher's own eval-rollout trait
    expression, the line CPSD must cross downward and the
    shuffled control should not.

PROVEN shape asserted below: CPSD removes on BOTH channels,
crossing below the plain floor, with utility retained (it
returns to at least the plain level -- removing the trait
frees the probability mass the trait was stealing); the ban
relocates (A screen clean, B rises, loading conserved);
off-policy CPSD lands measurably short of on-policy on both
channels and on the loading; the shuffled direction leaves
the trait at or above the plain floor. If off-policy had
matched on-policy, that would have revised #288's pricing --
the assertions encode the demonstrated margins only, with
modest headroom, and the measured table in the PR body is
the primary evidence either way.

Measured on pop-os (particlegan 0.10.1), 5 seeds
(installed exprA / plain floor / CPSD exprA / CPSD excess /
CPSD utility / off-policy exprA / off-policy excess gap /
ban B per-step rise / shuffled exprA):
  seed 0: 0.806 / 0.272 / 0.164 / -0.697 / 0.473 / 0.327 /
          +0.994 / +0.669 / 0.447
  seed 1: 0.806 / 0.261 / 0.162 / -0.671 / 0.472 / 0.331 /
          +1.135 / +0.656 / 0.421
  seed 2: 0.806 / 0.263 / 0.163 / -0.703 / 0.473 / 0.333 /
          +1.002 / +0.659 / 0.439
  seed 3: 0.809 / 0.255 / 0.163 / -0.723 / 0.474 / 0.325 /
          +1.283 / +0.660 / 0.452
  seed 4: 0.799 / 0.267 / 0.162 / -0.760 / 0.475 / 0.323 /
          +1.155 / +0.648 / 0.424
Installed excess loading is +5.200 on every seed (the
construction's mean alpha x 2 channels) and installed
utility 0.110 (plain 0.440); the ban's A screen is 0.000
and its excess ratio exactly 1.000 on every seed. One
mechanistic note the numbers carry: on contexts BOTH arms
cover, tau stalls near -0.9 rather than the -2.8 target --
probability-space KL cannot see the remaining logit gap
once trait mass is ~0.01-0.06 -- so the on/off-policy
separation is a coverage effect, not a per-context depth
effect, which is exactly #288's pricing in this toy's terms.

What is NOT asserted: the paper's exact numbers, anything
about real LM unlearning (contexts here are discrete and the
trait is a single planted writer scalar per context, where a
real trait is distributed), and NPO itself (no NPO arm; the
claim tested is CPSD's mechanism -- reversed direction,
on-policy distillation -- against its own ablations, with the
ban arm standing in for surface-level forgetting methods).
Pure torch toy on CPU, single-threaded, seeded. Skips cleanly
when the installed particlegan does not provide the EMPF
recipe (the house guard for this paper-test family, PR #273's
guarded-import pattern).
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

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
N_COMMON = 24
N_TAIL = 192
NC = N_COMMON + N_TAIL   # contexts: 0..23 common, 24..215 tail
V = 6                    # vocabulary; tokens 0,1 trait-preferred
TRAIT = (0, 1)
TRAIT_LOGIT = 0.3        # plain-context trait logit (the plain floor)
CORRECT_LOGIT = 1.5
ALPHA_COMMON = 1.0
ALPHA_TAIL = 2.8
STEPS = 16               # trajectory length
SEEDS = (0, 1, 2, 3, 4)

# Removal budget (identical for arms 1, 2 and 4).
ROUNDS = 20
TRAJ_PER_ROUND = 32
SGD_STEPS = 20
SGD_LR = 6.0
POOL_TRAJ = 32           # fixed off-policy pool (plain-teacher rollouts)
EVAL_TRAJ = 384


def _correct_token(c):
    return 2 + (c % 3)


def _alpha(c):
    return ALPHA_COMMON if c < N_COMMON else ALPHA_TAIL


def plain_logits():
    """(NC, V) plain-context teacher logits (same both channels)."""
    lg = np.zeros((NC, V), dtype=np.float64)
    for c in range(NC):
        lg[c, list(TRAIT)] = TRAIT_LOGIT
        lg[c, _correct_token(c)] = CORRECT_LOGIT
    return lg


def priv_logits():
    lg = plain_logits()
    for c in range(NC):
        lg[c, list(TRAIT)] += _alpha(c)
    return lg


PLAIN = plain_logits()          # (NC, V)
PRIV = priv_logits()            # (NC, V)
DELTA = PRIV - PLAIN            # the privileged-vs-plain direction
CF = PLAIN - DELTA              # CPSD counterfactual target (reversed)
PATTERN = np.zeros(V)
PATTERN[list(TRAIT)] = 1.0
NON_TRAIT = [i for i in range(V) if i not in TRAIT]


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


PLAIN_PROBS = np.stack([_softmax(PLAIN[c]) for c in range(NC)])


# ---------------------------------------------------------------- student
class Student:
    """Factorized student: frozen plain task component plus one
    writer scalar tau (NC,) shared across channels."""

    def __init__(self, tau_np):
        self.tau = torch.tensor(tau_np, dtype=torch.float32,
                                requires_grad=True)

    def logits_np(self):
        with torch.no_grad():
            lg = PLAIN[:, None, :] + self.tau.numpy()[:, None, None] \
                * PATTERN[None, None, :]
            return np.broadcast_to(lg, (NC, 2, V)).astype(np.float64)


def install_student(seed):
    """The #210 move: distill the PRIVILEGED teacher into the
    student over all contexts (MSE on logits, full batch; the
    writer tau is the only trainable -- the task component is
    the student's frozen plain capability)."""
    rng = np.random.default_rng(7000 + seed)
    stu = Student(rng.normal(0, 0.05, NC))
    target = torch.tensor(PRIV, dtype=torch.float32)
    plain_t = torch.tensor(PLAIN, dtype=torch.float32)
    pat = torch.tensor(PATTERN, dtype=torch.float32)
    opt = torch.optim.Adam([stu.tau], lr=0.05)
    for _ in range(300):
        logits = plain_t + stu.tau[:, None] * pat[None, :]
        loss = ((logits - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return stu


def clone_student(stu):
    return Student(stu.tau.detach().numpy().copy())


# ------------------------------------------------------------- trajectories
def _next_context(c, tok, rng):
    if tok in TRAIT:
        # Trait expression: enter the tail from common; in the
        # tail, stay in the SAME context (sticky tail).
        if c < N_COMMON:
            return N_COMMON + int(rng.integers(N_TAIL))
        return c
    if tok == _correct_token(c):
        # Task progress: advance within common, or exit the tail.
        return (c + 1) % N_COMMON if c < N_COMMON else c - N_COMMON
    # Neutral token: drift within the region.
    if c < N_COMMON:
        return int(rng.integers(N_COMMON))
    return N_COMMON + int(rng.integers(N_TAIL))


def rollout_contexts(probs_b, rng, n_traj):
    """Rollouts whose state advances on channel-B tokens drawn
    from probs_b (NC, V). Returns visited contexts
    (n_traj, STEPS)."""
    ctx = np.empty((n_traj, STEPS), dtype=np.int64)
    cur = rng.integers(0, N_COMMON, size=n_traj)
    for t in range(STEPS):
        ctx[:, t] = cur
        for i in range(n_traj):
            tok = int(rng.choice(V, p=probs_b[cur[i]]))
            cur[i] = _next_context(int(cur[i]), tok, rng)
    return ctx


def student_probs(stu):
    lg = stu.logits_np()
    return np.stack([_softmax(lg[c, 0]) for c in range(NC)])


# ---------------------------------------------------------------- removal
def _distill_round(stu, contexts, target_logits_a):
    """Forward-KL distillation on channel A over the round's
    visited contexts (weighted by visit counts); only the
    writer tau moves."""
    counts = np.bincount(contexts.ravel(), minlength=NC).astype(np.float64)
    visited = counts > 0
    if not visited.any():
        return
    idx = torch.tensor(np.where(visited)[0], dtype=torch.long)
    w = torch.tensor(counts[visited] / counts[visited].sum(),
                     dtype=torch.float32)
    tgt = torch.tensor(target_logits_a[visited], dtype=torch.float32)
    q = F.log_softmax(tgt, dim=-1).exp()
    plain_t = torch.tensor(PLAIN[visited], dtype=torch.float32)
    pat = torch.tensor(PATTERN, dtype=torch.float32)
    opt = torch.optim.SGD([stu.tau], lr=SGD_LR)
    for _ in range(SGD_STEPS):
        logits = plain_t + stu.tau[idx, None] * pat[None, :]
        logp = F.log_softmax(logits, dim=-1)
        loss = -(q * logp).sum(-1)
        loss = (loss * w).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()


def run_removal(installed, seed, arm):
    stu = clone_student(installed)
    if arm == "cpsd_off":
        target = CF
        # Fixed pool: plain-teacher rollouts, generated ONCE.
        rng_pool = np.random.default_rng(9100 + seed)
        pool = rollout_contexts(PLAIN_PROBS, rng_pool, POOL_TRAJ)
    elif arm == "cpsd_on":
        target = CF
        pool = None
    elif arm == "shuffled":
        rng_dir = np.random.default_rng(5500 + seed)
        u = rng_dir.normal(size=(NC, V))
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        norms = np.linalg.norm(DELTA, axis=1)  # (NC,)
        target = PLAIN - norms[:, None] * u
        pool = None
    else:
        raise ValueError(arm)
    for r in range(ROUNDS):
        if arm == "cpsd_off":
            rng = np.random.default_rng(32000 + seed * 1000 + r)
            take = rng.integers(0, pool.shape[0], size=TRAJ_PER_ROUND)
            ctx = pool[take]
        else:
            # On-policy states: fresh rollouts of the current
            # student (the shuffled arm isolates the DIRECTION,
            # so it shares CPSD's state distribution exactly).
            rng = np.random.default_rng(31000 + seed * 1000 + r)
            ctx = rollout_contexts(student_probs(stu), rng, TRAJ_PER_ROUND)
        _distill_round(stu, ctx, target)
    return stu


# ---------------------------------------------------------------- readouts
def eval_rollouts_logits(lg, seed, ban_a=False):
    """Sampled rollouts given (NC, 2, V) logits. The session
    state advances on the channel-B token. With ban_a=True,
    channel-A trait tokens are masked at inference and a
    masked-out step's expression falls through to channel B
    (one extra B emission for that step)."""
    probs = np.empty((NC, 2, V))
    for c in range(NC):
        for ch in (0, 1):
            probs[c, ch] = _softmax(lg[c, ch])
    rng = np.random.default_rng(88000 + seed)
    trait_a = trait_b = n_b = b_per_step = correct_a = steps = 0
    for _ in range(EVAL_TRAJ):
        c = int(rng.integers(0, N_COMMON))
        for _t in range(STEPS):
            pa = probs[c, 0].copy()
            fallthrough = False
            if ban_a:
                if rng.random() < pa[list(TRAIT)].sum():
                    fallthrough = True
                pa[list(TRAIT)] = 0.0
                pa = pa / pa.sum()
            tok_a = int(rng.choice(V, p=pa))
            trait_a += int(tok_a in TRAIT)
            correct_a += int(tok_a == _correct_token(c))
            tok_b = int(rng.choice(V, p=probs[c, 1]))
            trait_b += int(tok_b in TRAIT)
            n_b += 1
            b_here = int(tok_b in TRAIT)
            if fallthrough:
                tok_b2 = int(rng.choice(V, p=probs[c, 1]))
                trait_b += int(tok_b2 in TRAIT)
                n_b += 1
                b_here += int(tok_b2 in TRAIT)
            b_per_step += b_here
            c = _next_context(c, tok_b, rng)
            steps += 1
    return {
        "expr_a": trait_a / steps,
        "expr_b": trait_b / n_b,
        "b_per_step": b_per_step / steps,
        "correct_a": correct_a / steps,
    }


def probe_logits(lg):
    """Fixed-probe readouts over ALL contexts: per-channel
    displacement and channel-A correct-token probability."""
    disps = []
    for ch in (0, 1):
        d = (lg[:, ch][:, list(TRAIT)].mean(axis=1)
             - lg[:, ch][:, NON_TRAIT].mean(axis=1))
        disps.append(float(d.mean()))
    util = float(np.mean([
        _softmax(lg[c, 0])[_correct_token(c)] for c in range(NC)]))
    return {"disp_a": disps[0], "disp_b": disps[1], "utility": util}


PLAIN_LG = np.broadcast_to(PLAIN[:, None, :], (NC, 2, V)).copy()
PLAIN_PROBE = probe_logits(PLAIN_LG)


def excess_loading(pr):
    return (pr["disp_a"] + pr["disp_b"]) - (PLAIN_PROBE["disp_a"]
                                            + PLAIN_PROBE["disp_b"])


_CACHE = {}


def run_seed(seed):
    if seed in _CACHE:
        return _CACHE[seed]
    installed = install_student(seed)
    out = {}
    out["plain_floor"] = eval_rollouts_logits(PLAIN_LG, seed)["expr_a"]
    out["installed"] = {**eval_rollouts_logits(installed.logits_np(), seed),
                        **probe_logits(installed.logits_np())}
    out["ban"] = {**eval_rollouts_logits(installed.logits_np(), seed,
                                         ban_a=True),
                  **probe_logits(installed.logits_np())}
    for arm in ("cpsd_on", "cpsd_off", "shuffled"):
        stu = run_removal(installed, seed, arm)
        out[arm] = {**eval_rollouts_logits(stu.logits_np(), seed),
                    **probe_logits(stu.logits_np())}
    for key in ("installed", "ban", "cpsd_on", "cpsd_off", "shuffled"):
        out[key]["excess"] = excess_loading(out[key])
    _CACHE[seed] = out
    return out


# ------------------------------------------------------------------- tests
def test_trait_is_installed_by_acsd_distillation():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)["installed"]
        assert r["expr_a"] >= 0.55, (
            f"seed {seed}: installed A expression {r['expr_a']:.3f}")
        assert r["expr_b"] >= 0.55, (
            f"seed {seed}: installed B expression {r['expr_b']:.3f}")
        assert r["excess"] >= 3.5, (
            f"seed {seed}: installed excess loading {r['excess']:.3f}")


def test_cpsd_onpolicy_removes_below_plain_floor_with_utility():
    _empf_recipe()
    for seed in SEEDS:
        res = run_seed(seed)
        inst, arm, floor = res["installed"], res["cpsd_on"], res["plain_floor"]
        assert arm["expr_a"] <= 0.18, (
            f"seed {seed}: CPSD A expression {arm['expr_a']:.3f}")
        assert arm["expr_a"] <= inst["expr_a"] - 0.40, (
            f"seed {seed}: CPSD A {arm['expr_a']:.3f} vs installed "
            f"{inst['expr_a']:.3f}")
        # The reversal signature: BELOW the plain model's own
        # expression floor, on the trained channel and on the
        # held-out channel alike.
        assert arm["expr_a"] <= floor - 0.06, (
            f"seed {seed}: CPSD A {arm['expr_a']:.3f} vs plain floor "
            f"{floor:.3f}")
        assert arm["expr_b"] <= 0.18, (
            f"seed {seed}: CPSD B expression {arm['expr_b']:.3f}")
        assert arm["expr_b"] <= floor - 0.06, (
            f"seed {seed}: CPSD B {arm['expr_b']:.3f} vs plain floor "
            f"{floor:.3f}")
        assert arm["excess"] <= 0.25 * inst["excess"], (
            f"seed {seed}: CPSD excess {arm['excess']:.3f} vs installed "
            f"{inst['excess']:.3f}")
        # The paper's trade-off claim: removal WITHOUT utility
        # cost -- utility returns to at least the plain level.
        assert arm["utility"] >= PLAIN_PROBE["utility"] - 0.03, (
            f"seed {seed}: CPSD utility {arm['utility']:.3f} vs plain "
            f"{PLAIN_PROBE['utility']:.3f}")
        assert arm["utility"] >= inst["utility"] + 0.10, (
            f"seed {seed}: CPSD utility {arm['utility']:.3f} vs "
            f"installed {inst['utility']:.3f}")


def test_ban_relocates_and_shuffled_leaves_the_base_trait():
    _empf_recipe()
    for seed in SEEDS:
        res = run_seed(seed)
        inst, ban, shuf = res["installed"], res["ban"], res["shuffled"]
        floor = res["plain_floor"]
        # Ban: trained channel screens perfectly clean...
        assert ban["expr_a"] <= 0.01, (
            f"seed {seed}: ban A screen {ban['expr_a']:.4f}")
        # ...the trait bulges out of the unbanned channel: its
        # per-step trait emissions rise sharply (fallthrough
        # concentration; demonstrated +0.648..+0.669) and its
        # per-emission rate does not fall (demonstrated rise
        # +0.003..+0.016 -- small, because both channels read
        # the same writer at the same rate).
        assert ban["expr_b"] >= inst["expr_b"] - 0.002, (
            f"seed {seed}: ban B {ban['expr_b']:.3f} vs installed "
            f"{inst['expr_b']:.3f}")
        assert ban["b_per_step"] >= inst["b_per_step"] + 0.25, (
            f"seed {seed}: ban B/step {ban['b_per_step']:.3f} vs "
            f"installed {inst['b_per_step']:.3f}")
        # ...and the cross-channel loading is conserved (no
        # weight change, so the probe is the installed probe).
        ratio = ban["excess"] / inst["excess"]
        assert 0.90 <= ratio <= 1.10, (
            f"seed {seed}: ban excess ratio {ratio:.3f}")
        # Shuffled direction: reversing a random direction does
        # NOT cross below the plain floor -- the base trait
        # (the knowledge itself) is left standing.
        assert shuf["expr_a"] >= floor - 0.04, (
            f"seed {seed}: shuffled A {shuf['expr_a']:.3f} vs plain "
            f"floor {floor:.3f}")
        assert shuf["expr_a"] >= res["cpsd_on"]["expr_a"] + 0.15, (
            f"seed {seed}: shuffled A {shuf['expr_a']:.3f} vs CPSD "
            f"{res['cpsd_on']['expr_a']:.3f}")


def test_offpolicy_lands_short_of_onpolicy():
    _empf_recipe()
    for seed in SEEDS:
        res = run_seed(seed)
        on, off = res["cpsd_on"], res["cpsd_off"]
        # #288's pricing: the on-policy ingredient is load-bearing.
        assert off["expr_a"] >= on["expr_a"] + 0.10, (
            f"seed {seed}: off-policy A {off['expr_a']:.3f} vs "
            f"on-policy {on['expr_a']:.3f}")
        assert off["expr_b"] >= on["expr_b"] + 0.10, (
            f"seed {seed}: off-policy B {off['expr_b']:.3f} vs "
            f"on-policy {on['expr_b']:.3f}")
        assert off["excess"] >= on["excess"] + 0.75, (
            f"seed {seed}: off-policy excess {off['excess']:.3f} vs "
            f"on-policy {on['excess']:.3f}")
        # Utility follows the same ordering: the removal that
        # misses the tail leaves the tail's task distribution
        # trait-distorted.
        assert on["utility"] >= off["utility"] - 0.02, (
            f"seed {seed}: on-policy utility {on['utility']:.3f} vs "
            f"off-policy {off['utility']:.3f}")
