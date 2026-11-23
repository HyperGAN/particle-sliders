"""CPU gate for the liminal-training claims of Yanagisawa
et al., "On Mitigation of Subliminal Learning in LLMs"
(arXiv:2609.22215) -- PARTIAL verdict, with one claim
REVERSED in this toy class, stated plainly below.

The paper (1.5B-8B, Qwen/Gemma/Llama, number-sequence and
CoT settings) claims: (1) subliminal trait acquisition
tracked DURING fine-tuning is highly non-monotonic --
transient spikes and reversals, so a final-checkpoint
readout can miss the trajectory; (2) liminal training --
fine-tuning with a KL-to-base penalty (annealed, in the
paper) -- substantially reduces trait acquisition while
largely preserving task gains; (3) KL TIMING matters and
EARLY regularization beats LATE.

Toy operationalization: the coordinator prototypes of
cycle 18 (2 rounds, local scratch), rebuilt cleanly. A
2-layer linear net w_eff = W^T v (W: 16x32, v: 16, init
scale 0.1) is trained by SGD (lr 0.02, 160 steps) on a
mixed stream: task examples y = x . t_task (|t_task| = 2)
and carrier examples y = x . (t_task + 1.6 * tau), tau a
unit direction orthogonal to t_task. The trait loading
beta(t) = w_eff(t) . tau is logged EVERY step. Carriers
are denser mid-training (steps 50-89: 40 carrier vs 16 of
64 per batch) -- the burst that produces the transient
spike. The "KL-to-base" penalty is the Gaussian proxy used
throughout this family: lam * (|W - W_anchor|^2 +
|v - v_anchor|^2) / 2, anchored at the base init unless
stated. Arms share one data stream per seed (the data RNG
is reset per arm) and a matched total penalty integral
(lam_total = 400): none, const (lam_total / T throughout),
early (3x rate, first third), late (3x rate, last third),
and a specificity control: const penalty toward a
TRAIT-SHIFTED anchor, W_a with W_a^T v0 = W0^T v0 +
1.2 * tau -- an anchor that carries the trait, so a
penalty toward it should NOT suppress beta if suppression
is about anchoring to the (trait-free) BASE rather than
about any quadratic drag on learning.

What the toy shows (measured on pop-os, particlegan
0.10.1, 5 seeds; final beta / peak beta / task gain, task
gain = base task MSE - final task MSE on 2000 fresh x):

  seed:        0       1       2       3       4
  none final   0.388   0.386   0.395   0.429   0.375
  none peak    0.707   0.727   0.718   0.706   0.751
  const final  0.225   0.187   0.151   0.247   0.170
  early final  0.388   0.386   0.395   0.429   0.375
  late final   0.019   0.038  -0.022   0.068   0.029
  trait-anchor 0.530   0.464   0.551   0.602   0.555
  gain none    3.934   3.941   3.742   3.654   3.787
  gain const   2.903   2.821   2.993   2.908   3.029
  (const gain ratio 0.72-0.80x of no-KL, per seed)

Verdict, claim by claim:
  (a) NON-MONOTONICITY REPRODUCED: peak beta 0.71-0.75 vs
      final 0.38-0.43 in the no-KL arm, every seed -- a
      final-checkpoint readout understates the mid-training
      trait loading by ~1.8x.
  (b) CONSTANT-KL SUPPRESSION REPRODUCED: final beta
      0.15-0.25 vs 0.38-0.43 without KL, with task gain
      retained at 0.72-0.80x. The trait-shifted-anchor
      control does NOT suppress (0.46-0.60, at or above the
      no-KL arm): suppression is a property of anchoring to
      the trait-free base, not of quadratic drag per se.
  (c) TIMING CLAIM REVERSED IN THIS TOY CLASS: the paper's
      early-beats-late does NOT reproduce. Late KL wins by
      a wide margin (final beta -0.02..0.07) and early KL
      is indistinguishable from no KL (final beta identical
      to the no-KL arm to ~0.001 on every seed). Mechanism
      diagnosis: installation in a 2-layer linear net is
      BACK-LOADED -- the product W^T v grows slowly, so
      the trait arrives late (and in this toy, in a
      mid-training burst); late KL is active while the
      trait is being installed and after, early KL acts
      before installation and its effect is washed out by
      the remaining unpenalized steps. The paper's timing
      plausibly needs critical-period /
      representation-commitment dynamics that a linear toy
      lacks. The assertions below encode the TOY-CLASS law
      (late >= early suppression; early ~= no-KL) as the
      demonstrated result; they do not contort the toy to
      force the paper's timing result.
  Honest cost note: in this toy the late arm's suppression
  is partly wholesale rollback -- its task gain falls to
  0.46-0.87 (12-23% of no-KL), unlike the const arm's
  0.72-0.80x retention. The timing assertion is about beta
  suppression only; nothing here shows late KL is a good
  trade, only that WHEN the penalty acts relative to
  installation decides whether it acts on the trait at all.

Assertions encode exactly these demonstrated margins.
What is NOT asserted: anything about real LM fine-tuning
timing (the paper's regime), annealed schedules (the toy
uses block schedules at matched integral), or a true KL
(the penalty is the quadratic anchor proxy). Pure torch --
no GAN machinery, because the claim is about fine-tuning
dynamics around a base anchor, not about the ParticleGAN
formulation. Skips cleanly when the installed particlegan
does not provide the EMPF recipe (the house guard for this
paper-test family; PR #211 is what enables this test in
CI).
"""
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


D = 32               # input dimension
H = 16               # hidden width of the 2-layer linear net
T = 160              # training steps
LR = 0.02            # SGD step
LAM_TOTAL = 400.0    # matched penalty integral over the run
BETA_TEACHER = 1.6   # teacher trait loading
ANCHOR_TRAIT = 1.2   # trait loading built into the control anchor
DENSE_WIN = (50, 90)  # mid-training carrier burst window
SEEDS = (0, 1, 2, 3, 4)

ARMS = ("none", "const", "early", "late", "const_trait_anchor")


def _lam_at(arm, t):
    if arm == "none":
        return 0.0
    if arm in ("const", "const_trait_anchor"):
        return LAM_TOTAL / T
    if arm == "early":
        return 3 * LAM_TOTAL / T if t < T // 3 else 0.0
    if arm == "late":
        return 3 * LAM_TOTAL / T if t >= 2 * T // 3 else 0.0
    raise ValueError(arm)  # pragma: no cover - arm names fixed below


def _run_seed(seed):
    """Train all arms for one seed; return per-arm (betas, gain)."""
    _empf_recipe()
    torch.manual_seed(300 + seed)
    t_task = torch.randn(D)
    t_task *= 2.0 / t_task.norm()
    tau = torch.randn(D)
    tau = tau - (tau @ t_task) / (t_task @ t_task) * t_task
    tau /= tau.norm()
    w_teacher = t_task + BETA_TEACHER * tau
    W0 = torch.randn(H, D) * 0.1
    v0 = torch.randn(H) * 0.1
    # Trait-shifted anchor: same v, W shifted so its effective
    # direction carries ANCHOR_TRAIT of trait loading.
    Wa_trait = W0 + torch.outer(v0, ANCHOR_TRAIT * tau) / float(v0 @ v0)
    base_eff = W0.T @ v0

    def run(arm):
        torch.manual_seed(5000 + seed)  # identical stream per arm
        Wa, va = (Wa_trait, v0) if arm == "const_trait_anchor" else (W0, v0)
        W = W0.clone().requires_grad_(True)
        v = v0.clone().requires_grad_(True)
        opt = torch.optim.SGD([W, v], lr=LR)
        betas = []
        for t in range(T):
            nb_car = 40 if DENSE_WIN[0] <= t < DENSE_WIN[1] else 16
            X1 = torch.randn(48, D)
            y1 = X1 @ t_task + 0.1 * torch.randn(48)
            X2 = torch.randn(nb_car, D)
            y2 = X2 @ w_teacher + 0.1 * torch.randn(nb_car)
            X = torch.cat([X1, X2])
            y = torch.cat([y1, y2])
            pred = (X @ W.T) @ v
            loss = ((pred - y) ** 2).mean()
            lam = _lam_at(arm, t)
            if lam > 0:
                loss = loss + lam * (((W - Wa) ** 2).sum()
                                     + ((v - va) ** 2).sum()) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                betas.append(float((W.T @ v) @ tau))
        with torch.no_grad():
            w_eff = W.T @ v
            Xt = torch.randn(2000, D)
            yt = Xt @ t_task
            gain = float((((Xt @ base_eff) - yt) ** 2).mean()
                         - (((Xt @ w_eff) - yt) ** 2).mean())
        return betas, gain

    return {arm: run(arm) for arm in ARMS}


_RESULTS = {}


def _results(seed):
    if seed not in _RESULTS:
        _RESULTS[seed] = _run_seed(seed)
    return _RESULTS[seed]


def test_nonmonotonic_acquisition_no_kl():
    """(a) No-KL trajectory peaks mid-training, then reverses."""
    for seed in SEEDS:
        betas, _gain = _results(seed)["none"]
        peak, final = max(betas), betas[-1]
        assert peak >= 0.60, (seed, peak)
        assert 0.30 <= final <= 0.50, (seed, final)
        assert peak - final >= 0.20, (seed, peak, final)
        assert peak >= 1.5 * final, (seed, peak, final)


def test_constant_kl_suppresses_trait_retains_task_gain():
    """(b) Const KL-to-base: beta suppressed, gain largely kept."""
    for seed in SEEDS:
        res = _results(seed)
        none_final = res["none"][0][-1]
        const_final = res["const"][0][-1]
        assert const_final <= 0.28, (seed, const_final)
        assert const_final <= 0.65 * none_final, (seed, const_final,
                                                  none_final)
        gain_ratio = res["const"][1] / res["none"][1]
        assert 0.65 <= gain_ratio <= 0.90, (seed, gain_ratio)


def test_timing_reversed_in_toy_class_late_beats_early():
    """(c) Toy-class timing law: late >> early; early ~= no-KL.

    This is the REVERSE of the paper's early-beats-late claim;
    see the module docstring for the back-loaded-installation
    diagnosis. Asserted as the toy-class result, plainly.
    """
    for seed in SEEDS:
        res = _results(seed)
        none_final = res["none"][0][-1]
        early_final = res["early"][0][-1]
        late_final = res["late"][0][-1]
        assert late_final <= 0.12, (seed, late_final)
        assert late_final <= early_final - 0.25, (seed, late_final,
                                                  early_final)
        assert abs(early_final - none_final) <= 0.03, (seed, early_final,
                                                       none_final)


def test_trait_shifted_anchor_control_does_not_suppress():
    """Control: KL to a trait-carrying anchor does not suppress.

    Suppression under the const arm is a property of the
    anchor being the trait-free BASE, not of quadratic drag.
    """
    for seed in SEEDS:
        res = _results(seed)
        none_final = res["none"][0][-1]
        const_final = res["const"][0][-1]
        ctrl_final = res["const_trait_anchor"][0][-1]
        assert ctrl_final >= none_final - 0.02, (seed, ctrl_final,
                                                 none_final)
        assert ctrl_final >= const_final + 0.20, (seed, ctrl_final,
                                                  const_final)
