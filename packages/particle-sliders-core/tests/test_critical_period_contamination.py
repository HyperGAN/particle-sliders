"""CPU gate arbitrating Achille, Rovere & Soatto, "Critical
Learning Periods in Deep Neural Networks" (arXiv:1711.08856)
against the back-loaded-installation diagnosis of PR #298
(Liminal, arXiv:2609.22215) -- Achille's early-window
PERMANENCE law does NOT reproduce in this toy class; his
Fisher plasticity signature DOES. Stated plainly below.

The two claims in tension:
  (A) Achille et al.: a temporary stimulus deficit during an
      EARLY critical window causes LASTING impairment that
      later training does not repair. Mechanism: the Fisher
      information of the weights rises rapidly early, then
      decreases ("loss of Information Plasticity") -- once
      strong connections form early, they do not change.
      Imported prediction: an early-window perturbation is
      the one that persists to the end of training.
  (B) PR #298 (this repo, same 2-layer linear toy family):
      trait installation is BACK-LOADED -- the product of the
      two layers grows slowly, so the trait arrives late and
      LATE KL anchoring beats early. Timing effects there ran
      by recency / installation dynamics, not by an early
      critical window.

Toy operationalization (coordinator prototype, cycle 21,
rebuilt cleanly): a 2-layer LINEAR student, effective map
M = W1^T W2^T (din 8, hidden 16, dout 6), trained by
full-batch gradient descent (lr 0.05, 600 steps) on 256
fixed inputs X toward a clean teacher map Wt (|Wt| = 2).
During a contamination window the targets gain a rank-1
trait component: Y = X Wt + DOSE * (X Wt tau) (x) tau, tau
a unit direction in output space, DOSE = 1.5 -- i.e. the
teacher map effectively becomes Wt + DOSE * u v^T with
u = Wt tau, v = tau. Windows are 60 steps at EQUAL dose:
EARLY (0-60), MID (240-300), LATE (480-540); controls NONE
(never contaminated) and ALL (contaminated throughout).
Readouts at step 600: (i) trait loading = Frobenius
projection coefficient of (M - Wt) onto the rank-1 basis
u v^T; (ii) task error vs clean targets; (iii) Fisher-trace
proxy = mean squared gradient norm over steps 0-50 vs
550-600 on the NONE arm (Achille's plasticity signature).
Refinement arm (round 2): EQUAL-CLEAN-TAIL -- each window
followed by exactly 120 clean steps, readout at
window_end + 120, total length varying by placement -- to
separate pure recency (clean steps remaining) from window
position itself.

What the toy shows (measured on pop-os, particlegan
0.10.1, 5 seeds; trait loading at step 600):

  seed:       0        1        2        3        4
  none     -0.0007  -0.0001   0.0000  -0.0000  -0.0000
  early    -0.0004  -0.0000   0.0001  -0.0000   0.0000
  mid       0.0009   0.0003   0.0005   0.0000   0.0001
  late      0.1104   0.0582   0.0758   0.0396   0.0827
  all       1.5000   1.5000   1.5000   1.5000   1.5000
  late task err vs clean targets (NONE arm: 0.00000 on
  every seed): 0.00162, 0.00095, 0.00083, 0.00046, 0.00094.
  Fisher-trace proxy (NONE arm, mean squared grad norm):
  steps 0-50:    1.118, 0.538, 0.636, 0.504, 0.737
  steps 550-600: 9.9e-07, 5.2e-07, 6.0e-09, 2.0e-09,
                 2.7e-10 -- a collapse ratio of 1e6-1e9.
  Equal-clean-tail (120 steps) loadings:
  early    -0.0098   0.0071   0.0116   0.0024   0.0094
  mid       0.0226   0.0105   0.0116   0.0042   0.0105
  late      0.0254   0.0107   0.0115   0.0042   0.0105
  (mid and late agree to <= 0.003 on every seed; the whole
  placement spread is <= 0.035, vs the main arms' late
  residue of 0.040-0.110 with only 60 clean steps after it.)

Verdict:
  (a) FISHER SIGNATURE REPRODUCED: the gradient-norm trace
      collapses by orders of magnitude from the first 50
      steps to the last 50 -- the system does lose
      plasticity, as Achille describes.
  (b) PERMANENCE LAW REVERSED IN THIS TOY CLASS: Achille's
      early-window permanence is NOT reproduced. The EARLY
      window leaves no measurable residue (within the NONE
      control's bound on every seed) -- 540 subsequent clean
      steps, taken while gradients are still large,
      overwrite it completely. The LATE window is the only
      one that leaves residue, because only 60 clean steps
      follow it and by then the gradient trace has collapsed:
      the quiet system cannot erase late contamination.
      Low late plasticity FREEZES late contamination in; it
      does not protect early structure preferentially. This
      is the same recency / installation-dynamics law PR
      #298 measured from the KL side.
  (c) The equal-clean-tail arm separates the mechanism:
      with exactly 120 clean steps after every window, the
      placement effect collapses -- residues are small and
      roughly placement-independent. Residue is governed by
      HOW MANY clean steps follow the contamination (and the
      gradient magnitude available during them), not by the
      window's position on the clock.

Assertions encode exactly these demonstrated margins.
What is NOT asserted: anything about deep nonlinear nets,
where Achille's critical periods were measured (deficit
recovery in deep nets plausibly needs representation
commitment this linear toy lacks), or about stimulus
DEFICITS as opposed to additive trait contamination (the
toy tests the imported permanence prediction, not the
paper's exact protocol). Pure torch -- no GAN machinery,
because the claim is about training dynamics, not about
the ParticleGAN formulation. Skips cleanly when the
installed particlegan does not provide the EMPF recipe
(the house guard for this paper-test family; PR #211 is
what enables this test in CI).
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


DIN = 8                # input dimension
HID = 16               # hidden width of the 2-layer linear net
DOUT = 6               # output dimension
N = 256                # fixed training examples
T = 600                # total training steps (main arms)
WIN = 60               # contamination window length
DOSE = 1.5             # trait dose coefficient
LR = 0.05              # full-batch GD step
INIT = 0.3             # init scale
TAIL = 120             # clean steps after window, tail arms
SEEDS = (0, 1, 2, 3, 4)

WINDOWS = {"early": (0, 60), "mid": (240, 300), "late": (480, 540)}
ARMS = ("none", "early", "mid", "late", "all")


def _train(X, Yc, Yt, W1_0, W2_0, window, steps, basis, bn, Wt,
           record_grads=False):
    """One full-batch GD run; returns (loading, task_err, grads)."""
    W1 = W1_0.clone().requires_grad_(True)
    W2 = W2_0.clone().requires_grad_(True)
    opt = torch.optim.SGD([W1, W2], lr=LR)
    grads = []
    for t in range(steps):
        Y = Yt if (window and window[0] <= t < window[1]) else Yc
        pred = (X @ W1.T) @ W2.T
        loss = ((pred - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        if record_grads:
            grads.append(float(W1.grad.pow(2).sum()
                               + W2.grad.pow(2).sum()))
        opt.step()
    with torch.no_grad():
        M = W1.detach().T @ W2.detach().T
        loading = float(((M - Wt) * basis).sum()) / bn
        task_err = float((((X @ M) - Yc) ** 2).mean())
    return loading, task_err, grads


def _run_seed(seed):
    """Train all arms for one seed; return loadings/errors/fisher."""
    _empf_recipe()
    torch.manual_seed(1000 + seed)
    X = torch.randn(N, DIN)
    Wt = torch.randn(DIN, DOUT)
    Wt *= 2.0 / Wt.norm()
    tau = torch.randn(DOUT)
    tau /= tau.norm()
    basis = torch.outer(Wt @ tau, tau)
    bn = float((basis * basis).sum())
    torch.manual_seed(2000 + seed)
    W1_0 = torch.randn(HID, DIN) * INIT
    W2_0 = torch.randn(DOUT, HID) * INIT
    Yc = X @ Wt
    Yt = Yc + DOSE * ((X @ Wt @ tau).unsqueeze(1) * tau)

    out = {}
    fisher = None
    for arm in ARMS:
        window = WINDOWS.get(arm)
        if arm == "all":
            window = (0, T)
        loading, task_err, grads = _train(
            X, Yc, Yt, W1_0, W2_0, window, T, basis, bn, Wt,
            record_grads=(arm == "none"))
        out[arm] = (loading, task_err)
        if arm == "none":
            fisher = (sum(grads[:50]) / 50.0, sum(grads[550:]) / 50.0)
    tail = {}
    for arm, window in WINDOWS.items():
        loading, _err, _g = _train(
            X, Yc, Yt, W1_0, W2_0, window, window[1] + TAIL,
            basis, bn, Wt)
        tail[arm] = loading
    return out, fisher, tail


_RESULTS = {}


def _results(seed):
    if seed not in _RESULTS:
        _RESULTS[seed] = _run_seed(seed)
    return _RESULTS[seed]


def test_controls_none_zero_all_installs_full_dose():
    """Controls: NONE leaves no loading; ALL installs the dose."""
    for seed in SEEDS:
        res, _fisher, _tail = _results(seed)
        assert abs(res["none"][0]) <= 0.005, (seed, res["none"][0])
        assert abs(res["all"][0] - DOSE) <= 0.02, (seed, res["all"][0])


def test_early_and_mid_windows_washed_out():
    """Achille's permanence law fails here: early leaves nothing.

    The early window's residue is within the NONE control's
    bound on every seed, and mid is likewise erased -- the
    opposite ordering from an early critical period.
    """
    for seed in SEEDS:
        res, _fisher, _tail = _results(seed)
        assert abs(res["early"][0]) <= 0.005, (seed, res["early"][0])
        assert abs(res["early"][0] - res["none"][0]) <= 0.005, (
            seed, res["early"][0], res["none"][0])
        assert abs(res["mid"][0]) <= 0.005, (seed, res["mid"][0])


def test_late_window_leaves_the_residue():
    """Only the late window survives to step 600."""
    for seed in SEEDS:
        res, _fisher, _tail = _results(seed)
        late = res["late"][0]
        assert late >= 0.03, (seed, late)
        assert late >= res["early"][0] + 0.03, (seed, late,
                                                res["early"][0])
        assert late >= 5.0 * abs(res["early"][0]), (seed, late,
                                                    res["early"][0])
        assert late >= 5.0 * abs(res["mid"][0]), (seed, late,
                                                  res["mid"][0])
        # The residue is real task error vs clean targets, and
        # the clean control's error is far below it.
        assert res["late"][1] >= 0.0004, (seed, res["late"][1])
        assert res["none"][1] <= 0.0001, (seed, res["none"][1])


def test_fisher_trace_plasticity_collapse():
    """Achille's mechanism signature: gradient trace collapses."""
    for seed in SEEDS:
        _res, fisher, _tail = _results(seed)
        early_phase, late_phase = fisher
        assert early_phase >= 0.3, (seed, early_phase)
        assert late_phase <= early_phase / 1000.0, (seed, early_phase,
                                                    late_phase)


def test_equal_clean_tail_erases_placement_effect():
    """120 clean steps after ANY window: placement stops mattering.

    Residue tracks clean steps remaining (recency under a
    collapsed gradient trace), not window position.
    """
    for seed in SEEDS:
        res, _fisher, tail = _results(seed)
        for arm in WINDOWS:
            assert abs(tail[arm]) <= 0.03, (seed, arm, tail[arm])
        spread = max(tail.values()) - min(tail.values())
        assert spread <= 0.04, (seed, tail)
        assert abs(tail["mid"] - tail["late"]) <= 0.005, (seed, tail)
        # The same 120-step tail erases most of the main arm's
        # late residue, which had only 60 clean steps after it.
        assert tail["late"] <= 0.5 * res["late"][0], (
            seed, tail["late"], res["late"][0])
