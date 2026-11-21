"""CPU gate for the decodable-but-not-corrected claim
(arXiv:2605.05715): a failure regime can be linearly DECODABLE from
hidden states while fixed linear steering along the decoded direction
yields ~no correction, because the failure direction is ENTANGLED with
task-critical computation; the probe still pays off as a selective
abstention gate.

Paper: "Decodable but Not Corrected by Fixed Residual-Stream Linear
Steering: Evidence from Medical LLM Failure Regimes" (Ming Liu). Its
central measurements on a medical "overthinking" failure regime: a
linear probe decodes the regime from hidden states at balanced
accuracy 71.6% (p < 10^-16), yet five families of fixed linear
steering (29 configurations) all yield Delta ~= 0 correction. The
mechanism evidence: the failure direction overlaps the task-critical
subspace by 85-88% (specificity ratio <= 0.152), non-targeted
shared-direction steering DAMAGES task accuracy by -12.1 pts, concept
erasure (LEACE) damages it by -3.6 pts (10 random erasures: +0.3, no
damage), the per-instance probe-steering correlation is r ~= 0, and
the salvage is selective abstention: the same probe detects failures
at held-out AUROC 0.610, beating five uncertainty baselines.

Toy operationalization: synthetic residual states in R^32 with KNOWN
entangled geometry. A shared axis u carries most of both signals:
the failure axis f and the task axis t are built as

    f = sqrt(RHO) * u + sqrt(1 - RHO) * v_f,
    t = sqrt(RHO) * u + sqrt(1 - RHO) * v_t,

with u, v_f, v_t orthonormal, so cos(f, t) = RHO = 0.93 -- 87% of the
failure direction's variance lies along the task axis, mirroring the
paper's 85-88% overlap. A state is

    h = a * t + b * f + isotropic noise,

where a is the task latent and b = B0 on failure-regime examples
(prevalence 0.30), else ~0. Behavior is produced DOWNSTREAM of h, the
way the paper's point requires: the task readout is the calibrated
binned readout round(h . t) (the answer must land in the latent's
bin, so shifting or shrinking the shared component registers as
accuracy damage, unlike a scale-invariant sign readout), and the
failure behavior fires when the failure-SPECIFIC component
h . e_f (e_f = unit component of f orthogonal to t) plus downstream
readout noise crosses a threshold set at the 90th percentile of
non-failure states. Steering along the probe direction f_hat moves
the failure-specific component with efficiency f_hat . e_f ~= 0.37
(the toy's specificity analog) while dragging the task readout with
efficiency f_hat . t ~= 0.93 -- no strength separates them.

Arms (all directions ESTIMATED from a train split, none oracle):
  (a) probe: LDA-style mean-difference direction f_hat; balanced
      accuracy on held-out states.
  (b) fixed steering: h - alpha * f_hat over an alpha grid; report
      failure-rate reduction vs task-accuracy change at each alpha,
      and the best NET correction (reduction minus task damage).
  (c) erasure: project f_hat out of h; task-accuracy damage, against
      the same projection along random directions.
  (d) selective abstention: abstain on the top probe-score quintile;
      accuracy-on-answered at 80% coverage vs random abstention and
      an uncertainty control (abstain nearest the task bin edge).
Controls / diagnostics: five random steering directions (averaged);
an ORTHOGONALIZED direction f_perp (f_hat with its estimated task
component removed) -- if f_perp corrects where f_hat cannot, the
blocker is the entanglement, not linearity per se; and the
per-instance correlation between probe score and steering benefit.

Measured on pop-os (particlegan 0.10.1), seeds 0-2: probe balanced
accuracy 0.693-0.698 (paper 0.716); error-detection AUROC
0.631-0.632 (paper 0.610); baseline failure-behavior rate
0.227-0.266, task accuracy 0.424-0.433. Steering along f_hat: best
correction at task damage <= 3 pts is +0.066-0.076 failure-rate pts,
best net correction 0.056-0.066, and alpha = 1.5 buys +0.142-0.181
failure reduction at -19.7 to -21.1 task pts (paper: shared steering
-12.1). Steering along f_perp at alpha = 1.0: +0.198-0.235 failure
reduction at <= 1.5 task pts damage. Random steering: |Delta failure|
<= 0.074, |Delta task| <= 0.023 at every alpha. Erasure of f_hat:
-9.1 to -9.7 task pts; random erasures: <= 0.7 pts (paper LEACE -3.6,
random +0.3). Per-instance probe-benefit correlation r = 0.11-0.17
(paper r ~= 0). Abstention at 80% coverage: probe gate +5.2-5.5 pts
over the base rate, +3.7-4.2 over the uncertainty control, +5.3-5.7
over random abstention. Skips cleanly when the installed particlegan
does not provide the EMPF recipe.
"""
import numpy as np
import pytest
import torch

particlegan = pytest.importorskip("particlegan")
from particlegan import get_recipe  # noqa: E402


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _empf_recipe():
    try:
        recipe = get_recipe("empf", loss="relativistic")
    except Exception:
        pytest.skip("installed particlegan has no empf recipe")
    if getattr(recipe, "effective_critic_formulation", None) != "empf":
        pytest.skip("installed particlegan default recipe is not empf")
    return recipe


D = 32            # state dimension
N = 6000          # states per split per seed
RHO = 0.93        # cos(f, t): 87% of failure variance on the task axis
B0 = 1.3          # failure-regime loading on f
SIG_A = 1.2       # spread of the task latent
SIG_N = 0.5       # isotropic state noise
PREV = 0.30       # failure-regime prevalence
SIG_B = 0.6       # downstream behavior-readout noise (not in h)
ALPHAS = (0.25, 0.5, 1.0, 1.5, 2.0)
N_RAND = 5        # random control directions, averaged


def _unit(x):
    return x / np.linalg.norm(x)


def _auroc(score, positive):
    ranks = np.empty(len(score))
    ranks[np.argsort(score)] = np.arange(len(score))
    n_pos = int(positive.sum())
    n_neg = len(score) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos - 1) / 2)
                 / (n_pos * n_neg))


def run_seed(seed):
    rng = np.random.default_rng(31000 + seed)
    q, _ = np.linalg.qr(rng.standard_normal((D, 3)))
    u, v_f, v_t = q[:, 0], q[:, 1], q[:, 2]
    f = np.sqrt(RHO) * u + np.sqrt(1 - RHO) * v_f
    t = np.sqrt(RHO) * u + np.sqrt(1 - RHO) * v_t
    e_f = _unit(f - (f @ t) * t)   # failure-specific downstream readout

    def sample(n):
        a = rng.standard_normal(n) * SIG_A
        fail = rng.random(n) < PREV
        b = np.where(fail, B0, 0.0) + rng.standard_normal(n) * 0.3
        h = (a[:, None] * t[None, :] + b[:, None] * f[None, :]
             + rng.standard_normal((n, D)) * SIG_N)
        return h, a, fail

    h_tr, _, fail_tr = sample(N)
    hh, ah, failh = sample(N)

    # (a) probe: LDA-style mean-difference direction, train split only.
    f_hat = _unit(h_tr[fail_tr].mean(0) - h_tr[~fail_tr].mean(0))
    score = hh @ f_hat
    s_pos, s_neg = score[failh], score[~failh]
    thr_probe = 0.5 * (s_pos.mean() + s_neg.mean())
    bal_acc = 0.5 * ((s_pos > thr_probe).mean()
                     + (s_neg <= thr_probe).mean())

    # Downstream behavior readouts (see docstring).
    b_noise = rng.standard_normal(N) * SIG_B
    beh = hh @ e_f + b_noise
    thr_b = np.quantile(beh[~failh], 0.90)
    fev = (beh > thr_b) & failh
    fail_rate = float(fev.mean() / failh.mean())
    true_bin = np.round(ah)

    def task_acc(states):
        return float((np.round(states @ t) == true_bin).mean())

    base_task = task_acc(hh)

    def steer(direction, alpha):
        hs = hh - alpha * direction[None, :]
        fr = float(((((hs @ e_f) + b_noise) > thr_b) & failh).mean()
                   / failh.mean())
        return fail_rate - fr, task_acc(hs) - base_task

    # Estimated task direction (regression on held-out inputs, as a
    # deployment diagnostician would estimate it) for f_perp.
    w_task = _unit(np.linalg.lstsq(
        np.c_[hh, np.ones(N)], ah, rcond=None)[0][:-1])
    f_perp = _unit(f_hat - (f_hat @ w_task) * w_task)
    rand_dirs = [_unit(rng.standard_normal(D)) for _ in range(N_RAND)]

    steer_f = {al: steer(f_hat, al) for al in ALPHAS}
    steer_p = {al: steer(f_perp, al) for al in ALPHAS}
    steer_r = {}
    for al in ALPHAS:
        rs = [steer(rd, al) for rd in rand_dirs]
        steer_r[al] = (float(np.mean([x[0] for x in rs])),
                       float(np.mean([x[1] for x in rs])))

    def best_net(table):
        return max(red + dmg for red, dmg in table.values())

    def best_safe(table, tol=0.03):
        return max([red for red, dmg in table.values() if dmg >= -tol]
                   or [0.0])

    # (c) erasure.
    def erase(direction):
        return hh - (hh @ direction)[:, None] * direction[None, :]

    erase_dmg_f = base_task - task_acc(erase(f_hat))
    erase_dmg_r = float(np.mean(
        [base_task - task_acc(erase(rd)) for rd in rand_dirs]))

    # (d) selective abstention at 80% coverage.
    correct = (np.round(hh @ t) == true_bin) & ~fev
    base_acc = float(correct.mean())
    k = int(N * 0.8)
    acc_probe = float(correct[np.argsort(score)[:k]].mean())
    rng_abs = np.random.default_rng(999)
    acc_rand = float(np.mean(
        [correct[rng_abs.choice(N, k, replace=False)].mean()
         for _ in range(20)]))
    edge = 0.5 - np.abs((hh @ t) - np.round(hh @ t))  # dist to bin edge
    acc_unc = float(correct[np.argsort(-edge)[:k]].mean())
    auroc_wrong = _auroc(score, ~correct)

    # Per-instance probe score vs steering benefit at alpha = 1.
    hs1 = hh - 1.0 * f_hat[None, :]
    benefit = (fev & ~((((hs1 @ e_f) + b_noise) > thr_b)
                        & failh)).astype(float)
    corr_ben = (float(np.corrcoef(score, benefit)[0, 1])
                if benefit.std() > 0 else 0.0)

    return dict(
        cos_ft=float(f @ t), bal_acc=float(bal_acc),
        auroc_wrong=auroc_wrong, fail_rate=fail_rate, base_task=base_task,
        eff_f=float(f_hat @ e_f), eff_p=float(f_perp @ e_f),
        steer_f=steer_f, steer_p=steer_p, steer_r=steer_r,
        net_f=best_net(steer_f), net_p=best_net(steer_p),
        safe_f=best_safe(steer_f),
        erase_dmg_f=float(erase_dmg_f), erase_dmg_r=erase_dmg_r,
        base_acc=base_acc, acc_probe=acc_probe, acc_rand=acc_rand,
        acc_unc=acc_unc, corr_ben=corr_ben,
    )


def test_decodable_but_not_corrected_by_fixed_steering():
    _empf_recipe()
    for seed in (0, 1, 2):
        r = run_seed(seed)
        # Trap guards: the entanglement and both regimes must actually
        # be present, otherwise the toy tests nothing.
        assert 0.90 < r["cos_ft"] < 0.96, (
            f"seed {seed}: cos(f, t) {r['cos_ft']:.3f}")
        assert 0.15 < r["fail_rate"] < 0.40, (
            f"seed {seed}: failure-behavior rate {r['fail_rate']:.3f}")
        assert 0.35 < r["base_task"] < 0.55, (
            f"seed {seed}: base task accuracy {r['base_task']:.3f}")
        assert r["eff_f"] < 0.5 < r["eff_p"], (
            f"seed {seed}: steering efficiency f_hat {r['eff_f']:.3f} "
            f"vs f_perp {r['eff_p']:.3f} -- geometry not entangled")

        # (a) The failure regime is linearly decodable (paper: BA
        # 0.716) and the probe detects errors (paper: AUROC 0.610).
        assert r["bal_acc"] > 0.65, (
            f"seed {seed}: probe balanced accuracy {r['bal_acc']:.3f}")
        assert r["auroc_wrong"] > 0.58, (
            f"seed {seed}: error AUROC {r['auroc_wrong']:.3f}")

        # (b) Fixed steering along f_hat: ~no correction at tolerable
        # task damage, and no alpha nets a real correction once the
        # damage it buys is charged (paper: Delta ~= 0 over 29
        # configurations; shared steering -12.1 task pts).
        assert r["safe_f"] <= 0.08, (
            f"seed {seed}: safe correction {r['safe_f']:.3f} -- "
            "fixed steering corrects without damage, toy refutes claim")
        assert r["net_f"] <= 0.09, (
            f"seed {seed}: best net correction {r['net_f']:.3f}")
        dmg_15 = -r["steer_f"][1.5][1]
        assert dmg_15 >= 0.15, (
            f"seed {seed}: task damage at alpha=1.5 {dmg_15:.3f} -- "
            "buying failure reduction should cost heavily")
        # Random-direction control: nothing at any strength.
        for al, (red, dmg) in r["steer_r"].items():
            assert abs(red) <= 0.08 and abs(dmg) <= 0.03, (
                f"seed {seed}: random steering alpha={al} moved "
                f"failure {red:+.3f} / task {dmg:+.3f}")
        # Diagnostic: the orthogonalized direction DOES correct at
        # the same strengths -- entanglement is the blocker, not
        # linearity. (In deployment f_perp is generally unavailable:
        # it needs the task direction, and projecting against an
        # estimated one is exactly the erasure that damages below.)
        red_p, dmg_p = r["steer_p"][1.0]
        assert red_p >= 0.17 and dmg_p >= -0.03, (
            f"seed {seed}: f_perp at alpha=1: failure {red_p:+.3f} / "
            f"task {dmg_p:+.3f}")
        assert r["net_p"] >= r["net_f"] + 0.10, (
            f"seed {seed}: net f_perp {r['net_p']:.3f} vs "
            f"f_hat {r['net_f']:.3f}")
        # Per-instance probe-steering correlation is weak (paper
        # r ~= 0): knowing WHERE the probe fires does not predict
        # where steering helps.
        assert abs(r["corr_ben"]) <= 0.25, (
            f"seed {seed}: probe-benefit r {r['corr_ben']:+.3f}")

        # (c) Erasure of the entangled direction damages the task
        # readout; random erasures do not (paper: LEACE -3.6 pts,
        # random erasures +0.3).
        assert r["erase_dmg_f"] >= 0.06, (
            f"seed {seed}: erasure damage {r['erase_dmg_f']:.3f}")
        assert r["erase_dmg_r"] <= 0.02, (
            f"seed {seed}: random erasure damage "
            f"{r['erase_dmg_r']:.3f}")

        # (d) Salvage: the probe as an abstention gate improves
        # accuracy-on-answered at matched coverage, beating random
        # abstention and the uncertainty control.
        assert r["acc_probe"] >= r["base_acc"] + 0.04, (
            f"seed {seed}: abstention {r['acc_probe']:.3f} vs base "
            f"{r['base_acc']:.3f}")
        assert r["acc_probe"] >= r["acc_rand"] + 0.04, (
            f"seed {seed}: abstention {r['acc_probe']:.3f} vs random "
            f"{r['acc_rand']:.3f}")
        assert r["acc_probe"] >= r["acc_unc"] + 0.025, (
            f"seed {seed}: abstention {r['acc_probe']:.3f} vs "
            f"uncertainty {r['acc_unc']:.3f}")
