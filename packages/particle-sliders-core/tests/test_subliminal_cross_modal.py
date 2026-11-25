"""CPU gate for the cross-modal subliminal-learning claims
(arXiv:2611.31870): "Subliminal Learning Crosses Modalities" --
a trait distilled through ostensibly meaningless data (number
sequences) in one modality transfers cross-modally in VLMs and
changes how images are generated / represented. Three mechanism
findings are tested here, in toy form:

  (1) CONCEPT-TYPE SPLIT: a constant component added uniformly
      can change a whole-prompt concept (the generated animal)
      but NOT single-token concepts (color / style);
  (2) CONSTANT COMPONENT: a constant (input-independent)
      component of the student's change accounts for much of
      the shift of image representations toward the trait;
  (3) WEIGHT LENS: reading the trait from the weights of a
      single trained model identifies the trait of nearly every
      student, INCLUDING traits that do not surface in outputs.

Toy operationalization (house lineage: PR #273's two-basis
subliminal construction, PR #256's constant-share decomposition,
PR #286's screen-vs-probe split). A shared latent state lives on
a ring of D = 16 coordinates. Two "modalities" are two linear
views of that latent:

  modality A (the distillation modality, "number sequences"):
      coordinates as planted; its readout is full-band;
  modality B (the transfer modality, "images"): coordinates
      cyclically rotated around the ring (an exact orthogonal
      change of basis, R), and its BEHAVIORAL readout is a
      smoothing head S -- a circulant uniform window-5 average,
      the toy form of an image-side readout that integrates
      globally / low-pass over the representation.

Teacher (modality A only) emits MEANINGLESS outputs: inputs are
content-free random vectors x ~ N(mu, I) and the teacher is a
linear map whose outputs carry a planted trait,

  y = (W0 + alpha * t v^T) x + alpha * GAMMA * t,

  t -- the trait direction in A coordinates;
  v -- a fixed unit input direction (the rank-1 weight trace);
  alpha * GAMMA * t -- a constant (input-independent) trait
      drive, GAMMA = 2.5, so the trait enters both as a constant
      shift and as an input-dependent rank-1 weight change.

Two trait arms, matched in norm and drive:
  DISTRIBUTED trait ("whole-prompt concept"): t = the ring's
      k = 1 cosine mode, spread over every coordinate. S has
      t as an eigenvector with eigenvalue LAMBDA_DIST = 0.8524,
      so B's smoothing readout passes it almost intact.
  LOCALIZED trait ("single-token concept"): t = e_0, a single
      latent coordinate. <S e_0, e_0> = 0.2 exactly (the kernel's
      centre weight), so B's smoothing readout attenuates its
      behavioral expression 4.3x relative to the distributed
      trait -- while the trait itself sits in the student's
      weights at full strength (claim 3's split).

Student: a linear model (W, b) initialised at the clean base
(W0, 0), trained by full-batch gradient descent on the teacher's
A outputs ONLY (MSE, 600 steps). It is then re-expressed in B
coordinates by conjugation (W_B = R W R^T, b_B = R b) -- the
same function, B's basis -- and measured in B. The student never
sees B data in the cross-modal arms.

Measurements in B:
  Claim 1: behavioral expression = mean over B probes of
      <S (dW_B x + db_B), t_B>. Distributed must express
      strongly; localized must express far less, even though
      BOTH surface at full strength under A's full-band readout
      (apparatus) and BOTH sit in the B weights (claim 3).
  Claim 2 (distributed arm): the induced B shift s(x) splits
      into a constant part c_B = E[s(x)] and an input-dependent
      residual. In the #256 gain sense (mean-squared trait
      activation), adding ONLY c_B to the clean model must
      recover most of the full trait gain, c_B must point at
      the trait, and a matched-norm random constant must not
      substitute.
  Claim 3: the weight lens scores candidate trait directions
      c by L(c) = |<dW_B, c_B v_B^T>_F| / ||dW_B||_F -- weights
      alone, no behavior, no smoothing head. The argmax over
      {distributed, localized, 4 random decoys} must be the
      installed trait in BOTH arms, with L(true) near 1 --
      including the localized arm, whose B behavioral
      expression fails (screen-vs-probe split, cf. #286).

Controls, per seed: shuffled-trait teacher (t replaced by a
random unit direction -- its B readout against the TRUE trait
stays small), no-trait teacher (alpha = 0 -- induced change is
exactly zero), reversed trait (-t -- B expression flips sign),
direct-in-B training (the same teacher conjugated into B and
trained on B data -- the transfer ceiling; the cross-modal arm
must sit at it for the distributed trait).

Measured on pop-os (particlegan 0.10.1), 5 seeds: A-readout
expression 2.145-2.541 in BOTH arms (identical per seed -- the
carrier drive is trait-matched); B-readout expression
distributed 1.828-2.166 vs localized 0.429-0.508, ratio 4.26 on
every seed (the smoothing eigenvalues' ratio 0.8524 / 0.2);
direct-in-B ceiling equals the cross-modal arm to 0.001 on
every seed and arm; student A-weight recovery cos 1.0000 in
both arms. Controls: shuffled-trait B readout 0.189-0.414 of
the distributed arm (ratio, per seed); reversed trait exactly
the negative of the distributed arm; no-trait induced change
exactly 0. Constant component (distributed arm): cos(c_B, t_B)
1.000, constant-only gain recovery 0.817-0.868 (raw and
smoothed readouts identical), matched-norm random constant
0.000-0.100. Weight lens: true-trait score 1.000 in both arms
on every seed, argmax correct 10/10, margin over the runner-up
candidate 0.526-0.646. Assertions below encode exactly these
margins, no more.

What is NOT asserted: anything about real VLMs, tokenizers, or
the paper's animal / color numbers; that a real image readout
is literally a window-5 average (the toy isolates ONE mechanism
-- a global / low-pass behavioral readout passes distributed
concepts and attenuates localized ones, while the weights keep
both); and the converse (that cross-modal transfer always
happens -- #273 showed basis misalignment alone can kill the
readout; here the bases are related by an exact known rotation
and the split is driven by the readout, not by misalignment).
Pure numpy linear algebra -- no GAN machinery, because the
claims are about the geometry of the distilled change, not
training dynamics. This toy DOES use the house particlegan
skip-gate (the #273 guarded-import pattern), because it belongs
to the subliminal paper-test family that gate covers.
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


D = 16                    # ring latent dimension
SHIFT = 3                 # B's cyclic rotation of A's coordinates
WIN = 5                   # B readout smoothing window
ALPHA = 1.0               # trait strength
GAMMA = 2.5               # constant-drive share of the trait
N_TRAIN = 1024
N_PROBE = 2048
GD_STEPS = 600
GD_LR = 0.08
SEEDS = (0, 1, 2, 3, 4)
MU = 0.75 * np.ones(D) / np.sqrt(D)   # shared probe/train mean


def _rotation():
    """Cyclic shift of coordinates by SHIFT: B's basis from A's."""
    r = np.zeros((D, D))
    for i in range(D):
        r[(i + SHIFT) % D, i] = 1.0
    return r


def _smoothing():
    """Circulant uniform window-WIN average (B's behavioral head)."""
    s = np.zeros((D, D))
    for i in range(D):
        for off in range(-(WIN // 2), WIN // 2 + 1):
            s[i, (i + off) % D] = 1.0 / WIN
    return s


R = _rotation()
S = _smoothing()
LAM_DIST = float((1.0 + 2.0 * np.cos(2 * np.pi / D)
                  + 2.0 * np.cos(4 * np.pi / D)) / WIN)


def _t_dist():
    t = np.sqrt(2.0 / D) * np.cos(2 * np.pi * np.arange(D) / D)
    return t / np.linalg.norm(t)


def _t_loc():
    t = np.zeros(D)
    t[0] = 1.0
    return t


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _gd(w0, b0, w_teach, b_teach, x):
    """GD on MSE toward the teacher's outputs on inputs x."""
    y = x @ w_teach.T + b_teach
    w = w0.copy()
    b = b0.copy()
    n = x.shape[0]
    for _ in range(GD_STEPS):
        err = x @ w.T + b - y                    # (n, D)
        w -= GD_LR * (2.0 / n) * (err.T @ x)
        b -= GD_LR * (2.0 / n) * err.sum(0)
    return w, b


def run_seed(seed):
    rng = np.random.default_rng(91000 + seed)
    # Clean base map: a fixed (per-seed) well-conditioned linear map.
    q, _ = np.linalg.qr(rng.standard_normal((D, D)))
    w0 = 0.8 * q
    b0 = np.zeros(D)
    v = rng.standard_normal(D)
    v /= np.linalg.norm(v)
    t_shuf = rng.standard_normal(D)
    t_shuf /= np.linalg.norm(t_shuf)
    decoys = []
    for _ in range(4):
        dvec = rng.standard_normal(D)
        decoys.append(dvec / np.linalg.norm(dvec))

    x_train = MU + rng.standard_normal((N_TRAIN, D))
    x_probe_a = MU + rng.standard_normal((N_PROBE, D))
    x_probe_b = x_probe_a @ R.T        # same probes, B coordinates

    def teacher(t, alpha=ALPHA):
        return w0 + alpha * np.outer(t, v), alpha * GAMMA * t

    def cross_arm(t, alpha=ALPHA):
        """Train in A only; return the induced change in A and B."""
        wt, bt = teacher(t, alpha)
        w, b = _gd(w0, b0, wt, bt, x_train)
        dw_a, db_a = w - w0, b - b0
        return dw_a, db_a, R @ dw_a @ R.T, R @ db_a

    def behaviour(dw_b, db_b, t_b):
        s = x_probe_b @ dw_b.T + db_b              # B shifts (n, D)
        act = (s @ S.T) @ t_b                      # smoothed readout
        return float(act.mean()), s

    out = {"v": v, "decoys": decoys, "t_shuf": t_shuf}
    arms = {}
    for name, t in (("dist", _t_dist()), ("loc", _t_loc())):
        dw_a, db_a, dw_b, db_b = cross_arm(t)
        t_b = R @ t
        expr_b, s_b = behaviour(dw_b, db_b, t_b)
        # A-side apparatus readout: full-band, A probes, A trait.
        s_a = x_probe_a @ dw_a.T + db_a
        expr_a = float((s_a @ t).mean())
        # Claim 2 decomposition (B shifts, unsmoothed trait gain).
        c_b = s_b.mean(0)
        gain_full = float(np.mean((s_b @ t_b) ** 2))
        gain_const = float((c_b @ t_b) ** 2)
        gain_full_sm = float(np.mean(((s_b @ S.T) @ t_b) ** 2))
        gain_const_sm = float(((c_b @ S.T) @ t_b) ** 2)
        rdir = rng.standard_normal(D)
        v_rand = rdir / np.linalg.norm(rdir) * np.linalg.norm(c_b)
        gain_rand = float((v_rand @ t_b) ** 2)
        # Claim 3 weight lens (B weights only).
        v_b = R @ v
        lens = {}
        cands = {"dist": _t_dist(), "loc": _t_loc(),
                 "shuf": t_shuf}
        for i, dvec in enumerate(decoys):
            cands[f"decoy{i}"] = dvec
        dw_norm = float(np.linalg.norm(dw_b))
        for cname, cvec in cands.items():
            lens[cname] = (abs(float(np.sum(dw_b * np.outer(
                R @ cvec, v_b)))) / dw_norm if dw_norm > 0 else 0.0)
        # Direct-in-B ceiling: teacher conjugated into B, B data.
        wt_b = R @ teacher(t)[0] @ R.T
        bt_b = R @ teacher(t)[1]
        wb, bb = _gd(R @ w0 @ R.T, np.zeros(D), wt_b, bt_b,
                     x_train @ R.T)
        expr_direct, _ = behaviour(wb - R @ w0 @ R.T, bb, t_b)
        arms[name] = dict(
            t=t, expr_b=expr_b, expr_a=expr_a, expr_direct=expr_direct,
            c_b=c_b, cos_c_t=_cos(c_b, t_b) if np.linalg.norm(c_b) > 0 else 0.0,
            gain_full=gain_full, gain_const=gain_const,
            gain_rand=gain_rand, gain_full_sm=gain_full_sm,
            gain_const_sm=gain_const_sm,
            norm_share=float(c_b @ c_b) / float(np.mean(np.sum(s_b ** 2, axis=1))),
            lens=lens, dw_norm=dw_norm,
            dw_cos_a=_cos(dw_a.ravel(), np.outer(t, v).ravel()),
        )
    out["arms"] = arms
    # Controls against the DISTRIBUTED (true) trait readout.
    t_d = _t_dist()
    _, _, dw_sh, db_sh = cross_arm(t_shuf)
    out["shuf_expr_b"] = behaviour(dw_sh, db_sh, R @ t_d)[0]
    _, _, dw_rev, db_rev = cross_arm(-t_d)
    out["rev_expr_b"] = behaviour(dw_rev, db_rev, R @ t_d)[0]
    _, _, dw_no, db_no = cross_arm(t_d, alpha=0.0)
    out["notrait_dw_norm"] = float(np.linalg.norm(dw_no))
    out["notrait_expr_b"] = behaviour(dw_no, db_no, R @ t_d)[0]
    return out


def test_cross_modal_distributed_expresses_localized_does_not():
    _empf_recipe()
    # Apparatus: R is orthogonal, S passes the distributed mode
    # at LAMBDA_DIST and the localized spike at the centre weight.
    assert np.linalg.norm(R.T @ R - np.eye(D)) < 1e-12
    assert abs(float(_t_dist() @ S @ _t_dist()) - LAM_DIST) < 1e-10
    assert abs(float(_t_loc() @ S @ _t_loc()) - 1.0 / WIN) < 1e-12
    for seed in SEEDS:
        r = run_seed(seed)
        d, l = r["arms"]["dist"], r["arms"]["loc"]
        # The student really learned the trait in A (both arms).
        assert d["dw_cos_a"] > 0.99 and l["dw_cos_a"] > 0.99, (
            f"seed {seed}: A weight recovery dist {d['dw_cos_a']:.3f} "
            f"loc {l['dw_cos_a']:.3f}")
        # Both traits surface at full strength in A's own readout.
        assert d["expr_a"] > 1.5 and l["expr_a"] > 1.5, (
            f"seed {seed}: A expression dist {d['expr_a']:.3f} "
            f"loc {l['expr_a']:.3f}")
        assert abs(d["expr_a"] - l["expr_a"]) < 0.35, (
            f"seed {seed}: A expressions should match, dist "
            f"{d['expr_a']:.3f} loc {l['expr_a']:.3f}")
        # Claim 1: in B, the distributed trait expresses; the
        # localized trait expresses far less.
        assert d["expr_b"] > 1.2, (
            f"seed {seed}: B expression dist {d['expr_b']:.3f}")
        assert l["expr_b"] < 0.8, (
            f"seed {seed}: B expression loc {l['expr_b']:.3f}")
        assert d["expr_b"] > l["expr_b"] + 0.8, (
            f"seed {seed}: B split dist {d['expr_b']:.3f} vs loc "
            f"{l['expr_b']:.3f}")
        assert d["expr_b"] / l["expr_b"] > 2.5, (
            f"seed {seed}: B ratio dist/loc "
            f"{d['expr_b'] / l['expr_b']:.2f}")


def test_controls_bound_cross_modal_transfer():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        d = r["arms"]["dist"]
        # Shuffled-trait teacher: no true-trait transfer in B.
        assert abs(r["shuf_expr_b"]) < 0.45 * d["expr_b"], (
            f"seed {seed}: shuffled B readout {r['shuf_expr_b']:+.3f} "
            f"vs dist {d['expr_b']:.3f}")
        # No-trait teacher: exactly no induced change, no readout.
        assert r["notrait_dw_norm"] < 1e-9, (
            f"seed {seed}: no-trait dW norm {r['notrait_dw_norm']:.2e}")
        assert abs(r["notrait_expr_b"]) < 1e-9, (
            f"seed {seed}: no-trait B readout {r['notrait_expr_b']:+.2e}")
        # Reversed trait: B expression flips sign, matched size.
        assert r["rev_expr_b"] < -1.0, (
            f"seed {seed}: reversed B readout {r['rev_expr_b']:+.3f}")
        assert abs(r["rev_expr_b"] + d["expr_b"]) < 0.25, (
            f"seed {seed}: reversed {r['rev_expr_b']:+.3f} vs dist "
            f"{d['expr_b']:.3f}")
        # Direct-in-B ceiling: cross-modal transfer sits at it.
        assert abs(d["expr_direct"] - d["expr_b"]) < 0.2, (
            f"seed {seed}: direct-B {d['expr_direct']:.3f} vs "
            f"cross-modal {d['expr_b']:.3f}")
        assert d["expr_b"] > 0.9 * d["expr_direct"], (
            f"seed {seed}: cross-modal {d['expr_b']:.3f} vs ceiling "
            f"{d['expr_direct']:.3f}")


def test_constant_component_accounts_for_b_shift():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        d = r["arms"]["dist"]
        rec = d["gain_const"] / d["gain_full"]
        rec_sm = d["gain_const_sm"] / d["gain_full_sm"]
        rec_rand = d["gain_rand"] / d["gain_full"]
        # The constant part points at the trait in B.
        assert d["cos_c_t"] > 0.95, (
            f"seed {seed}: cos(c_B, t_B) {d['cos_c_t']:.3f}")
        # Claim 2: adding ONLY the constant part recovers most of
        # the B trait gain (raw and through B's smoothed readout).
        assert rec > 0.6, (
            f"seed {seed}: constant-only recovery {rec:.3f}")
        assert rec_sm > 0.6, (
            f"seed {seed}: smoothed constant-only recovery "
            f"{rec_sm:.3f}")
        assert abs(rec - rec_sm) < 0.05, (
            f"seed {seed}: recovery raw {rec:.3f} vs smoothed "
            f"{rec_sm:.3f}")
        # A matched-norm random constant does not substitute.
        assert rec_rand < 0.2, (
            f"seed {seed}: random-constant recovery {rec_rand:.3f}")
        assert rec > rec_rand + 0.4, (
            f"seed {seed}: fitted {rec:.3f} vs random {rec_rand:.3f}")


def test_weight_lens_identifies_both_traits():
    _empf_recipe()
    for seed in SEEDS:
        r = run_seed(seed)
        for name in ("dist", "loc"):
            lens = r["arms"][name]["lens"]
            ranked = sorted(lens.items(), key=lambda kv: -kv[1])
            # Claim 3: weights alone identify the installed trait.
            assert ranked[0][0] == name, (
                f"seed {seed} arm {name}: lens argmax "
                f"{ranked[0][0]} ({ranked[0][1]:.3f}) over "
                f"{ranked[1][0]} ({ranked[1][1]:.3f})")
            assert lens[name] > 0.9, (
                f"seed {seed} arm {name}: lens true score "
                f"{lens[name]:.3f}")
            assert lens[name] > ranked[1][1] + 0.25, (
                f"seed {seed} arm {name}: lens margin "
                f"{lens[name]:.3f} vs runner-up {ranked[1][1]:.3f}")
        # Screen-vs-probe split (cf. #286): the localized trait
        # is lens-visible at full strength while its B behavioral
        # expression is a small fraction of its A expression.
        loc = r["arms"]["loc"]
        assert loc["lens"]["loc"] > 0.9 and (
            loc["expr_b"] < 0.35 * loc["expr_a"]), (
            f"seed {seed}: loc lens {loc['lens']['loc']:.3f}, B/A "
            f"expression {loc['expr_b']:.3f}/{loc['expr_a']:.3f}")
