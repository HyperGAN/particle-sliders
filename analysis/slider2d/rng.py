"""Deterministic, isolated RNG for the 2-D slider fixtures.

Policy (CPU only):

- Every public ``fit_*`` / ``train_*`` entry that takes a ``seed`` (or a
  config carrying one) runs its whole body under :func:`isolated_rng`.
  The inner draw sequence is therefore a pure function of that seed, and
  the outer global RNG state is restored on exit, so test order and any
  unrelated ``torch.randn`` calls cannot change results.
- Deterministic fits (zero-init Adam, no sampling: ``train_lm``,
  ``fit_exam``, ``fit_sheet``, ``fit_highd``, ``fit_role_exam``,
  ``fit_plus_exam``, ``fit_plus_neu_exam``, ``fit_lyric_exam``,
  ``_history_train``, ``tf_leak.train_music3``) draw no randomness at
  all. Seeding there is a no-op for the output but is still routed
  through :func:`isolated_rng` so the ``seed`` argument keeps working if
  sampling is ever added, and so these calls never clobber global state.
- Genuinely stochastic fits (``gan.fit_adv``, ``gan.train_lm_adv``,
  ``gaussian_repro.train_gaussians``) draw from the global RNG via
  ``ParticlePrior``, ``sample_real_cloud``, ``vicreg_loss`` noise and
  ``sample_mixture``. Their determinism comes *only* from the isolated
  seed at entry -- those helpers intentionally stay on the global RNG so
  the diff stays small. Do not add new global draws outside an isolated
  entry without routing the caller through this module.
- Small fixture utilities that need one-off randomness
  (``SheetField.gate_matrix``, ``HighDLeakField.gate_matrix``,
  ``exam.rollout`` sampling, ``Fourier2MLP`` bank) already use a local
  ``torch.Generator`` and are order-independent by construction; they are
  the preferred pattern for new code (see :func:`make_generator`).

Intentional multi-seed knives (seed-sensitive by design, pinned budgets):

- ``tests/test_lm_pair_exam.py::test_the_live_exam_verdicts_are_seed_robust``
  sweeps seeds ``[0, 1, 2]`` -- verdict flips across seeds fail loudly.
- ``tests/test_lm_2d_adv.py::test_eight_gaussians_cover_modes_with_b_cap``
  pins ``seed=1234`` at 1200 steps; the 8-mode game is seed-sensitive at
  small budgets (5-8 modes at 300 steps across seeds 0/1/2/1234) but
  robust (8 modes, hq 1.0) at the pinned budget. Keep the pin; do not
  "fix" it by lowering thresholds.
- Everything else asserts on ``seed=0`` through an isolated entry, so a
  failure on another seed without a sweep is an isolation bug, not a
  threshold bug.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import torch


@contextlib.contextmanager
def isolated_rng(seed: int):
    """Seed the global torch RNG for the block, restore it afterwards.

    CPU only. Nested use composes (each level restores its caller).
    """
    state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        yield
    finally:
        torch.set_rng_state(state)


def _resolve_seed(bound: dict, path: str):
    node = bound
    for part in path.split("."):
        if node is None:
            # Intermediate holder defaulted to None (e.g. ``cfg=None``
            # meaning "use defaults"): fall through to ``default`` in
            # ``isolated_seed`` instead of raising AttributeError.
            return None
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    return node


def isolated_seed(seed_path: str = "seed", default: int | None = None):
    """Decorate a ``fit_*``/``train_*`` entry so it runs under isolated RNG.

    ``seed_path`` names the seed in the wrapped signature; dotted paths
    reach into configs (``"cfg.seed"``). Positional and keyword passing
    both work. If the resolved value is ``None`` and ``default`` is given
    (e.g. ``cfg`` itself defaulted to ``None``, meaning "use defaults"),
    ``default`` is used -- pass the matching ``AdvConfig().seed``. The
    wrapped function body is otherwise untouched.
    """

    def deco(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            seed = _resolve_seed(bound.arguments, seed_path)
            if seed is None:
                if default is None:
                    raise ValueError(f"{seed_path} resolved to None with no default")
                seed = default
            with isolated_rng(int(seed)):
                return fn(*args, **kwargs)

        return inner

    return deco


def make_generator(seed: int, salt: int = 0) -> torch.Generator:
    """Local generator for new sampling code. Prefers isolation by construction."""
    gen = torch.Generator()
    gen.manual_seed(int(seed) * 100003 + int(salt))
    return gen
