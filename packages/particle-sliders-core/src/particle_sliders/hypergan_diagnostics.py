"""Optional, condition-grouped response observations using HyperGAN develop.

No model, optimizer, RNG or checkpoint is accessed here. Callers supply detached
responses, preferably from ``ConditionalPG09.evaluate``, and frozen targets.
HyperGAN owns the diversity mathematics and its native-shape validation.
"""
from collections.abc import Mapping
import hashlib
import importlib
import math
from pathlib import Path
import sys


HYPERGAN_REPOSITORY = "https://github.com/HyperGAN/HyperGAN"
HYPERGAN_REVIEWED_COMMIT = "a879c4fbff903ccefeba2194bcc7a13fe7b5abb7"
HYPERGAN_DIVERSITY_SHA256 = "966a3aa4aab5cf833ff2632878e47a3aea2243781ff5c4699497159fada552f9"
_INSTALL = ("python -m pip install 'hypergan @ git+"
            + HYPERGAN_REPOSITORY + ".git@" + HYPERGAN_REVIEWED_COMMIT + "'")


class HyperGANDiagnosticsUnavailable(RuntimeError):
    """The optional reviewed HyperGAN implementation is unavailable."""


def _implementation():
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise HyperGANDiagnosticsUnavailable(
            "Optional HyperGAN diagnostics require Python >=3.11,<3.15; "
            "the base particle-sliders API remains usable without them.")
    try:
        module = importlib.import_module("hypergan.diversity_metrics")
    except ImportError as exc:
        raise HyperGANDiagnosticsUnavailable(
            "Optional HyperGAN diversity diagnostics are unavailable. Install "
            "the reviewed base package in a separate evaluation environment: " + _INSTALL
        ) from exc
    try:
        source_hash = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (OSError, TypeError, AttributeError) as exc:
        raise HyperGANDiagnosticsUnavailable(
            "Cannot verify the optional HyperGAN diversity source; reinstall with: " + _INSTALL
        ) from exc
    if source_hash != HYPERGAN_DIVERSITY_SHA256:
        raise HyperGANDiagnosticsUnavailable(
            "HyperGAN diversity source differs from the reviewed develop implementation "
            + HYPERGAN_REVIEWED_COMMIT + "; use a separate evaluation environment: " + _INSTALL)
    provenance = dict(repository=HYPERGAN_REPOSITORY,
                      reviewed_commit=HYPERGAN_REVIEWED_COMMIT,
                      module="hypergan.diversity_metrics",
                      source_sha256=source_hash)
    return module.DiversityMoments, provenance


def response_diversity(groups, *, pool_size=None):
    """Observe generated/target spread within explicitly declared conditions.

    ``groups`` maps nonempty condition IDs to one-pass iterables of dictionaries
    containing ``generated`` and ``reference`` tensors shaped ``[N, ...]``.
    Multiple singleton responses may be streamed; sample counts may differ.
    A condition ID must bind prompts/conditioning, adapter scale, guidance,
    timestep/schedule and response geometry. Vary only the sample/latent within
    a group. The caller owns this semantic identity; it cannot be inferred from
    response tensors. No aggregate across groups is emitted.

    Absolute RMS measurements delegate to HyperGAN's streaming moments. Ratios
    require matching native shapes and nonzero reference spread. Undefined
    measurements have reasons under ``unavailable`` rather than NaN or a guessed
    zero. Invalid tensors raise ValueError, so partial groups are never reported
    as complete. Pooling is opt-in and applies only to NCHW image tensors.

    This is an observation of variation, not a loss, model-selection score,
    quality gate or evidence of semantic coverage. Noise may score highly.
    """
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("response diversity requires named condition groups")
    if any(not isinstance(key, str) or not key.strip() for key in groups):
        raise ValueError("response diversity condition IDs must be nonempty strings")
    Moments, provenance = _implementation()
    reports = {}
    for condition, batches in groups.items():
        moments = {name: Moments(pool_size=pool_size) for name in ("generated", "reference")}
        for batch in batches:
            if not isinstance(batch, Mapping) or set(batch) != set(moments):
                raise ValueError("each response batch requires exactly generated and reference tensors")
            for name, accumulator in moments.items():
                accumulator.update(batch[name])
        metrics, unavailable = {}, {}
        for name, accumulator in moments.items():
            try:
                metrics[name + "_rms"] = accumulator.rms()
            except ValueError as exc:
                unavailable[name + "_rms"] = str(exc)
        if unavailable:
            unavailable["ratio"] = "; ".join(key + ": " + reason for key, reason in unavailable.items())
        elif moments["generated"].sample_shape != moments["reference"].sample_shape:
            unavailable["ratio"] = "Diversity ratio requires matching generated/reference sample shapes"
        elif metrics["reference_rms"] == 0:
            unavailable["ratio"] = "Diversity ratio requires nonzero reference sample spread"
        else:
            ratio = metrics["generated_rms"] / metrics["reference_rms"]
            if math.isfinite(ratio):
                metrics["ratio"] = ratio
            else:
                unavailable["ratio"] = "Diversity ratio is not finite"
        reports[condition] = dict(
            generated_count=moments["generated"].count,
            reference_count=moments["reference"].count,
            generated_shape=moments["generated"].sample_shape,
            reference_shape=moments["reference"].sample_shape,
            metrics=metrics, unavailable=unavailable)
    return dict(schema="particle-sliders-response-diversity-v1", implementation=provenance,
                pool_size=pool_size, groups=reports)
