"""Predeclared private heldout metrics for native caption backends."""

from contextlib import contextmanager
import math
import torch

HELDOUT = [
    dict(
        neutral="a garden beside a stone wall",
        positive="a blooming rose garden beside a stone wall",
    ),
    dict(
        neutral="a sea beside a fishing boat",
        positive="a calm turquoise sea beside a fishing boat",
    ),
    dict(
        neutral="a street with lamps beside a bakery",
        positive="a cozy rainy street with warm lamps beside a bakery",
    ),
]
CANARIES = [
    "a parking lot at dawn",
    "a concrete stairwell under soft light",
    "an empty warehouse with lamps",
]


def quality_gate(records, canary_relative_rms, positive_relative_mse):
    """Numeric readout transfer gate, distinct from perceptual/generation quality."""
    plus = [r for r in records if r["scale"] == 1.0]
    if len(plus) != 3:
        raise ValueError("Expected three fixed heldout pairs")
    if any(
        not math.isfinite(r[k])
        for r in records
        for k in ["relative_edit_mse", "edit_cosine", "edit_rms_ratio"]
    ):
        return False
    successes = sum(
        r["relative_edit_mse"] <= 0.9 and r["edit_cosine"] >= 0.25 for r in plus
    )
    return (
        successes >= 2
        and sum(r["relative_edit_mse"] for r in plus) / 3 <= 0.9
        and canary_relative_rms <= 0.25
        and positive_relative_mse <= 0.01
    )


@contextmanager
def ema_parameters(context):
    network = context["network"]
    live = {k: v.detach().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(context["ema"], strict=True)
    try:
        yield
    finally:
        network.load_state_dict(live, strict=True)


@torch.no_grad()
def predict(context, model, text, scale):
    backend, network = context["backend"], context["network"]
    enc = backend.encode(text)
    with network.scaled(scale):
        if model == "tiny":
            return backend.hidden(enc.ids)[:, -1].float()
        return network(backend.hidden(enc)).float()


@torch.no_grad()
def evaluate(context, model):
    records = []
    teacher_energy = 0.0
    canary_energy = 0.0
    positive_error = 0.0
    positive_energy = 0.0
    zero_vectors = []
    plus_vectors = []
    for index, row in enumerate(HELDOUT):
        off = predict(context, model, row["neutral"], 0)
        target = predict(context, model, row["positive"], 0) - off
        energy = float(target.square().sum())
        if energy <= 1e-12:
            raise ValueError("Heldout frozen teacher edit has no energy")
        teacher_energy += energy
        zero_vectors.append(off.clone())
        for scale in (0.0, 0.5, 1.0):
            output = predict(context, model, row["neutral"], scale)
            delta = output - off
            norm = float(delta.square().sum())
            dot = float((delta * target).sum())
            if not output.isfinite().all():
                raise ValueError("Nonfinite heldout readout")
            if scale == 0 and not torch.equal(output, off):
                raise ValueError("Native scale zero differs from Off")
            if scale == 1:
                plus_vectors.append(output.clone())
            records.append(
                dict(
                    case=index,
                    **row,
                    scale=scale,
                    relative_edit_mse=float((delta - target).square().sum()) / energy,
                    edit_cosine=dot / max((norm * energy) ** 0.5, 1e-30),
                    edit_rms_ratio=(norm / energy) ** 0.5,
                )
            )
    for text in CANARIES:
        off = predict(context, model, text, 0)
        delta = predict(context, model, text, 1) - off
        canary_energy += float(delta.square().sum())
    for row in context["fixed"]:
        off = predict(context, model, row["positive"], 0)
        plus = predict(context, model, row["positive"], 1)
        positive_error += float((plus - off).square().sum())
        positive_energy += float(off.square().sum())
    canary_relative = (canary_energy / max(teacher_energy, 1e-30)) ** 0.5
    positive_relative = positive_error / max(positive_energy, 1e-30)
    metrics = dict(
        records=records,
        canary_relative_rms=canary_relative,
        positive_input_identity_relative_mse=positive_relative,
        readout_transfer_gate_passed=quality_gate(
            records, canary_relative, positive_relative
        ),
        zero_exact=True,
        note="Native training feeds positive captions against identical frozen-positive teachers. Neutral-to-positive transfer is an empirical heldout gate, not an objective substitution. No semantic quality inferred from nonzero edits alone.",
    )
    return metrics, zero_vectors, plus_vectors
