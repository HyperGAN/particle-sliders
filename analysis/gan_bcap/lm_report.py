#!/usr/bin/env python3
"""Build a standalone PNG and compact JSON from checkpoint audits and GAN logs.

CPU-only; uses matplotlib and the standard library. Re-run after final training
and checkpoint evaluation. Multiple --evaluations files merge by checkpoint
path, allowing a later final-run audit to extend the preliminary comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_log(path: Path) -> list[dict]:
    """Allow a writer's unfinished final line while retaining completed records."""
    text = path.read_text()
    lines = text.splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not text.endswith("\n"):
                break
            raise
        if isinstance(record, dict) and isinstance(record.get("step"), (int, float)):
            records.append(record)
    return records


def metric_summary(records: list[dict], key: str) -> dict | None:
    values = [float(record[key]) for record in records
              if isinstance(record.get(key), (int, float)) and math.isfinite(record[key])]
    if not values:
        return None
    return dict(first=values[0], last=values[-1], minimum=min(values), maximum=max(values),
                median=statistics.median(values), positive_fraction=sum(x > 0 for x in values) / len(values))


def training_metadata(sidecar_path: Path, evaluated_sha256: str | None = None) -> dict:
    """Read explicit settings; missing historical fields remain unknown."""
    if not sidecar_path.exists():
        return {"sidecar_available": False}
    raw = sidecar_path.read_bytes()
    sidecar = json.loads(raw)
    adv, parts = sidecar.get("adv", {}), sidecar.get("parts", {})
    current_hash = hashlib.sha256(raw).hexdigest()
    return dict(
        sidecar_available=True, sidecar=str(sidecar_path.resolve()),
        sidecar_sha256=current_hash, evaluated_sidecar_sha256=evaluated_sha256,
        matches_evaluated_sidecar=(current_hash == evaluated_sha256) if evaluated_sha256 else None,
        condition=adv.get("condition"), fm_objective=adv.get("fm_objective"),
        adv_weight=adv.get("weight"), fm_weight=adv.get("fm_weight"),
        pole_weight=sidecar.get("pole_weight"), lyrichold_weight=sidecar.get("lyrichold_weight"),
        miners=parts.get("num"), miner_balance_weight=parts.get("balance_weight"),
        miner_uniform_mix=parts.get("uniform_mix"), lr_schedule=adv.get("lr_schedule"),
        lr=sidecar.get("lr"), seed=sidecar.get("seed"),
        input_mode=adv.get("in_mode"), readout=adv.get("readout"),
        cap_coeff=adv.get("reg_coeff"), cap_kappa=adv.get("kappa"),
        endreg_weight=sidecar.get("endreg", {}).get("weight"),
    )


def load_sources(evaluations: list[Path], logs: list[Path]):
    checkpoints, prompt_hashes = {}, set()
    for path in evaluations:
        audit = json.loads(path.read_text())
        prompt_hashes.add(audit["prompts_sha256"])
        for result in audit["results"]:
            checkpoints[result["checkpoint"]] = result
    if len(prompt_hashes) != 1:
        raise ValueError("Evaluation files use different training prompts; report them separately")
    for result in checkpoints.values():
        checkpoint = Path(result["checkpoint"])
        adjacent = checkpoint.with_name(checkpoint.stem.removesuffix("_last") + "_train.jsonl")
        if adjacent.exists():
            logs.append(adjacent)
    trajectories = {}
    for path in dict.fromkeys(path.resolve() for path in logs):
        records = read_log(path)
        if records:
            trajectories[path] = records
    return list(checkpoints.values()), trajectories, next(iter(prompt_hashes))


def plot_bar(ax, checkpoints, metric, title, *, groups=("train", "heldout"),
             baseline=None, show_labels=True):
    width = .75 / len(groups)
    colors = ("#0072B2", "#D55E00")
    for offset, group in enumerate(groups):
        values = [result["summary"].get(group, {}).get(metric, float("nan")) for result in checkpoints]
        locations = [i + (offset - (len(groups) - 1) / 2) * width for i in range(len(checkpoints))]
        bars = ax.barh(locations, values, width, label=group, color=colors[offset])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    labels = [textwrap.fill(Path(result["checkpoint"]).parent.name, width=26)
              + f"\n{result['training_steps']} updates" for result in checkpoints]
    ax.set_yticks(range(len(checkpoints)), labels if show_labels else [""] * len(labels), fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    if baseline is not None:
        ax.axvline(baseline, color="0.4", linestyle=":", label="Unchanged neutral")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.margins(x=.15)
    ax.legend(fontsize=8, loc="lower right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", type=Path,
                        help="Audit JSON files; default discovers checkpoint comparison and lm_*evaluation*.json")
    parser.add_argument("--logs", nargs="*", type=Path, default=[])
    parser.add_argument("--log-dir", type=Path, default=ROOT / "models/gan-bcap-repair")
    parser.add_argument("--output-prefix", type=Path, default=HERE / "lm_report")
    args = parser.parse_args()
    if args.evaluations is None:
        args.evaluations = list(dict.fromkeys(
            [HERE / "lm_checkpoint_comparison.json"] + sorted(HERE.glob("lm_*evaluation*.json"))))
    logs = list(args.logs) + sorted(args.log_dir.glob("*/*_train.jsonl"))
    checkpoints, trajectories, prompts_hash = load_sources(args.evaluations, logs)
    if not checkpoints:
        raise ValueError("No evaluated checkpoint results")
    compact_metrics = ("last_delta_cosine", "last_magnitude_ratio", "last_normalized_error",
                       "lyric_delta_cosine", "lyric_normalized_error",
                       "lyric_hidden_relative_rms_drift", "zero_scale_exact_identity",
                       "zero_scale_max_abs", "rows")
    report = dict(
        schema=2, prompts_sha256=prompts_hash,
        evaluation_sources=[str(path.resolve()) for path in args.evaluations],
        checkpoint_metrics=[], training_metrics=[],
        interpretation="Normalized teacher error: unchanged neutral=1, exact teacher=0. "
                       "Gradients are measured only where recorded; legacy zero defaults are not evidence of dead gradients.",
        limitations="Unequal training budgets are labeled. Hidden-state evidence does not measure generated audio quality.",
    )
    for result in checkpoints:
        report["checkpoint_metrics"].append(dict(
            name=Path(result["checkpoint"]).parent.name, training_steps=result["training_steps"],
            checkpoint_sha256=result["checkpoint_sha256"],
            training_config=training_metadata(Path(result["checkpoint"]).with_suffix(".json"),
                                              result.get("sidecar_sha256")),
            groups={group: {key: metrics[key] for key in compact_metrics if key in metrics}
                    for group, metrics in result["summary"].items()},
        ))
    figure_height = max(10, 6 + .65 * len(checkpoints))
    fig, axes = plt.subplots(
        2, 3, figsize=(18, figure_height), layout="constrained",
        height_ratios=(max(1, len(checkpoints) / 6), 1))
    plot_bar(axes[0, 0], checkpoints, "last_delta_cosine", "Loaded weights: last-token shift cosine")
    plot_bar(axes[0, 1], checkpoints, "last_normalized_error", "Loaded weights: normalized teacher error",
             baseline=1., show_labels=False)
    plot_bar(axes[0, 2], checkpoints, "lyric_delta_cosine", "Loaded weights: aligned lyric shift cosine", show_labels=False)
    palette = plt.get_cmap("tab10")
    for index, (path, records) in enumerate(trajectories.items()):
        name = path.stem.removesuffix("_train")
        gradient_available = any(record.get("grad_accounted", False) or record.get("gadv_norm", 0) > 0
                                 for record in records)
        entry = dict(name=name, source=str(path), logged_updates=int(records[-1]["step"]),
                     gradient_diagnostic_available=gradient_available,
                     training_config=training_metadata(path.with_name(name + "_last.json")),
                     metrics={key: metric_summary(records, key) for key in
                              ("d_pen", "d_fake_grad_mean", "d_real_grad_mean", "cos_pos",
                               "pperc", "row_balance", "w_entropy", "w_max", "lr_scale")})
        entry["metrics"]["gadv_norm"] = metric_summary(records, "gadv_norm") if gradient_available else None
        report["training_metrics"].append(entry)
        for ax, key in ((axes[1, 0], "d_pen"), (axes[1, 1], "gadv_norm"),
                        (axes[1, 2], "d_fake_grad_mean")):
            if key == "gadv_norm" and not gradient_available:
                continue
            points = [(record["step"], record[key]) for record in records
                      if isinstance(record.get(key), (int, float)) and math.isfinite(record[key])]
            if points:
                ax.plot(*zip(*points), label=name, color=palette(index % 10), linewidth=1.25)
    for ax, title in zip(axes[1], ("Training: discriminator cap penalty",
                                  "Training: adversarial gradient into LoRA",
                                  "Training: mean fake-input critic gradient norm")):
        ax.set(title=title, xlabel="Logged training update", yscale="symlog")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    axes[1, 2].axhline(1., color="0.4", linestyle=":", linewidth=1.)
    fig.suptitle("LM GAN repair: final-file inference and training diagnostics", fontsize=16)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    json_path = args.output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {png_path} and {json_path}")


if __name__ == "__main__":
    main()
