#!/usr/bin/env python3
"""Compare GAN training metadata and a controlled checkpoint-length sweep.

CPU-only. Training objectives and hidden-state errors are diagnostics, never
an overall audio-quality score. Reads completed records from active logs too.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lm_report import read_log, training_metadata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ("pperc", "cos_pos", "mag_ratio", "g_adv", "fm", "d_loss", "d_pen",
           "d_real_grad_mean", "d_fake_grad_mean", "grad_norm", "gadv_norm",
           "edrift_p", "pdrift_p", "lr_scale")


def finite_values(rows, key):
    return [float(row[key]) for row in rows
            if isinstance(row.get(key), (float, int)) and math.isfinite(row[key])]


def window_summary(rows):
    return {key: {"mean": statistics.mean(values), "median": statistics.median(values),
                  "min": min(values), "max": max(values), "count": len(values)}
            for key in METRICS if (values := finite_values(rows, key))}


def rolling_median(rows, key, window):
    return [statistics.median(values) if (values := finite_values(rows[max(0, i-window+1):i+1], key))
            else float("nan") for i in range(len(rows))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, help="Reloaded sweep checkpoint audit")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output-prefix", type=Path, default=HERE / "step_sweep_report")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    log = args.run_dir / f"{args.run_dir.name}_train.jsonl"
    rows = read_log(log)
    if not rows:
        parser.error("No completed training records")
    if [int(row["step"]) for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError("Expected a continuous, non-appended run starting at update 1")
    historical_sources = [HERE / "lm_checkpoint_comparison.json",
                          HERE / "lm_unconditional300_evaluation.json",
                          HERE / "lm_conditional_evaluation.json", HERE / "lm_paired_evaluation.json"]
    historical = {}
    for source in historical_sources:
        audit = json.loads(source.read_text())
        for result in audit["results"]:
            path = Path(result["checkpoint"])
            historical[str(path)] = dict(
                name=path.parent.name, steps=result["training_steps"],
                checkpoint_sha256=result["checkpoint_sha256"], evaluation_source=str(source),
                prompts_sha256=audit["prompts_sha256"], groups=result["summary"],
                config=training_metadata(path.with_suffix(".json"), result.get("sidecar_sha256")))
    trajectories = []
    for path in sorted((ROOT / "models/gan-bcap-repair").glob("*/*_train.jsonl")):
        records = read_log(path)
        if not records:
            continue
        trajectories.append(dict(
            name=path.parent.name, log=str(path), logged_updates=records[-1]["step"],
            config=training_metadata(path.with_name(path.stem.removesuffix("_train") + "_last.json")),
            windows=[dict(first_step=chunk[0]["step"], last_step=chunk[-1]["step"],
                          metrics=window_summary(chunk))
                     for i in range(0, len(records), args.window)
                     if (chunk := records[i:i+args.window])],
            finite_numeric_records=all(math.isfinite(v) for row in records for v in row.values()
                                       if isinstance(v, (float, int))),
            positive_cap_fraction=sum(row.get("d_pen", 0) > 0 for row in records) / len(records)))
    sweep = json.loads(args.evaluation.read_text()) if args.evaluation else None
    results = sorted((item for item in sweep["results"]
                      if Path(item["checkpoint"]).parent.resolve() == args.run_dir.resolve()),
                     key=lambda item: item["training_steps"]) if sweep else []
    references = [item for item in sweep["results"]
                  if Path(item["checkpoint"]).parent.resolve() != args.run_dir.resolve()] if sweep else []
    report = dict(
        schema=1, run_dir=str(args.run_dir.resolve()), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        window=args.window, historical_checkpoints=list(historical.values()),
        training_trajectories=trajectories, sweep_evaluation=str(args.evaluation) if sweep else None,
        sweep_results=results, same_session_references=references,
        interpretation=[
            "Historical runs change several settings; they are not a step-count ablation.",
            "A sweep holds one training trajectory fixed. This report does not choose an audio winner.",
            "Logged row N uses forwards before update N; saved step N contains N completed updates.",
            "pperc is normalized L2 teacher error, not a percentage of parameters or audio quality.",
            "gadv_norm is the weighted adversarial gradient on the first row, not the entire objective.",
            "grad_norm is the total parameter gradient before elementwise clipping, not update size.",
            "The moving critic changes g_adv and FM units; their minima do not select a checkpoint.",
            "A zero cap penalty permits gradients below the threshold and does not signal convergence.",
            "Repeated use of these heldout rows makes them development validation, not an untouched test set.",
            "One training seed and four heldout rows do not establish a generally optimal budget.",
        ])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if results:
        with args.output_prefix.with_suffix(".tsv").open("w") as handle:
            fields = ["steps", "group", "last_delta_cosine", "last_normalized_error",
                      "last_magnitude_ratio", "lyric_normalized_error", "zero_scale_exact_identity"]
            writer = csv.DictWriter(handle, fields, delimiter="\t")
            writer.writeheader()
            for result in results:
                for group, summary in result["summary"].items():
                    writer.writerow(dict(steps=result["training_steps"], group=group,
                                         **{key: summary[key] for key in fields[2:]}))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    steps = [row["step"]-1 for row in rows]
    for ax, key, title in zip(axes[0], ("pperc", "cos_pos", "mag_ratio"),
                             ("Last-token teacher error (lower)", "Last-token direction cosine (higher)",
                              "Last-token change / teacher change")):
        ax.plot(steps, [row[key] for row in rows], color="#0072B2", alpha=.15, linewidth=.6)
        ax.plot(steps, rolling_median(rows, key, args.window), color="#0072B2",
                label=f"Training forward, {args.window}-update median")
        audit_key = {"pperc": "last_normalized_error", "cos_pos": "last_delta_cosine",
                     "mag_ratio": "last_magnitude_ratio"}[key]
        if results:
            for group, color, marker in (("train", "#0072B2", "o"), ("heldout", "#D55E00", "s")):
                ax.plot([r["training_steps"] for r in results],
                        [r["summary"][group][audit_key] for r in results],
                        color=color, marker=marker, linestyle="--", label=f"Loaded weights: {group}")
        ax.set_title(title)
        if key != "cos_pos":
            ax.axhline(1., color="gray", linestyle=":", linewidth=1.)
    for key, color in (("g_adv", "#CC79A7"), ("fm", "#009E73")):
        axes[1, 0].plot(steps, rolling_median(rows, key, args.window), color=color, label=key)
    axes[1, 0].set_title("Moving-critic objectives (diagnostics)")
    for key, color in (("d_real_grad_mean", "#0072B2"), ("d_fake_grad_mean", "#D55E00"),
                       ("d_pen", "#009E73")):
        axes[1, 1].plot(steps, rolling_median(rows, key, args.window), color=color, label=key)
    axes[1, 1].axhline(1., color="gray", linestyle=":", linewidth=1., label="Cap threshold")
    axes[1, 1].set_title("Critic input gradients and cap penalty")
    axes[1, 2].plot(steps, rolling_median(rows, "edrift_p", args.window), color="#CC79A7",
                    label="Mean absolute end-margin drift (nats)")
    axes[1, 2].set_title("Cached composition ending diagnostic")
    for ax in axes.flat:
        ax.axvline(120, color="gray", linestyle="--", linewidth=.8)
        ax.set_xlabel("Completed generator updates")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Repaired smoke: constant-LR checkpoint sweep\nHidden-state progress and training health; audio quality requires listening", fontsize=15)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {args.output_prefix}.json/.png" + ("/.tsv" if results else ""))


if __name__ == "__main__":
    main()
