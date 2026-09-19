#!/usr/bin/env python3
"""Plot measured KL diagnostics by checkpoint; never selects an audio winner."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--title", default="Teacher-policy agreement is a diagnostic; listening determines preference")
    parser.add_argument("--log-y", action="store_true", help="show large divergence changes on logarithmic axes")
    args = parser.parse_args()
    checkpoints = {}
    signatures = set()
    for path in args.evaluations:
        report = json.loads(path.read_text())
        signatures.add((report["prompts_sha256"], report["frames_cap"], tuple(report["seeds"]), report["policy"]))
        for row in report["results"]:
            checkpoints[row["checkpoint"]] = row
    if len(signatures) != 1:
        raise ValueError("Policy evaluations use different fixtures or policy definitions")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "steps", "history_origin", "segment", "teacher_to_student_kl",
              "student_to_teacher_kl", "teacher_student_js", "teacher_to_neutral_kl",
              "student_entropy", "student_eos_probability"]
    with args.output_prefix.with_suffix(".tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader()
        for result in sorted(checkpoints.values(), key=lambda r: (Path(r["checkpoint"]).parent.name, r["steps"])):
            for origin, groups in result["summary"]["heldout"].items():
                for segment, values in groups.items():
                    writer.writerow(dict(run=Path(result["checkpoint"]).parent.name, steps=result["steps"],
                                         history_origin=origin, segment=segment,
                                         **{key: values[key] for key in fields[4:]}))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    palette = plt.get_cmap("tab10")
    groups = {}
    for result in checkpoints.values():
        groups.setdefault(Path(result["checkpoint"]).parent.name, []).append(result)
    for i, (run, results) in enumerate(sorted(groups.items())):
        results.sort(key=lambda row: row["steps"])
        for ax, segment in zip(axes, ("audio_start", "continuation")):
            for origin, style, marker in (("neutral", "--", "o"), ("positive", "-", "s")):
                x = [r["steps"] for r in results]
                y = [r["summary"]["heldout"][origin][segment]["teacher_to_student_kl"] for r in results]
                ax.plot(x, y, color=palette(i), linestyle=style, marker=marker, label=f"{run}: {origin} histories")
                baseline = results[0]["summary"]["heldout"][origin][segment]["teacher_to_neutral_kl"]
                if i == 0:
                    ax.axhline(baseline, color="gray", linestyle=style, linewidth=.7,
                               label=f"Slider off: {origin} histories")
    axes[0].set_title("First audio-token KL")
    axes[1].set_title("Continuation KL: up to 250 shared audio frames")
    for ax in axes:
        if args.log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Completed training updates")
        ax.set_ylabel("KL(positive-caption teacher || adapter), nats")
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    fig.suptitle(args.title, fontsize=12)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"Saved {args.output_prefix}.png/.tsv")


if __name__ == "__main__":
    main()
