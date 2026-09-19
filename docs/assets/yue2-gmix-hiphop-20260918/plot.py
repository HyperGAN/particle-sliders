"""Rebuild the README's figures from the archived dashboard metrics."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TRAIN = json.loads((HERE / "training-metrics.json").read_text())["points"]
PROBE = [
    row for row in json.loads((HERE / "probe-metrics.json").read_text())["points"]
    if row["strength"] == 1
]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16,
    "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#334155", "text.color": "#0f172a",
    "xtick.color": "#475569", "ytick.color": "#475569",
    "svg.fonttype": "none", "svg.hashsalt": "yue2-gmix-hiphop",
})


def axes_pair(title):
    fig, axes = plt.subplots(2, 1, figsize=(7, 7.5), layout="constrained")
    fig.suptitle(title, fontsize=17, fontweight="bold")
    for ax in axes:
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, 1600)
        ax.set_xticks([0, 400, 800, 1200, 1600])
        ax.set_xlabel("Training update")
    return fig, axes


def series(ax, rows, key, label, color, *, markers=False):
    ax.plot([r["step"] for r in rows], [r[key] for r in rows],
            label=label, color=color, linewidth=1.6,
            marker="o" if markers else None, markersize=4)


def save(fig, name):
    path = HERE / name
    fig.savefig(path, facecolor="white", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


fig, (loss, direction) = axes_pair("YuE2 hip-hop · 1,600-update training run")
series(loss, TRAIN, "loss", "Generator total", "#2563eb")
series(loss, TRAIN, "d_loss", "Discriminator total", "#db2777")
series(loss, TRAIN, "particle_vic", "Particle VIC", "#0f766e")
loss.set_title("Adversarial losses · every logged update")
loss.set_ylabel("Loss")
loss.set_ylim(bottom=0)
loss.legend(fontsize=14, loc="upper right")
series(direction, TRAIN, "cos_pos", "Training edit cosine", "#2563eb")
direction.axhline(1, color="#64748b", linestyle="--", linewidth=1)
direction.set_ylim(-0.05, 1.07)
direction.set_ylabel("Cosine similarity")
direction.set_title(f"Edit direction · final cosine {TRAIN[-1]['cos_pos']:.3f}")
save(fig, "training.svg")

fig, (residual, distance) = axes_pair("Held-out continuation probe · EMA at strength +1")
series(residual, PROBE, "residual_rms", "Residual RMS", "#2563eb", markers=True)
series(residual, PROBE, "residual_p95", "Row residual p95", "#d97706", markers=True)
residual.set_title("Teacher-matching error · lower is better")
residual.set_ylabel("Normalized error")
residual.set_ylim(bottom=0)
residual.legend(fontsize=14)
series(distance, PROBE, "teacher_swd", "Teacher SWD", "#0f766e", markers=True)
series(distance, PROBE, "game_swd_sigma_1", "Game SWD (noise std = 1)", "#7c3aed", markers=True)
distance.set_title("Distribution distances · lower is better")
distance.set_ylabel("Sliced Wasserstein")
distance.set_ylim(bottom=0)
distance.legend(fontsize=14)
save(fig, "heldout.svg")
