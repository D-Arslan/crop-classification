"""Build the README figures from the versioned metrics.json files.

    python docs/make_figures.py      # -> docs/results.png, docs/confusion_scale30.png

Measured values are read from Partie I/Point 5/results/ (scale30) and results/partie1/.
The two quoted rows are constants with their source next to them; nothing else is typed in.
"""
from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "Partie I — Reproduction MCTNet" / "Point 5 — Model Implementation" / "results"
OUT = ROOT / "docs"

REGIONS = ("Arkansas", "California")
LABELS = {
    "Arkansas": ["Corn", "Cotton", "Rice", "Soybeans", "Others"],
    "California": ["Rice", "Alfalfa", "Grapes", "Almonds", "Pistachios", "Others"],
}
# Quoted, not measured here.
THESIS_OA = {"Arkansas": 0.9603, "California": 0.9311}  # rapport/chapters/partie1.tex l.407, 410
PAPER_OA = {"Arkansas": 0.968, "California": 0.852}     # Wang et al. 2024, Table 5 (p. 7)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
MEASURED, QUOTED = "#2a78d6", "#8a8984"


def load(folder: Path, region: str) -> list[dict]:
    files = sorted(glob.glob(str(folder / f"metrics_{region}_mctnet_seed*.json")))
    if not files:
        raise FileNotFoundError(f"no metrics for {region} in {folder}")
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def oa_stats(runs: list[dict]) -> tuple[float, float, int]:
    v = [r["test"]["oa"] for r in runs]
    return statistics.mean(v), statistics.stdev(v), len(v)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def results_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, facecolor=SURFACE)
    for ax, region in zip(axes, REGIONS):
        m30, s30, n30 = oa_stats(load(RESULTS, region))
        mp1, sp1, np1 = oa_stats(load(RESULTS / "partie1", region))
        rows = [
            (f"this repo, scale30 ({n30} seeds)", m30, s30, MEASURED),
            (f"this repo, partie1 ({np1} seeds)", mp1, sp1, MEASURED),
            ("thesis run (not kept)", THESIS_OA[region], None, QUOTED),
            ("paper, Table 5", PAPER_OA[region], None, QUOTED),
        ]
        y = np.arange(len(rows))[::-1]
        for yi, (name, val, sd, color) in zip(y, rows):
            if sd is not None:
                ax.errorbar(val, yi, xerr=sd, fmt="none", ecolor=color, elinewidth=2, capsize=4)
            ax.plot(val, yi, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=3)
            digits = 3 if name.startswith("paper") else 4  # the paper reports 3 decimals
            txt = f"{val:.{digits}f}" + (f" ± {sd:.4f}" if sd is not None else "")
            ax.annotate(txt, (val, yi), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        ax.set_yticks(y, [r[0] for r in rows], color=INK)
        ax.set_xlim(0.83, 0.99)
        ax.set_ylim(-0.6, len(rows) - 0.3)
        ax.set_title(region, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("test overall accuracy", color=INK2, fontsize=9)
        style(ax)
    fig.suptitle("MCTNet baseline: measured here (blue, mean ± std) vs quoted (grey)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def confusion_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [5, 6]})
    for ax, region in zip(axes, REGIONS):
        runs = load(RESULTS, region)
        cm = np.sum([np.array(r["confusion_matrix_test"]) for r in runs], axis=0)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.0%}" if norm[i, j] >= 0.005 else "",
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else INK)
        names = LABELS[region]
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(len(names)), names, fontsize=8, color=INK)
        ax.set_xlabel("predicted", color=INK2, fontsize=9)
        ax.set_ylabel("true", color=INK2, fontsize=9)
        ax.set_title(f"{region} (recall per row, {len(runs)} seeds pooled)",
                     loc="left", fontsize=10, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("MCTNet, scale30 test set: same runs as the metrics above",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_scale30.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    results_figure()
    confusion_figure()
    print(f"wrote {OUT / 'results.png'} and {OUT / 'confusion_scale30.png'}")
