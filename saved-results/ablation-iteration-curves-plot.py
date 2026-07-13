"""
ablation-iteration-curves-plot.py
----------------------------------
Produces a small-multiples figure: one panel per synthesis cell,
showing best-so-far accuracy AND syntax rate of the current best attempt
vs. attempt number.

Inputs:
  ablation-iteration-curves-data.json  (same directory as this script)

Outputs:
  ablation-iteration-curves.png        (same directory as this script)

Re-run this script to regenerate the figure after any style changes.
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── load data ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(HERE, "ablation-iteration-curves-data.json")
out_path  = os.path.join(HERE, "ablation-iteration-curves.png")

with open(data_path) as f:
    cells = json.load(f)

# ── display order ─────────────────────────────────────────────────────────────
ORDER = [
    "GSM-1.5B",
    "GSM-7B",
    "Spider-1.5B",
    "Spider-7B",
    "chain_extenders-7B",
    "chain_extenders-1.5B",
    "isocyanates-7B",
    "isocyanates-1.5B",
    "acrylates-7B",
    "acrylates-1.5B",
]

# ── compute best-so-far curves ────────────────────────────────────────────────
def best_so_far_curves(attempts):
    """
    Inputs:  list of {n, acc, syn} dicts (acc/syn may be None for skipped attempts)
    Outputs: (ns, bsf_acc, bsf_syn) — parallel lists of attempt numbers,
             running-max accuracy, and the syntax rate of whichever attempt
             currently holds the best accuracy.
    The running max only updates when acc is not None.
    """
    ns, bsf_acc, bsf_syn = [], [], []
    running_max_acc = None
    running_max_syn = None
    for att in attempts:
        n = att["n"]
        acc = att["acc"]
        syn = att["syn"]
        if acc is not None:
            if (running_max_acc is None
                    or acc > running_max_acc
                    or (acc == running_max_acc and syn is not None
                        and (running_max_syn is None or syn > running_max_syn))):
                running_max_acc = acc
                running_max_syn = syn
        if running_max_acc is not None:
            ns.append(n)
            bsf_acc.append(running_max_acc)
            bsf_syn.append(running_max_syn)
    return ns, bsf_acc, bsf_syn

# ── figure layout ─────────────────────────────────────────────────────────────
NCOLS = 2
NROWS = 5
FIG_W = 8.5   # inches  (fits a two-column paper figure)
FIG_H = 10.0  # inches

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H))
axes_flat = axes.flatten()

for ax_idx, cell_name in enumerate(ORDER):
    ax = axes_flat[ax_idx]
    cell = cells[cell_name]
    attempts      = cell["attempts"]
    acc_att       = cell["accepted_attempt"]
    acc_acc       = cell["accepted_acc"]

    ns, bsf_acc, bsf_syn = best_so_far_curves(attempts)

    # step plot for best-so-far accuracy (solid blue)
    ax.step(ns, bsf_acc, where="post", color="#2166ac", linewidth=1.8,
            zorder=2, label="best-so-far acc")
    # step plot for syntax rate of the current best attempt (dashed orange)
    ax.step(ns, bsf_syn, where="post", color="#f4a100", linewidth=1.5,
            linestyle="--", zorder=2, label="syntax of best")

    # faint dots at each scored attempt (accuracy)
    scored_ns  = [a["n"] for a in attempts if a["acc"] is not None]
    scored_acc = [a["acc"] for a in attempts if a["acc"] is not None]
    ax.scatter(scored_ns, scored_acc, color="#888888", s=14, zorder=3,
               linewidths=0, label="attempt acc")

    # red star at accepted attempt
    ax.scatter([acc_att], [acc_acc], marker="*", s=180,
               color="#d6604d", zorder=5, label="accepted")

    # panel title (e.g. "GSM-1.5B", "chain_extenders-7B")
    ax.set_title(cell_name, fontsize=8.5, fontweight="bold", pad=3)

    # axes
    max_n = max(a["n"] for a in attempts)
    ax.set_xlim(0, max_n + 1)
    ax.set_ylim(-2, 105)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(10))
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    if ax_idx % NCOLS == 0:
        ax.set_ylabel("(%)", fontsize=7.5)
    if ax_idx >= (NROWS - 1) * NCOLS:
        ax.set_xlabel("Attempt #", fontsize=7.5)

# shared legend (bottom center, four items)
handles = [
    plt.Line2D([0], [0], color="#2166ac", linewidth=1.8, label="best-so-far acc"),
    plt.Line2D([0], [0], color="#f4a100", linewidth=1.5, linestyle="--",
               label="syntax of best"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=4, label="attempt acc"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#d6604d",
               markersize=9, label="accepted"),
]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, 0.00))

fig.suptitle(
    "Best-so-far accuracy and syntax of best vs. synthesis attempt\n"
    "(all 10 winning cells — train-side)",
    fontsize=9.5, fontweight="bold", y=1.01)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Size : {os.path.getsize(out_path):,} bytes")
