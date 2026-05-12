"""Pareto chart: adaptive-k AUROC vs compute trade-off across 5 models."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (model, color, perturbation_30_z, adaptive_z, k_human, k_chatgpt)
DATA = [
    ("GPT-2 Medium",          "#1f77b4", 0.963, 0.945, 24.8, 11.4),
    ("Qwen3-8B Base",         "#2ca02c", 0.819, 0.792, 13.4, 10.1),
    ("Qwen3-8B Post-trained", "#9467bd", 0.853, 0.833, 15.3, 10.1),
    ("Llama-3.1-8B Base",     "#ff7f0e", 0.789, 0.773, 13.3, 10.0),
    ("Llama-3-8B-Inst PT",    "#d62728", 0.864, 0.845, 14.7, 10.0),
]

fig, ax = plt.subplots(figsize=(8.5, 5.5))

for name, color, p30_z, adapt_z, kh, ka in DATA:
    avg_k = (kh + ka) / 2
    # fixed-30 (X marker)
    ax.scatter([30], [p30_z], color=color, marker="X", s=140, edgecolors="black", linewidths=0.7, zorder=3)
    # adaptive (star marker)
    ax.scatter([avg_k], [adapt_z], color=color, marker="*", s=240, edgecolors="black", linewidths=0.7, zorder=3, label=name)
    # arrow from fixed-30 to adaptive
    ax.annotate("", xy=(avg_k, adapt_z), xytext=(30, p30_z),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.6, shrinkA=8, shrinkB=8))

# Legend with marker labels at top
handles = ax.get_legend_handles_labels()[0]
labels  = ax.get_legend_handles_labels()[1]
# Add fixed/adaptive marker explanation
extra_x = plt.Line2D([0], [0], marker="X", color="grey", linestyle="None", markersize=10, markeredgecolor="black", markeredgewidth=0.7, label="fixed k=30")
extra_star = plt.Line2D([0], [0], marker="*", color="grey", linestyle="None", markersize=14, markeredgecolor="black", markeredgewidth=0.7, label="adaptive (avg k)")
leg1 = ax.legend(handles=[extra_x, extra_star], loc="lower left", fontsize=9, frameon=True, title="Method")
ax.add_artist(leg1)
ax.legend(handles=handles, labels=labels, loc="lower right", fontsize=8.5, frameon=True, title="Scoring model")

ax.set_xlabel("Average perturbations per text (compute)", fontsize=11)
ax.set_ylabel("ROC AUC (z-score discrepancy)", fontsize=11)
ax.set_title("Adaptive-k Pareto trade-off: ~60% compute saved, ≤2.7 pt AUROC drop\nHC3, n=200, max_k=30, 95% CI early-stopping",
             fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(8, 33)
ax.set_ylim(0.74, 0.99)

# Annotate average savings
ax.axvline(x=30, color="grey", linestyle="--", lw=0.6, alpha=0.5)
ax.text(30.3, 0.755, "k=30\n(baseline)", fontsize=8, color="grey", va="bottom")

# Add savings % near the arrow midpoints (skip GPT-2 to avoid clutter)
for name, color, p30_z, adapt_z, kh, ka in DATA[1:]:  # skip GPT-2
    avg_k = (kh + ka) / 2
    saved_pct = (1 - avg_k / 30) * 100
    drop_pt = (p30_z - adapt_z) * 100
    mid_x = (30 + avg_k) / 2
    mid_y = (p30_z + adapt_z) / 2 - 0.005
    ax.text(mid_x, mid_y, f"-{saved_pct:.0f}% comp\n-{drop_pt:.1f}pt",
            fontsize=7, color=color, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor=color, alpha=0.85, lw=0.5))

plt.tight_layout()
out = "results/adaptive_pareto.png"
import os; os.makedirs("results", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"Saved to {out}")
