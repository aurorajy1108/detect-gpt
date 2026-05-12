"""Adaptive-k extension figure — ACL/EMNLP paper style."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paper-style: serif (Times-like), modest sizes, clean look
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.grid": False,
})

DATA = [
    ("GPT-2\nMedium",          0.963, 0.945, 18.1),
    ("Qwen3-8B\nBase",         0.819, 0.792, 11.75),
    ("Qwen3-8B\nPost-train.",  0.853, 0.833, 12.7),
    ("LLaMA-3.1-8B\nBase",     0.789, 0.773, 11.65),
    ("LLaMA-3-8B\nInstruct",   0.864, 0.845, 12.35),
]

models = [d[0] for d in DATA]
fixed  = [d[1] for d in DATA]
adapt  = [d[2] for d in DATA]
ks     = [d[3] for d in DATA]
fixed_k = [30] * len(DATA)

x = np.arange(len(models))
w = 0.34

# ACL-friendly palette (distinguishable in grayscale via lightness)
NAVY  = "#3a5a80"   # for Fixed
RUST  = "#c97b4a"   # for Adaptive
ANNO  = "#8b1a1a"   # for savings annotation

# Double-column figure width (~7 in), modest height
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.6),
                                gridspec_kw={"hspace": 0.55})

# ── Top: accuracy ────────────────────────────────────────────────────
b1 = ax1.bar(x - w/2, fixed, w, label=r"Fixed $k{=}30$", color=NAVY,
             edgecolor="white", linewidth=0.5)
b2 = ax1.bar(x + w/2, adapt, w, label=r"Adaptive-$k$", color=RUST,
             edgecolor="white", linewidth=0.5)
for b in [*b1, *b2]:
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5)
ax1.set_xticks(x); ax1.set_xticklabels(models)
ax1.set_ylabel("ROC AUC ($z$-score)")
ax1.set_ylim(0.6, 1.10)
ax1.set_title(r"(a) Accuracy: adaptive $\approx$ fixed $k{=}30$",
              loc="left", pad=4)
ax1.legend(loc="upper right", frameon=False, ncol=2,
           handlelength=1.2, columnspacing=1.2)
ax1.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax1.set_axisbelow(True)

# ── Bottom: compute ──────────────────────────────────────────────────
b3 = ax2.bar(x - w/2, fixed_k, w, label=r"Fixed $k{=}30$", color=NAVY,
             edgecolor="white", linewidth=0.5)
b4 = ax2.bar(x + w/2, ks, w, label=r"Adaptive-$k$", color=RUST,
             edgecolor="white", linewidth=0.5)
for b in b3:
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
             "30", ha="center", va="bottom", fontsize=7.5)
for b, k in zip(b4, ks):
    saved = (1 - k/30) * 100
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
             f"{k:.1f}\n$-${saved:.0f}%", ha="center", va="bottom",
             fontsize=7.5, color=ANNO)
ax2.set_xticks(x); ax2.set_xticklabels(models)
ax2.set_ylabel("Avg perturbations / text")
ax2.set_ylim(0, 42)
ax2.set_title(r"(b) Compute: adaptive uses $\sim$60% fewer perturbations",
              loc="left", pad=4)
ax2.legend(loc="upper right", frameon=False, ncol=2,
           handlelength=1.2, columnspacing=1.2)
ax2.yaxis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
ax2.set_axisbelow(True)

plt.tight_layout()
out = "results/adaptive_k_extension.png"
out_pdf = "results/adaptive_k_extension.pdf"
import os; os.makedirs("results", exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved {out} and {out_pdf}")
