import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_PERT = "tmp_results/gpt2-medium-t5-large-temp/2026-04-22-18-31-34-550287-fp32-0.3-1-hc3-100/"
BASE_ADAP = "results/gpt2-medium-t5-large-temp/2026-04-23-10-19-55-879290-fp32-0.3-1-hc3-100/"
BASE_FAST = "results/gpt2-medium-t5-large-temp/2026-04-23-11-32-16-804702-fp32-0.3-1-hc3-100/"

RESULTS = {
    "Pert-1 (d)":       BASE_PERT + "perturbation_1_d_results.json",
    "Pert-1 (z)":       BASE_PERT + "perturbation_1_z_results.json",
    "Pert-10 (d)":      BASE_PERT + "perturbation_10_d_results.json",
    "Pert-10 (z)":      BASE_PERT + "perturbation_10_z_results.json",
    "Adaptive-k (d)":   BASE_ADAP + "adaptive_d_conf95_results.json",
    "Adaptive-k (z)":   BASE_ADAP + "adaptive_z_conf95_results.json",
    "Fast-DetectGPT":   BASE_FAST + "fast_detect_gpt_results.json",
    "Qwen3-8B loglik":  "results/qwen3_8b_base_loglik_hc3.json",
}

COLORS = {
    "Pert-1 (d)":     "#aec7e8",
    "Pert-1 (z)":     "#6baed6",
    "Pert-10 (d)":    "#fdae6b",
    "Pert-10 (z)":    "#e6550d",
    "Adaptive-k (d)": "#bcbddc",
    "Adaptive-k (z)": "#756bb1",
    "Fast-DetectGPT": "#31a354",
    "Qwen3-8B loglik": "#d62728",
}

data = {name: json.load(open(path)) for name, path in RESULTS.items()}

# avg k for adaptive methods
adap_d = data["Adaptive-k (d)"]["raw_results"]
adap_z = data["Adaptive-k (z)"]["raw_results"]
avg_k_d = np.mean([r["k_used_original"] for r in adap_d] + [r["k_used_sampled"] for r in adap_d])
avg_k_z = np.mean([r["k_used_original"] for r in adap_z] + [r["k_used_sampled"] for r in adap_z])

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ── Left: ROC curves ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
for name, d in data.items():
    fpr = d["metrics"]["fpr"]
    tpr = d["metrics"]["tpr"]
    auc = d["metrics"]["roc_auc"]
    lw  = 2.5 if name == "Fast-DetectGPT" else 1.8
    ls  = "--" if ("Adaptive" in name or "Qwen3" in name) else "-"
    ax1.plot(fpr, tpr, label=f"{name}  ({auc:.4f})",
             color=COLORS[name], lw=lw, ls=ls)

ax1.set_xlabel("False Positive Rate", fontsize=10)
ax1.set_ylabel("True Positive Rate", fontsize=10)
ax1.set_title("ROC Curves — HC3\n(GPT-2 scorer, ChatGPT source)", fontsize=10)
ax1.legend(fontsize=7.5, loc="lower right")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)

# ── Middle: AUROC bar chart ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
names = list(data.keys())
aucs  = [data[n]["metrics"]["roc_auc"] for n in names]
bars  = ax2.bar(range(len(names)), aucs,
                color=[COLORS[n] for n in names],
                edgecolor="white", width=0.6)
for bar, auc in zip(bars, aucs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
             f"{auc:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax2.set_ylabel("ROC AUC", fontsize=10)
ax2.set_title("AUROC Comparison — HC3", fontsize=10)
ax2.set_ylim(0.3, 1.07)
ax2.axhline(0.9, color="grey", lw=0.8, ls="--", alpha=0.5)
ax2.grid(True, axis="y", alpha=0.3)

# ── Right: k distribution for adaptive-k ─────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ks_z = ([r["k_used_original"] for r in adap_z] +
        [r["k_used_sampled"]  for r in adap_z])
bins = range(5, 36, 5)
ax3.hist(ks_z, bins=bins, color="#756bb1", edgecolor="white", alpha=0.85, align="left")
ax3.axvline(avg_k_z, color="#31a354", lw=2, ls="--", label=f"Mean k = {avg_k_z:.1f}")
ax3.set_xlabel("k (perturbations used)", fontsize=10)
ax3.set_ylabel("Count", fontsize=10)
ax3.set_title(f"Adaptive-k Distribution (z)\nHC3  |  avg k = {avg_k_z:.1f}  (max=30)", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)

plt.suptitle(
    "DetectGPT Methods on HC3  |  Scorer: GPT-2 Medium  |  Source: ChatGPT  |  n=100",
    fontsize=10, y=1.01)

out = "results/hc3_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
for name, auc in zip(names, aucs):
    print(f"  {name:<18} {auc:.4f}")
print(f"\n  Avg k (adaptive-z): {avg_k_z:.1f} / 30 max  →  {avg_k_z/30*100:.0f}% of max budget")
