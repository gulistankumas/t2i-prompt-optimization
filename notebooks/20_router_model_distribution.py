"""Gün 13 ek — 3 router'ın model dağılım figürü (eco + premium yan yana)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df_all = pd.read_csv(ROOT / "data" / "processed" / "three_router_eval.csv")

modes = ["economic", "premium"]
routers_list = ["Lookup", "ML", "LLM"]
all_models = ["SD1.5", "FLUX.1-schnell", "SDXL", "FLUX.1-dev",
              "SD3.5-Large", "Gemini2.5-Flash-Image", "Qwen-Image", "GPT-Image-1"]
colors_models = plt.cm.tab10(np.linspace(0, 1, len(all_models)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax_idx, mode in enumerate(modes):
    ax = axes[ax_idx]
    counts = np.zeros((len(routers_list), len(all_models)))
    for i, router in enumerate(routers_list):
        sub = df_all[(df_all["router"] == router) & (df_all["mode"] == mode)]
        vc = sub["model"].value_counts()
        for j, m in enumerate(all_models):
            counts[i, j] = vc.get(m, 0)

    # stacked horizontal bar
    bottom = np.zeros(len(routers_list))
    for j, m in enumerate(all_models):
        ax.bar(routers_list, counts[:, j], bottom=bottom,
               label=m, color=colors_models[j], edgecolor="white", linewidth=0.5)
        # in-bar count text
        for i, val in enumerate(counts[:, j]):
            if val >= 10:
                ax.text(i, bottom[i] + val / 2, f"{int(val)}",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if val > 30 else "black")
        bottom += counts[:, j]

    ax.set_title(f"{mode.upper()} mod (140 prompt karar)", fontweight="bold")
    ax.set_ylabel("Kac kez secildi")
    ax.set_ylim(0, 145)

# common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.05), fontsize=10)

plt.suptitle("3 Router'in Model Secim Dagilimi (eco + premium)",
             fontsize=14, fontweight="bold", y=1.00)
plt.tight_layout()
out = ROOT / "figures" / "router_model_distribution.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.close()
print(f">> {out}")

# ekstra: concentration metrigi (top-1 model'e dusen yuzde)
print("\nConcentration (top-1 modelin payi):")
for mode in modes:
    print(f"\n--- {mode} ---")
    for router in routers_list:
        sub = df_all[(df_all["router"] == router) & (df_all["mode"] == mode)]
        top1_share = sub["model"].value_counts().iloc[0] / len(sub) * 100
        top1_name = sub["model"].value_counts().index[0]
        n_unique = sub["model"].nunique()
        print(f"  {router:<8s}: top-1 = {top1_name} ({top1_share:.1f}%), {n_unique} farkli model")
