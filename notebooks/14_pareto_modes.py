"""Gün 9 — Pareto frontier + lookup mod seçimleri (görsel).

Mevcut Pareto grafiğini al, lookup table'ın seçtiği modelleri renkli
kenarlıkla işaretle. Her track için ayrı subplot.

Çıktı: figures/pareto_with_modes.png
"""
# %% imports
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

from src.data_loader.prism import PRISM_TRACKS

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "master_final.csv")
with open(ROOT / "data" / "processed" / "lookup_table_2mode.json", "r", encoding="utf-8") as f:
    lookup = json.load(f)

# Pareto modeller master_final'daki pareto_{track} bayraklarından geliyor
pareto_dict = {
    t: df[df[f"pareto_{t}"] == True]["model"].tolist() for t in PRISM_TRACKS
}

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
axes_flat = axes.flatten()

for i, track in enumerate(PRISM_TRACKS):
    ax = axes_flat[i]
    track_col = f"{track}_avg"

    eco_pick = lookup["economic"][track]["model"]
    pre_pick = lookup["premium"][track]["model"]

    # Tüm modelleri scatter
    for _, row in df.iterrows():
        if pd.isna(row[track_col]):
            continue
        color = "steelblue" if row["access_type"] == "open" else "crimson"

        is_eco = row["model"] == eco_pick
        is_pre = row["model"] == pre_pick

        if is_eco and is_pre:
            size, edge, ew = 420, "gold", 3.5
        elif is_eco:
            size, edge, ew = 320, "green", 3.0
        elif is_pre:
            size, edge, ew = 320, "purple", 3.0
        else:
            size, edge, ew = 110, "black", 0.5

        ax.scatter(row["cost_usd"], row[track_col],
                   s=size, c=color, alpha=0.75,
                   edgecolors=edge, linewidth=ew, zorder=3)
        ax.annotate(row["model"], (row["cost_usd"], row[track_col]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=8, zorder=4)

    # Pareto çizgisi
    pareto_data = df[df["model"].isin(pareto_dict[track])].sort_values("cost_usd")
    ax.plot(pareto_data["cost_usd"], pareto_data[track_col],
            "g--", linewidth=2, alpha=0.4, zorder=1)

    # Eşik çizgileri
    ax.axhline(y=60, color="blue", linestyle=":", alpha=0.6, linewidth=1.2, zorder=0)
    ax.axhline(y=75, color="purple", linestyle=":", alpha=0.6, linewidth=1.2, zorder=0)

    ax.set_xlabel("Cost (USD per image)")
    ax.set_ylabel(f"{track.capitalize()} Quality")
    ax.set_title(track.capitalize(), fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

# Legend
axes_flat[7].axis("off")
legend_elements = [
    Patch(facecolor="steelblue", label="Open Source"),
    Patch(facecolor="crimson", label="Closed Source"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=14,
           markeredgecolor="green", markeredgewidth=3, label="Ekonomik mod secim"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=14,
           markeredgecolor="purple", markeredgewidth=3, label="Premium mod secim"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=17,
           markeredgecolor="gold", markeredgewidth=3.5, label="Her iki modda secim"),
    Line2D([0], [0], color="blue", linestyle=":", linewidth=1.2, label="Ekonomik esigi (60)"),
    Line2D([0], [0], color="purple", linestyle=":", linewidth=1.2, label="Premium esigi (75)"),
    Line2D([0], [0], color="g", linestyle="--", linewidth=2, alpha=0.4, label="Pareto frontier"),
]
axes_flat[7].legend(handles=legend_elements, loc="center", fontsize=11)

plt.suptitle("Track-Based Pareto Frontier with 2-Mode Lookup Selections",
             fontsize=16, fontweight="bold", y=1.00)
plt.tight_layout()
out = ROOT / "figures" / "pareto_with_modes.png"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f">> {out}")

# Özet: hangi track'te ne işaretlendi
print("\n=== Lookup secimleri ===")
for t in PRISM_TRACKS:
    eco = lookup["economic"][t]["model"]
    pre = lookup["premium"][t]["model"]
    same = " (HER IKI MOD)" if eco == pre else ""
    print(f"  {t:<16s}: eco={eco}, pre={pre}{same}")
