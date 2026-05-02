"""Gün 8 — Lookup table çıktıları: CSV + heatmap + findings doc.

Girdi: data/processed/lookup_table_2mode.json
Çıktılar:
  data/processed/lookup_table_2mode.csv
  figures/lookup_heatmap.png
"""
# %% imports
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.data_loader.prism import PRISM_TRACKS

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10

ROOT = Path(__file__).resolve().parents[1]
LOOKUP_JSON = ROOT / "data" / "processed" / "lookup_table_2mode.json"
LOOKUP_CSV = ROOT / "data" / "processed" / "lookup_table_2mode.csv"
HEATMAP = ROOT / "figures" / "lookup_heatmap.png"

with open(LOOKUP_JSON, "r", encoding="utf-8") as f:
    lookup = json.load(f)


# ============================================================
# 1. CSV — long format
# ============================================================
rows = []
for mode in ["economic", "premium"]:
    for track in PRISM_TRACKS:
        info = lookup[mode][track]
        all_q = info.get("all_qualifying") or []
        rows.append({
            "mode": mode,
            "track": track,
            "selected_model": info["model"],
            "quality": info["quality"],
            "cost_usd": info["cost"],
            "fallback": info["fallback"],
            "n_qualifying": len(all_q),
            "all_qualifying": ",".join(all_q),
        })
csv_df = pd.DataFrame(rows)
csv_df.to_csv(LOOKUP_CSV, index=False)
print(f">> {LOOKUP_CSV}")


# ============================================================
# 2. Heatmap — 7 track × 2 mod, hücre rengi: cost
# ============================================================
modes = ["economic", "premium"]
mode_labels = ["Ekonomik (>=60)", "Premium (>=75)"]
cost_matrix = np.zeros((len(PRISM_TRACKS), len(modes)))
annot_matrix = []
for i, t in enumerate(PRISM_TRACKS):
    row_text = []
    for j, m in enumerate(modes):
        info = lookup[m][t]
        cost_matrix[i, j] = info["cost"]
        row_text.append(f"{info['model']}\nq={info['quality']:.1f}\n${info['cost']:.4f}")
    annot_matrix.append(row_text)

annot = np.array(annot_matrix)

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(
    pd.DataFrame(cost_matrix, index=PRISM_TRACKS, columns=mode_labels),
    annot=annot, fmt="", cmap="RdYlGn_r",
    cbar_kws={"label": "Cost (USD per image)"},
    ax=ax, linewidths=2, linecolor="white",
    annot_kws={"size": 9, "fontweight": "bold"},
)
ax.set_title("Router Lookup Table — Selected Model per (Track, Mode)\n"
             "Cell color: cost (green=cheap, red=expensive)",
             fontsize=12, fontweight="bold", pad=14)
ax.set_xlabel("")
ax.set_ylabel("PRISM Track")
plt.tight_layout()
plt.savefig(HEATMAP, bbox_inches="tight")
plt.close()
print(f">> {HEATMAP}")


# ============================================================
# 3. CSV preview
# ============================================================
print("\n=== Lookup CSV preview ===")
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
print(csv_df.to_string(index=False))
