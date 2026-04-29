"""Gün 5 — Tezin 5 ana grafiği. Çıktı: figures/*.png

1. Track bazında Pareto frontier (2x4)
2. Track-arası korelasyon ısı haritası (8 vs 19 model)
3. Segment-bazlı skor yayılımı (boxplot)
4. İki jüri anlaşması (scatter, 7 track)
5. Premium paradoksu (bar chart, GPT-Image-1 vs Gemini)
"""
# %% setup
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
import numpy as np
from scipy.stats import spearmanr

from src.data_loader.prism import load_prism, PRISM_TRACKS

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "data" / "processed" / "master_table.csv")
df_19_gpt = load_prism("gpt41")
df_19_qwen = load_prism("qwen25vl")
tracks = PRISM_TRACKS


# ============================================================
# Graph 1: Pareto per track
# ----
# Analiz: Her PRISM track için kalite-maliyet uzayında Pareto frontier;
#   modeller açık/kapalı kaynak rengine göre.
# Kanıt: SDXL track-aware specialist (entity/style/affection'da Pareto'da
#   olduğu kanıtlanır); GPT-Image-1 affection ve long_text'te dominated;
#   Qwen-Image 7/7 track'te dominated. Track-aware routing'in değer
#   ürettiğini gösterir.
# ============================================================
def find_pareto(d: pd.DataFrame, quality_col: str, cost_col: str):
    pareto = []
    for i, row in d.iterrows():
        if pd.isna(row[quality_col]):
            continue
        dominated = False
        for j, other in d.iterrows():
            if i == j or pd.isna(other[quality_col]):
                continue
            if (other[quality_col] >= row[quality_col] and
                other[cost_col] <= row[cost_col] and
                (other[quality_col] > row[quality_col] or other[cost_col] < row[cost_col])):
                dominated = True
                break
        if not dominated:
            pareto.append((row["model"], row[cost_col], row[quality_col]))
    pareto.sort(key=lambda x: x[1])
    return pareto


fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()
for i, track in enumerate(tracks):
    ax = axes_flat[i]
    for _, row in df.iterrows():
        if pd.isna(row[track]):
            continue
        color = "steelblue" if row["access_type"] == "open" else "crimson"
        ax.scatter(row["cost_usd"], row[track], s=150, c=color, alpha=0.7,
                   edgecolors="black", linewidth=1)
        ax.annotate(row["model"], (row["cost_usd"], row[track]),
                    xytext=(7, 7), textcoords="offset points", fontsize=8)

    pareto = find_pareto(df, track, "cost_usd")
    if pareto:
        cs = [p[1] for p in pareto]
        qs = [p[2] for p in pareto]
        ax.plot(cs, qs, "g-", alpha=0.5, linewidth=2.5, zorder=0)

    ax.set_xlabel("Cost (USD per image)")
    ax.set_ylabel(f"{track.capitalize()} Quality")
    ax.set_title(track.capitalize(), fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

axes_flat[7].axis("off")
legend_elements = [
    Patch(facecolor="steelblue", label="Open Source"),
    Patch(facecolor="crimson", label="Closed Source"),
    Line2D([0], [0], color="g", alpha=0.5, linewidth=2.5, label="Pareto frontier"),
]
axes_flat[7].legend(handles=legend_elements, loc="center", fontsize=14)

plt.suptitle("Track-Based Pareto Frontier Analysis (2-judge avg)",
             fontsize=16, fontweight="bold", y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / "pareto_frontier_per_track.png", bbox_inches="tight")
plt.close()
print("[1/5] pareto_frontier_per_track.png")


# ============================================================
# Graph 2: Correlation heatmaps (8 vs 19, both GPT-4.1 for consistency)
# ----
# Analiz: 8-model router havuzu ve 19-model PRISM havuzunun track'ler arası
#   Spearman korelasyon matrisleri.
# Kanıt: 8-model'da ortalama 0.94 (uçtan uca havuz seçimi yapay yüksek
#   korelasyon yaratıyor); 19-model'da 0.84, en düşük çift entity↔long_text
#   0.64. Havuz çeşitliliği track-bağımsızlığını ortaya çıkarır → routing
#   değeri görmek için zenginleştirilmiş havuz gerekir.
# ============================================================
target_8 = ["SD1.5", "SDXL", "SD3.5-Large", "FLUX.1-schnell", "FLUX.1-dev",
            "Qwen-Image", "GPT-Image-1", "Gemini2.5-Flash-Image"]
df_8_gpt = df_19_gpt[df_19_gpt["model"].isin(target_8)]
corr_8 = df_8_gpt[tracks].corr(method="spearman")
corr_19 = df_19_gpt[tracks].corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_8, annot=True, cmap="RdYlGn", vmin=0.5, vmax=1.0,
            fmt=".2f", square=True, ax=axes[0],
            cbar_kws={"label": "Spearman rho"})
axes[0].set_title("8 Target Models (Router pool)", fontweight="bold")
axes[0].tick_params(axis="x", rotation=30)
axes[0].tick_params(axis="y", rotation=0)

sns.heatmap(corr_19, annot=True, cmap="RdYlGn", vmin=0.5, vmax=1.0,
            fmt=".2f", square=True, ax=axes[1],
            cbar_kws={"label": "Spearman rho"})
axes[1].set_title("19 Models (Full PRISM-Bench)", fontweight="bold")
axes[1].tick_params(axis="x", rotation=30)
axes[1].tick_params(axis="y", rotation=0)

plt.suptitle("Track-Wise Spearman Correlation: pool diversity reduces redundancy",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "track_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("[2/5] track_correlation_heatmap.png")


# ============================================================
# Graph 3: Segment spread (boxplot)
# ----
# Analiz: Premium / Mid / Cheap segmentlerinin her PRISM track'teki skor
#   dağılımı.
# Kanıt: Premium dar yayılım (modeller arası fark <15 puan), Cheap geniş
#   yayılım (text_rendering'de >50 puan). Routing'in değer ürettiği nokta
#   ucuz/orta segment; premium'da seçim trivial.
# ============================================================
df_long = df.dropna(subset=tracks).melt(id_vars=["model", "access_type"],
                                         value_vars=tracks,
                                         var_name="track", value_name="score")


def get_segment(model: str) -> str:
    if model in ["GPT-Image-1", "Gemini2.5-Flash-Image", "Qwen-Image"]:
        return "Premium"
    if model in ["SD3.5-Large", "FLUX.1-dev"]:
        return "Mid"
    return "Cheap"


df_long["segment"] = df_long["model"].apply(get_segment)

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_long, x="track", y="score", hue="segment",
            palette={"Premium": "crimson", "Mid": "goldenrod", "Cheap": "steelblue"},
            ax=ax)
ax.set_title("Score Distribution by Segment and Track", fontweight="bold")
ax.set_xlabel("PRISM Track")
ax.set_ylabel("Quality Score (avg of 2 judges)")
plt.xticks(rotation=20)
plt.legend(title="Segment")
plt.tight_layout()
plt.savefig(FIG_DIR / "segment_track_distribution.png", bbox_inches="tight")
plt.close()
print("[3/5] segment_track_distribution.png")


# ============================================================
# Graph 4: Judge agreement
# ----
# Analiz: GPT-4.1 vs Qwen2.5-VL jürilerinin 19 model × 7 track skorları
#   scatter olarak; her panele Spearman ρ ve y=x referansı.
# Kanıt: Her track'te ρ ≥ 0.95 (sıralama anlaşması yüksek) ama noktalar
#   diagonalden sistematik kayık → mutlak skor düzleminde bias var.
#   Tek-jüriye dayalı router'lar absolute-skor algoritmasıyla risk taşır;
#   master tablo iki jüri ortalaması üzerinden kurulmalıdır.
# ============================================================
df_judges = df_19_gpt.merge(df_19_qwen, on="model", suffixes=("_gpt", "_qwen"))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes_flat = axes.flatten()
for i, track in enumerate(tracks):
    ax = axes_flat[i]
    x = df_judges[f"{track}_gpt"]
    y = df_judges[f"{track}_qwen"]
    ax.scatter(x, y, s=80, alpha=0.6, edgecolors="black")
    lim = [0, 100]
    ax.plot(lim, lim, "k--", alpha=0.3)
    rho, _ = spearmanr(x, y)
    ax.text(0.05, 0.95, f"rho = {rho:.2f}", transform=ax.transAxes,
            fontsize=10, fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("GPT-4.1 Judge")
    ax.set_ylabel("Qwen2.5-VL Judge")
    ax.set_title(track.capitalize(), fontweight="bold")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
axes_flat[7].axis("off")
plt.suptitle("Inter-Judge Agreement Across Tracks (19 PRISM models)",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "judge_agreement.png", bbox_inches="tight")
plt.close()
print("[4/5] judge_agreement.png")


# ============================================================
# Graph 5: Premium paradox
# ----
# Analiz: GPT-Image-1 ($0.167) vs Gemini 2.5 Flash ($0.039) — 7 track yan
#   yana bar chart, her track'te puan farkı anotasyonlu.
# Kanıt: Overall fark sadece 0.65 puan, fiyat farkı 4.3x. Affection ve
#   long_text'te Gemini önde (negatif fark). "Premium = GPT-Image-1" naif
#   kuralının çürütülmesi; routing'in en kritik kararı bu segmentte.
# ============================================================
df_premium = df[df["model"].isin(["GPT-Image-1", "Gemini2.5-Flash-Image"])].set_index("model")

fig, ax = plt.subplots(figsize=(12, 6))
xs = np.arange(len(tracks))
width = 0.35
gpt_scores = df_premium.loc["GPT-Image-1", tracks].values.astype(float)
gemini_scores = df_premium.loc["Gemini2.5-Flash-Image", tracks].values.astype(float)

ax.bar(xs - width / 2, gpt_scores, width,
       label="GPT-Image-1 ($0.167/img)", color="crimson")
ax.bar(xs + width / 2, gemini_scores, width,
       label="Gemini 2.5 Flash ($0.039/img)", color="steelblue")

ax.set_xlabel("PRISM Track")
ax.set_ylabel("Quality Score (avg of 2 judges)")
ax.set_title("Premium Paradox: 0.65 pt overall difference at 4.3x cost",
             fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels(tracks, rotation=20)
ax.legend()

for i, (g, gem) in enumerate(zip(gpt_scores, gemini_scores)):
    diff = g - gem
    color = "darkred" if diff < 0 else "darkgreen"
    ax.annotate(f"{diff:+.1f}", xy=(i, max(g, gem) + 2),
                ha="center", fontsize=9, color=color, fontweight="bold")

ax.set_ylim(0, max(max(gpt_scores), max(gemini_scores)) + 8)
plt.tight_layout()
plt.savefig(FIG_DIR / "premium_paradox.png", bbox_inches="tight")
plt.close()
print("[5/5] premium_paradox.png")

print(f"\nTum grafikler: {FIG_DIR}")
