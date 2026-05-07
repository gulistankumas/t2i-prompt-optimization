"""Gün 13 — Üç router karşılaştırması (Lookup, ML, LLM) 140 test × 2 mod = 280 karar.

Çıktılar:
  data/processed/three_router_eval.csv
  figures/three_router_comparison.png
"""
# %% imports
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.router.lookup_router import LookupRouter
from src.router.ml_router import MLRouter
from src.router.llm_router import LLMRouter

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"


# %% load
df_test = pd.read_csv(PROC / "test_predictions_3way.csv")
df_master = pd.read_csv(PROC / "master_final.csv")


def get_quality(track, model):
    row = df_master[df_master["model"] == model]
    if len(row) == 0:
        return np.nan
    val = row[f"{track}_avg"].iloc[0]
    return float(val) if pd.notna(val) else np.nan


# %% router'ları yükle
print("Router'lar yukleniyor...")
routers = {
    "Lookup": LookupRouter(),
    "ML": MLRouter(),
    "LLM": LLMRouter(),
}


# %% 280 routing × 3 router
all_results = []
for name, router in routers.items():
    print(f"\n=== {name} ===")
    t0 = time.time()
    for idx, row in df_test.iterrows():
        for mode in ["economic", "premium"]:
            d = router.route(row["prompt"], mode)
            all_results.append({
                "router": name,
                "prompt_id": idx,
                "actual_track": row["track"],
                "predicted_track": d.predicted_track,
                "mode": mode,
                "model": d.model,
                "expected_quality": d.expected_quality,
                "expected_cost": d.expected_cost,
                "actual_quality": get_quality(row["track"], d.model),
            })
        if (idx + 1) % 20 == 0 and name == "LLM":
            print(f"  [{idx + 1}/140]", flush=True)
    elapsed = time.time() - t0
    print(f"  Toplam: {elapsed:.1f}s ({elapsed / 280 * 1000:.0f}ms per karar)")

dfa = pd.DataFrame(all_results)
dfa.to_csv(PROC / "three_router_eval.csv", index=False)
print(f"\n>> {PROC / 'three_router_eval.csv'}")


# ============================================================
# Karşılaştırma tablosu
# ============================================================
print("\n\n=== 3 ROUTER KARSILASTIRMASI ===")
print(f"\n{'Router':<10s} {'Mode':<10s} {'Quality':>10s} {'Cost ($)':>12s} {'Q/$':>10s}")
print("-" * 60)
summary_rows = []
for name in ["Lookup", "ML", "LLM"]:
    for mode in ["economic", "premium"]:
        sub = dfa[(dfa["router"] == name) & (dfa["mode"] == mode)]
        q = sub["actual_quality"].mean()
        c = sub["expected_cost"].mean()
        print(f"{name:<10s} {mode:<10s} {q:>10.2f} {c:>12.4f} {q / c:>10.0f}")
        summary_rows.append({"router": name, "mode": mode,
                              "quality": q, "cost": c, "q_per_cost": q / c})

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# Görsel: 6 strateji yan yana (3 router × 2 mod)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

methods = [f"{r}\n({m})" for r, m in zip(summary_df["router"], summary_df["mode"])]
colors_map = {"Lookup": "steelblue", "ML": "mediumseagreen", "LLM": "crimson"}
colors = [colors_map[r] for r in summary_df["router"]]
qualities = summary_df["quality"].tolist()
costs = summary_df["cost"].tolist()

bars1 = ax1.bar(methods, qualities, color=colors, edgecolor="black", linewidth=0.7)
ax1.set_ylabel("Ortalama Gerceği Kalite (oracle)")
ax1.set_title("Kalite", fontweight="bold")
ax1.axhline(y=60, color="blue", linestyle=":", alpha=0.5)
ax1.axhline(y=75, color="purple", linestyle=":", alpha=0.5)
ax1.set_ylim(0, max(qualities) * 1.15)
for b, v in zip(bars1, qualities):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
             ha="center", fontsize=9)

bars2 = ax2.bar(methods, costs, color=colors, edgecolor="black", linewidth=0.7)
ax2.set_ylabel("Ortalama Maliyet (USD/img)")
ax2.set_title("Maliyet (log scale)", fontweight="bold")
ax2.set_yscale("log")
for b, v in zip(bars2, costs):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, f"${v:.4f}",
             ha="center", fontsize=8)

plt.suptitle("3 Router Karsilastirmasi (140 test × 2 mod = 280 karar)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG / "three_router_comparison.png", bbox_inches="tight", dpi=200)
plt.close()
print(f">> {FIG / 'three_router_comparison.png'}")


# ============================================================
# Mod bazinda model dagilimi (3 router yan yana)
# ============================================================
print("\n=== Model dagilimi (kac kez secildi) ===")
for mode in ["economic", "premium"]:
    print(f"\n--- {mode.upper()} ---")
    for name in ["Lookup", "ML", "LLM"]:
        sub = dfa[(dfa["router"] == name) & (dfa["mode"] == mode)]
        top3 = sub["model"].value_counts().head(3)
        print(f"  {name:<8s}: {dict(top3)}")
