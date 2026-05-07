"""Gün 12 — Lookup router 140 test prompt'unda + baseline karşılaştırması.

Akış:
  1. Lookup router (LR + lookup table) 280 karar (140 prompt × 2 mod)
  2. Her karar için "actual quality" = oracle(actual_track, recommended_model)
  3. 4 baseline ile karşılaştır:
       always_cheapest (SD1.5), always_premium (GPT-Image-1),
       always_popular (FLUX.1-dev), random
  4. Görsel: lookup_vs_baselines.png (kalite + maliyet 2 panel)
  5. Çıktı: data/processed/lookup_router_eval.csv
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
from src.data_loader.prism import PRISM_TRACKS

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"


# %% load
df_test = pd.read_csv(PROC / "test_predictions_3way.csv")
df_master = pd.read_csv(PROC / "master_final.csv")
print(f"Test prompts: {len(df_test)}")


# %% router
print("\nLookup router (LR + lookup table) yukleniyor...")
router = LookupRouter()


# %% 140 prompt × 2 mod = 280 karar
print("\n140 prompt x 2 mod routing...")
t0 = time.time()
results = []
for idx, row in df_test.iterrows():
    prompt = row["prompt"]
    actual = row["track"]
    for mode in ["economic", "premium"]:
        d = router.route(prompt, mode)
        results.append({
            "prompt_id": idx,
            "actual_track": actual,
            "predicted_track": d.predicted_track,
            "track_correct": actual == d.predicted_track,
            "mode": mode,
            "model": d.model,
            "expected_quality": d.expected_quality,
            "expected_cost": d.expected_cost,
        })
elapsed = time.time() - t0
print(f"Tamamlandi: {elapsed:.1f}s ({elapsed / len(df_test) * 1000:.1f}ms per prompt × 2 mod)")

dfr = pd.DataFrame(results)


# %% actual quality hesabı (oracle: gerçek track × önerilen model)
def get_quality(track, model):
    row = df_master[df_master["model"] == model]
    if len(row) == 0:
        return np.nan
    val = row[f"{track}_avg"].iloc[0]
    return float(val) if pd.notna(val) else np.nan


dfr["actual_quality"] = dfr.apply(
    lambda r: get_quality(r["actual_track"], r["model"]), axis=1
)
dfr.to_csv(PROC / "lookup_router_eval.csv", index=False)
print(f"\n>> {PROC / 'lookup_router_eval.csv'}")


# %% router performansı
print("\n=== ROUTER PERFORMANSI ===")
print(f"Track tahmin doğruluğu: {dfr['track_correct'].mean() * 100:.1f}%  (LR baseline ile uyumlu olmalı)")

for mode in ["economic", "premium"]:
    sub = dfr[dfr["mode"] == mode]
    print(f"\n--- {mode.upper()} mod ---")
    print(f"  Ortalama beklenen kalite: {sub['expected_quality'].mean():.2f}")
    print(f"  Ortalama gerçek kalite (oracle): {sub['actual_quality'].mean():.2f}")
    print(f"  Ortalama maliyet: ${sub['expected_cost'].mean():.4f}")
    print(f"  Önerilen modeller:\n{sub['model'].value_counts().to_string()}")


# ============================================================
# Baseline karşılaştırması
# ============================================================
print("\n\n=== BASELINE KARŞILAŞTIRMASI ===")
baselines = {
    "always_cheapest": "SD1.5",
    "always_popular": "FLUX.1-dev",
    "always_premium": "GPT-Image-1",
}

baseline_stats = {}
for name, model in baselines.items():
    qs = [get_quality(t, model) for t in df_test["track"]]
    cost = float(df_master[df_master["model"] == model]["cost_usd"].iloc[0])
    baseline_stats[name] = {
        "model": model,
        "avg_quality": float(np.nanmean(qs)),
        "avg_cost": cost,
    }

# random: her prompt için uniform random model (8 PRISM model)
rng = np.random.default_rng(42)
prism_models = df_master[df_master["overall_avg"].notna()]["model"].tolist()
random_qs, random_cs = [], []
for t in df_test["track"]:
    m = rng.choice(prism_models)
    random_qs.append(get_quality(t, m))
    random_cs.append(float(df_master[df_master["model"] == m]["cost_usd"].iloc[0]))
baseline_stats["random"] = {
    "model": "random",
    "avg_quality": float(np.nanmean(random_qs)),
    "avg_cost": float(np.mean(random_cs)),
}

print(f"\n{'Method':<25s} {'Quality':>10s} {'Cost ($)':>12s}")
print("-" * 50)

# router
for mode in ["economic", "premium"]:
    sub = dfr[dfr["mode"] == mode]
    print(f"{'Lookup ' + mode:<25s} {sub['actual_quality'].mean():>10.2f} {sub['expected_cost'].mean():>12.4f}")

# baselines
for name, st in baseline_stats.items():
    print(f"{name + ' (' + st['model'] + ')':<25s} {st['avg_quality']:>10.2f} {st['avg_cost']:>12.4f}")


# ============================================================
# Görsel: 2 panel (kalite + maliyet)
# ============================================================
methods = ["Lookup\n(eco)", "Lookup\n(premium)", "Always\nSD1.5", "Always\nFLUX.1-dev", "Always\nGPT-Image-1", "Random"]
qualities = [
    dfr[dfr["mode"] == "economic"]["actual_quality"].mean(),
    dfr[dfr["mode"] == "premium"]["actual_quality"].mean(),
    baseline_stats["always_cheapest"]["avg_quality"],
    baseline_stats["always_popular"]["avg_quality"],
    baseline_stats["always_premium"]["avg_quality"],
    baseline_stats["random"]["avg_quality"],
]
costs = [
    dfr[dfr["mode"] == "economic"]["expected_cost"].mean(),
    dfr[dfr["mode"] == "premium"]["expected_cost"].mean(),
    baseline_stats["always_cheapest"]["avg_cost"],
    baseline_stats["always_popular"]["avg_cost"],
    baseline_stats["always_premium"]["avg_cost"],
    baseline_stats["random"]["avg_cost"],
]
colors = ["steelblue", "crimson", "gray", "gray", "gray", "lightgray"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(methods, qualities, color=colors, edgecolor="black", linewidth=0.7)
ax1.set_ylabel("Ortalama Kalite Skoru (oracle)")
ax1.set_title("Kalite Karşılaştırması", fontweight="bold")
ax1.axhline(y=60, color="blue", linestyle=":", alpha=0.5, label="Ekonomik eşiği (60)")
ax1.axhline(y=75, color="purple", linestyle=":", alpha=0.5, label="Premium eşiği (75)")
ax1.legend(loc="lower right", fontsize=9)
ax1.set_ylim(0, max(qualities) * 1.15)
for b, v in zip(bars1, qualities):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

bars2 = ax2.bar(methods, costs, color=colors, edgecolor="black", linewidth=0.7)
ax2.set_ylabel("Ortalama Maliyet (USD/img)")
ax2.set_title("Maliyet Karşılaştırması (log scale)", fontweight="bold")
ax2.set_yscale("log")
for b, v in zip(bars2, costs):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.15, f"${v:.4f}", ha="center", fontsize=8)

plt.suptitle("Lookup Router vs Baseline Stratejiler (140 test prompt)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG / "lookup_vs_baselines.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"\n>> {FIG / 'lookup_vs_baselines.png'}")
