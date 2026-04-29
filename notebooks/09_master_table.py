"""Gün 4 — Master tablo final hâli.

İki jüri ortalaması (PRISM tracks) + overall + maliyet + tezin nihai
karar tablosu. Çıktı: data/processed/master_table.csv
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader.prism import load_prism, PRISM_TRACKS
from src.data_loader.costs import load_costs

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

# %% load
gpt = load_prism("gpt41").set_index("model")
qwen = load_prism("qwen25vl").set_index("model")
qwen = qwen.reindex(gpt.index)
costs = load_costs().set_index("model")

# %% iki jüri ortalaması
score_avg = (gpt[PRISM_TRACKS + ["overall"]] + qwen[PRISM_TRACKS + ["overall"]]) / 2
score_avg = score_avg.round(2)

# %% 9 hedef model
TARGET_9 = [
    "SD1.5", "SDXL", "SD3.5-Large", "FLUX.1-schnell", "FLUX.1-dev",
    "Qwen-Image", "DALL-E 2", "GPT-Image-1", "Gemini2.5-Flash-Image",
]

rows = []
for m in TARGET_9:
    row = {"model": m}
    if m in score_avg.index:
        for col in PRISM_TRACKS + ["overall"]:
            row[col] = score_avg.loc[m, col]
    else:
        # DALL-E 2 — PRISM'de yok, NaN
        for col in PRISM_TRACKS + ["overall"]:
            row[col] = np.nan
    if m in costs.index:
        row["vendor"] = costs.loc[m, "vendor"]
        row["access_type"] = costs.loc[m, "access_type"]
        row["params_b"] = costs.loc[m, "params_b"]
        row["cost_usd"] = costs.loc[m, "cost_per_image_usd"]
    rows.append(row)

master = pd.DataFrame(rows)
# kolon sırası
cols = ["model", "vendor", "access_type", "params_b"] + PRISM_TRACKS + ["overall", "cost_usd"]
master = master[cols]

# %% kalite/dolar oranı
master["quality_per_usd"] = (master["overall"] / master["cost_usd"]).round(0)

# overall'a göre sırala (NaN'ları sona)
master_sorted = master.sort_values("overall", ascending=False, na_position="last")
print("=" * 100)
print("MASTER TABLO (iki jüri ortalaması, 9 model)")
print("=" * 100)
print(master_sorted.to_string(index=False))

# %% Pareto frontier (overall vs cost)
print("\n" + "=" * 100)
print("PARETO FRONTIER (kalite vs maliyet)")
print("=" * 100)
prism_only = master.dropna(subset=["overall"]).copy()
prism_only_sorted = prism_only.sort_values("cost_usd")
pareto_models = []
best_quality_so_far = -1
for _, r in prism_only_sorted.iterrows():
    if r["overall"] > best_quality_so_far:
        pareto_models.append(r["model"])
        best_quality_so_far = r["overall"]

print(f"\nPareto-optimal modeller (kalite-maliyet sınırında):")
for m in pareto_models:
    r = master.loc[master["model"] == m].iloc[0]
    print(f"  {m:25s} kalite={r['overall']:5.2f}  maliyet=${r['cost_usd']:.4f}  q/$={r['quality_per_usd']:.0f}")

dominated = [m for m in prism_only["model"] if m not in pareto_models]
print(f"\nDominated modeller (Pareto'da olmayan):")
for m in dominated:
    r = master.loc[master["model"] == m].iloc[0]
    print(f"  {m:25s} kalite={r['overall']:5.2f}  maliyet=${r['cost_usd']:.4f}")

# %% kaydet
out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "master_table.csv"
master.to_csv(out_path, index=False)
print(f"\n>> Master tablo kaydedildi: {out_path}")

# %% mod-bazlı en iyi (kullanıcı modları için lookup)
print("\n" + "=" * 100)
print("MOD BAZLI EN İYİ TAVSİYE (Pareto üzerinden)")
print("=" * 100)
pareto_set = master[master["model"].isin(pareto_models)].sort_values("cost_usd").reset_index(drop=True)
print(pareto_set[["model", "overall", "cost_usd", "quality_per_usd"]].to_string(index=False))
print(f"\n  Ucuz mod   -> en düşük maliyet Pareto modeli: {pareto_set.iloc[0]['model']} (${pareto_set.iloc[0]['cost_usd']:.4f})")
print(f"  Dengeli mod -> en yüksek q/$ Pareto modeli:    {pareto_set.loc[pareto_set['quality_per_usd'].idxmax(), 'model']} (q/$={pareto_set['quality_per_usd'].max():.0f})")
print(f"  Premium mod -> en yüksek kalite Pareto modeli: {pareto_set.iloc[-1]['model']} (kalite={pareto_set.iloc[-1]['overall']:.1f})")
