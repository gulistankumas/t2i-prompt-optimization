"""Gün 6 — Tek master final tablo. Hafta 2'de router kuracağımız tek kaynak.

Çıktı: data/processed/master_final.csv

Sütunlar:
  model, vendor, access_type, params_b, segment,
  {track}_gpt, {track}_qwen, {track}_avg  (7 track × 3 = 21 kolon),
  overall_gpt, overall_qwen, overall_avg,
  cost_usd, quality_per_usd,
  pareto_{track}  (7 track bool),
  pareto_overall  (bool)
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from src.data_loader.prism import load_prism, PRISM_TRACKS
from src.data_loader.costs import load_costs

ROOT = Path(__file__).resolve().parents[1]

# 9 hedef model (DALL-E 2 dahil — PRISM yok, NaN kalacak)
TARGET_9 = [
    "SD1.5", "SDXL", "SD3.5-Large", "FLUX.1-schnell", "FLUX.1-dev",
    "Qwen-Image", "DALL-E 2", "GPT-Image-1", "Gemini2.5-Flash-Image",
]

SEGMENT = {
    "GPT-Image-1": "Premium",
    "Gemini2.5-Flash-Image": "Premium",
    "Qwen-Image": "Premium",
    "SD3.5-Large": "Mid",
    "FLUX.1-dev": "Mid",
    "FLUX.1-schnell": "Cheap",
    "SDXL": "Cheap",
    "SD1.5": "Cheap",
    "DALL-E 2": "Historical",
}


# %% load skor + maliyet
gpt = load_prism("gpt41").set_index("model")
qwen = load_prism("qwen25vl").set_index("model").reindex(gpt.index)
costs = load_costs().set_index("model")

# %% per-model satır kur
rows = []
for m in TARGET_9:
    r = {"model": m}
    if m in costs.index:
        r["vendor"] = costs.loc[m, "vendor"]
        r["access_type"] = costs.loc[m, "access_type"]
        r["params_b"] = costs.loc[m, "params_b"]
        r["cost_usd"] = costs.loc[m, "cost_per_image_usd"]
    r["segment"] = SEGMENT.get(m, np.nan)

    for t in PRISM_TRACKS + ["overall"]:
        if m in gpt.index:
            r[f"{t}_gpt"] = gpt.loc[m, t]
            r[f"{t}_qwen"] = qwen.loc[m, t]
            r[f"{t}_avg"] = round((gpt.loc[m, t] + qwen.loc[m, t]) / 2, 2)
        else:
            r[f"{t}_gpt"] = np.nan
            r[f"{t}_qwen"] = np.nan
            r[f"{t}_avg"] = np.nan
    rows.append(r)

master = pd.DataFrame(rows)
master["quality_per_usd"] = (master["overall_avg"] / master["cost_usd"]).round(0)


# %% Pareto bayrakları (avg skor + cost_usd üzerinde)
def pareto_flag(df: pd.DataFrame, q_col: str, c_col: str) -> pd.Series:
    flags = []
    for i, row in df.iterrows():
        if pd.isna(row[q_col]) or pd.isna(row[c_col]):
            flags.append(False)
            continue
        dominated = False
        for j, other in df.iterrows():
            if i == j or pd.isna(other[q_col]):
                continue
            if (other[q_col] >= row[q_col] and
                other[c_col] <= row[c_col] and
                (other[q_col] > row[q_col] or other[c_col] < row[c_col])):
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


for t in PRISM_TRACKS:
    master[f"pareto_{t}"] = pareto_flag(master, f"{t}_avg", "cost_usd").values
master["pareto_overall"] = pareto_flag(master, "overall_avg", "cost_usd").values

# %% kolon sırası
ordered_cols = ["model", "vendor", "access_type", "params_b", "segment"]
for t in PRISM_TRACKS:
    ordered_cols += [f"{t}_gpt", f"{t}_qwen", f"{t}_avg"]
ordered_cols += ["overall_gpt", "overall_qwen", "overall_avg",
                 "cost_usd", "quality_per_usd"]
for t in PRISM_TRACKS:
    ordered_cols += [f"pareto_{t}"]
ordered_cols += ["pareto_overall"]
master = master[ordered_cols]

# %% sanity
print("=== KONTROL LİSTESİ ===")
print(f"Satır sayısı: {len(master)} (beklenen 9)")
print(f"DALL-E 2 NaN: {pd.isna(master.loc[master['model']=='DALL-E 2', 'overall_avg'].values[0])}")
print(f"DALL-E 2 cost: {master.loc[master['model']=='DALL-E 2', 'cost_usd'].values[0]} (beklenen 0.020)")
print(f"Track avg = (gpt+qwen)/2 doğru mu (örnek SD1.5 imagination):")
sd_g = master.loc[master['model']=='SD1.5', 'imagination_gpt'].values[0]
sd_q = master.loc[master['model']=='SD1.5', 'imagination_qwen'].values[0]
sd_a = master.loc[master['model']=='SD1.5', 'imagination_avg'].values[0]
print(f"  gpt={sd_g}, qwen={sd_q}, avg={sd_a}, kontrol={(sd_g+sd_q)/2:.2f}")
print(f"Pareto-overall sayısı: {master['pareto_overall'].sum()} (beklenen 6)")
print(f"Track bazlı Pareto sayıları:")
for t in PRISM_TRACKS:
    n = master[f"pareto_{t}"].sum()
    print(f"  {t:<16s}: {n} model")

print(f"\nEksik değer kontrolü (DALL-E 2 hariç):")
non_dalle = master[master["model"] != "DALL-E 2"]
score_cols = [f"{t}_avg" for t in PRISM_TRACKS]
n_nan = non_dalle[score_cols].isna().sum().sum()
print(f"  PRISM 8 modelinde NaN sayısı: {n_nan} (beklenen 0)")

# %% kaydet
out_path = ROOT / "data" / "processed" / "master_final.csv"
master.to_csv(out_path, index=False)
print(f"\n>> Kaydedildi: {out_path}")
print(f">> Toplam {master.shape[1]} kolon × {master.shape[0]} satır")

# %% özet preview
print("\n=== KORELASYONA HAZIR ÖZET ===")
print(master[["model", "segment", "overall_avg", "cost_usd", "quality_per_usd",
              "pareto_overall"]].to_string(index=False))
