"""Gün 4 — 19-model PRISM havuzu üzerinde genişletilmiş analiz.

Hedef: Bizim 8-model alt-örneklemimizdeki yapay yüksek korelasyon yerine,
PRISM'in tam 19-model havuzunda ne tür capability ayrışmaları var?
"""
# %% imports
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader.prism import load_prism, PRISM_TRACKS

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

df = load_prism("gpt41")
mat = df.set_index("model")[PRISM_TRACKS]

# ============================================================
# 1. 19-model Spearman + en bağımsız 5 track çifti
# ============================================================
print("=" * 70)
print("1. SPEARMAN korelasyon (19 model)")
print("=" * 70)
corr = mat.corr(method="spearman").round(3)
print(corr)

print("\n-- En düşük korelasyonlu 5 çift --")
pair_rows = []
for a, b in combinations(PRISM_TRACKS, 2):
    pair_rows.append({"track_a": a, "track_b": b, "spearman": corr.loc[a, b]})
pair_df = pd.DataFrame(pair_rows).sort_values("spearman").reset_index(drop=True)
print(pair_df.head(5).to_string(index=False))

# ============================================================
# 2. 19-model bottleneck profili
# ============================================================
print("\n" + "=" * 70)
print("2. BOTTLENECK PROFİLİ (19 model, GPT-4.1 jüri)")
print("=" * 70)

bn = mat.copy()
bn["mean"] = mat.mean(axis=1)
bn["min"] = mat.min(axis=1)
bn["max"] = mat.max(axis=1)
bn["range"] = bn["max"] - bn["min"]
bn["std"] = mat.std(axis=1)
bn["weakest_track"] = mat.idxmin(axis=1)
bn["strongest_track"] = mat.idxmax(axis=1)

bn_sorted = bn.sort_values("range", ascending=False)
print("\n-- En dengesiz (range büyük) --")
print(bn_sorted[["mean", "min", "max", "range", "std", "weakest_track"]].head(8).round(1).to_string())

print("\n-- En dengeli (range küçük, mean yüksek tercih) --")
print(bn_sorted[["mean", "min", "max", "range", "std", "weakest_track"]].tail(6).round(1).to_string())

# weakest track distribution
print("\n-- Hangi track 'en zayıf halka' rolünde sıkça çıkıyor? --")
print(bn["weakest_track"].value_counts())

# ============================================================
# 3. Specialist / Generalist skoru (rank disparity)
# ============================================================
print("\n" + "=" * 70)
print("3. SPECIALIST/GENERALIST skoru (max rank disparity)")
print("=" * 70)
print("(Tüm 21 çift üzerinde modelin maksimum rank farkı — büyükse uzman,")
print(" küçükse generalist)")

# Her model için 21 çift üzerinde max rank farkı hesapla
rank_per_track = mat.rank(ascending=False, method="min")
rank_diffs = []
for model in mat.index:
    max_diff = 0
    diff_pair = None
    for a, b in combinations(PRISM_TRACKS, 2):
        d = abs(rank_per_track.loc[model, a] - rank_per_track.loc[model, b])
        if d > max_diff:
            max_diff = d
            diff_pair = (a, b)
    rank_diffs.append({
        "model": model,
        "mean_score": mat.loc[model].mean(),
        "max_rank_disparity": max_diff,
        "disparity_pair": diff_pair,
    })
rd = pd.DataFrame(rank_diffs).sort_values("max_rank_disparity", ascending=False)
print(rd.round(1).to_string(index=False))

print("\n-- En uzman 5 model (büyük disparity) --")
print(rd.head(5).round(1).to_string(index=False))
print("\n-- En generalist 5 model --")
print(rd.tail(5).round(1).to_string(index=False))

# ============================================================
# 4. Combined skor stratejileri (mean vs min vs hybrid)
# ============================================================
print("\n" + "=" * 70)
print("4. KOMBİNE SKOR STRATEJİ KARŞILAŞTIRMASI")
print("=" * 70)
print("Eğer prompt 'tüm 7 track' istiyorsa hangi model?")
agg = pd.DataFrame({
    "mean_7tracks": mat.mean(axis=1),
    "min_7tracks": mat.min(axis=1),
    "harmonic_mean": 7 / (1.0 / mat).sum(axis=1),
}).round(1).sort_values("min_7tracks", ascending=False)

print("\n(min_7tracks = en kötü track — bottleneck-aware kalite tahmini)")
print(agg.to_string())

# Sıralamada fark eden modeller
mean_rank = agg["mean_7tracks"].rank(ascending=False)
min_rank = agg["min_7tracks"].rank(ascending=False)
inversion = (mean_rank - min_rank).abs().sort_values(ascending=False)
print("\n-- Mean vs min sıralamada en büyük fark --")
print(inversion.head(5).to_string())
