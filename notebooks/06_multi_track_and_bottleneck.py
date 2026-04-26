"""Gün 3 — Çoklu-track analizi + Bottleneck model analizi.

Soru 1: Track çiftleri arasındaki en düşük korelasyon hangisi? Kombine skor
sıralamasında yön değişiyor mu (ranking inversion)?

Soru 2: Bazı modeller bir track'te yüksek, başka track'te dramatik düşüyor mu?
Eğer prompt çoklu yetenek istiyorsa bu modeller "bottleneck" olur.
"""
# %% imports
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader.prism import load_prism, filter_target_models, PRISM_TRACKS

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# %% load — both 8-model and 19-model
df_all = load_prism("gpt41")  # 19 models
df8 = filter_target_models(df_all)  # 8 target

# ============================================================
# BÖLÜM 1: En düşük korelasyonlu track çiftleri ve kombinasyonu
# ============================================================
print("=" * 70)
print("1. EN DÜŞÜK KORELASYONLU 3 TRACK ÇİFTİ (19 model üzerinde, Spearman)")
print("=" * 70)

corr_19 = df_all[PRISM_TRACKS].corr(method="spearman")
pair_rows = []
for a, b in combinations(PRISM_TRACKS, 2):
    pair_rows.append({"track_a": a, "track_b": b, "spearman": corr_19.loc[a, b]})
pair_df = pd.DataFrame(pair_rows).sort_values("spearman").reset_index(drop=True)
print(pair_df.head(10).round(3).to_string(index=False))

# 3 en düşük çift
top3_independent = pair_df.head(3)[["track_a", "track_b"]].values.tolist()
print(f"\nİncelenen 3 çift: {top3_independent}")

# %% her çift için kombine skor sıralama analizi
def analyze_pair(df: pd.DataFrame, ta: str, tb: str, label: str):
    print(f"\n{'-' * 70}")
    print(f"PAIR: {ta} <-> {tb}    [{label}]")
    print('-' * 70)
    sub = df[[ "model", ta, tb]].copy()
    sub["mean"] = (sub[ta] + sub[tb]) / 2
    sub["min"]  = sub[[ta, tb]].min(axis=1)  # bottleneck-aware aggregation
    sub = sub.sort_values("mean", ascending=False).reset_index(drop=True)

    # rank inversions
    rank_a = sub[ta].rank(ascending=False, method="min")
    rank_b = sub[tb].rank(ascending=False, method="min")
    rank_mean = sub["mean"].rank(ascending=False, method="min")
    rank_min = sub["min"].rank(ascending=False, method="min")

    out = sub.copy()
    out["rank_a"] = rank_a.astype(int)
    out["rank_b"] = rank_b.astype(int)
    out["rank_mean"] = rank_mean.astype(int)
    out["rank_min"] = rank_min.astype(int)
    out["|rank_a - rank_b|"] = (rank_a - rank_b).abs().astype(int)

    print(out[["model", ta, tb, "mean", "min", "rank_a", "rank_b", "rank_mean", "rank_min", "|rank_a - rank_b|"]].round(1).to_string(index=False))

    # specialist / generalist verdict
    print(f"\n  En büyük rank-farkı: {out['|rank_a - rank_b|'].max()} → "
          f"{out.loc[out['|rank_a - rank_b|'].idxmax(), 'model']} (uzman, sadece bir track'te iyi)")

# her çift için 19-model üzerinde
for ta, tb in top3_independent:
    analyze_pair(df_all, ta, tb, "19 model")

# ============================================================
# BÖLÜM 2: Bottleneck Model Analizi
# ============================================================
print("\n\n" + "=" * 70)
print("2. BOTTLENECK MODEL ANALİZİ (8 hedef model)")
print("=" * 70)

bn = df8.set_index("model")[PRISM_TRACKS].copy()
bn["mean"] = bn[PRISM_TRACKS].mean(axis=1)
bn["min"] = bn[PRISM_TRACKS].min(axis=1)
bn["max"] = bn[PRISM_TRACKS].max(axis=1)
bn["range"] = bn["max"] - bn["min"]
bn["std"] = bn[PRISM_TRACKS].std(axis=1)
bn["weakest_track"] = bn[PRISM_TRACKS].idxmin(axis=1)
bn["strongest_track"] = bn[PRISM_TRACKS].idxmax(axis=1)
bn = bn.sort_values("range", ascending=False)

print("\n-- Track profili dengesizlik (range = max-min) --")
print(bn[["mean", "max", "min", "range", "std", "weakest_track", "strongest_track"]].round(1).to_string())

# %% multiplicative penalty: AND-prompt aggregation
print("\n-- Bottleneck ÖRNEĞİ: hangi model çoklu-yetenek prompt'unda düşer? --")
print("(min(track_a, track_b) = AND-prompt için pessimistic kalite tahmini)")
print()

DEMO_COMBOS = [
    ("style", "text_rendering"),
    ("entity", "long_text"),
    ("style", "long_text"),
]
for ta, tb in DEMO_COMBOS:
    print(f"\n  Kombo: {ta} + {tb}")
    sub = df8.set_index("model")[[ta, tb]].copy()
    sub["solo_a"] = sub[ta]
    sub["solo_b"] = sub[tb]
    sub["combo_min"] = sub[[ta, tb]].min(axis=1)
    sub["drop_from_solo_a"] = sub["solo_a"] - sub["combo_min"]
    sub = sub.sort_values("combo_min", ascending=False)
    print(sub[["solo_a", "solo_b", "combo_min", "drop_from_solo_a"]].round(1).to_string())
    biggest_drop = sub["drop_from_solo_a"].idxmax()
    print(f"  En dramatik düşüş: {biggest_drop} ({sub.loc[biggest_drop, 'drop_from_solo_a']:.1f} puan)")
