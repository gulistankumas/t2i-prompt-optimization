"""Day 2: PRISM skorlarının ilk keşfi — track bazında model sıralaması ne kadar farklı?

Bu dosya jupyter notebook gibi çalışacak şekilde # %% cell marker'ları ile yazıldı.
VS Code veya PyCharm'da "Run Cell" ile interaktif çalıştırılabilir, ya da düz python
olarak `python notebooks/01_prism_exploration.py` şeklinde koşulur.
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader.prism import load_prism_gpt41_english, filter_target_models, PRISM_TRACKS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# %% load
df_all = load_prism_gpt41_english()
df = filter_target_models(df_all)
print(f"8 hedef model, 7 track. Overall skor aralığı: {df['overall'].min():.1f} – {df['overall'].max():.1f}")

# %% per-track best model
print("\n=== Track bazında BİRİNCİ model ===")
best_per_track = {t: df.loc[df[t].idxmax(), "model"] for t in PRISM_TRACKS}
for t, m in best_per_track.items():
    print(f"  {t:<16s} -> {m}")

unique_winners = set(best_per_track.values())
print(f"\nFarklı birinci sayısı: {len(unique_winners)}/7 -> {sorted(unique_winners)}")

# %% per-track ranking (who is top-3 in each track?)
print("\n=== Track bazında TOP-3 ===")
for t in PRISM_TRACKS:
    top3 = df.nlargest(3, t)[["model", t]]
    ranking = " > ".join(f"{row.model} ({row[t]:.1f})" for _, row in top3.iterrows())
    print(f"  {t:<16s}: {ranking}")

# %% spread per track (does routing matter?)
print("\n=== Track içi skor yayılımı ===")
for t in PRISM_TRACKS:
    vals = df[t]
    spread = vals.max() - vals.min()
    print(f"  {t:<16s}: min={vals.min():.1f}, max={vals.max():.1f}, spread={spread:.1f}")

# %% correlation between tracks (are some tracks redundant?)
print("\n=== Track skorları arası korelasyon ===")
corr = df[PRISM_TRACKS].corr().round(2)
print(corr)

# %% routing signal: if best model by overall is always the best per-track, routing is useless
overall_winner = df.loc[df["overall"].idxmax(), "model"]
always_best = all(best_per_track[t] == overall_winner for t in PRISM_TRACKS)
print(f"\n=== Routing sinyali ===")
print(f"Overall birinci: {overall_winner}")
print(f"Tüm track'lerde de birinci mi? {always_best}")
if not always_best:
    non_overall_wins = {t: m for t, m in best_per_track.items() if m != overall_winner}
    print(f"Overall-birinci-OLMAYAN track'ler: {non_overall_wins}")
