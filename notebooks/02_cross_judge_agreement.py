"""Gün 2+: İki PRISM jürisi (GPT-4.1 ve Qwen2.5-VL) arasındaki anlaşma.

Jüri seçimi router kararını etkiliyor mu? Eğer iki jüri benzer sıralamaları
veriyorsa tezin "gerçek sinyal" iddiası güçlenir.
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader.prism import load_prism, filter_target_models, PRISM_TRACKS

pd.set_option("display.width", 200)

# %% load both
gpt = filter_target_models(load_prism("gpt41")).set_index("model")
qwen = filter_target_models(load_prism("qwen25vl")).set_index("model")
# keep same model order
qwen = qwen.reindex(gpt.index)

# %% per-model score delta (overall)
print("=== Overall skor karşılaştırması ===")
delta = pd.DataFrame({
    "gpt41": gpt["overall"],
    "qwen25vl": qwen["overall"],
    "delta": gpt["overall"] - qwen["overall"],
})
print(delta.round(1).sort_values("delta", ascending=False))
print(f"\nMean |delta|: {delta['delta'].abs().mean():.2f}")

# %% per-track rank correlation (Spearman)
print("\n=== Track başına iki jüri arası sıralama korelasyonu (Spearman) ===")
for t in PRISM_TRACKS + ["overall"]:
    rho = gpt[t].rank().corr(qwen[t].rank())  # Pearson-on-ranks = Spearman
    print(f"  {t:<16s}: rho = {rho:.3f}")

# %% per-track top-1 agreement
print("\n=== Track başına en iyi model anlaşması ===")
disagreements = []
for t in PRISM_TRACKS:
    g_best = gpt[t].idxmax()
    q_best = qwen[t].idxmax()
    match = "MATCH" if g_best == q_best else "DIFFER"
    print(f"  {t:<16s}: GPT41={g_best}, Qwen25VL={q_best}  [{match}]")
    if match == "DIFFER":
        disagreements.append(t)

# %% final verdict
print(f"\n=== Sonuç ===")
print(f"Top-1 anlaşmazlık olan track sayısı: {len(disagreements)}/7")
if disagreements:
    print(f"Anlaşmazlık track'leri: {disagreements}")
    print("Tez için: jüri seçimi bu track'lerde router kararını değiştirebilir.")
else:
    print("İki jüri tüm track'lerde aynı top-1'i veriyor -> router kararı jüri'den bağımsız.")
