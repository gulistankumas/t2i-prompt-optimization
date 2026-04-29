"""Gün 4 — Self-preference bias formal analizi.

Hedef: GPT-4.1 ve Qwen2.5-VL jürilerinin sistematik bias'larını ölç.
Hipotez: Jüri sahibi şirketin modeline yüksek puan verme eğilimi var.
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data_loader.prism import load_prism, PRISM_TRACKS

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

# Vendor mapping
VENDOR = {
    "GPT-Image-1": "OpenAI",
    "Gemini2.5-Flash-Image": "Google",
    "Qwen-Image": "Alibaba",
    "SEEDream 3.0": "ByteDance",
    "HiDream-I1-Full": "HiDream",
    "FLUX.1-Krea-dev": "BFL",
    "FLUX.1-dev": "BFL",
    "SD3.5-Large": "Stability",
    "HiDream-I1-Dev": "HiDream",
    "SD3.5-Medium": "Stability",
    "SD3-Medium": "Stability",
    "Bagel-CoT": "ByteDance",
    "Bagel": "ByteDance",
    "FLUX.1-schnell": "BFL",
    "Playground": "Playground",
    "JanusPro-7B": "DeepSeek",
    "SDXL": "Stability",
    "SD2.1": "Stability",
    "SD1.5": "Runway",
}

# %% load both judges aligned
gpt = load_prism("gpt41").set_index("model")
qwen = load_prism("qwen25vl").set_index("model")
qwen = qwen.reindex(gpt.index)

# delta per model per track
delta = (gpt[PRISM_TRACKS] - qwen[PRISM_TRACKS])
delta["overall_delta"] = gpt["overall"] - qwen["overall"]
delta["mean_track_delta"] = delta[PRISM_TRACKS].mean(axis=1)
delta["max_track_delta"] = delta[PRISM_TRACKS].max(axis=1)
delta["min_track_delta"] = delta[PRISM_TRACKS].min(axis=1)
delta["vendor"] = [VENDOR[m] for m in delta.index]
delta = delta.sort_values("overall_delta", ascending=False)

print("=" * 70)
print("1. GPT-4.1 - Qwen2.5-VL skor delta'sı (model bazında)")
print("=" * 70)
print("Pozitif: GPT-4.1 daha cömert. Negatif: Qwen2.5-VL daha cömert.")
print()
print(delta[["vendor", "overall_delta", "mean_track_delta", "max_track_delta", "min_track_delta"]].round(2).to_string())

# %% vendor-bazlı toplam
print("\n" + "=" * 70)
print("2. VENDOR bazında ortalama delta (kaç model olduğu # ile)")
print("=" * 70)
vd = delta.groupby("vendor").agg(
    n_models=("overall_delta", "count"),
    mean_delta=("overall_delta", "mean"),
    median_delta=("overall_delta", "median"),
).sort_values("mean_delta", ascending=False).round(2)
print(vd.to_string())

# %% track bazında bias yönü (her track'te 19 model üzerinde GPT4.1 ortalama - Qwen ortalama)
print("\n" + "=" * 70)
print("3. TRACK bazında jüri tercih farkı (19 model toplam)")
print("=" * 70)
track_bias = pd.DataFrame({
    "gpt41_mean": gpt[PRISM_TRACKS].mean(),
    "qwen_mean": qwen[PRISM_TRACKS].mean(),
    "delta": gpt[PRISM_TRACKS].mean() - qwen[PRISM_TRACKS].mean(),
}).round(2).sort_values("delta", ascending=False)
print(track_bias.to_string())

# %% sıralama anlaşmazlığı: top-1 değişen track'ler
print("\n" + "=" * 70)
print("4. TOP-1 ANLAŞMAZLIĞI (her track'te kim birinci?)")
print("=" * 70)
disagree_rows = []
for t in PRISM_TRACKS:
    g_top = gpt[t].idxmax()
    q_top = qwen[t].idxmax()
    disagree_rows.append({
        "track": t,
        "gpt41_winner": g_top,
        "qwen_winner": q_top,
        "match": g_top == q_top,
        "gpt41_top_score": gpt.loc[g_top, t],
        "qwen_top_score": qwen.loc[q_top, t],
    })
print(pd.DataFrame(disagree_rows).round(1).to_string(index=False))

# %% paired test (parametric değil, sign test)
from scipy.stats import wilcoxon
print("\n" + "=" * 70)
print("5. WILCOXON SIGNED-RANK testi (her model 7 track delta'sı 0'dan farklı mı?)")
print("=" * 70)
print("H0: GPT4.1 ve Qwen aynı skoru verir; H1: sistematik fark var")
sig_rows = []
for model in delta.index:
    arr = delta.loc[model, PRISM_TRACKS].values.astype(float)
    if np.all(arr == 0):
        stat, p = (np.nan, np.nan)
    else:
        try:
            stat, p = wilcoxon(arr)
        except ValueError:
            stat, p = (np.nan, np.nan)
    sig_rows.append({
        "model": model,
        "vendor": VENDOR[model],
        "mean_delta": arr.mean(),
        "wilcoxon_p": p,
        "significant_at_0.05": (p < 0.05) if not np.isnan(p) else False,
    })
sig = pd.DataFrame(sig_rows).sort_values("mean_delta", ascending=False)
print(sig.round(3).to_string(index=False))

# %% verdict
print("\n" + "=" * 70)
print("ÖZET")
print("=" * 70)
gpt_advantage = (delta["overall_delta"] > 0).sum()
qwen_advantage = (delta["overall_delta"] < 0).sum()
print(f"GPT-4.1 daha yüksek skor verdiği model sayısı: {gpt_advantage}/19")
print(f"Qwen2.5-VL daha yüksek skor verdiği model sayısı: {qwen_advantage}/19")
print(f"En büyük GPT-4.1 avantajı: {delta.iloc[0].name} (+{delta.iloc[0]['overall_delta']:.2f})")
print(f"En büyük Qwen2.5-VL avantajı: {delta.iloc[-1].name} ({delta.iloc[-1]['overall_delta']:+.2f})")
