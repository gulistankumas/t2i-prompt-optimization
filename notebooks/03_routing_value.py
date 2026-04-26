"""Yüksek korelasyona rağmen routing değer üretiyor mu?

Hipotez: Üst tier'da routing trivial; orta/ucuz tier'da track-spesifik
seçim ciddi kalite kazanımı sağlıyor.
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader.prism import load_prism, filter_target_models, PRISM_TRACKS

pd.set_option("display.width", 200)

df = filter_target_models(load_prism("gpt41")).set_index("model")

# %% segmentlere böl
df_sorted = df.sort_values("overall", ascending=False)
print("Modeller overall'a göre sıralı:")
print(df_sorted[["overall"]])

PREMIUM = ["GPT-Image-1", "Gemini2.5-Flash-Image", "Qwen-Image"]
MID = ["SD3.5-Large", "FLUX.1-dev", "FLUX.1-schnell", "SDXL"]
LOW = ["SD1.5"]

# %% her segmentte: track-aware vs track-agnostic
def compare_strategies(segment_models, segment_name):
    sub = df.loc[segment_models]
    # track-agnostic: her zaman overall'da en iyi olanı seç
    best_overall = sub["overall"].idxmax()
    track_agnostic_quality = {t: sub.loc[best_overall, t] for t in PRISM_TRACKS}

    # track-aware: her track için o track'te en iyi olanı seç
    track_aware_quality = {t: sub[t].max() for t in PRISM_TRACKS}
    track_aware_winner = {t: sub[t].idxmax() for t in PRISM_TRACKS}

    rows = []
    for t in PRISM_TRACKS:
        gain = track_aware_quality[t] - track_agnostic_quality[t]
        rows.append({
            "track": t,
            "agnostic_pick": best_overall,
            "agnostic_score": track_agnostic_quality[t],
            "aware_pick": track_aware_winner[t],
            "aware_score": track_aware_quality[t],
            "gain": gain,
        })
    out = pd.DataFrame(rows)
    print(f"\n=== {segment_name} segmenti ===")
    print(out.to_string(index=False))
    print(f"  Toplam routing kazancı (7 track ortalaması): {out['gain'].mean():.2f} puan")
    print(f"  Aware seçimde kaç farklı model kullanıldı: {out['aware_pick'].nunique()}")

compare_strategies(PREMIUM, "PREMIUM (top-3)")
compare_strategies(MID, "MID-TIER (4 model)")

# %% en kritik: ucuz segmentte track'e göre seçim
print("\n=== UCUZ ÖZEL: SDXL vs FLUX.1-schnell ===")
print("(her ikisi de düşük maliyet segmenti, hangisi hangi track'te iyi?)")
cheap_pair = df.loc[["SDXL", "FLUX.1-schnell"]][PRISM_TRACKS].T
cheap_pair["fark"] = cheap_pair["FLUX.1-schnell"] - cheap_pair["SDXL"]
cheap_pair["pick"] = cheap_pair.apply(lambda r: "FLUX.1-schnell" if r["fark"] > 0 else "SDXL", axis=1)
print(cheap_pair.round(1))
print(f"\n  En büyük fark: {cheap_pair['fark'].abs().max():.1f} puan ({cheap_pair['fark'].abs().idxmax()})")
print(f"  Yön değişiyor mu? {len(cheap_pair['pick'].unique())} farklı kazanan")
