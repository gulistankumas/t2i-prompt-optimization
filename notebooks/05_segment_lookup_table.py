"""Yüksek + Orta + Ucuz segment tüm tracklerde model skorları, kazanan ve spread.

Soru: Track-aware routing, lookup table mantığında anlamlı bir tablo veriyor mu?
"""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader.prism import load_prism, PRISM_TRACKS

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

SEGMENTS = {
    "PREMIUM": ["GPT-Image-1", "Gemini2.5-Flash-Image", "Qwen-Image"],
    "MID":     ["SD3.5-Large", "FLUX.1-dev"],
    "UCUZ":    ["FLUX.1-schnell", "SDXL", "SD1.5"],
}


def show_segment(df: pd.DataFrame, models: list[str], name: str):
    sub = df.loc[models, PRISM_TRACKS + ["overall"]].round(1)
    print(f"\n{'=' * 70}")
    print(f"{name} segmenti (overall sıralı)")
    print('=' * 70)
    print(sub.sort_values("overall", ascending=False).to_string())


def lookup_row(df: pd.DataFrame, models: list[str]) -> dict:
    sub = df.loc[models]
    out = {}
    for t in PRISM_TRACKS:
        out[t] = (sub[t].idxmax(), float(sub[t].max()), float(sub[t].max() - sub[t].min()))
    return out


def render_lookup(df: pd.DataFrame, judge_name: str):
    print(f"\n{'=' * 70}")
    print(f"LOOKUP TABLE — {judge_name}: (segment, track) -> kazanan (skor, spread)")
    print('=' * 70)
    rows = []
    for seg, models in SEGMENTS.items():
        rec = lookup_row(df, models)
        for t in PRISM_TRACKS:
            winner, score, spread = rec[t]
            rows.append({"segment": seg, "track": t, "winner": winner, "score": score, "spread": spread})
    out = pd.DataFrame(rows)
    pivot_winner = out.pivot(index="track", columns="segment", values="winner").reindex(PRISM_TRACKS)
    pivot_score = out.pivot(index="track", columns="segment", values="score").reindex(PRISM_TRACKS).round(1)
    pivot_spread = out.pivot(index="track", columns="segment", values="spread").reindex(PRISM_TRACKS).round(1)

    print("\n-- Kazanan model --")
    print(pivot_winner[["PREMIUM", "MID", "UCUZ"]].to_string())
    print("\n-- Kazananın skoru --")
    print(pivot_score[["PREMIUM", "MID", "UCUZ"]].to_string())
    print("\n-- Segment-içi spread (max - min) --")
    print(pivot_spread[["PREMIUM", "MID", "UCUZ"]].to_string())

    # routing değer ölçütü: kazananın segment overall-en-iyi'siyle aynı olmadığı satırlar
    print("\n-- Routing yön değişikliği (kazanan != segmentin overall lideri) --")
    for seg, models in SEGMENTS.items():
        seg_df = df.loc[models]
        overall_top = seg_df["overall"].idxmax()
        diff_tracks = [t for t in PRISM_TRACKS if seg_df[t].idxmax() != overall_top]
        print(f"  {seg:8s} (overall lideri: {overall_top}) -> yön değiştiren track sayısı: {len(diff_tracks)}/7  {diff_tracks}")


# %% her iki jüri için ayrı ayrı
for judge in ("gpt41", "qwen25vl"):
    df = load_prism(judge).set_index("model")
    print(f"\n\n{'#' * 70}")
    print(f"## JÜRI: {judge}")
    print('#' * 70)
    for seg_name, models in SEGMENTS.items():
        show_segment(df, models, seg_name)
    render_lookup(df, judge)
