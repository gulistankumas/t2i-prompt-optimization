"""Ucuz açık-kaynak segmentin tüm tracklerdeki skorları, iki jüri yan yana."""
# %% imports
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader.prism import load_prism, PRISM_TRACKS

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

CHEAP_OPEN = ["SD1.5", "SDXL", "FLUX.1-schnell"]


def cheap_table(judge: str) -> pd.DataFrame:
    df = load_prism(judge).set_index("model")
    return df.loc[CHEAP_OPEN, PRISM_TRACKS + ["overall"]]


# %% iki jüri yan yana
gpt = cheap_table("gpt41")
qwen = cheap_table("qwen25vl")

print("=" * 60)
print("GPT-4.1 jürisi")
print("=" * 60)
print(gpt.round(1).to_string())

print("\n" + "=" * 60)
print("Qwen2.5-VL jürisi")
print("=" * 60)
print(qwen.round(1).to_string())

# %% per-track: kazanan + spread
def winner_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in PRISM_TRACKS:
        col = df[t]
        rows.append({
            "track": t,
            "min_score": col.min(),
            "max_score": col.max(),
            "spread": col.max() - col.min(),
            "winner": col.idxmax(),
            "loser": col.idxmin(),
        })
    return pd.DataFrame(rows)

print("\n" + "=" * 60)
print("Track bazında (GPT-4.1)")
print("=" * 60)
print(winner_table(gpt).round(1).to_string(index=False))

print("\n" + "=" * 60)
print("Track bazında (Qwen2.5-VL)")
print("=" * 60)
print(winner_table(qwen).round(1).to_string(index=False))

# %% iki jüri kazanan tutarlılığı
print("\n" + "=" * 60)
print("İki jüri 'kazanan' anlaşması")
print("=" * 60)
g_win = winner_table(gpt).set_index("track")["winner"]
q_win = winner_table(qwen).set_index("track")["winner"]
agree = pd.DataFrame({"gpt41_winner": g_win, "qwen25vl_winner": q_win})
agree["match"] = agree["gpt41_winner"] == agree["qwen25vl_winner"]
print(agree.to_string())
print(f"\nAnlaşma: {agree['match'].sum()}/{len(agree)} track")
