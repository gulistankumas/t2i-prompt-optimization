"""PRISM-Bench leaderboard loader."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PRISM_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "prism_bench_gpt41_english.csv"

PRISM_TRACKS = [
    "imagination",
    "entity",
    "text_rendering",
    "style",
    "affection",
    "composition",
    "long_text",
]

TARGET_MODELS_IN_PRISM = [
    "SD1.5",
    "SDXL",
    "SD3.5-Large",
    "FLUX.1-schnell",
    "FLUX.1-dev",
    "Qwen-Image",
    "GPT-Image-1",
    "Gemini2.5-Flash-Image",
]


def load_prism_gpt41_english() -> pd.DataFrame:
    df = pd.read_csv(PRISM_CSV_PATH)
    assert set(PRISM_TRACKS).issubset(df.columns), f"Missing track columns: {set(PRISM_TRACKS) - set(df.columns)}"
    assert "overall" in df.columns
    assert "model" in df.columns
    return df


def filter_target_models(df: pd.DataFrame) -> pd.DataFrame:
    missing = [m for m in TARGET_MODELS_IN_PRISM if m not in df["model"].values]
    if missing:
        raise ValueError(f"Target models not found in PRISM CSV: {missing}")
    return df[df["model"].isin(TARGET_MODELS_IN_PRISM)].reset_index(drop=True)


if __name__ == "__main__":
    df = load_prism_gpt41_english()
    print(f"Loaded {len(df)} models, {len(PRISM_TRACKS)} tracks")
    targets = filter_target_models(df)
    print(f"\nTarget models ({len(targets)}):")
    print(targets.to_string(index=False))
