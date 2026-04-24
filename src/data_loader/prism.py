"""PRISM-Bench leaderboard loader."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

PRISM_FILES = {
    "gpt41": _DATA_DIR / "prism_bench_gpt41_english.csv",
    "qwen25vl": _DATA_DIR / "prism_bench_qwen25vl_english.csv",
}

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


def load_prism(judge: Literal["gpt41", "qwen25vl"] = "gpt41") -> pd.DataFrame:
    path = PRISM_FILES[judge]
    df = pd.read_csv(path)
    assert set(PRISM_TRACKS).issubset(df.columns), f"Missing track columns: {set(PRISM_TRACKS) - set(df.columns)}"
    assert "overall" in df.columns
    assert "model" in df.columns
    return df


def load_prism_gpt41_english() -> pd.DataFrame:
    return load_prism("gpt41")


def filter_target_models(df: pd.DataFrame) -> pd.DataFrame:
    missing = [m for m in TARGET_MODELS_IN_PRISM if m not in df["model"].values]
    if missing:
        raise ValueError(f"Target models not found in PRISM CSV: {missing}")
    return df[df["model"].isin(TARGET_MODELS_IN_PRISM)].reset_index(drop=True)


if __name__ == "__main__":
    for judge in ("gpt41", "qwen25vl"):
        df = load_prism(judge)
        tgt = filter_target_models(df)
        print(f"[{judge}] {len(df)} models loaded, {len(tgt)} target models, overall range {tgt['overall'].min():.1f}-{tgt['overall'].max():.1f}")
