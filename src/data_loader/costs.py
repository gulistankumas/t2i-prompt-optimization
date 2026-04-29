"""Maliyet verisi loader."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

COSTS_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "model_costs.csv"


def load_costs() -> pd.DataFrame:
    df = pd.read_csv(COSTS_PATH)
    assert {"model", "vendor", "access_type", "cost_per_image_usd"}.issubset(df.columns)
    return df


if __name__ == "__main__":
    df = load_costs()
    print(df.to_string(index=False))
