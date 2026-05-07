"""Lookup-table tabanlı router.

Pipeline:
  prompt -> kategorizer -> predicted_track
  (predicted_track, mode) -> lookup table -> model + meta

Kategorizer interface'i değiştirilebilir (default: MLCategorizer LR).
Lookup table Gün 8'de hazırlandı: data/processed/lookup_table_2mode.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.router.base_router import BaseRouter, RouterDecision
from src.router.ml_categorizer import MLCategorizer


_DEFAULT_LOOKUP = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "lookup_table_2mode.json"
)

VALID_MODES = ("economic", "premium")


class LookupRouter(BaseRouter):
    def __init__(self, categorizer: Any = None,
                 lookup_path: str | Path = _DEFAULT_LOOKUP):
        super().__init__(name="LookupRouter")
        self.categorizer = categorizer or MLCategorizer(classifier="lr")
        with open(lookup_path, "r", encoding="utf-8") as f:
            self.lookup = json.load(f)
        self.valid_tracks = list(self.lookup["economic"].keys())

    def route(self, prompt: str, mode: str = "economic") -> RouterDecision:
        if mode not in VALID_MODES:
            raise ValueError(f"Geçersiz mod: {mode}. Seçenekler: {VALID_MODES}")

        predicted = self.categorizer.classify(prompt)
        if predicted not in self.valid_tracks:
            predicted = "entity"  # safety fallback

        entry = self.lookup[mode][predicted]
        return RouterDecision(
            model=entry["model"],
            expected_quality=float(entry["quality"]),
            expected_cost=float(entry["cost"]),
            predicted_track=predicted,
            mode=mode,
            fallback=bool(entry.get("fallback", False)),
            fallback_note=entry.get("note"),
        )


if __name__ == "__main__":
    router = LookupRouter()
    cases = [
        ('a futuristic city with the word TOKYO in neon', "economic"),
        ('a futuristic city with the word TOKYO in neon', "premium"),
        ('an elephant standing in tall grass', "economic"),
        ('an elephant standing in tall grass', "premium"),
        ('a melancholic woman looking out at the rain', "economic"),
        ('a cat sitting next to a vase on a wooden table', "economic"),
        ('a Picasso-style cubist portrait', "premium"),
    ]
    print("=== LookupRouter Self-Test ===\n")
    for prompt, mode in cases:
        d = router.route(prompt, mode)
        print(f"[{mode:8s}] {prompt[:55]}")
        print(f"           -> {d}\n")
