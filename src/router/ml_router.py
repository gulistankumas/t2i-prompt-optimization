"""ML-based T2I router (end-to-end SBERT + Logistic Regression).

Pipeline: prompt SBERT embedding + mod one-hot -> 8-class model classifier.
Kategorizer atlanır; track tahmini yok.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.router.base_router import BaseRouter, RouterDecision

_ROOT = Path(__file__).resolve().parents[2]
_MODELS = _ROOT / "models"
_PROC = _ROOT / "data" / "processed"

_TRACKS = ("imagination", "entity", "text_rendering", "style",
           "affection", "composition", "long_text")


class MLRouter(BaseRouter):
    def __init__(self,
                 model_path: Path = _MODELS / "ml_router_lr.pkl",
                 encoder_path: Path = _MODELS / "ml_router_label_encoder.pkl",
                 master_path: Path = _PROC / "master_final.csv",
                 sbert_name: str = "all-MiniLM-L6-v2"):
        super().__init__(name="MLRouter")
        self.classifier = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.sbert = SentenceTransformer(sbert_name)
        self.master = pd.read_csv(master_path)

    def _build_features(self, prompt: str, mode: str) -> np.ndarray:
        emb = self.sbert.encode([prompt], convert_to_numpy=True)
        mode_oh = np.array([[1, 0]]) if mode == "economic" else np.array([[0, 1]])
        return np.hstack([emb, mode_oh])

    def route(self, prompt: str, mode: str = "economic") -> RouterDecision:
        if mode not in ("economic", "premium"):
            raise ValueError(f"Geçersiz mod: {mode}")
        feat = self._build_features(prompt, mode)
        pred = self.classifier.predict(feat)[0]
        model_name = str(self.label_encoder.inverse_transform([pred])[0])

        row = self.master[self.master["model"] == model_name].iloc[0]
        avg_quality = float(row[[f"{t}_avg" for t in _TRACKS]].mean())
        cost = float(row["cost_usd"])

        return RouterDecision(
            model=model_name,
            expected_quality=avg_quality,
            expected_cost=cost,
            predicted_track="N/A",  # ML router track tahmin etmiyor
            mode=mode,
            fallback=False,
        )


if __name__ == "__main__":
    router = MLRouter()
    cases = [
        ('a futuristic city with the word TOKYO in neon', "economic"),
        ('a futuristic city with the word TOKYO in neon', "premium"),
        ('a melancholic woman looking out at the rain', "economic"),
        ('a Picasso-style cubist portrait', "premium"),
        ('a cat sitting next to a vase on a wooden table', "economic"),
        ('an elephant standing in tall grass', "premium"),
    ]
    print("=== MLRouter Self-Test ===\n")
    for prompt, mode in cases:
        d = router.route(prompt, mode)
        print(f"[{mode:8s}] {prompt[:55]}")
        print(f"           -> {d}\n")
