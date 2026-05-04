"""ML tabanlı PRISM track kategorizer.

SBERT prompt embedding + classifier (default XGBoost). Gün 10'da eğitildi.

Kullanım:
    from src.router.ml_categorizer import MLCategorizer
    cat = MLCategorizer()                       # XGBoost yükler
    cat = MLCategorizer(classifier="lr")        # Logistic Regression yükler
    track = cat.classify("a happy dog")
    track, conf = cat.classify_with_confidence("...")
    tracks = cat.classify_batch(["...", "..."])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import joblib
from sentence_transformers import SentenceTransformer

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


class MLCategorizer:
    def __init__(self, classifier: Literal["lr", "xgb"] = "lr",
                 models_dir: Path | None = None):
        d = models_dir or _MODELS_DIR
        clf_path = d / f"track_classifier_{classifier}.pkl"
        self.classifier_name = classifier
        self.classifier = joblib.load(clf_path)
        self.label_encoder = joblib.load(d / "label_encoder.pkl")
        with open(d / "sbert_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.sbert = SentenceTransformer(cfg["model_name"])

    def classify(self, prompt: str) -> str:
        emb = self.sbert.encode([prompt], convert_to_numpy=True)
        pred = self.classifier.predict(emb)[0]
        return str(self.label_encoder.inverse_transform([pred])[0])

    def classify_batch(self, prompts: list[str]) -> list[str]:
        emb = self.sbert.encode(prompts, convert_to_numpy=True)
        preds = self.classifier.predict(emb)
        return [str(t) for t in self.label_encoder.inverse_transform(preds)]

    def classify_with_confidence(self, prompt: str) -> tuple[str, float]:
        emb = self.sbert.encode([prompt], convert_to_numpy=True)
        probs = self.classifier.predict_proba(emb)[0]
        idx = int(probs.argmax())
        track = str(self.label_encoder.inverse_transform([idx])[0])
        return track, float(probs[idx])


if __name__ == "__main__":
    cat = MLCategorizer()
    test_prompts = [
        'a futuristic city with the word "TOKYO" in neon',
        'a surreal painting of floating clocks',
        'a happy golden retriever in a park',
        'an elephant',
        'a cat sitting next to a vase',
        'a melancholic woman looking out at the rain through a window in deep contemplation',
    ]
    print(f"=== MLCategorizer ({cat.classifier_name}) ===")
    for p in test_prompts:
        track, conf = cat.classify_with_confidence(p)
        print(f"  {track:<16s} (conf={conf:.2f}) | {p[:55]}")
