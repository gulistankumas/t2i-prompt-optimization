"""Base router class — tüm router yaklaşımlarının ortak interface'i."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RouterDecision:
    """Router'ın verdiği kararın yapısı."""
    model: str                      # Önerilen model adı
    expected_quality: float         # Beklenen kalite skoru (0-100)
    expected_cost: float            # Beklenen maliyet (USD/image)
    predicted_track: str            # Tahmin edilen PRISM track
    mode: str                       # "economic" / "premium"
    fallback: bool = False
    fallback_note: Optional[str] = None

    def __repr__(self):
        flag = " [FALLBACK]" if self.fallback else ""
        return (f"RouterDecision(model={self.model}, "
                f"quality={self.expected_quality:.1f}, "
                f"cost=${self.expected_cost:.4f}, "
                f"track={self.predicted_track}, "
                f"mode={self.mode}){flag}")


class BaseRouter:
    """Tüm router'ların türeyeceği soyut sınıf."""

    def __init__(self, name: str):
        self.name = name

    def route(self, prompt: str, mode: str = "economic") -> RouterDecision:
        raise NotImplementedError("Subclasses must implement route()")

    def route_batch(self, prompts: list[str], mode: str = "economic") -> list[RouterDecision]:
        return [self.route(p, mode) for p in prompts]
