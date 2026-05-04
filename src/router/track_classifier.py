"""Kural tabanlı PRISM track sınıflandırıcı.

Prompt -> 7 PRISM track'inden biri.
Sıralı kontrol: spesifik kurallar önce, default 'entity' en sonda.

Tezde Lookup-rule router'ın kategorizer ayağı; LLM tabanlı sürümle (Gün 10)
karşılaştırılacak.
"""
from __future__ import annotations

import re

PRISM_TRACKS_LIST = [
    "imagination", "entity", "text_rendering", "style",
    "affection", "composition", "long_text",
]

_STYLE_KW = [
    "style", "painting", "monet", "van gogh", "picasso", "cubist",
    "cyberpunk", "anime", "oil painting", "watercolor", "sketch",
    "impressionist", "renaissance", "abstract painting",
]
_COMPOSITION_KW = [
    "next to", "on top of", "beside", "behind",
    "in front of", "between", "under", "above",
    "to the right of", "to the left of",
]
_IMAGINATION_KW = [
    "surreal", "fantasy", "dream", "impossible",
    "magical", "futuristic", "mythical", "ethereal",
    "alternate", "parallel world",
]
_AFFECTION_KW = [
    "happy", "sad", "joyful", "lonely", "melancholic",
    "cheerful", "sorrowful", "peaceful", "angry", "wistful",
]


def classify_track_rules(prompt: str) -> str:
    """Kural sırası: text_rendering -> long_text -> style -> composition
    -> imagination -> affection -> entity (default)."""
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    # 1. text_rendering — tırnak içinde harfler, "text", "word", "letters"
    if re.search(r'"[^"]+"', prompt) or any(
        kw in prompt_lower for kw in ("text", "word", "letters", "inscription", "sign")
    ):
        return "text_rendering"

    # 2. long_text — 25+ kelime
    if word_count >= 25:
        return "long_text"

    # 3. style
    if any(kw in prompt_lower for kw in _STYLE_KW):
        return "style"

    # 4. composition — uzamsal ilişki
    if any(kw in prompt_lower for kw in _COMPOSITION_KW):
        return "composition"

    # 5. imagination — yaratıcı/imkansız
    if any(kw in prompt_lower for kw in _IMAGINATION_KW):
        return "imagination"

    # 6. affection — duygu
    if any(kw in prompt_lower for kw in _AFFECTION_KW):
        return "affection"

    # 7. default
    return "entity"
