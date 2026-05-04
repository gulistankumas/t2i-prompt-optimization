"""Kural tabanlı PRISM track sınıflandırıcı.

v2: PRISM prompt'ları uzun yazıldığı için v1'in 25-kelime long_text eşiği
diğer track sinyallerini boğuyordu. v2 kuralları:
- text_rendering: tırnak (çift veya tek-büyük), 'the word', 'reads', 'written'
- style: sanatçı/medium isimleri genişletildi
- composition: uzamsal preposition'lar
- imagination: yaratıcı/imkansız anahtar kelimeler genişletildi
- affection: duygu/atmosfer kelimeleri genişletildi
- long_text: SADECE 50+ kelime VE adım-adım talimat (first/then/step) işaret kelimesi
- entity: default

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
    "style of", "painting", "monet", "van gogh", "picasso",
    "cyberpunk", "anime", "oil painting", "watercolor",
    "sketch", "illustration", "abstract art", "baroque",
    "impressionist", "photographic",
]
_COMPOSITION_KW = [
    "next to", "on top of", "beside", "behind",
    "in front of", "between", "under", "above",
    "to the left", "to the right", "arranged",
]
_IMAGINATION_KW = [
    "surreal", "fantasy", "dream", "impossible",
    "magical", "mythical", "ethereal", "made of",
    "floating", "mythological", "unreal",
]
_AFFECTION_KW = [
    "happy", "sad", "joyful", "lonely", "melancholic",
    "cheerful", "sorrowful", "peaceful", "angry",
    "serene", "gloomy", "nostalgic", "tender",
    "atmosphere", "mood", "feeling",
]
_LONG_TEXT_MARKERS = ("first", "then", "finally", "step")


def classify_track_rules(prompt: str) -> str:
    """v2 sıralama: text_rendering -> style -> composition -> imagination
    -> affection -> long_text (50+ words AND step markers) -> entity (default)."""
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    # 1. text_rendering — tırnak veya açık metin işareti
    if re.search(r'"[^"]+"', prompt):
        return "text_rendering"
    if re.search(r"'[A-Z][^']{2,}'", prompt):  # 'TOKYO' gibi caps-lock
        return "text_rendering"
    if "the word" in prompt_lower or "reads" in prompt_lower or "written" in prompt_lower:
        return "text_rendering"

    # 2. style
    if any(kw in prompt_lower for kw in _STYLE_KW):
        return "style"

    # 3. composition
    if any(kw in prompt_lower for kw in _COMPOSITION_KW):
        return "composition"

    # 4. imagination
    if any(kw in prompt_lower for kw in _IMAGINATION_KW):
        return "imagination"

    # 5. affection
    if any(kw in prompt_lower for kw in _AFFECTION_KW):
        return "affection"

    # 6. long_text — SADECE çok-koşullu uzun prompt
    if word_count > 50 and any(m in prompt_lower for m in _LONG_TEXT_MARKERS):
        return "long_text"

    # 7. default
    return "entity"
