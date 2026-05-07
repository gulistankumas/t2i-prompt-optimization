"""LLM-based PRISM track categorizer.

Few-shot prompting with GPT-4o-mini (default). System prompt + 7 examples
for zero-shot-like consistent predictions.

Usage:
    from src.router.llm_categorizer import LLMCategorizer
    cat = LLMCategorizer()
    track = cat.classify("a happy dog")
    tracks = cat.classify_batch(["...", "..."], delay=0.1)
"""
from __future__ import annotations

import os
import time
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VALID_TRACKS = (
    "imagination", "entity", "text_rendering", "style",
    "affection", "composition", "long_text",
)

SYSTEM_PROMPT = """You are an expert T2I prompt classifier.
Assign the given prompt to the MOST dominant of PRISM-Bench's 7 tracks.

Track definitions:
- imagination: Creative, abstract, impossible, or surreal scenes
- entity: Rendering of a defined object/being
- text_rendering: Readable text/letters within the image
- style: Imitation of a specific art style (artist, genre, era)
- affection: Conveying emotion, expression, or atmosphere
- composition: Spatial relationships, multi-object arrangement
- long_text: Long, multi-conditional, detailed instructions (typically 30+ words, multi-step)

Decision rules (priority order):
1. If the prompt explicitly contains text/letters (e.g., 'TOKYO' in quotes) -> text_rendering
2. If the prompt names a specific artist/style -> style
3. If the prompt is 30+ words with multi-conditional details -> long_text
4. If the prompt specifies spatial relationships (X to the right of Y, on top of) -> composition
5. If the prompt is a creative/impossible combination -> imagination
6. If the prompt emphasizes emotion/atmosphere -> affection
7. Otherwise, default: entity

Output format: ONLY the track name (single word, lowercase), NO explanation."""

FEW_SHOT_EXAMPLES = """Examples:
Prompt: "a futuristic city with the word 'TOKYO' in neon"
Track: text_rendering

Prompt: "a surreal painting of floating clocks melting over a tree, in the style of Salvador Dali"
Track: style

Prompt: "a happy golden retriever sitting in a sunlit park"
Track: affection

Prompt: "an elephant"
Track: entity

Prompt: "a red box to the right of a green sphere, both on a wooden table"
Track: composition

Prompt: "First, imagine a vast desert. Then add three pyramids in the distance. Finally, place a camel in the foreground walking towards the viewer with detailed sand textures."
Track: long_text

Prompt: "a city made of glass where rivers of light flow between crystalline towers"
Track: imagination
"""


class LLMCategorizer:
    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: str | None = None,
                 max_tokens: int = 15):
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _parse(self, raw: str) -> str:
        ans = raw.strip().lower()
        if ans in VALID_TRACKS:
            return ans
        # loose match: check if track name appears in answer
        for t in VALID_TRACKS:
            if t in ans:
                return t
        # fallback: default
        return "entity"

    def classify(self, prompt: str, retry: int = 3) -> str:
        user_msg = f"{FEW_SHOT_EXAMPLES}\n\nPrompt: \"{prompt}\"\nTrack:"
        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = response.choices[0].message.content
                return self._parse(raw)
            except Exception as e:
                if attempt == retry - 1:
                    print(f"  [LLM error, default='entity']: {e}")
                    return "entity"
                time.sleep(2 ** attempt)  # exponential backoff
        return "entity"

    def classify_batch(self, prompts: Iterable[str], delay: float = 0.1,
                       progress: bool = True) -> list[str]:
        results = []
        prompts = list(prompts)
        for i, p in enumerate(prompts):
            if progress and (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(prompts)}]", flush=True)
            results.append(self.classify(p))
            time.sleep(delay)
        return results


if __name__ == "__main__":
    cat = LLMCategorizer()
    test_samples = [
        ('a futuristic city with the word "TOKYO" in neon', "text_rendering"),
        ('a surreal painting of floating clocks', "imagination/style"),
        ('a happy golden retriever in a park', "affection/entity"),
        ('an elephant', "entity"),
        ('a cat sitting next to a vase', "composition"),
        ('a melancholic woman looking out at the rain', "affection"),
        ('a dragon made entirely of rainbow ice cream', "imagination"),
        ('two boxes stacked on top of each other', "composition"),
        ('first imagine a forest, then add three deer grazing peacefully', "long_text"),
        ('a Picasso-style cubist portrait', "style"),
    ]
    print("=== LLMCategorizer (10 sample sanity test) ===")
    correct = 0
    for prompt, expected in test_samples:
        pred = cat.classify(prompt)
        # loose correct: if expected has slash, any match is enough
        possible = [x.strip() for x in expected.split("/")]
        ok = pred in possible
        correct += ok
        mark = "OK" if ok else "??"
        print(f"  [{mark}] pred={pred:<16s} expected={expected:<25s} | {prompt[:55]}")
    print(f"\n{correct}/10 correct (loose match)")