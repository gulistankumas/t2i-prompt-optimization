"""LLM-based T2I router (few-shot prompting).

Pipeline: prompt + mod -> LLM few-shot -> model adı.
Lookup table'ı LLM'e few-shot örneklerle öğretir.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.router.base_router import BaseRouter, RouterDecision

load_dotenv()

_ROOT = Path(__file__).resolve().parents[2]
_PROC = _ROOT / "data" / "processed"

_TRACKS = ("imagination", "entity", "text_rendering", "style",
           "affection", "composition", "long_text")

VALID_MODELS = (
    "SD1.5", "FLUX.1-schnell", "SDXL", "FLUX.1-dev",
    "SD3.5-Large", "Gemini2.5-Flash-Image", "Qwen-Image", "GPT-Image-1",
)

LLM_ROUTER_SYSTEM = """You are a T2I model selection expert.
Given a prompt and a quality mode, select the best model from this list:
- SD1.5 (cheapest, $0.0023)
- FLUX.1-schnell ($0.0030)
- SDXL ($0.0046)
- FLUX.1-dev ($0.0300)
- SD3.5-Large ($0.0350)
- Gemini2.5-Flash-Image ($0.0390)
- Qwen-Image ($0.0400)
- GPT-Image-1 ($0.1670)

Selection rules (based on PRISM-Bench analysis):
- "economic" mode: balance quality 60+ with lowest cost
- "premium" mode: balance quality 75+ with lowest cost

Track-aware preferences (most common selections):
- imagination + economic -> FLUX.1-dev | + premium -> Gemini2.5-Flash-Image
- entity + economic -> SDXL | + premium -> Gemini2.5-Flash-Image
- text_rendering + economic -> SD3.5-Large | + premium -> GPT-Image-1
- style + economic -> FLUX.1-schnell | + premium -> SD3.5-Large
- affection + economic -> SD1.5 | + premium -> SDXL
- composition + economic -> FLUX.1-schnell | + premium -> FLUX.1-schnell
- long_text + economic -> FLUX.1-schnell | + premium -> Gemini2.5-Flash-Image

Output format: ONLY the model name (exact match from the list above), no explanation."""

LLM_ROUTER_FEWSHOT = """Examples:

Prompt: "a futuristic city with the word 'TOKYO' in neon"
Mode: economic
Model: SD3.5-Large

Prompt: "a futuristic city with the word 'TOKYO' in neon"
Mode: premium
Model: GPT-Image-1

Prompt: "a happy golden retriever in a park"
Mode: economic
Model: SD1.5

Prompt: "a happy golden retriever in a park"
Mode: premium
Model: SDXL

Prompt: "a Picasso-style cubist portrait"
Mode: economic
Model: FLUX.1-schnell

Prompt: "a Picasso-style cubist portrait"
Mode: premium
Model: SD3.5-Large

Prompt: "a cat sitting next to a vase"
Mode: economic
Model: FLUX.1-schnell

Prompt: "two boxes stacked on top of each other"
Mode: premium
Model: FLUX.1-schnell
"""


class LLMRouter(BaseRouter):
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 api_key: str | None = None,
                 master_path: Path = _PROC / "master_final.csv"):
        super().__init__(name="LLMRouter")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.master = pd.read_csv(master_path)

    def _parse(self, raw: str) -> str:
        ans = raw.strip()
        if ans in VALID_MODELS:
            return ans
        for m in VALID_MODELS:
            if m in ans:
                return m
        return "FLUX.1-schnell"  # safe fallback

    def _quality_cost(self, model_name: str) -> tuple[float, float]:
        row = self.master[self.master["model"] == model_name].iloc[0]
        avg = float(row[[f"{t}_avg" for t in _TRACKS]].mean())
        return avg, float(row["cost_usd"])

    def route(self, prompt: str, mode: str = "economic", retry: int = 3) -> RouterDecision:
        user_msg = f'{LLM_ROUTER_FEWSHOT}\n\nPrompt: "{prompt}"\nMode: {mode}\nModel:'
        for attempt in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=20,
                    messages=[
                        {"role": "system", "content": LLM_ROUTER_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = response.choices[0].message.content
                selected = self._parse(raw)
                q, c = self._quality_cost(selected)
                return RouterDecision(
                    model=selected, expected_quality=q, expected_cost=c,
                    predicted_track="N/A", mode=mode, fallback=False,
                )
            except Exception as e:
                if attempt == retry - 1:
                    print(f"  [LLMRouter error]: {e}")
                    q, c = self._quality_cost("FLUX.1-schnell")
                    return RouterDecision(
                        model="FLUX.1-schnell", expected_quality=q, expected_cost=c,
                        predicted_track="N/A", mode=mode, fallback=True,
                    )
                time.sleep(2 ** attempt)

    def route_batch(self, prompts: Iterable[str], mode: str = "economic",
                    delay: float = 0.1, progress: bool = True) -> list[RouterDecision]:
        results = []
        prompts = list(prompts)
        for i, p in enumerate(prompts):
            if progress and (i + 1) % 20 == 0:
                print(f"  [{i + 1}/{len(prompts)}]", flush=True)
            results.append(self.route(p, mode))
            time.sleep(delay)
        return results


if __name__ == "__main__":
    router = LLMRouter()
    cases = [
        ('a futuristic city with the word TOKYO in neon', "economic"),
        ('a futuristic city with the word TOKYO in neon', "premium"),
        ('a melancholic woman looking out at the rain', "economic"),
        ('a Picasso-style cubist portrait', "premium"),
    ]
    print("=== LLMRouter Self-Test ===\n")
    for prompt, mode in cases:
        d = router.route(prompt, mode)
        print(f"[{mode:8s}] {prompt[:55]}")
        print(f"           -> {d}\n")
