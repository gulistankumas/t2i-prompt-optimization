"""Adim 4/4: Ekonomik mod validation.

Aynı 5 prompt + ekonomik router modelleri (FLUX-schnell, SD1.5, SD3.5-Large).
Baseline (FLUX.1-dev) zaten premium turunda uretildi → reuse edilir.

Adımlar:
  1) Ekonomik router modellerini belirle (track → model)
  2) 5 ekonomik router gorseli uret (rate-limit aware, 12s delay)
  3) GPT-4.1 ile skorla
  4) Premium ile karsilastir

Cikti: data/validation/vision_scores_economic.csv
"""
import base64
import json
import sys
import time
from pathlib import Path

import importlib.util
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "validation"
IMG_DIR = VAL_DIR / "images"

# 24'teki generate_one fonksiyonunu kullan
spec = importlib.util.spec_from_file_location(
    "_gen", Path(__file__).parent / "24_generate_images.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

# Ekonomik lookup
with open(ROOT / "data" / "processed" / "lookup_table_2mode.json", "r", encoding="utf-8") as f:
    lookup = json.load(f)
ECO_MODELS = {t: lookup["economic"][t]["model"] for t in lookup["economic"]}
ECO_COSTS = {t: lookup["economic"][t]["cost"] for t in lookup["economic"]}

print("Ekonomik mod modelleri:")
for t, m in ECO_MODELS.items():
    print(f"  {t:18s} -> {m}")


# %% prompt'lari yukle, ekonomik router'i ekle
df = pd.read_csv(VAL_DIR / "selected_5_prompts.csv").copy()
df["eco_router_model"] = df["track"].map(ECO_MODELS)
df["eco_router_cost"] = df["track"].map(ECO_COSTS)
df["eco_router_api"] = df["eco_router_model"].apply(
    lambda m: "replicate" if m in ("SD1.5", "SDXL", "SD3.5-Large", "FLUX.1-schnell", "FLUX.1-dev") else "openai"
)

print("\nEkonomik 5 prompt:")
for _, r in df.iterrows():
    print(f"  [{r['validation_id']}] {r['track']:14s} -> {r['eco_router_model']:14s} "
          f"(${r['eco_router_cost']:.4f}, {r['eco_router_api']})")
print(f"Toplam ekonomik router maliyeti: ${df['eco_router_cost'].sum():.3f}")


# %% gorselleri uret (rate limit aware, 12s delay)
DELAY = 12.0
gen_log = []
for i, row in df.iterrows():
    vid = row["validation_id"]
    model = row["eco_router_model"]
    api = row["eco_router_api"]
    prompt = row["prompt"]
    out_path = IMG_DIR / f"{vid}_eco_router_{model.replace('.', '').replace('/', '_')}.png"

    if out_path.exists():
        size_kb = out_path.stat().st_size / 1024
        print(f"\n[{vid}] {model:14s} - mevcut, atlandi ({size_kb:.0f} KB)")
        gen_log.append({
            "validation_id": vid, "model": model, "api": api,
            "track": row["track"], "image_path": str(out_path.relative_to(ROOT)),
            "elapsed_s": 0, "cost_usd": row["eco_router_cost"], "status": "OK_cached",
        })
        continue

    print(f"\n[{vid}] {model:14s} - uretiliyor...")
    for attempt in range(3):
        try:
            elapsed = gen.generate_one(model, api, prompt, out_path)
            size_kb = out_path.stat().st_size / 1024
            print(f"  OK ({elapsed:.1f}s, {size_kb:.0f} KB)")
            gen_log.append({
                "validation_id": vid, "model": model, "api": api,
                "track": row["track"], "image_path": str(out_path.relative_to(ROOT)),
                "elapsed_s": round(elapsed, 1), "cost_usd": row["eco_router_cost"],
                "status": "OK",
            })
            break
        except Exception as e:
            err = str(e)[:120]
            print(f"  deneme {attempt + 1} fail: {err}")
            if "429" in err or "throttl" in err.lower():
                wait = 15 + attempt * 5
                print(f"  rate limit; {wait}s bekleniyor...")
                time.sleep(wait)
            else:
                gen_log.append({
                    "validation_id": vid, "model": model, "api": api,
                    "track": row["track"], "image_path": "", "elapsed_s": 0,
                    "cost_usd": row["eco_router_cost"], "status": f"ERROR: {err}",
                })
                break
    if api == "replicate":
        print(f"  bir sonraki cagri icin {DELAY}s bekleniyor...")
        time.sleep(DELAY)

gen_df = pd.DataFrame(gen_log)
gen_df.to_csv(VAL_DIR / "generation_log_economic.csv", index=False)


# %% GPT-4.1 ile skorla
client = OpenAI()
JUDGE_PROMPT = """You are an expert T2I image evaluator following PRISM-Bench methodology.

Evaluate this image for the prompt:
"{prompt}"

Score on two dimensions (each 0-100):
1. ALIGNMENT: How well does the image match what the prompt describes?
   (objects, attributes, scene, composition, text)
2. AESTHETIC: Visual quality, composition, detail, polish.

Output ONLY two integers separated by a comma, NO explanation:
ALIGNMENT,AESTHETIC

Example: 85,72"""


def score_one(prompt: str, image_path: Path):
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4.1",
        max_tokens=15,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": JUDGE_PROMPT.format(prompt=prompt)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    raw = response.choices[0].message.content.strip()
    try:
        ali, aes = [int(x.strip()) for x in raw.split(",")[:2]]
        return ali, aes
    except Exception:
        return None, None


print("\n" + "=" * 70)
print("GPT-4.1 ile skorlama (5 ekonomik router gorseli)")
print("=" * 70)
score_rows = []
for _, row in df.iterrows():
    vid = row["validation_id"]
    img_path = IMG_DIR / f"{vid}_eco_router_{row['eco_router_model'].replace('.', '').replace('/', '_')}.png"
    if not img_path.exists():
        print(f"[{vid}] EKSIK gorsel, skipping")
        continue
    print(f"\n[{vid}] {row['eco_router_model']:14s}")
    t0 = time.time()
    ali, aes = score_one(row["prompt"], img_path)
    elapsed = time.time() - t0
    avg = (ali + aes) / 2 if ali is not None else None
    avg_str = f"{avg:.1f}" if avg is not None else "N/A"
    print(f"  alignment={ali}, aesthetic={aes}, avg={avg_str} ({elapsed:.1f}s)")
    score_rows.append({
        "validation_id": vid, "role": "eco_router", "model": row["eco_router_model"],
        "track": row["track"], "alignment": ali, "aesthetic": aes, "avg_score": avg,
    })
    time.sleep(0.5)

eco_scores = pd.DataFrame(score_rows)
eco_scores.to_csv(VAL_DIR / "vision_scores_economic.csv", index=False)


# %% Premium ile karsilastirma + baseline
prem = pd.read_csv(VAL_DIR / "vision_scores.csv")
print("\n" + "=" * 70)
print("EKONOMIK vs PREMIUM vs BASELINE (validation_id bazli)")
print("=" * 70)
print(f"{'ID':<8s} {'Track':<14s} {'Eco':<28s} {'Prem':<28s} {'Base':<14s}")
print(f"{'':<8s} {'':<14s} {'model':<14s} {'avg':>6s} | {'model':<14s} {'avg':>6s} | {'avg':>6s}")
print("-" * 100)

for vid in eco_scores["validation_id"].unique():
    e = eco_scores[eco_scores["validation_id"] == vid].iloc[0]
    p_router = prem[(prem["validation_id"] == vid) & (prem["role"] == "router")].iloc[0]
    p_base = prem[(prem["validation_id"] == vid) & (prem["role"] == "baseline")].iloc[0]
    print(f"{vid:<8s} {e['track']:<14s} "
          f"{e['model']:<14s} {e['avg_score']:>6.1f} | "
          f"{p_router['model']:<14s} {p_router['avg_score']:>6.1f} | "
          f"{p_base['avg_score']:>6.1f}")

# Ortalamalar
eco_avg = eco_scores["avg_score"].mean()
prem_avg = prem[prem["role"] == "router"]["avg_score"].mean()
base_avg = prem[prem["role"] == "baseline"]["avg_score"].mean()
print()
print(f"{'TOPLAM':<8s} {'(5)':<14s} "
      f"{'eco-routed':<14s} {eco_avg:>6.2f} | "
      f"{'premium-rt':<14s} {prem_avg:>6.2f} | "
      f"{base_avg:>6.2f}")

# Maliyet
eco_cost = df["eco_router_cost"].sum()
prem_cost = df["router_cost"].sum()
base_cost = df["baseline_cost"].sum()
print(f"\n{'MALIYET':<8s} {'($/5img)':<14s} "
      f"{'eco':<14s} {eco_cost:>6.3f} | "
      f"{'premium':<14s} {prem_cost:>6.3f} | "
      f"{base_cost:>6.3f}")

# Q/$ ratio
print(f"\n{'Q/$':<8s} {'oranlari':<14s} "
      f"{'eco':<14s} {eco_avg / max(eco_cost, 0.001):>6.0f} | "
      f"{'premium':<14s} {prem_avg / max(prem_cost, 0.001):>6.0f} | "
      f"{base_avg / max(base_cost, 0.001):>6.0f}")

print(f"\n>> {VAL_DIR / 'vision_scores_economic.csv'}")
