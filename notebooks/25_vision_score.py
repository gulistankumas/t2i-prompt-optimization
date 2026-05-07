"""Adim 3/4: GPT-4.1 vision ile 10 gorseli skorla.

PRISM jurileri ile uyumlu: alignment + aesthetic 0-100. Her gorsel icin tek
API call. Cikti: data/validation/vision_scores.csv ve router-vs-baseline rapor.
"""
import base64
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "validation"

log_df = pd.read_csv(VAL_DIR / "generation_log.csv")
log_df = log_df[log_df["status"] == "OK"].reset_index(drop=True)
print(f"{len(log_df)} gorsel skorlanacak.")

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


def encode_image(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def score_one(prompt: str, image_path: Path, model: str = "gpt-4.1") -> tuple:
    b64 = encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        max_tokens=15,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": JUDGE_PROMPT.format(prompt=prompt)},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    raw = response.choices[0].message.content.strip()
    try:
        ali_str, aes_str = raw.split(",")[:2]
        return int(ali_str.strip()), int(aes_str.strip()), raw
    except Exception:
        return None, None, raw


# %% skorlamayi calistir
results = []
for i, row in log_df.iterrows():
    img_path = ROOT / row["image_path"]
    if not img_path.exists():
        print(f"  EKSIK: {img_path}"); continue

    print(f"\n[{row['validation_id']}] {row['role']:8s} {row['model']:18s}")
    t0 = time.time()
    try:
        ali, aes, raw = score_one(row["prompt"], img_path)
    except Exception as e:
        print(f"  HATA: {str(e)[:120]}")
        ali, aes, raw = None, None, str(e)[:50]
    elapsed = time.time() - t0
    avg = (ali + aes) / 2 if ali is not None and aes is not None else None
    avg_str = f"{avg:.1f}" if avg is not None else "N/A"
    print(f"  alignment={ali}, aesthetic={aes}, avg={avg_str} ({elapsed:.1f}s)")
    results.append({
        "validation_id": row["validation_id"],
        "role": row["role"], "model": row["model"], "track": row["track"],
        "image_path": row["image_path"],
        "alignment": ali, "aesthetic": aes, "avg_score": avg,
        "raw": raw, "elapsed_s": round(elapsed, 1),
    })
    time.sleep(0.5)

scores_df = pd.DataFrame(results)
scores_df.to_csv(VAL_DIR / "vision_scores.csv", index=False)
print(f"\n>> {VAL_DIR / 'vision_scores.csv'}")


# %% router vs baseline karşılaştırma
print("\n" + "=" * 70)
print("ROUTER vs BASELINE (validation_id bazli)")
print("=" * 70)
print(f"{'ID':<8s} {'Track':<14s} {'Router':<22s} {'Base':<14s} {'R Avg':>7s} {'B Avg':>7s} {'Diff':>7s}")
diffs = []
for vid in scores_df["validation_id"].unique():
    sub = scores_df[scores_df["validation_id"] == vid]
    r = sub[sub["role"] == "router"].iloc[0]
    b = sub[sub["role"] == "baseline"].iloc[0]
    if r["avg_score"] is None or b["avg_score"] is None:
        continue
    diff = r["avg_score"] - b["avg_score"]
    diffs.append(diff)
    print(f"{vid:<8s} {r['track']:<14s} {r['model']:<22s} {b['model']:<14s} "
          f"{r['avg_score']:>7.1f} {b['avg_score']:>7.1f} {diff:>+7.1f}")

# Toplam
router_avg = scores_df[scores_df["role"] == "router"]["avg_score"].mean()
base_avg = scores_df[scores_df["role"] == "baseline"]["avg_score"].mean()
print()
print(f"{'TOPLAM':<8s} {'all':<14s} {'(routed)':<22s} {'FLUX.1-dev':<14s} "
      f"{router_avg:>7.2f} {base_avg:>7.2f} {router_avg - base_avg:>+7.2f}")

print(f"\nRouter > Baseline: {sum(d > 0 for d in diffs)}/5 vakada")
print(f"Router < Baseline: {sum(d < 0 for d in diffs)}/5 vakada")
print(f"Beraberlik       : {sum(d == 0 for d in diffs)}/5 vakada")
