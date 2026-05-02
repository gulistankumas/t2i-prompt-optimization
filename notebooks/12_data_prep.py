"""Gün 8 — Veri hazırlığı: PRISM 700 prompt'u, train/test split, threshold map.

Çıktılar:
  data/raw/prism_prompts.csv         — 700 prompt × {prompt, track}
  data/processed/train_prompts.csv   — 560 prompt (80%, stratified by track)
  data/processed/test_prompts.csv    — 140 prompt (20%)
  data/processed/threshold_map.csv   — (track × mod) -> eligible model listesi

Not: 08 numarası önceki nb tarafından alındı, bu dosya 12_data_prep.
"""
# %% imports
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader.prism import PRISM_TRACKS

ROOT = Path(__file__).resolve().parents[1]
PRISM_CAP_DIR = ROOT / "external" / "prism-bench" / "captions" / "en"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"


# ============================================================
# 1. 700 prompt'u yükle
# ============================================================
records = []
for track in PRISM_TRACKS:
    jl = PRISM_CAP_DIR / f"{track}.jsonl"
    with open(jl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            records.append({
                "prompt_id": f"{track}_{i:03d}",
                "track": track,
                "prompt": d["prompt"],
            })

prompts = pd.DataFrame(records)
print(f"Toplam {len(prompts)} prompt yüklendi (beklenen 700)")
print(f"Track dağılımı:\n{prompts['track'].value_counts().sort_index()}")

prompts.to_csv(RAW_DIR / "prism_prompts.csv", index=False)
print(f"\n>> {RAW_DIR / 'prism_prompts.csv'}")


# ============================================================
# 2. Stratified train/test split (80/20)
# ============================================================
train, test = train_test_split(
    prompts,
    test_size=0.2,
    stratify=prompts["track"],
    random_state=42,
)
print(f"\nSplit: {len(train)} train + {len(test)} test")
print(f"Train track dağılımı:\n{train['track'].value_counts().sort_index()}")
print(f"Test track dağılımı:\n{test['track'].value_counts().sort_index()}")

train.sort_values("prompt_id").to_csv(PROC_DIR / "train_prompts.csv", index=False)
test.sort_values("prompt_id").to_csv(PROC_DIR / "test_prompts.csv", index=False)
print(f"\n>> {PROC_DIR / 'train_prompts.csv'}")
print(f">> {PROC_DIR / 'test_prompts.csv'}")


# ============================================================
# 3. Quality threshold haritası
# ============================================================
master = pd.read_csv(PROC_DIR / "master_final.csv")
master_active = master[master["model"] != "DALL-E 2"].copy()  # PRISM yok

MODES = {
    "low": 45,
    "mid": 65,
    "high": 80,
}

threshold_rows = []
for track in PRISM_TRACKS:
    track_col = f"{track}_avg"
    for mode_name, threshold in MODES.items():
        eligible = master_active[master_active[track_col] >= threshold].sort_values(
            "cost_usd"  # ucuza göre sırala router kolaylığı için
        )
        threshold_rows.append({
            "track": track,
            "mode": mode_name,
            "min_score_threshold": threshold,
            "n_eligible": len(eligible),
            "cheapest_eligible": eligible.iloc[0]["model"] if len(eligible) else None,
            "cheapest_eligible_cost": eligible.iloc[0]["cost_usd"] if len(eligible) else None,
            "best_quality_eligible": eligible.sort_values(track_col, ascending=False).iloc[0]["model"] if len(eligible) else None,
            "eligible_models": ",".join(eligible["model"].values) if len(eligible) else "",
        })

thr = pd.DataFrame(threshold_rows)
thr.to_csv(PROC_DIR / "threshold_map.csv", index=False)
print(f"\n>> {PROC_DIR / 'threshold_map.csv'}")

# %% rapor
print("\n=== Threshold Haritası (özet) ===")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)
print(thr[["track", "mode", "min_score_threshold", "n_eligible",
           "cheapest_eligible", "best_quality_eligible"]].to_string(index=False))


# ============================================================
# 4. Skor dağılımı kontrolü (track × segment)
# ============================================================
print("\n=== Track × Segment skor dağılımı (avg jüri) ===")
seg_summary = []
for track in PRISM_TRACKS:
    col = f"{track}_avg"
    for seg in ["Premium", "Mid", "Cheap"]:
        sub = master_active[master_active["segment"] == seg][col]
        seg_summary.append({
            "track": track,
            "segment": seg,
            "n": len(sub),
            "min": sub.min(),
            "median": sub.median(),
            "max": sub.max(),
        })
seg_df = pd.DataFrame(seg_summary)
print(seg_df.round(1).to_string(index=False))


# ============================================================
# 5. Sanity check
# ============================================================
print("\n=== KONTROL ===")
print(f"prism_prompts.csv satır: {len(prompts)} (beklenen 700)")
print(f"train_prompts.csv satır: {len(train)} (beklenen 560)")
print(f"test_prompts.csv satır:  {len(test)} (beklenen 140)")
print(f"threshold_map.csv satır: {len(thr)} (beklenen 21)")
assert len(prompts) == 700
assert len(train) == 560
assert len(test) == 140
assert len(thr) == 21
print("Tüm kontrolller geçti.")
