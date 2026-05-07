"""Gün 11 — LLM kategorizer 140 test prompt'unda + 3-yollu karşılaştırma."""
# %% imports
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.router.track_classifier import classify_track_rules
from src.router.llm_categorizer import LLMCategorizer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

TRACKS = ["affection", "composition", "entity", "imagination",
          "long_text", "style", "text_rendering"]


# %% load
df_test = pd.read_csv(PROC / "test_prompts.csv")
print(f"Test seti: {len(df_test)} prompt")

cache = np.load(PROC / "sbert_embeddings.npz")
X_test = cache["X_test"]
le = joblib.load(ROOT / "models" / "label_encoder.pkl")
lr = joblib.load(ROOT / "models" / "track_classifier_lr.pkl")
y_test = le.transform(df_test["track"])


# %% LLM tahminleri
print("\n=== LLM kategorizer (GPT-4o-mini, few-shot) ===")
print(f"140 prompt, delay 0.1s, tahmini sure: ~80sn, maliyet: ~$0.003")
t0 = time.time()
cat = LLMCategorizer()
df_test["llm_pred"] = cat.classify_batch(df_test["prompt"].tolist(), delay=0.1)
print(f"Tamamlandi ({time.time() - t0:.1f}s)")


# %% diğer modellerin tahminleri
df_test["rule_pred"] = df_test["prompt"].apply(classify_track_rules)
df_test["lr_pred"] = le.inverse_transform(lr.predict(X_test))


# %% accuracies
acc_rule = accuracy_score(df_test["track"], df_test["rule_pred"])
acc_lr = accuracy_score(df_test["track"], df_test["lr_pred"])
acc_llm = accuracy_score(df_test["track"], df_test["llm_pred"])

print("\n=== ACCURACY OZETI ===")
print(f"Rule v2:        {acc_rule * 100:.1f}%")
print(f"LR + SBERT:     {acc_lr * 100:.1f}%")
print(f"LLM (4o-mini):  {acc_llm * 100:.1f}%")


# %% classification report (LLM)
print("\n=== LLM SINIF BAZINDA ===")
print(classification_report(df_test["track"], df_test["llm_pred"], digits=3))


# %% per-track accuracy karşılaştırması
def per_track_acc(true_col, pred_col):
    out = {}
    for t in TRACKS:
        mask = df_test[true_col] == t
        out[t] = (df_test[mask][pred_col] == t).mean() * 100 if mask.sum() else 0
    return out

rule_pt = per_track_acc("track", "rule_pred")
lr_pt = per_track_acc("track", "lr_pred")
llm_pt = per_track_acc("track", "llm_pred")

print("\n=== TRACK BAZINDA ACCURACY (test 140) ===")
comp = pd.DataFrame({"rule_v2": rule_pt, "lr_sbert": lr_pt, "llm_4omini": llm_pt}).round(1)
print(comp.to_string())


# %% predictions kaydet
df_test.to_csv(PROC / "test_predictions_3way.csv", index=False)
print(f"\n>> {PROC / 'test_predictions_3way.csv'}")


# %% 3-yollu karşılaştırma görseli
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(TRACKS))
width = 0.27
ax.bar(x - width, [rule_pt[t] for t in TRACKS], width,
       label=f"Rule-based v2 ({acc_rule * 100:.1f}%)", color="lightcoral")
ax.bar(x, [lr_pt[t] for t in TRACKS], width,
       label=f"LR + SBERT ({acc_lr * 100:.1f}%)", color="steelblue")
ax.bar(x + width, [llm_pt[t] for t in TRACKS], width,
       label=f"LLM 4o-mini ({acc_llm * 100:.1f}%)", color="mediumseagreen")
ax.set_xlabel("PRISM Track")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Kategorizer Karsilastirmasi: Track Bazinda Dogruluk (test 140)",
             fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(TRACKS, rotation=20)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(FIG / "categorizer_3way_comparison.png", bbox_inches="tight", dpi=200)
plt.close()
print(f">> {FIG / 'categorizer_3way_comparison.png'}")


# %% özet tablosu
summary = pd.DataFrame({
    "Kategorizer": ["Kural v2", "LR + SBERT", "LLM (GPT-4o-mini)"],
    "Test Accuracy": [f"{acc_rule * 100:.1f}%", f"{acc_lr * 100:.1f}%", f"{acc_llm * 100:.1f}%"],
    "Egitim": ["-", "4.4s", "0 (zero/few-shot)"],
    "Inference (per prompt)": ["<1ms", "<5ms", "~500ms"],
    "Maliyet (1000 prompt)": ["$0", "$0", "~$0.02 (4o-mini)"],
    "Model boyutu": ["-", "21KB", "API"],
})
print("\n=== KATEGORIZER OZET TABLO ===")
print(summary.to_string(index=False))
