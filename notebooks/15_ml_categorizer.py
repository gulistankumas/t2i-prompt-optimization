"""Gün 10 — ML tabanlı PRISM track kategorizer.

Pipeline: SBERT embedding -> (LogReg | XGBoost) classifier
Veri: data/processed/{train,test}_prompts.csv (560/140, stratified)
Çıktılar:
  models/track_classifier_xgb.pkl
  models/track_classifier_lr.pkl
  models/label_encoder.pkl
  models/sbert_config.json
  figures/categorizer_comparison.png
"""
# %% imports
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from src.router.track_classifier import classify_track_rules

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIG = ROOT / "figures"
MODELS.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

TRACKS = ["affection", "composition", "entity", "imagination",
          "long_text", "style", "text_rendering"]


# ============================================================
# 1. Veri
# ============================================================
df_train = pd.read_csv(PROC / "train_prompts.csv")
df_test = pd.read_csv(PROC / "test_prompts.csv")
print(f"Train: {len(df_train)} | Test: {len(df_test)}")
print(f"Train track dağılımı:\n{df_train['track'].value_counts().sort_index()}")


# ============================================================
# 2. SBERT embed (ilk çalıştırmada ~80MB indirir)
# ============================================================
SBERT_NAME = "all-MiniLM-L6-v2"
EMB_CACHE = PROC / "sbert_embeddings.npz"

if EMB_CACHE.exists():
    print("\nSBERT cache yüklendi.")
    cache = np.load(EMB_CACHE)
    X_train = cache["X_train"]
    X_test = cache["X_test"]
else:
    print(f"\nSBERT modeli yükleniyor: {SBERT_NAME}")
    t0 = time.time()
    sbert = SentenceTransformer(SBERT_NAME)
    print(f"  yüklendi ({time.time() - t0:.1f}s); dim={sbert.get_sentence_embedding_dimension()}")

    print("Train embed ediliyor...")
    t0 = time.time()
    X_train = sbert.encode(df_train["prompt"].tolist(),
                            convert_to_numpy=True, show_progress_bar=False)
    print(f"  {time.time() - t0:.1f}s, shape={X_train.shape}")

    print("Test embed ediliyor...")
    t0 = time.time()
    X_test = sbert.encode(df_test["prompt"].tolist(),
                           convert_to_numpy=True, show_progress_bar=False)
    print(f"  {time.time() - t0:.1f}s, shape={X_test.shape}")

    np.savez(EMB_CACHE, X_train=X_train, X_test=X_test)
    print(f"  cache: {EMB_CACHE}")


# ============================================================
# 3. Label encode
# ============================================================
le = LabelEncoder()
le.fit(TRACKS)
y_train = le.transform(df_train["track"])
y_test = le.transform(df_test["track"])
print(f"\nLabel sınıfları: {list(le.classes_)}")


# ============================================================
# 4a. Logistic Regression (basit baseline)
# ============================================================
print("\n=== LOGISTIC REGRESSION ===")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
print(f"Eğitim: {time.time() - t0:.1f}s")
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"Test accuracy: {acc_lr * 100:.1f}%")


# ============================================================
# 4b. XGBoost
# ============================================================
print("\n=== XGBOOST ===")
t0 = time.time()
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective="multi:softmax", num_class=7,
    random_state=42, n_jobs=-1,
)
xgb_clf.fit(X_train, y_train)
print(f"Eğitim: {time.time() - t0:.1f}s")
y_pred_xgb = xgb_clf.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Test accuracy: {acc_xgb * 100:.1f}%")


# ============================================================
# 5. Rule-based v2 (test setinde)
# ============================================================
print("\n=== RULE V2 (test seti) ===")
rule_pred_strs = df_test["prompt"].apply(classify_track_rules).tolist()
y_pred_rule = le.transform(rule_pred_strs)
acc_rule = accuracy_score(y_test, y_pred_rule)
print(f"Test accuracy: {acc_rule * 100:.1f}%")


# ============================================================
# 6. Karşılaştırma
# ============================================================
print(f"\n=== KARŞILAŞTIRMA (test 140 prompt) ===")
print(f"Rule-based v2  : {acc_rule * 100:.1f}%")
print(f"LogReg + SBERT : {acc_lr * 100:.1f}%   (rule'a göre +{(acc_lr - acc_rule) * 100:.1f} puan)")
print(f"XGBoost + SBERT: {acc_xgb * 100:.1f}%   (rule'a göre +{(acc_xgb - acc_rule) * 100:.1f} puan)")


# ============================================================
# 7. Per-class report (XGBoost — best/main)
# ============================================================
print(f"\n=== XGBOOST — sınıf bazında report ===")
print(classification_report(y_test, y_pred_xgb,
                             target_names=le.classes_, digits=3))


# ============================================================
# 8. Confusion matrix figure (3 panel: rule, LR, XGB)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
for ax, (preds, name, acc, cmap) in zip(axes, [
    (y_pred_rule, "Rule-based v2", acc_rule, "Blues"),
    (y_pred_lr, "LogReg + SBERT", acc_lr, "Oranges"),
    (y_pred_xgb, "XGBoost + SBERT", acc_xgb, "Greens"),
]):
    cm = confusion_matrix(y_test, preds, labels=range(len(le.classes_)))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap=cmap, ax=ax, cbar=False)
    ax.set_title(f"{name}\nAcc: {acc * 100:.1f}%", fontweight="bold")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(FIG / "categorizer_comparison.png", bbox_inches="tight", dpi=200)
plt.close()
print(f"\n>> {FIG / 'categorizer_comparison.png'}")


# ============================================================
# 9. Modelleri kaydet
# ============================================================
joblib.dump(xgb_clf, MODELS / "track_classifier_xgb.pkl")
joblib.dump(lr, MODELS / "track_classifier_lr.pkl")
joblib.dump(le, MODELS / "label_encoder.pkl")
with open(MODELS / "sbert_config.json", "w", encoding="utf-8") as f:
    json.dump({"model_name": SBERT_NAME, "embedding_dim": int(X_train.shape[1])}, f, indent=2)
print(f">> {MODELS}")
print("\nKaydedilen dosyalar:")
for p in MODELS.iterdir():
    print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
