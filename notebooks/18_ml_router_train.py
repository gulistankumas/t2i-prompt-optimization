"""Gün 13 — ML router eğitimi.

Pipeline: prompt SBERT embedding + mod one-hot -> 8-class model classifier.
Eğitim verisi: 560 train prompt × 2 mod = 1120 örnek.
Label: lookup table'dan gelen optimal model (track + mod biliniyor).

Çıktı: models/ml_router_lr.pkl, models/ml_router_label_encoder.pkl
"""
# %% imports
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


# ============================================================
# 1. Veri yükle
# ============================================================
df_train = pd.read_csv(PROC / "train_prompts.csv").reset_index(drop=True)
df_test = pd.read_csv(PROC / "test_prompts.csv").reset_index(drop=True)
with open(PROC / "lookup_table_2mode.json", "r", encoding="utf-8") as f:
    lookup = json.load(f)

cache = np.load(PROC / "sbert_embeddings.npz")
X_train_unique = cache["X_train"]   # (560, 384)
X_test_unique = cache["X_test"]      # (140, 384)


# ============================================================
# 2. (prompt, mod) çiftlerini oluştur ve label ata
# ============================================================
def build_router_data(df, X_unique):
    rows, X_emb, mode_oh = [], [], []
    for i, row in df.iterrows():
        track = row["track"]
        for mode_idx, mode in enumerate(["economic", "premium"]):
            rows.append({
                "prompt_idx": i, "mode": mode, "track": track,
                "optimal_model": lookup[mode][track]["model"],
            })
            X_emb.append(X_unique[i])
            mode_oh.append([1, 0] if mode == "economic" else [0, 1])
    df_r = pd.DataFrame(rows)
    X = np.hstack([np.array(X_emb), np.array(mode_oh)])
    return df_r, X


df_tr_r, X_train = build_router_data(df_train, X_train_unique)
df_te_r, X_test = build_router_data(df_test, X_test_unique)

print(f"Train router: {X_train.shape}, Test router: {X_test.shape}")
print(f"Optimal model dağılımı (train):\n{df_tr_r['optimal_model'].value_counts().to_string()}")


# ============================================================
# 3. Logistic Regression eğit
# ============================================================
le = LabelEncoder()
y_train = le.fit_transform(df_tr_r["optimal_model"])
y_test = le.transform(df_te_r["optimal_model"])

print("\n=== ML ROUTER (LogReg) ===")
t0 = time.time()
clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print(f"Eğitim: {time.time() - t0:.1f}s")

acc_train = accuracy_score(y_train, clf.predict(X_train))
acc_test = accuracy_score(y_test, clf.predict(X_test))
print(f"Train: {acc_train * 100:.1f}%   Test: {acc_test * 100:.1f}%   Gap: {(acc_train - acc_test) * 100:.1f}")
print(f"Sınıflar (model adları): {list(le.classes_)}")


# ============================================================
# 4. Kaydet
# ============================================================
joblib.dump(clf, MODELS / "ml_router_lr.pkl")
joblib.dump(le, MODELS / "ml_router_label_encoder.pkl")
print(f"\n>> {MODELS / 'ml_router_lr.pkl'}")
print(f">> {MODELS / 'ml_router_label_encoder.pkl'}")
