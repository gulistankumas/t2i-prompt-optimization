"""Gün 14 — OOD Generalization Testi.

Senaryo: 5 in-distribution track'te kategorizer eğit; entity + long_text track'lerini
hiç görme, sadece test'te değerlendir.

Seçenek B (entity + long_text): Spearman korelasyonu en düşük çift (0.644). En zor
ama metodolojik olarak savunulabilir OOD seçimi.

Akış:
  1. Train ve test setini in-dist (5 track) ve OOD (2 track) olarak böl
  2. OOD-LR kategorizer eğit (5 track, 400 train)
  3. In-dist test (100 prompt) ve OOD test (40 prompt) accuracy
  4. OOD prompt'lar için lookup router → yanlış track → seçilen model
  5. Oracle kalite (gerçek track × seçilen model)
  6. In-dist vs OOD karşılaştırma + figure
"""
# %% imports
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"

ALL_TRACKS = ["affection", "composition", "entity", "imagination",
              "long_text", "style", "text_rendering"]
OOD_TRACKS = ["entity", "long_text"]
IN_DIST_TRACKS = [t for t in ALL_TRACKS if t not in OOD_TRACKS]

print(f"In-distribution (5): {IN_DIST_TRACKS}")
print(f"OOD (2): {OOD_TRACKS}")


# ============================================================
# 1. Veri böl + cached embeddings'i yeniden indeksle
# ============================================================
df_train = pd.read_csv(PROC / "train_prompts.csv").reset_index(drop=True)
df_test = pd.read_csv(PROC / "test_prompts.csv").reset_index(drop=True)

cache = np.load(PROC / "sbert_embeddings.npz")
X_train_all = cache["X_train"]   # (560, 384)
X_test_all = cache["X_test"]      # (140, 384)

# In-dist train
mask_train_in = df_train["track"].isin(IN_DIST_TRACKS)
df_train_in = df_train[mask_train_in].reset_index(drop=True)
X_train_in = X_train_all[mask_train_in.values]

# Test (hem in-dist hem OOD)
mask_test_in = df_test["track"].isin(IN_DIST_TRACKS)
mask_test_ood = df_test["track"].isin(OOD_TRACKS)
df_test_in = df_test[mask_test_in].reset_index(drop=True)
df_test_ood = df_test[mask_test_ood].reset_index(drop=True)
X_test_in = X_test_all[mask_test_in.values]
X_test_ood = X_test_all[mask_test_ood.values]

print(f"\nTrain in-dist: {len(df_train_in)} (beklenen 400)")
print(f"Test in-dist: {len(df_test_in)} (beklenen 100)")
print(f"Test OOD: {len(df_test_ood)} (beklenen 40)")


# ============================================================
# 2. OOD-LR kategorizer eğit
# ============================================================
le_ood = LabelEncoder()
le_ood.fit(IN_DIST_TRACKS)
y_train_in = le_ood.transform(df_train_in["track"])
y_test_in = le_ood.transform(df_test_in["track"])

print("\n=== OOD-LR KATEGORIZER (5 in-dist track) ===")
lr_ood = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_ood.fit(X_train_in, y_train_in)

acc_train = accuracy_score(y_train_in, lr_ood.predict(X_train_in))
acc_in_test = accuracy_score(y_test_in, lr_ood.predict(X_test_in))
print(f"Train acc (5 track): {acc_train * 100:.1f}%")
print(f"In-dist test acc (5 track): {acc_in_test * 100:.1f}%")


# ============================================================
# 3. OOD prompt tahmin dağılımı
# ============================================================
ood_pred = lr_ood.predict(X_test_ood)
df_test_ood = df_test_ood.copy()
df_test_ood["predicted_track"] = le_ood.inverse_transform(ood_pred)

print("\n=== OOD TAHMİN DAĞILIMI ===")
for ood_track in OOD_TRACKS:
    sub = df_test_ood[df_test_ood["track"] == ood_track]
    print(f"\nGerçek '{ood_track}' (20 prompt) -> {dict(sub['predicted_track'].value_counts())}")


# ============================================================
# 4. OOD lookup routing + oracle kalite
# ============================================================
df_master = pd.read_csv(PROC / "master_final.csv")
with open(PROC / "lookup_table_2mode.json", "r", encoding="utf-8") as f:
    lookup = json.load(f)


def oracle_quality(track: str, model: str) -> float:
    row = df_master[df_master["model"] == model]
    if len(row) == 0:
        return np.nan
    val = row[f"{track}_avg"].iloc[0]
    return float(val) if pd.notna(val) else np.nan


ood_records = []
for _, row in df_test_ood.iterrows():
    actual = row["track"]
    predicted = row["predicted_track"]
    for mode in ["economic", "premium"]:
        entry = lookup[mode][predicted]  # yanlış (5 in-dist) track'in lookup hücresi
        rec_model = entry["model"]
        ood_records.append({
            "actual_track": actual,
            "predicted_track": predicted,
            "mode": mode,
            "recommended_model": rec_model,
            "actual_quality": oracle_quality(actual, rec_model),
            "cost": float(entry["cost"]),
        })
df_ood = pd.DataFrame(ood_records)


# ============================================================
# 5. In-dist vs OOD karşılaştırma
# ============================================================
df_lookup_eval = pd.read_csv(PROC / "three_router_eval.csv")
df_lookup_eval = df_lookup_eval[df_lookup_eval["router"] == "Lookup"]

print("\n\n=== IN-DIST vs OOD KALITE DÜŞÜŞÜ ===")
comparison = []
for mode in ["economic", "premium"]:
    df_in = df_lookup_eval[(df_lookup_eval["mode"] == mode) &
                            (df_lookup_eval["actual_track"].isin(IN_DIST_TRACKS))]
    in_q = df_in["actual_quality"].mean()
    in_c = df_in["expected_cost"].mean()

    df_o = df_ood[df_ood["mode"] == mode]
    ood_q = df_o["actual_quality"].mean()
    ood_c = df_o["cost"].mean()

    drop = in_q - ood_q
    comparison.append({
        "mode": mode, "in_quality": in_q, "ood_quality": ood_q, "drop": drop,
        "in_cost": in_c, "ood_cost": ood_c,
    })
    print(f"\n{mode.upper()}:")
    print(f"  In-dist kalite: {in_q:.2f} | OOD kalite: {ood_q:.2f} | Düşüş: {drop:.2f} puan")
    print(f"  In-dist maliyet: ${in_c:.4f} | OOD maliyet: ${ood_c:.4f}")


# Track bazında detay
print("\n=== OOD TRACK BAZINDA DETAY ===")
for track in OOD_TRACKS:
    print(f"\n{track}:")
    for mode in ["economic", "premium"]:
        sub = df_ood[(df_ood["actual_track"] == track) & (df_ood["mode"] == mode)]
        most_picked = sub["recommended_model"].value_counts().head(1)
        print(f"  {mode}: kalite={sub['actual_quality'].mean():.2f}, "
              f"maliyet=${sub['cost'].mean():.4f}, "
              f"en sık model={most_picked.index[0]} ({most_picked.iloc[0]}x)")


# ============================================================
# 6. Görsel
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, mode in enumerate(["economic", "premium"]):
    ax = axes[ax_idx]

    # in-dist track'lerin oracle kalitesi
    in_q = []
    for t in IN_DIST_TRACKS:
        sub = df_lookup_eval[(df_lookup_eval["mode"] == mode) &
                              (df_lookup_eval["actual_track"] == t)]
        in_q.append(sub["actual_quality"].mean())

    # ood track'lerin oracle kalitesi
    ood_q = []
    for t in OOD_TRACKS:
        sub = df_ood[(df_ood["mode"] == mode) & (df_ood["actual_track"] == t)]
        ood_q.append(sub["actual_quality"].mean())

    qualities = in_q + ood_q
    labels = IN_DIST_TRACKS + OOD_TRACKS
    colors = ["steelblue"] * len(IN_DIST_TRACKS) + ["crimson"] * len(OOD_TRACKS)

    bars = ax.bar(np.arange(len(labels)), qualities, color=colors,
                   edgecolor="black", linewidth=0.7)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Ortalama Oracle Kalite")
    threshold = 60 if mode == "economic" else 75
    ax.axhline(threshold, color="black", linestyle=":", alpha=0.5)
    ax.set_title(f"{mode.capitalize()} mod (esik {threshold})", fontweight="bold")
    for b, v in zip(bars, qualities):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", fontsize=9)

    ax.legend(handles=[
        Patch(facecolor="steelblue", label="In-Distribution (egitilmis)"),
        Patch(facecolor="crimson", label="OOD (gorulmemis)"),
    ], loc="lower right", fontsize=9)

plt.suptitle("OOD Generalization Test — Lookup Router (5 train track + 2 OOD)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = FIG / "ood_test_results.png"
plt.savefig(out, bbox_inches="tight", dpi=200)
plt.close()
print(f"\n>> {out}")

# kaydet
df_ood.to_csv(PROC / "ood_test_eval.csv", index=False)
print(f">> {PROC / 'ood_test_eval.csv'}")
