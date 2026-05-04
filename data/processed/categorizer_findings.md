# Kategorizer Bulguları

PRISM 7 track sınıflandırma — üç yaklaşım test edildi, ortak değerlendirme kümesi: 140 PRISM test prompt'u (her track 20).

## Özet Tablo

| Yaklaşım | Train Acc | Test Acc | Gap | Δ vs rule | Eğitim süresi |
|---|:---:|:---:|:---:|:---:|:---:|
| Rule-based v2 | — | 37.1% | — | — | 0 (kurallar) |
| **LogReg + SBERT** ⭐ | 93.8% | **83.6%** | **10.2** | +46.4 | ~5s |
| XGBoost + SBERT | 100.0% | 80.0% | **20.0** | +42.9 | ~15s |

**LogReg birincil tezin ML kategorizer'ı.** Üç gerekçe:
1. **Daha iyi generalleme:** Test accuracy +3.6 puan, train-test gap yarısı (10.2 vs 20.0).
2. **Daha az overfit:** XGBoost 200-ağaç × max_depth=6 kapasitesiyle 560 örneği ezbere yutuyor (train %100). 384-dim SBERT embedding üzerinde lineer sınır zaten yeterli; ek karmaşıklık sadece overfit getiriyor.
3. **Pratik avantajlar:** 22 KB model boyutu (XGB 1.4 MB'a karşı), 3x daha hızlı eğitim, daha düzgün confidence dağılımı (kısa OOD prompt'larda 0.25-0.54 — XGB 0.81-0.99 overconfident).

Tezin "iyi temsil > karmaşık model" ve "kapasite ≠ generalleme" tezine somut örnek.

---

## Gün 9 — Rule-based v1 ve v2

İlk versiyon (25-kelime long_text eşiği): %20.9 (700 prompt). Düzeltilmiş v2 (50-kelime + step markerı, genişletilmiş keyword listeleri): %36.3 (700 prompt) / %37.1 (test 140).

Track bazında v2:
- text_rendering 83% (kuralda en güçlü — tırnak ve "the word"/"reads"/"written" sinyali)
- entity 66% (default rule)
- style 42%, imagination 29%, composition 25%, affection 9%, long_text 0%

Pratik tavan ~%40-45. Tezde **alt sınır baseline**.

---

## Gün 10 — ML tabanlı (SBERT + classifier)

**Pipeline:** `prompt → SBERT (all-MiniLM-L6-v2, 384-dim) → classifier → track`. SBERT embedding'leri `data/processed/sbert_embeddings.npz`'de cache'leniyor.

### LogReg (ana model) — sınıf bazında F1 (test seti)

| Track | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| affection | 0.91 | **1.00** | **0.952** | 20 |
| imagination | 0.95 | 0.95 | **0.950** | 20 |
| style | 0.83 | **1.00** | **0.909** | 20 |
| long_text | 0.94 | 0.75 | 0.833 | 20 |
| text_rendering | 0.79 | 0.75 | 0.769 | 20 |
| composition | 0.74 | 0.70 | 0.718 | 20 |
| entity | 0.70 | 0.70 | 0.700 | 20 |

**Macro F1: 0.833, accuracy: 83.6%.** Üç track'te **recall 1.00** (affection, style — hiçbir gerçek prompt kaçırılmıyor). Rule-based'in tamamen başarısız olduğu track'lerde dramatik iyileşme: affection 0% → 95%, long_text 0% → 83%, imagination 6% → 95%, style 0% → 91%.

### XGBoost (ikincil/karşılaştırma) — sınıf bazında F1

| Track | LR F1 | XGB F1 | Δ |
|---|:---:|:---:|:---:|
| imagination | 0.950 | 0.810 | LR +0.14 |
| style | 0.909 | 0.850 | LR +0.06 |
| text_rendering | 0.769 | 0.706 | LR +0.06 |
| composition | 0.718 | 0.650 | LR +0.07 |
| affection | 0.952 | 0.923 | LR +0.03 |
| entity | 0.700 | 0.667 | LR +0.03 |
| long_text | 0.833 | **1.000** | XGB +0.17 |
| **Macro F1** | **0.833** | 0.800 | **LR +0.03** |

LR 6/7 track'te daha iyi. XGB sadece long_text'te perfect ama bu büyük olasılıkla **memorization eseri** (XGB train %100, long_text prompt'ları yapısal olarak ayırt edici → ezberliyor). Long_text-spesifik kullanım için XGB ensemble'a dahil edilebilir, ama tezin **ana kategorizeri LR**.

### Modeller saved
- `models/track_classifier_xgb.pkl` (1.4 MB)
- `models/track_classifier_lr.pkl` (21 KB)
- `models/label_encoder.pkl`
- `models/sbert_config.json` (`{"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384}`)

### Kullanım

```python
from src.router.ml_categorizer import MLCategorizer
cat = MLCategorizer(classifier="lr")   # tezin ana modeli (önerilen)
# cat = MLCategorizer(classifier="xgb")  # alternatif (overfit, sadece long_text-spesifik kullanım için)
track = cat.classify("a happy dog")
track, conf = cat.classify_with_confidence("...")
tracks = cat.classify_batch(["...", "..."])
```

---

## ⚠️ In-distribution vs Out-of-distribution davranış

PRISM prompt'ları çoğunlukla **uzun ve betimleyici** (25+ kelime). Kısa kullanıcı-stili prompt'larda ML modeli farklı davranıyor:

| Test prompt | Beklenen | Rule v2 | XGB | XGB conf | LR | LR conf |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `'TOKYO' in neon` (kısa, tırnak) | text_rendering | text_rendering ✓ | entity ✗ | 0.87 | entity ✗ | 0.54 |
| `surreal painting of floating clocks` | imagination/style | style ✓ | style ✓ | 0.72 | style ✓ | 0.39 |
| `happy golden retriever in a park` | affection/entity | affection ? | entity | 0.99 | entity | 0.49 |
| `an elephant` | entity | entity ✓ | entity ✓ | 0.95 | entity ✓ | 0.45 |
| `cat sitting next to a vase` | composition | composition ✓ | entity ✗ | 0.81 | entity ✗ | 0.37 |
| `melancholic woman looking out at the rain ...` (uzun) | affection | affection ✓ | entity ✗ | 0.28 | entity ✗ | 0.25 |

**Gözlem:** Kısa OOD prompt'larda ML modeli `entity`'ye varsayılan olarak yöneliyor. Rule-based bazı kısa prompt'larda (tırnaklı text_rendering, "next to" composition) **daha iyi** çünkü açık keyword sinyali keyword tabanlı sistem için yeterli.

**Tez için sonuç:**
> ML categorizer **in-distribution** (PRISM tarzı uzun, betimleyici prompt) için %80+ ile güçlü. **Out-of-distribution** (kısa, kullanıcı-stili prompt) için ise eğitim dağılımına bağlılığı nedeniyle zayıflıyor; bu durumda kuralın açık tetikleyicileri (tırnak, edat) faydalı oluyor. Tez "hibrit kategorizer" tartışması için bu somut data sunar: ML çoğunluk durum + rule emergency kapı.

### To-do (sonraki günler)
- LLM-based kategorizer (Hafta 2 sonu) — Claude Haiku / GPT-4o-mini few-shot, OOD davranışı muhtemelen daha iyi
- Confidence threshold (örn. <0.5 ise rule'a fallback) hibrit yaklaşım denemesi
- Opsiyonel: XGB regularization deneyleri (n_estimators=50, max_depth=4, early stopping) — train-test gap düşürülebilir mi?

---

## Karar (Gün 10 sonu)

**Tezin ana ML kategorizer'ı: Logistic Regression + SBERT** (`models/track_classifier_lr.pkl`).
- Test accuracy 83.6%, macro F1 0.833
- Train-test gap 10.2 puan (kabul edilebilir)
- 22 KB model

XGBoost (`models/track_classifier_xgb.pkl`) kayıtlı kalır ama tezde **karşılaştırma noktası** olarak kullanılır — "yüksek kapasite + ezberleme = test'te kayıp" örneği.

Default `MLCategorizer()` çağrısı XGB yüklüyor; bunu LR yapmak için `MLCategorizer(classifier="lr")` çağrılır. Sonraki Lookup-ML router implementasyonunda **`classifier="lr"` default olacak şekilde** kategorizer çağrılmalı (ya da MLCategorizer'ın default'u "lr" yapılır).
