# Kategorizer Bulguları

PRISM 7 track sınıflandırma — üç yaklaşım test edildi, ortak değerlendirme kümesi: 140 PRISM test prompt'u (her track 20).

## Özet Tablo

| Yaklaşım | Test Acc | Δ | Eğitim süresi |
|---|:---:|:---:|:---:|
| Rule-based v2 | 37.1% | — | 0 (kurallar) |
| **LogReg + SBERT** | **83.6%** | +46.4 | ~5s |
| XGBoost + SBERT | 80.0% | +42.9 | ~15s |

LogReg, XGBoost'u 3.6 puan geçti — 384-dim SBERT embedding üzerinde lineer sınır yeterli, ek ağaç karmaşıklığı kazanım sağlamadı.

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

### XGBoost — sınıf bazında F1 (test seti)

| Track | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| affection | 0.95 | 0.90 | **0.92** | 20 |
| long_text | 1.00 | 1.00 | **1.00** | 20 |
| style | 0.85 | 0.85 | 0.85 | 20 |
| imagination | 0.77 | 0.85 | 0.81 | 20 |
| text_rendering | 0.86 | 0.60 | 0.71 | 20 |
| entity | 0.60 | 0.75 | 0.67 | 20 |
| composition | 0.65 | 0.65 | 0.65 | 20 |

**Macro F1: 0.80, weighted F1: 0.80.** Rule-based'in zayıf olduğu track'lerde dramatik iyileşme: affection 0% → 92%, long_text 0% → 100%, imagination 6% → 81%.

### Modeller saved
- `models/track_classifier_xgb.pkl` (1.4 MB)
- `models/track_classifier_lr.pkl` (21 KB)
- `models/label_encoder.pkl`
- `models/sbert_config.json` (`{"model_name": "all-MiniLM-L6-v2", "embedding_dim": 384}`)

### Kullanım

```python
from src.router.ml_categorizer import MLCategorizer
cat = MLCategorizer(classifier="xgb")  # veya "lr"
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
