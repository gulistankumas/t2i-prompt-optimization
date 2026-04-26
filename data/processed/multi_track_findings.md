# Gün 3 — Çoklu-Track ve Bottleneck Analizi

**Script:** [notebooks/06_multi_track_and_bottleneck.py](../../notebooks/06_multi_track_and_bottleneck.py)
**Veri:** PRISM 19 model (havuz analizi) + 8 hedef model (bottleneck), GPT-4.1 İngilizce jüri

## 1. En Düşük Korelasyonlu 3 Track Çifti (19-model Spearman)

| Çift | ρ |
|---|:---:|
| **entity ↔ long_text** | **0.644** |
| style ↔ long_text | 0.692 |
| affection ↔ long_text | 0.779 |

**Bulgu:** `long_text` PRISM'in en bağımsız track'i — diğer 6 track'in 3'üyle en düşük korelasyona sahip. Entity tanıma, stil ve duygu rendering "tek-aspect" yetenekler; long_text uzun talimatları takip etmeyi ölçüyor — farklı bilişsel boyut.

## 2. Çoklu-Track Sıralama Yön Değişimleri (Specialist Modeller)

`entity ↔ long_text` (en bağımsız çift) için 19-model sıralamasında en büyük rank kaymaları:

| Model | entity rank | long_text rank | Fark | Uzmanlığı |
|-------|:---:|:---:|:---:|---|
| **Bagel** | 17 | 7 | **10** | Long-text uzmanı, entity'de zayıf |
| **FLUX.1-schnell** | 16 | 8 | 8 | Long-text iyi, entity'de zayıf |
| **SDXL** | 9 | 17 | 8 | Entity'de iyi, long-text dipte |
| **Playground** | 8 | 16 | 8 | Style/entity uzmanı, long-text zayıf |

`affection ↔ long_text`'te en büyük rank-farkı **FLUX.1-schnell** (9). `style ↔ long_text`'te yine **Bagel** (8).

**Generalist modeller** (rank farkı ≤ 1): GPT-Image-1, Gemini 2.5, Qwen-Image, SEEDream 3.0. Premium tier hep generalist; uzmanlık alt segmentlerde ortaya çıkıyor.

## 3. Bottleneck Modeli — Track Profili Dengesizliği (8 hedef model)

| Model | Mean | Min | Range | Std | En Zayıf Track |
|-------|:---:|:---:|:---:|:---:|---|
| **SDXL** | 60.4 | 25.4 | **52.6** | **19.9** | text_rendering |
| **SD1.5** | 44.3 | 20.6 | 40.4 | 14.7 | text_rendering |
| FLUX.1-dev | 73.7 | 56.3 | 33.4 | 11.8 | text_rendering |
| Qwen-Image | 79.9 | 61.6 | 28.8 | 10.3 | text_rendering |
| SD3.5-Large | 73.8 | 58.0 | 28.2 | 11.7 | long_text |
| FLUX.1-schnell | 64.2 | 50.2 | 26.1 | 9.3 | text_rendering |
| Gemini 2.5 | 85.3 | 69.7 | 22.4 | 7.9 | text_rendering |
| GPT-Image-1 | 86.3 | 74.5 | 18.6 | **7.3** | text_rendering |

**Bulgular:**
- **SDXL en dengesiz model** — range 52.6, text_rendering 25.4'te dramatik düşüş
- **GPT-Image-1 en dengeli** — std 7.3, en zayıfı bile 74.5
- **text_rendering 8 modelden 7'si için en zayıf track** → modern T2I'da bile yazı rendering global olarak zayıf yetenek

## 4. AND-Prompt Kalite Tahmini (min-aggregation)

Eğer prompt iki yetenek de gerektiriyorsa, model performansı **min(track_a, track_b)** ile sınırlanır. SDXL'in solo skorlarından gerçek combo skoruna düşüş:

| Kombinasyon | SDXL solo (track 1) | SDXL combo min | Düşüş |
|---|:---:|:---:|:---:|
| style + text_rendering | 73.9 (style) | 25.4 | **-48.5** |
| style + long_text | 73.9 (style) | 41.9 | -32.0 |
| entity + long_text | 70.0 (entity) | 41.9 | -28.1 |

**SDXL'in style skoru solo değerlendirmede üstte tutuyor; ama style+text_rendering çoklu prompt'unda 48.5 puan düşüş yaşıyor.** Bu büyük bir routing sinyali — solo-track lookup, combo prompt'larda yanıltıcı olur.

## 5. Tezin Routing Stratejisine Etkisi

**5.1 Solo-track lookup yetersiz.** "Prompt kategorisi → en iyi model" basit lookup, çoklu yetenek isteyen prompt'larda bottleneck'i göremez. SDXL style sorgusunda çıkar ama prompt hem style hem yazı içeriyorsa felaket.

**5.2 Önerilen genişletme: capability-min routing.**
LLM kategorizer çıktısını **tek track yerine track listesi** olarak alır. Lookup → modellerin `min(track_listesi)` skoruna göre sıralanır. Bu yaklaşım:
- Bagel gibi long-text uzmanı modelleri yalın long-text prompt'larda öne çıkarır
- SDXL gibi bottleneck'li modelleri çoklu yetenek prompt'larında dezavantaja sokar
- Tek satırla matematiksel temellendirme: `score(model, prompt) = min_{t ∈ required(prompt)} skor(model, t)`

**5.3 Tezin "track-aware routing" iddiasını güçlendiriyor.**
Yüksek korelasyon argümanına karşı en güçlü cevap. Track-aware routing'in değer ürettiği yer:
- Bottleneck modelleri devre dışı bırakmak (SDXL'i text_rendering içeren prompt'larda eleme)
- Specialist modelleri spesifik kullanım alanlarında öne çıkarmak (Bagel long-text'te)
- Pareto frontier'ı combo-skor üzerinden çıkarmak

## To-Do (sonraki günler)
- AND-aggregation'lı router prototipi (Hafta 2'de lookup table'ı min-based yap)
- LLM kategorizer'ı **multi-label** yap (tek kategori değil, gerekli track listesi döndürür)
- 19-model genişletme kararıyla yeniden çalıştır (havuz büyürse Bagel/Playground gibi specialist'leri analize dahil et)
