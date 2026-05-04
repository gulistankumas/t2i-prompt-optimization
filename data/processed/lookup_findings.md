# Lookup Table Bulguları (2 Mod)

## Ana sonuç
14 hücreden 14'ü doldu, hiçbir fallback gerekmedi. 2 mod sistemi (Ekonomik ≥60, Premium ≥75) PRISM'in tüm 7 track'inde uygulanabilir.

## Sürpriz bulgular

### 1. SDXL premium affection'da kazandı
- Skor: 75.3 (eşiği 0.3 puanla geçti)
- Maliyet: $0.0046 (premium fiyat olmamasına rağmen)
- Yorum: SDXL'in specialist davranışı premium segmente uzanıyor

### 2. FLUX.1-schnell premium composition'da kazandı
- Skor: 77.45
- Maliyet: $0.003 (en ucuz model!)
- Yorum: "Pahalı = iyi" varsayımı çürütüldü; track-bazlı performans segment etiketinden önemli

## Maliyet aralığı analizi
- Premium mod: $0.003 - $0.167 (55x fark)
- Ekonomik mod: $0.002 - $0.035 (17x fark)
- Aynı mod içinde bu kadar fark, router'ın somut değer önerisidir

---

## Gün 9 ek bulgular

- Pareto frontier üzerinde lookup tablo seçimleri görselleştirildi → [`figures/pareto_with_modes.png`](../../figures/pareto_with_modes.png)
- **composition**: FLUX.1-schnell hem ekonomik hem premium modda seçildi (altın çerçeve)
- **affection**: premium modda SDXL seçildi (75.3 kıl payı qualifying — eşiği +0.3 ile geçti)
- Eşik çizgileri (60 ve 75) görselde net görünür; üstte qualifying, altta dışlanan modeller
- Kural tabanlı track classifier yazıldı: [`src/router/track_classifier.py`](../../src/router/track_classifier.py)

### Rule-based classifier — PRISM 700 prompt accuracy

Genel accuracy: **%20.9** (700/700 etiketli ground truth üzerinde test edildi).

Track bazına accuracy:
| Track | Accuracy | Not |
|---|:---:|---|
| text_rendering | **77%** | tırnak ve keyword kuralları işe yarıyor |
| entity | 41% | default'a düşen prompt'lar |
| long_text | 22% | 25-kelime eşiği zayıf sinyal |
| imagination | 6% | prompt'lar uzun yazılmış, long_text'e kaçıyor |
| affection | **0%** | 100/100'ün 88'i long_text predicted |
| composition | 0% | 100/100'ün 77'si long_text predicted |
| style | 0% | 100/100'ün 71'i long_text predicted |

**Sebep:** PRISM prompt'larının çoğu 25+ kelime. 25-kelime long_text kuralı **diğer 5 track'in sinyalini boğuyor**. Eşik 50'ye çıkarılsa veya long_text en sona alınsa kısmi düzelir, ama o durumda yine `text_rendering` haricinde kuralların discriminative gücü zayıf.

**Tez için sonuç:**
> Rule-based classifier alt sınır baseline (text_rendering ve entity hariç pratikte yetersiz). LLM tabanlı kategorizer (Gün 10) **zorunlu**, "lüks" değil. Bu, tezin "üç router yaklaşımı karşılaştırması" bölümünde lookup-rule vs lookup-LLM kontrastını sayısal olarak güçlendirir.
