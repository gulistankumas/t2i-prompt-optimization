# Gün 2 — PRISM İlk Keşif Bulguları

**Veri:** 19 model × 7 track (GPT-4.1 İngilizce jüri) → 8 hedef modelimizin tüm skorları mevcut (DALL-E 2 hariç).

## 1. Track Başına Birinci Model

| Track | Birinci | Skor |
|-------|---------|:---:|
| imagination | Gemini 2.5 | 88.6 |
| entity | GPT-Image-1 | 88.2 |
| text_rendering | GPT-Image-1 | 74.5 |
| style | GPT-Image-1 | 93.1 |
| affection | Gemini 2.5 | 92.1 |
| composition | GPT-Image-1 | 92.8 |
| long_text | Gemini 2.5 | 81.1 |

Sadece 2 farklı birinci var. GPT-Image-1 (entity, text, style, composition) analitik/objektif tracklerde; Gemini 2.5 (imagination, affection, long_text) yaratıcılık/duygu tracklerinde önde.

## 2. Track-İçi Skor Yayılımı

| Track | Min | Max | Yayılım |
|-------|:---:|:---:|:-------:|
| text_rendering | 20.6 (SD 1.5) | 74.5 (GPT-Image-1) | **53.9** |
| imagination | 36.4 | 88.6 | 52.2 |
| long_text | 32.9 | 81.1 | 48.2 |
| entity | 47.5 | 88.2 | 40.7 |
| style | 55.3 | 93.1 | 37.8 |
| composition | 56.1 | 92.8 | 36.7 |
| affection | 61.0 | 92.1 | **31.1** |

**text_rendering** en ayırt edici track — modeller arasında en büyük fark. **affection** en az ayırt edici — herkes makul.

## 3. Track-Arası Korelasyon

Tüm track çiftleri **0.80-0.98** aralığında korelasyon gösteriyor. Pratikte bu, modellerin track'ler arası performansının **büyük ölçüde tek boyutlu** olduğu anlamına gelir.

**Tez için implikasyon:**
- ✅ Routing sinyali var — track bazında birinci değişiyor
- ⚠️ Ama sinyal zayıf — bir modelin bir track skorunu biliyorsan diğerlerini ~%95 isabetle tahmin edebilirsin
- ⚠️ **LOCO-CV genelleme iddiası (hafif (c))** bu yüksek korelasyon yüzünden trivial olabilir. Tezin bu sonucu dürüstçe raporlaması lazım: "ML router kategorilere genelleme yapıyor, ama kategoriler yüksek korelasyonlu olduğu için bu zorlu bir genelleme değil."

## 4. Routing Değeri — Segmentasyon

Top-3 her track'te aynı: **GPT-Image-1 > Gemini 2.5 > Qwen-Image**. Premium modda routing neredeyse trivial — her prompt için GPT-Image-1.

**Routing asıl değerini orta/ucuz segmentlerde kazanıyor:**
- FLUX.1-schnell vs SDXL vs SD 3.5 arası fark track'e göre değişiyor
- Örn: text_rendering'de **FLUX.1-schnell (50.2)** >> SDXL (25.4), ama style'da **SDXL (73.9)** yaklaşık FLUX-schnell (69.4)
- Maliyet-duyarlı modların hedeflediği segment burası

**Tez hikâyesinin düzeltmesi:** "Router premium mod için trivial, asıl fayda maliyet-kalite dengesi orta segmentte; düşük maliyetli modellerin yetenek profillerinin prompt'a uydurulması, %X-Y kalite kaybı ile %Z maliyet tasarrufu sağlar."

## 5. İki Jüri Arasındaki Anlaşma (Qwen2.5-VL eklendi)

Kullanıcının manuel pastelediği PRISM paper Tablo 2 (Qwen2.5-VL jüri) verisi işlendi. Analiz scripti: [notebooks/02_cross_judge_agreement.py](../../notebooks/02_cross_judge_agreement.py).

### Sıralama korelasyonu (Spearman'a eşdeğer)

| Track | ρ |
|-------|:---:|
| long_text | 1.00 |
| overall | 0.99 |
| imagination | 0.98 |
| text_rendering | 0.98 |
| style | 0.98 |
| affection | 0.98 |
| entity | 0.95 |
| composition | 0.95 |

İki jüri **sıralama** açısından neredeyse tam anlaşmada (tüm track'lerde ρ ≥ 0.95).

### Top-1 anlaşma (asıl kritik)

| Track | GPT-4.1 seçer | Qwen2.5-VL seçer | Anlaşma |
|-------|:---|:---|:---:|
| imagination | Gemini 2.5 | GPT-Image-1 | ❌ |
| entity | GPT-Image-1 | GPT-Image-1 | ✅ |
| text_rendering | GPT-Image-1 | Gemini 2.5 | ❌ |
| style | GPT-Image-1 | Gemini 2.5 | ❌ |
| affection | Gemini 2.5 | Gemini 2.5 | ✅ |
| composition | GPT-Image-1 | Gemini 2.5 | ❌ |
| long_text | Gemini 2.5 | Gemini 2.5 | ✅ |

**4/7 track'te iki jüri FARKLI birinci modeli seçiyor.** Router lookup'ı hangi jüriye güvendiğine göre farklı çıktı verebilir.

### Skor deltası ve olası bias

| Model | GPT-4.1 | Qwen2.5-VL | Fark |
|-------|:---:|:---:|:---:|
| Qwen-Image | 79.9 | 74.1 | **+5.8** (GPT yüksek) |
| GPT-Image-1 | 86.3 | 80.7 | **+5.6** |
| FLUX.1-dev | 73.7 | 68.5 | +5.2 |
| Gemini 2.5 | 85.3 | 80.4 | +4.9 |
| SDXL | 60.4 | 57.0 | +3.4 |
| SD3.5-Large | 73.7 | 70.6 | +3.1 |
| FLUX.1-schnell | 64.2 | 64.7 | -0.5 |
| SD1.5 | 44.2 | 46.0 | -1.8 |

GPT-4.1 sistematik olarak üst-segmente yüksek skor veriyor (+3 ila +6); Qwen2.5-VL düşük-segmentte biraz cömert. Bu, top-1 anlaşmazlığının neden **üst tier** track'lerde yoğunlaştığını açıklar: iki top model (GPT-Image-1 vs Gemini) arasında ince fark, jüri tercihine bağlı.

**Potansiyel bias notu:** GPT-4.1 (OpenAI jürisi) GPT-Image-1'i (OpenAI modeli) 4 track'te birinci seçiyor; Qwen2.5-VL aynı 4 track'in 3'ünde Gemini'yi birinci seçiyor. Tezde "LLM-as-judge self-preference bias" başlığı altında tartışılmalı — literatürde bilinen bir fenomen.

### Tez için sonuç

Router dizaynında **tek jüriye** dayanmak risk. Öneri:
- Master skor matrisi **iki jüri ortalamasıyla** kurulur
- Ek analiz: "jüri seçiminin router kararına etkisi" — tezin robustness bölümü

## 6. Veri Eksiklikleri ve To-Do

- ❌ **DALL-E 2 PRISM'de yok** — Gün 3: HEIM'den alınıp SD 1.5 köprüsüyle projekte edilecek
- ❌ **Chinese leaderboard** — kapsam dışı
- ⚠️ **Manuel spot-check:** iki CSV'den 2-3'er hücre tarayıcıdan doğrulansın
- ❌ **Alignment/Aesthetic ayrı kolonlar** — elimizde sadece Avg var; Qwen2.5-VL paster'ından Ali/Aes de çıkarılabilir, gerekirse genişletilir
