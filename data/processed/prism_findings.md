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

## 5. Veri Eksiklikleri ve To-Do

- ❌ **DALL-E 2 PRISM'de yok** — HEIM'den alınıp SD 1.5 köprüsüyle projekte edilecek (Gün 3)
- ❌ **Qwen2.5-VL jüri skorları alınmadı** — WebFetch iki tabloyu ayıramadı, manuel çekim veya farklı yöntem gerekli (çapraz doğrulama için)
- ❌ **Chinese leaderboard** — bu tezin kapsamı dışı, yoksayılabilir
- ⚠️ **Manuel spot-check önerilir:** CSV'deki 2-3 hücre leaderboard'dan karşılaştırılmalı
- ❌ **Alignment/Aesthetic ayrı ayrı** — sadece "Avg" elimizde, Align ve Aes ayrı skorlar tezin daha zengin analiz yapmasına imkân verebilir
