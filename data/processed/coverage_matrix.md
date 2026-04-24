# Revize Model Havuzu × HEIM × PRISM Kapsama Matrisi

**Güncelleme:** 2026-04-24 (revize sonrası)
**Model sayısı:** 9 (önceki 8'e FLUX.1-schnell eklendi, DALL-E 2 tarihsel baseline olarak korundu)

## Havuz

| # | Model | Ana Kaynak | Kategori | HEIM | PRISM |
|---|-------|:---------:|----------|:----:|:-----:|
| 1 | Stable Diffusion 1.5 | HEIM | Eski nesil, ucuz, baseline | ✅ | ✅ **Köprü** |
| 2 | SDXL | PRISM | 2023-2024 orta segment | ❌ | ✅ |
| 3 | Stable Diffusion 3.5 Large | PRISM | Modern açık kaynak | ❌ | ✅ |
| 4 | FLUX.1-schnell | PRISM | Ucuz, hızlı, modern | ❌ | ✅ |
| 5 | FLUX.1-dev | PRISM | Premium açık kaynak | ❌ | ✅ |
| 6 | Qwen-Image | PRISM | Yeni, metin-kompozisyon | ❌ | ✅ |
| 7 | DALL-E 2 | HEIM | Kapalı kaynak pioneer | ✅ | ❌ |
| 8 | GPT-Image-1 | PRISM | Modern premium kapalı | ❌ | ✅ |
| 9 | Gemini 2.5 Flash-Image | PRISM | Alternatif premium kapalı | ❌ | ✅ |

## Veri Akışı

- **PRISM cluster** (7 model): SDXL, SD 3.5 Large, FLUX.1-schnell, FLUX.1-dev, Qwen-Image, GPT-Image-1, Gemini 2.5 — ana skor PRISM'den
- **HEIM cluster** (2 model): SD 1.5, DALL-E 2 — ana skor HEIM'den
- **Köprü noktası:** SD 1.5 her iki benchmark'ta da skorlanmış → HEIM↔PRISM ölçek kalibrasyonu için tek anchor

## Katkı İddiası Üzerine Etkisi

Tezin orijinal katkı (1) iddiası — "HEIM + PRISM uyumlandırılması" — şimdi somut ve savunulabilir:

> "SD 1.5 modeli hem HEIM hem PRISM tarafından değerlendirildiği için, bu modelin iki sistemdeki skorları arasında regresyon tabanlı bir eşleme fonksiyonu öğrenilir. Bu eşleme ile DALL-E 2'nin HEIM skoru PRISM ölçeğine projekte edilir, böylece tek bir karşılaştırılabilir kalite eksenine sahip 9 modelli bir master tablo elde edilir."

## Karar (2026-04-24)

**Master tablo PRISM ölçeğinde kurulur.**

- 8 model (SDXL, SD 3.5 Large, FLUX.1-schnell, FLUX.1-dev, Qwen-Image, GPT-Image-1, Gemini 2.5, **ve** SD 1.5) doğrudan PRISM skorunu kullanır
- DALL-E 2 HEIM skoruna sahip → SD 1.5 köprüsüyle öğrenilen `f: HEIM_score → PRISM_score` fonksiyonu ile projekte edilir
- SD 1.5'in HEIM skoru yalnızca `f` fonksiyonunu fit etmek için kullanılır, master tabloda görünmez

**Gerekçe:** Minimum ölçek çevrimi → minimum kalibrasyon hatası.

## Tez İçin Açık Risk

`f` fonksiyonu **tek veri noktası** (SD 1.5) üzerinden fit edilecek. Bu istatistiksel olarak zayıf — lineer regresyonla bile anlamlı confidence interval çıkaramayız. Olası hafifletmeler:

- **Prompt-düzeyi köprü:** SD 1.5'in N prompt'u için HEIM skoru ve PRISM skoru ayrı ayrı alınır; bu N değer üzerinden regresyon fit edilir. Tek skalar değil, N skalar karşılaştırması.
- **Boyut-düzeyi köprü:** HEIM'ın 12 boyutundan hangilerinin PRISM track'leriyle örtüştüğü tespit edilir, boyut başına ayrı `f_d` fit edilir.
- **Sade yol:** DALL-E 2'yi tamamen PRISM ölçeğinden dışarıda bırak, "HEIM-skorlu tarihsel referans" olarak ayrı bir sütunda sun. Router kararına girmesin.

Gün 3 sonunda PRISM veri yapısı görüldükten sonra bu üç yoldan biri seçilir.
