# Gün 4 — Master Tablo + Pareto Bulguları

**Script:** [notebooks/09_master_table.py](../../notebooks/09_master_table.py)
**Çıktı:** [master_table.csv](master_table.csv) (9 model × [vendor + access + params + 7 track + overall + cost + q/$])
**Maliyet kaynağı:** [model_costs.csv](../raw/model_costs.csv) (kaynaklar belirtildi)

## Master Tablo (overall'a göre sıralı)

| Model | Vendor | Access | Params | Overall (avg of 2 judges) | Cost ($/img) | Q/$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| GPT-Image-1 | OpenAI | closed | — | 83.50 | 0.1670 | 500 |
| Gemini 2.5 Flash-Image | Google | closed | — | 82.85 | 0.0390 | **2124** |
| Qwen-Image | Alibaba | open | 20B | 77.00 | 0.0400 | 1925 |
| SD3.5-Large | Stability | open | 8B | 72.15 | 0.0350 | 2061 |
| FLUX.1-dev | BFL | open | 12B | 71.10 | 0.0300 | 2370 |
| FLUX.1-schnell | BFL | open | 12B | 64.45 | 0.0030 | **21483** |
| SDXL | Stability | open | 3.5B | 58.70 | 0.0046 | 12761 |
| SD1.5 | Runway/CompVis | open | 1B | 45.10 | 0.0023 | 19609 |
| DALL-E 2 | OpenAI | closed | — | (NaN, HEIM-only) | 0.0200 | — |

## Pareto Frontier — En Kritik Bulgu

**6 Pareto-optimal model:**
1. SD1.5 (kalite 45.10 / $0.0023) — uç ucuz
2. FLUX.1-schnell (64.45 / $0.0030) — sweet spot
3. FLUX.1-dev (71.10 / $0.0300)
4. SD3.5-Large (72.15 / $0.0350)
5. Gemini 2.5 (82.85 / $0.0390)
6. GPT-Image-1 (83.50 / $0.1670) — uç premium

**2 Pareto-DOMINATED model (havuzdan çıkması düşünülmeli):**
- ❌ **SDXL** (58.70 / $0.0046) — FLUX.1-schnell hem **daha ucuz** ($0.0030) hem **daha kaliteli** (64.45). SDXL hiçbir modda optimal değil.
- ❌ **Qwen-Image** (77.00 / $0.0400) — Gemini 2.5 hemen hemen aynı fiyata ($0.0390) **5.85 puan daha kaliteli** (82.85). Yine optimal değil.

## Mod Bazlı Tavsiye (Pareto frontier'dan)

| Mod | Strateji | Önerilen Model | Gerekçe |
|---|---|---|---|
| Ucuz | En düşük maliyet | **SD1.5** | $0.0023, ama kalite çok düşük (45.1) |
| Ucuz (gerçekçi) | En iyi q/$ | **FLUX.1-schnell** | $0.0030 ile 64.5 kalite — q/$ = 21483, en yüksek |
| Dengeli | Mid-tier kalite | FLUX.1-dev veya SD3.5-Large | ~$0.03 ile ~71-72 kalite |
| Premium | En yüksek kalite | GPT-Image-1 | 83.5 kalite, ama $0.167 (5x Gemini fiyatı) |
| Premium (rasyonel) | İkinci-en-iyi kalite | **Gemini 2.5** | 82.85 kalite, $0.039 — sadece 0.65 puan altında, **4.3x daha ucuz** |

**Premium mod paradoksu:** GPT-Image-1 ile Gemini 2.5 arası kalite farkı **0.65 puan** ama maliyet farkı **4.3x**. Routing'in en kritik kararı burada — kullanıcı "premium" derken her zaman GPT-Image-1 mi istiyor, yoksa "kaliteli ama makul" Gemini'yi mi?

## DALL-E 2 Durumu

PRISM'de yok, master tabloda NaN olarak duruyor. Bridge kalibrasyonu (SD 1.5 köprüsü) tek-noktayla istatistiksel olarak zayıf olduğu için **router değerlendirmesinden çıkarıldı**. Tezde "tarihsel referans" olarak ayrı bölümde işlenir, HEIM skorlarıyla.

## Track-Bazlı Pareto Frontier (kritik düzeltme)

Overall-Pareto'da SDXL ve Qwen-Image dominated görünüyordu. Ama track-bazlı Pareto **farklı resim** çiziyor:

| Model | Pareto-optimal track sayısı | Hangi track'lerde |
|---|:---:|---|
| SD1.5, FLUX.1-schnell, Gemini 2.5 | **7/7** | Tüm track'ler |
| FLUX.1-dev | 6/7 | entity hariç |
| SD3.5-Large | 5/7 | affection ve long_text hariç |
| GPT-Image-1 | 5/7 | **affection ve long_text'te dominated** |
| SDXL | **3/7** | entity, style, affection (specialist) |
| Qwen-Image | **0/7** | hiçbir track'te Pareto-optimal değil |
| DALL-E 2 | 0/7 | PRISM verisi yok |

### Üç çarpıcı bulgu

**1. SDXL track-aware specialist olarak değer üretiyor.** Overall'da dominated görünmesine rağmen entity/style/affection'da Pareto frontier'da. Bu, tezin **"overall skor track-spesifik kararı temsil etmez"** iddiasının canlı kanıtı.

**2. GPT-Image-1 affection ve long_text'te dominated.** Premium fiyatı (5x Gemini) iki track'te justify edilmiyor; Gemini 2.5 hem ucuz hem daha kaliteli. Naif "Premium = GPT-Image-1" kuralının karşı-örneği.

**3. Qwen-Image hiçbir track'te Pareto-optimal değil.** Her track'te Gemini 2.5 tarafından dominate ediliyor (Gemini $0.039 < Qwen $0.040 ve her track'te daha kaliteli). Açık-kaynak premium isteyene SD3.5-Large veya FLUX.1-dev makul, Qwen-Image rasyonel hiçbir kararla seçilmez. Buna rağmen **tezde "vendor-marketing iddiası vs Pareto sınaması" örneği olarak değer üretir**: Qwen-Image text_rendering ile pazarlandı ama PRISM'de text_rendering skoru 63.05 (Gemini 74.10'un altında).

## Tezin Model Havuzu Üzerine Karar

Veri açıkça gösteriyor: **SDXL ve Qwen-Image hiçbir mod için optimal değil**. Üç olası yaklaşım:

**A) 9'u koru ama Pareto-dominated olduklarını dürüstçe raporla.** Akademik dürüstlük açısından güçlü; tezde "neden bu modeller dahil edildi" sorusu sorulduğunda "çeşitlilik için, ama Pareto'da olmadıklarını gösteriyoruz" denir.

**B) Pareto-dominated'leri çıkar, 7 model ile devam.** SDXL ve Qwen-Image silinir; havuz: SD1.5, FLUX.1-schnell, FLUX.1-dev, SD3.5-Large, DALL-E 2 (tarihsel), Gemini 2.5, GPT-Image-1. Daha temiz tablo, ama "neden Stability ve Alibaba modelleri yok" sorusuna cevap zorlaşır.

**C) SDXL'i çıkar (gerçekten dominated), Qwen-Image'i tut (yakın, sınır vakası).** Qwen-Image vs Gemini 2.5 farkı sadece $0.001 maliyet ve 5.85 kalite. Kullanıcının açık-kaynak tercih etmesi durumunda Qwen-Image hâlâ rasyonel seçim.

**Önerim revize: A — 9 modeli koru.** Track-bazlı Pareto analizi gösterdi ki SDXL specialist (3/7 track), Qwen-Image her track'te dominated. İkisi de tezin temel iddiasının kanıtı:
- SDXL → "track-aware routing değer üretiyor" örneği
- Qwen-Image → "vendor-marketing iddiası Pareto'da test edilince çürüyor" örneği

Bu havuz, akademik dürüstlük ve sınama gücü açısından 7-modelli temizlenmiş havuzdan **daha güçlü**.

## To-Do
- Karar verildikten sonra [bitirme.md](../../bitirme.md) Model Havuzu güncellenir
- Pareto grafiği (matplotlib) `figures/pareto_frontier.png` olarak çizilir (Hafta 2)
- Track bazında Pareto da ayrı analiz edilebilir (her track için ayrı frontier)
