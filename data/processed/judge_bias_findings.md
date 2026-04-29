# Gün 4 — Jüri Bias Formal Analizi

**Script:** [notebooks/08_judge_bias_analysis.py](../../notebooks/08_judge_bias_analysis.py)
**Veri:** PRISM 19 model × 7 track, GPT-4.1 ve Qwen2.5-VL-72B jürileri

## 1. Sıralama vs Mutlak Skor — İki Farklı Tablo

| Boyut | Bulgu |
|-------|-------|
| Sıralama (Spearman) | ρ ≥ 0.95 her track'te → iki jüri **aynı sırayla** sıralıyor |
| Mutlak skor delta | Track ortalaması bazında **±5 puana kadar** sistematik fark |
| Top-1 anlaşmazlığı | 4/7 track'te farklı birinci |

## 2. Track-Bazında Sistematik Bias (19 model toplam)

| Track | GPT-4.1 ortalama | Qwen2.5-VL ortalama | Delta | Yorum |
|-------|:---:|:---:|:---:|---|
| **imagination** | 68.65 | 52.70 | **+15.95** | GPT-4.1 yaratıcılığa **çok daha cömert** |
| affection | 82.47 | 77.32 | +5.16 | GPT-4.1 duygu rendering'inde cömert |
| composition | 81.27 | 79.23 | +2.04 | Yakın |
| style | 76.54 | 75.94 | +0.59 | Eşit |
| long_text | 58.23 | 59.36 | -1.13 | Yakın |
| entity | 67.64 | 68.86 | -1.22 | Qwen biraz cömert |
| **text_rendering** | 46.45 | 51.31 | **-4.85** | Qwen2.5-VL **teknik track'te cömert** |

### Pattern
GPT-4.1 **sübjektif/yaratıcı** track'lerde (imagination +16, affection +5) cömert; Qwen2.5-VL **teknik/objektif** track'lerde (text_rendering +5, entity +1) cömert. Bu, LLM-as-judge literatüründe bilinen bir fenomen — dil tabanlı jüriler subjektif değerlendirmede cömertleşir.

## 3. Model-Bazında Delta (sorted)

GPT-4.1 daha cömert (top 5):
- **Playground +7.0** (Wilcoxon p=0.047, **anlamlı**)
- Qwen-Image +5.8
- **GPT-Image-1 +5.6** (p=0.047, **anlamlı**)
- FLUX.1-dev +5.2
- Gemini 2.5 +4.9

Qwen2.5-VL daha cömert (top 5):
- SD2.1 -2.8
- SD1.5 -1.8
- JanusPro-7B -1.4
- SD3-Medium -0.7
- FLUX.1-schnell -0.5

**14/19 modelde GPT-4.1 daha yüksek skor verdi**, 5/19'da Qwen2.5-VL.

## 4. Self-Preference Bias Sorgusu

GPT-4.1 OpenAI'nin GPT-Image-1 modelini (+5.6) sistematik daha yüksek skorluyor — Wilcoxon p=0.047 (n=7 ile sınırlı güç). Ama:
- Aynı zamanda Playground'u (+7.0, OpenAI değil) ve Qwen-Image'i (+5.8, Alibaba) da yüksek skorluyor
- Yani bu **vendor self-preference değil**, **stilistik tercih**: GPT-4.1 muhtemelen "fotogerçekçi, kompozisyonu güçlü" görselleri yapısal olarak yüksek skorluyor

## 5. Tez İçin Sonuç ve Öneri

**Tek jüriye dayalı router riski:**
- Routing **kararı** sıralamada büyük fark yapmaz (ρ 0.95+)
- Ama **mutlak skor** üzerinden bağlayan algoritmaları (örn. ML regression, threshold tabanlı routing) etkiler
- Premium modda iki jüri farklı top-1 verir → kullanıcıya gösterilen "%X kalite kazandın" iddiası jüri-bağımlı

**Öneri:**
1. **Master skor matrisi her hücre için iki jürinin ortalaması** (uygulandı: [master_table.csv](master_table.csv))
2. Tezde "robustness" bölümünde tek-jüri vs çift-jüri router kararları karşılaştırılır
3. **Track-spesifik bias** (imagination'da +16) tezde tartışılır, "evaluation methodology" alt-bölümünde
