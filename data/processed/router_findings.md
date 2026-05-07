# Router Bulguları

## Gün 12 — Lookup Router

### Mimari ve performans
- Kategorizer: LR + SBERT (%83.6 track accuracy)
- Latency: ~32ms per routing
- Maliyet: $0 (yerel pipeline)

### Test seti sonuçları (140 PRISM prompt)

#### Ekonomik mod (eşik 60+)
- Beklenen kalite: 66.36
- Gerçek kalite: 64.77
- Maliyet: $0.0113
- En sık seçilen: FLUX.1-schnell (59), SD1.5 (22), SDXL (20)

#### Premium mod (eşik 75+)
- Beklenen kalite: 77.84
- Gerçek kalite: 76.84
- Maliyet: $0.0454
- En sık seçilen: Gemini 2.5 (56), SD3.5-Large (24), SDXL (22)

### Baseline karşılaştırması (kalite-maliyet trade-off)

| Strateji | Kalite | Maliyet | Yorum |
|----------|--------|---------|-------|
| Always Cheapest (SD1.5) | 45.14 | $0.0023 | Ucuz ama kalite düşük |
| **Lookup Ekonomik** | **64.77** | **$0.0113** | Dengeli |
| Always Popular (FLUX.1-dev) | 71.10 | $0.0300 | Sektör standardı |
| Random | 70.44 | $0.0492 | Karışık |
| **Lookup Premium** | **76.84** | **$0.0454** | Yüksek kalite, makul maliyet |
| Always Premium (GPT-Image-1) | 83.50 | $0.1670 | En yüksek kalite, çok pahalı |

### Ana bulgular

**1. Lookup ekonomik vs always_popular:**
- %91 kalite korunması (64.77/71.10)
- %62 maliyet tasarrufu ($0.0113 vs $0.0300)

**2. Lookup premium vs always_premium:**
- %92 kalite korunması (76.84/83.50)
- %73 maliyet tasarrufu ($0.0454 vs $0.1670)

**3. Beklenen-gerçek fark:**
- 1-1.6 puan, kategorizer %16.4 hatasından kaynaklı
- Track doğruluğu router performansının üst sınırı

### Tek cümlelik özet
> Lookup router, sektör standardı "always FLUX.1-dev" stratejisine kıyasla kalitenin **%91'ini koruyarak maliyeti %62 azaltır**; "always GPT-Image-1" stratejisine kıyasla kalitenin **%92'sini koruyarak maliyeti %73 azaltır**. Tezin temel iddiası: track-aware routing kalite-maliyet eksende dominant pratik çözüm.

### Tablo 5.1: Routing Stratejilerinin Performans Karşılaştırması (140 PRISM test prompt)

| Strateji | Kalite | Maliyet | Kalite/Maliyet |
|----------|:---:|:---:|:---:|
| Always Cheapest | 45.14 | $0.0023 | 19,626 puan/$ |
| **Lookup Ekonomik** ★ | **64.77** | **$0.0113** | **5,732 puan/$** |
| Always Popular | 71.10 | $0.0300 | 2,370 puan/$ |
| Random | 70.44 | $0.0492 | 1,432 puan/$ |
| **Lookup Premium** ★ | **76.84** | **$0.0454** | **1,693 puan/$** |
| Always Premium | 83.50 | $0.1670 | 500 puan/$ |

★ = Lookup router (bu çalışmanın yöntemi). Lookup ekonomik **2. en yüksek kalite/maliyet** oranı (yalnızca extreme-ucuz Always SD1.5'in altında ama 19.6 puan daha kaliteli).

### Latency karşılaştırması

| Bileşen | Süre |
|---|:---:|
| LR + SBERT kategorizer | ~28 ms |
| Lookup tablo erişimi | <1 ms |
| **Toplam (per prompt)** | **~32 ms** |

LLM kategorizer kullansak ~500-1500ms olurdu (15-50x daha yavaş). Bu, demo'da kullanıcının "loading" bile görmeden öneri alabilmesini sağlar.

### Tezde konum
- Bölüm 5.X: Lookup Router Performansı (ana grafik: lookup_vs_baselines.png)
- Bölüm 6.X: Kalite-maliyet trade-off tartışması
- Bölüm 6.Y: Kategorizer doğruluğunun router etkisi (%16.4 hata → 1-1.6 puan kalite cezası)

### Çıktılar
- [`data/processed/lookup_router_eval.csv`](lookup_router_eval.csv) — 280 routing kararı
- [`figures/lookup_vs_baselines.png`](../../figures/lookup_vs_baselines.png) — kalite + maliyet 2 panel

### Sonraki Adımlar (Gün 13-14)
- ML router (SBERT classifier doğrudan model seçer, lookup tablosuz)
- LLM router (few-shot prompting ile direkt model önerisi)
- Hepsi ile lookup karşılaştırması — Gün 14 check-point
