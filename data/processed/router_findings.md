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

---

## Gün 13 — Üç Router Karşılaştırması

3 farklı routing yaklaşımı 140 test × 2 mod = 280 karar üzerinde karşılaştırıldı.

### Mimariler

| Router | Pipeline | Eğitim | Latency | Çalışma Maliyeti |
|---|---|:---:|:---:|:---:|
| Lookup | prompt → kategorizer → track → tablo → model | Yok (kural + LR cat.) | 32ms | $0 |
| ML | prompt + mod → SBERT + LogReg → model | 1120 (track→model) | 33ms | $0 |
| LLM | prompt + mod → GPT-4o-mini few-shot → model | Few-shot (8 örnek) | 855ms | ~$0.0001/prompt |

### Sonuçlar (140 prompt × 2 mod = 280 karar)

| Router | Mode | Gerçek Kalite | Maliyet ($) | Kalite/Maliyet |
|---|---|:---:|:---:|:---:|
| Lookup | economic | 64.77 | 0.0113 | 5,723 |
| ML | economic | 64.41 | 0.0092 | 7,030 |
| **LLM** | economic | 63.82 | **0.0084** | **7,583** |
| Lookup | premium | 76.84 | 0.0454 | 1,693 |
| ML | premium | 79.13 | 0.0421 | 1,878 |
| **LLM** | premium | **82.07** | **0.0388** | **2,114** |

### Sürpriz Bulgu — End-to-End > Lookup

**LLM ve ML router lookup'tan daha iyi performans veriyor**, hem kalite hem maliyet açısından. Premium modda LLM Pareto-dominant: en yüksek kalite (82.07) **ve** en düşük maliyet ($0.0388).

**Açıklama (kategorizer hata zinciri):**
Lookup pipeline'ı `prompt → kategorizer (LR %83.6) → track → tablo → model` şeklinde. Kategorizer'ın %16.4 hatası "yanlış lookup hücresi" demek; o hücredeki model, prompt'un gerçek track'inde optimal değil. Track tahmin cezası 1-1.6 puan (Gün 12 bulgusu).

ML ve LLM router track tahmin adımını atlıyor — doğrudan `(prompt, mod) → model`. Kategorizer hata zinciri kırılıyor.

### Concentration Trade-Off — Çeşitlilik vs Performans

LLM premium modda 280 kararın **133'ünde Gemini2.5-Flash** seçiyor (%48). ML 87 (%31). Lookup 56 (%20).

| Mode | Lookup top-3 | ML top-3 | LLM top-3 |
|---|---|---|---|
| eco | FLUX-schnell (59), SD1.5 (22), SDXL (20) | FLUX-schnell (81), FLUX-dev (20), SD1.5 (19) | FLUX-schnell (74), SDXL (40), FLUX-dev (21) |
| premium | Gemini (56), SD3.5-L (24), SDXL (22) | Gemini (87), SDXL (17), FLUX-schnell (13) | **Gemini (133)**, FLUX-schnell (4), SD3.5-L (2) |

LLM premium modu pratik olarak **"always Gemini"**. Yüksek skor alıyor ama genuine routing değil — concentration. Lookup track-aware seçim yapıyor (7 farklı model dağılmış), kategorizer hatası onu yaviş vuruyor.

### Tezde Kullanım

**3 router'ın doğru kıyaslama mesajı:**
- **Çeşitlilik:** Lookup ≫ ML > LLM
- **Kalite (oracle):** LLM > ML > Lookup
- **Maliyet:** LLM = ML < Lookup
- **Latency:** Lookup = ML ≪ LLM (26x)
- **Açıklanabilirlik:** Lookup ≫ ML ≈ LLM (track tahmini transparan)

**Tezdeki ana iddia revize:**
> Track-aware lookup router en yüksek **çeşitlilik** ve **açıklanabilirlik** sunar. End-to-end ML/LLM router daha **yüksek kalite-maliyet etkinliği** sunar ama concentration eğilimi taşır. Production seçimi kullanıcı önceliğine bağlı: explainable+diverse → Lookup; cost-quality optimal → ML (lokal+ücretsiz) ya da LLM (API maliyeti kabul edilirse).

### Çıktılar
- [`data/processed/three_router_eval.csv`](three_router_eval.csv) — 840 karar (3 router × 280)
- [`figures/three_router_comparison.png`](../../figures/three_router_comparison.png) — kalite + maliyet 6 strateji yan yana
- [`src/router/ml_router.py`](../../src/router/ml_router.py), [`src/router/llm_router.py`](../../src/router/llm_router.py)
- [`models/ml_router_lr.pkl`](../../models/ml_router_lr.pkl) — train %82.9, test %74.3, gap 8.7
