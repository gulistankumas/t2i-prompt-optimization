# Router Bulguları — Hafta 2 Özeti

**Son güncelleme:** Gün 13 sonu
**Test seti:** PRISM-Bench 140 prompt (her track 20)
**Değerlendirme:** Her prompt × 2 mod = 280 routing kararı

---

## 1. Mimari Özet

Bu çalışmada üç router yaklaşımı geliştirilmiş ve karşılaştırılmıştır:

### 1.1 Lookup Router (iki aşamalı)
- **Adım 1:** ML Kategorizer (LR + SBERT) → prompt → PRISM track tahmini
- **Adım 2:** Lookup tablosu (track × mod → model) deterministic sorgu
- **Eğitim:** Kategorizer 560 PRISM train prompt'u üzerinde eğitildi
- **Veri:** Lookup tablosu master_final.csv'den hesaplandı (Gün 8)

### 1.2 ML Router (tek aşamalı, end-to-end)
- **Pipeline:** prompt + mod → SBERT embedding (384) + mod one-hot (2) → 386 boyut → Logistic Regression → 8-sınıflı model tahmini
- **Eğitim:** 1120 örnek (560 train prompt × 2 mod)
- **Etiket kaynağı:** Her prompt'un track'inde lookup tablosundaki optimal model

### 1.3 LLM Router (zero/few-shot)
- **API:** GPT-4o-mini (OpenAI)
- **System prompt:** Lookup tablosu kuralları (track × mod → model haritası)
- **Few-shot:** 8 örnek (her track için en az 1, mod karışık)
- **Eğitim:** Yok, in-context learning

---

## 2. Sayısal Performans Karşılaştırması

### 2.1 Ana metrikler (140 test × 2 mod = 280 karar)

| Router | Mode | Avg Quality | Avg Cost | Q/$ Ratio | Latency |
|--------|------|-------------|----------|-----------|---------|
| Lookup | economic | 64.77 | $0.0113 | 5,723 | 32ms |
| ML | economic | 64.41 | $0.0092 | 7,030 | 33ms |
| LLM | economic | 63.82 | $0.0084 | 7,583 | 855ms |
| Lookup | premium | 76.84 | $0.0454 | 1,693 | 32ms |
| ML | premium | 79.13 | $0.0421 | 1,878 | 33ms |
| LLM | premium | 82.07 | $0.0388 | 2,114 | 855ms |

### 2.2 Beklenen vs gerçek kalite farkı

Lookup router için (kategorizer hatasından kaynaklı):
- Economic: beklenen 66.36, gerçek 64.77 → fark **-1.59 puan**
- Premium: beklenen 77.84, gerçek 76.84 → fark **-1.00 puan**

Bu fark, kategorizer'ın %16.4 hata oranı tarafından açıklanmaktadır.

### 2.3 Sayısal performans yorumu

LLM router her iki modda da **en yüksek Q/$ ratio** veriyor görünüyor. Ancak Bölüm 3'te gösterileceği gibi, bu **gerçek routing'den ziyade akıllı default davranışından** kaynaklanıyor.

---

## 3. Karar Çeşitlilik Analizi

### 3.1 Premium modda model seçim dağılımı (140 karar)

| Router | En sık 3 model | Aktif model sayısı |
|--------|----------------|---------------------|
| Lookup | Gemini(56), SD3.5-Large(24), SDXL(22) | 5+ model |
| ML | Gemini(87), SDXL(17), FLUX.1-schnell(13) | 3 model dominant |
| LLM | Gemini(133), FLUX.1-schnell(4), SD3.5-Large(2) | %95 monoseçim |

### 3.2 Economic modda model seçim dağılımı (140 karar)

| Router | En sık 3 model |
|--------|----------------|
| Lookup | FLUX.1-schnell(59), SD1.5(22), SDXL(20) |
| ML | FLUX.1-schnell(81), FLUX.1-dev(20), SD1.5(19) |
| LLM | FLUX.1-schnell(74), SDXL(40), FLUX.1-dev(21) |

### 3.3 LLM monoseçim bulgusu — kritik

LLM premium modda 140 kararın 133'ünde (%95) Gemini 2.5 seçti.

**"Always Gemini 2.5" baseline karşılaştırması:**
- Always Gemini: kalite 82.85, maliyet $0.039
- LLM premium: kalite 82.07, maliyet $0.0388
- **Fark: 0.78 puan kalite, $0.0002 maliyet**

LLM router prompt'a göre farklılaştırma yapmak yerine "premium = Gemini" akıllı default'una yakınsadı. Few-shot prompting, lookup tablosundaki track-mode etkileşimlerini öğretmek için **yetersiz** kalmıştır.

### 3.4 Çeşitlilik tezdeki anlamı

Sayısal performans hiyerarşisi: **LLM > ML > Lookup**
Karar çeşitliliği hiyerarşisi: **Lookup > ML > LLM**

Bu **ters orantı**, "router etkinliği"nin sadece sayısal metriklerle değerlendirilemeyeceğini gösterir.

---

## 4. Karar Uyumluluk Analizi

### 4.1 Pairwise uyumluluk (280 karar üzerinden)

| Karşılaştırma | Economic | Premium | Genel |
|---------------|----------|---------|-------|
| ML vs Lookup | 76.4% | 72.1% | 74.3% |
| LLM vs Lookup | 52.1% | 39.3% | 45.7% |
| ML vs LLM | 51.4% | 60.7% | 56.1% |
| Üçü birden uyumlu | - | - | 40.7% |

### 4.2 Yorum

**ML router lookup'ın yaklaşık öğrenilmiş hali:** %74.3 uyum, ML router'ın lookup mantığını büyük ölçüde öğrendiğini gösteriyor. Kalan %25.7 ayrılma, ML router'ın majority bias'ı (popüler modellere kayma) nedeniyle.

**LLM router farklı strateji:** Lookup ile sadece %45.7 uyum. LLM lookup'ın iç mantığını **takip etmiyor**, kendi "akıllı default" stratejisini geliştiriyor.

**Premium modda divergence artıyor:** ML-Lookup uyumu 76.4%'ten 72.1%'e düşüyor, LLM-Lookup uyumu 52.1%'den 39.3%'e düşüyor. Premium modda her iki end-to-end yaklaşım da Gemini'ye kayıyor, ortak bias'a sahip oluyorlar (ML-LLM uyumu premium'da %60.7, economic'te %51.4).

**Üçü birden uyum %40.7:** Bu, üç router'ın **gerçekten farklı routing stratejileri** öğrendiğinin kanıtı. Eğer hepsi %95 uyumlu olsaydı, 3 router gereksiz olurdu. %40.7 uyum karşılaştırmanın akademik anlamlılığını doğruluyor.

---

## 5. Baseline Karşılaştırması (Lookup Router)

### 5.1 Sabit stratejilerle karşılaştırma

| Strateji | Kalite | Maliyet | Q/$ Ratio |
|----------|--------|---------|-----------|
| Always Cheapest (SD1.5) | 45.14 | $0.0023 | 19,626 |
| **Lookup Economic** | **64.77** | **$0.0113** | **5,723** |
| Always Popular (FLUX.1-dev) | 71.10 | $0.0300 | 2,370 |
| Random | 70.44 | $0.0492 | 1,432 |
| **Lookup Premium** | **76.84** | **$0.0454** | **1,693** |
| Always Premium (GPT-Image-1) | 83.50 | $0.1670 | 500 |

### 5.2 Maliyet tasarrufu hesabı

**Lookup Economic vs Always Popular:**
- Kalite korunması: 64.77/71.10 = **%91**
- Maliyet tasarrufu: ($0.0300 - $0.0113)/$0.0300 = **%62**

**Lookup Premium vs Always Premium:**
- Kalite korunması: 76.84/83.50 = **%92**
- Maliyet tasarrufu: ($0.1670 - $0.0454)/$0.1670 = **%73**

### 5.3 Tezdeki konum

Bu bulgular RouteLLM (Ong et al., 2024) çalışmasındaki "%85 maliyet tasarrufu, %95 kalite korunması" sonuçlarıyla **kavramsal olarak paralel**. Bu çalışma LLM domain'inden T2I domain'ine taşıma yaparak benzer trade-off'ların **görüntü üretimi alanında da** mümkün olduğunu doğruluyor.

---

## 6. Methodological Findings

### 6.1 Kategorizer karşılaştırması (Gün 9-11)

| Yaklaşım | Test Acc | Train Acc | Gap | Latency | Maliyet |
|----------|----------|-----------|-----|---------|---------|
| Rule-based v2 | 37.1% | - | - | <1ms | $0 |
| **LR + SBERT** | **83.6%** | 93.8% | 10.2 pt | <5ms | $0 |
| XGBoost + SBERT | 80.0% | 100% | 20.0 pt | <5ms | $0 |
| LLM (GPT-4o-mini) | 72.9% | - | - | 500ms | ~$0.30/1K |

### 6.2 Sürpriz bulgu: yerel ML, LLM'i geçti

**LR + SBERT (%83.6) > LLM (%72.9), 10.7 puan fark**

Per-track analizi:
- LLM güçlü: long_text (100%), text_rendering (95%), imagination (100%), affection (100%)
- LLM zayıf: composition (45%), entity (25%), style (45%)
- LR güçlü: style (100%), affection (100%), imagination (95%)
- LR zayıf: composition (70%), entity (70%)

### 6.3 LLM kategorizer'ın uzunluk önyargısı

LLM toplam 33 prompt'u long_text'e atadı:
- 20 gerçek long_text (doğru)
- 13 yanlış aktarım (composition: 3, entity: 4, style: 6)

**LLM, prompt uzunluğunu semantik içerikten önce baz alarak sistematik hata yapıyor.** Bu, few-shot prompting'in alana özgü track tanımlarını yakalamada yetersizliğinin somut göstergesi.

### 6.4 Overfit kontrolü (kategorizer)

- LR: train 93.8%, test 83.6% → 10.2 puan gap (sağlıklı)
- XGBoost: train 100%, test 80.0% → 20.0 puan gap (overfit)

**Yorum:** Sınırlı veri (560 örnek) + yüksek boyutlu embedding (384 boyut) kombinasyonunda lineer modeller daha güvenli seçim. Ağaç tabanlı XGBoost dense embedding'lere uygun değil.

### 6.5 Akademik bulgu cümlesi

> "Few-shot LLM yaklaşımı, alana-özgü benchmark'ın iç dilini yakalamada, benchmark verisiyle eğitilmiş yerel ML modelinden daha az başarılıdır."

---

## 7. Tasarım Kararları (Veri-Driven)

### 7.1 3 modtan 2 moda geçiş (Gün 8)

İlk plan: 3 mod (düşük ≥45, orta ≥65, yüksek ≥80)

Master tablodan kontrol edildiğinde:
- imagination max: 76.5 (80 imkansız)
- text_rendering max: 75.7 (80 imkansız)
- long_text max: 79.8 (80 kıl payı imkansız)

**3 track'te yüksek mod ulaşılamaz** olduğu için 2 mod sistemi tasarlandı:
- Ekonomik (≥60): tüm track'lerde uygulanabilir
- Premium (≥75): tüm track'lerde uygulanabilir (long_text 79.75 ile zorla)

### 7.2 Best-effort fallback mekanizması

Lookup tablosu her hücreyi doldurabildi (14/14), fallback gerekmedi. Ama mimari olarak ekleniyor: gelecekte yeni track eklenirse veya eşik yükseltilirse fallback devreye girer.

### 7.3 Sürpriz lookup seçimleri

İki dikkat çekici sonuç:

**Premium affection → SDXL ($0.0046):**
SDXL premium fiyatlı bir model olmamasına rağmen affection track'inde 75.3 puan ile eşiği zorla geçer. Track-bazlı specialist davranışın premium segmente uzanması.

**Premium composition → FLUX.1-schnell ($0.003):**
En ucuz model premium kalite eşiğini karşılıyor (77.45 puan). "Pahalı = iyi" varsayımı çürütüldü; track-bazlı performans segment etiketinden önemli.

---

## 8. Demo Tasarımı için Kararlar

### 8.1 Ana router: Lookup

**Sebepler:**
- Karar çeşitliliği yüksek (5+ model aktif)
- Açıklanabilir (track + mod → model, her iki adım net)
- Hızlı (32ms)
- Ücretsiz (yerel)
- Demo'da kullanıcı deneyimi için kritik

### 8.2 Yan karşılaştırma: ML + LLM

Demo'da kullanıcı dropdown'dan seçebilir:
- "Lookup Router" (default)
- "ML Router"
- "LLM Router"

Aynı prompt için 3 router'ın farklı önerileri görülebilir, akademik karşılaştırma görselleşir.

### 8.3 Demo'da fallback uyarısı

Eğer gelecekte fallback tetiklenirse: "Bu kombinasyonda 75+ kalite eşiği zorlanıyor, en yakın model önerildi" mesajı.

---

## 9. Tezdeki Konum (Bölümlere Eşleme)

| Tez Bölümü | Bu Belgedeki Bölüm |
|-------------|---------------------|
| 3. Metodoloji | 1. Mimari Özet |
| 4. Deneyler | 2. Sayısal Performans |
| 5.1 Kategorizer Sonuçları | 6. Methodological Findings |
| 5.2 Router Karşılaştırması | 2. Sayısal Performans + 4. Uyumluluk |
| 5.3 Karar Çeşitliliği | 3. Çeşitlilik Analizi |
| 5.4 Baseline Karşılaştırması | 5. Baseline Karşılaştırması |
| 6.1 Tasarım Kararları | 7. Tasarım Kararları |
| 6.2 LLM Sınırlılıkları | 3.3 + 6.3 |
| 6.3 Demo İmplementasyonu | 8. Demo Tasarımı |

---

## 10. OOD Generalization Testi (Gün 14)

### 10.1 Setup

- **In-distribution (5 track):** affection, composition, imagination, style, text_rendering
- **OOD (2 track):** entity, long_text — Spearman korelasyonu en düşük çift (0.644), metodolojik olarak en zor transfer
- Kategorizer 400 prompt × 5 track ile eğitildi (entity/long_text hiç görmedi)
- Test: 100 in-dist + 40 OOD prompt

### 10.2 OOD-LR kategorizer

- Train accuracy: %96.8 (5 sınıf, 400 örnek)
- In-dist test accuracy: %94.0 (sınıf sayısı 7→5 düştüğünden tüm sistem accuracy'sinin üstünde)

### 10.3 OOD prompt tahmin dağılımı

Router OOD prompt'larını yakın in-dist track'lere maple yebiliyor:

| Gerçek track | En sık tahmin | Dağılım |
|---|---|---|
| entity (20) | composition (10) | composition 10, style 6, text_rendering 2, imagination 2 |
| long_text (20) | style (9) | style 9, composition 5, text_rendering 4, affection 1, imagination 1 |

Mantıklı eşlemeler: entity prompt'ları sahne tarif ettiği için composition'a, long_text prompt'ları uzun anlatı içerdiği için style'a kayıyor.

### 10.4 In-dist vs OOD kalite düşüşü

| Mode | In-dist Q | OOD Q | Düşüş | In-dist $ | OOD $ |
|---|:---:|:---:|:---:|:---:|:---:|
| economic | 65.50 | 60.12 | **−5.38** | $0.0132 | $0.0098 |
| premium | 76.55 | 65.88 | **−10.67** | $0.0457 | $0.0423 |

Premium modda OOD düşüşü 2x daha büyük (track-spesifiklik premium'da daha kritik). OOD maliyeti **daha düşük** çünkü yanlış mapping FLUX.1-schnell gibi ucuz modellere yönlendiriyor.

### 10.5 OOD track bazında detay

| Track × Mode | Kalite | Maliyet | En sık model |
|---|:---:|:---:|---|
| entity / eco | 61.23 | $0.0089 | FLUX.1-schnell (16/20) |
| entity / premium | **68.58** ❌ | $0.0326 | FLUX.1-schnell (10/20) |
| long_text / eco | **59.00** ❌ | $0.0107 | FLUX.1-schnell (14/20) |
| long_text / premium | **63.17** ❌ | $0.0521 | SD3.5-Large (9/20) |

❌ = eşik (60 eko / 75 premium) **karşılanmadı**. 4 OOD hücresinin 3'ünde kullanıcıya verilen kalite garantisi tutmuyor.

### 10.6 Ek bulgu: in-distribution track içi varyans

In-distribution track'ler arasında bile performans dağılımı geniştir:
- Economic in-dist: text_rendering 56.8 (eşik altı), composition 76.9 (eşik üstü)
- Premium in-dist: text_rendering 73.4 (eşik altı), composition 81.0 (eşik üstü)

Bu, router'ın track-doğru tahmin etse bile lookup tablosundaki modelin o track'te yapısal sınırı olduğunu gösterir. T2I modellerinin text_rendering ve long_text gibi track'lerdeki zayıflıkları **"OOD problemi" değil, "yapısal kısıt"tır**.

### 10.7 Tezde kullanım

> "Router 5 in-dist track ile eğitildiğinde, görmediği 2 track'te (entity + long_text) ortalama 5.4 puan (eko) ve 10.7 puan (premium) kalite düşüşü gösterir. Premium modda 4 OOD hücresinin 3'ü kalite eşiğini karşılayamaz. Bu, router'ın **kategori-bağımlı genelleme sınırlarını** somutlaştırır. Ek olarak, in-dist track'ler arasında bile yapısal varyans (text_rendering 56-73 vs composition 76-81) gözlenmiştir; T2I model uzayının track-spesifik darboğazları routing'in üst sınırını koymaktadır — bu sınır kategorizer doğruluğundan bağımsızdır."

### 10.8 Çıktılar
- [`notebooks/21_ood_test.py`](../../notebooks/21_ood_test.py)
- [`data/processed/ood_test_eval.csv`](ood_test_eval.csv) — 80 OOD karar (40 prompt × 2 mod)
- [`figures/ood_test_results.png`](../../figures/ood_test_results.png) — eko + premium yan yana, in-dist mavi / OOD kırmızı

---

## Hafta 3 İçin Bekleyen Analizler

- **Latency-quality trade-off detayı:** Router seçim kriteri olarak (Hafta 3 başı)
- **Capability-min routing (opsiyonel):** Yedek hatırlatma mevcut, check-point sonrası karar
- **Streamlit demo entegrasyonu:** Hafta 3 ana iş

---

## 11. Açık Sorular ve Sınırlılıklar

1. **PRISM track-bazlı veri:** Router her prompt için **ortalama track skoru** kullanıyor (prompt-bazlı skor PRISM'de yok). Bu, router'ın varsayımsal "oracle" senaryoda performansını yansıtıyor; gerçek dünyada prompt-içi varyans olabilir.

2. **8 model havuzu:** PRISM'in 19 modelinden seçilen 8 model. Daha geniş havuz (Stable Cascade, Playground v2 vs.) test edilmedi.

3. **Tek dil (İngilizce):** Tüm prompt'lar İngilizce. Çok dilli routing test edilmedi.

4. **2026 modelleri:** GPT-Image-2 veya yeni Gemini sürümleri test edilmedi (PRISM benchmark Eylül 2025 sürümü).

5. **Cost API fiyatları sabit:** Replicate ve API fiyatları zaman içinde değişebilir; bu çalışma Mart 2026 fiyatlarını baz alıyor.

---

**Belge sonu.**
