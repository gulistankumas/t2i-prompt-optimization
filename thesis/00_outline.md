# Bitirme Tezi — Outline (Hafta 3'te Açılacak Çatı)

**Tez:** T2I Modelleri için Prompt Tabanlı Otomatik Router Sistemi
**Tahmini sayfa:** 35-45
**Yazım tarihi:** 2026-04-29 (Gün 6)

> Her alt bölüm bir paragraf — içeride hangi bulgu/argüman/şekil olacağını söylüyor.
> Hafta 3'te bunlar genişletilecek. Yapı sabit, içerik açık.

---

## Ön Materyal

- **Türkçe Özet (1 sayfa):** Problem, yöntem, ana bulgular, katkılar.
- **İngilizce Özet (1 sayfa):** Türkçe özetin literal çevirisi değil, akıcı İngilizce.
- **İçindekiler / Şekil ve Tablo Listesi / Sembol-Kısaltma Listesi**

---

## Bölüm 1 — Giriş (3-4 sayfa)

### 1.1 Problem Tanımı
T2I (text-to-image) modellerinin sayısı son 3 yılda hızla arttı; her birinin kalite, maliyet ve yetenek profili farklı. Kullanıcı bir prompt için "hangi modeli kullanmalıyım" sorusuna cevap verirken seçim yapacağı modeller arasında trade-off'ları manuel değerlendirmek pratik değil. Bu bölümde T2I model seçim sorununun maliyet-kalite ekseninde formüle edilmesi, prompt'un *capability profile*'ı kavramı ve görsel üretmeden önce karar verme zorunluluğu (latency + cost) tartışılacak.

### 1.2 Motivasyon
Bu çalışmanın iki yönlü motivasyonu var: pratik (T2I API maliyetleri 2025-2026'da kullanıcılar için ciddi yük; en pahalı vs en ucuz model arasında 70x maliyet farkı) ve akademik (mevcut benchmark'lar [HEIM, PRISM] statik liderlik tablosu sunuyor, ancak prompt-spesifik karar mimarisi yok). RouteT2I'nin (ICCV 2025) iki-modelli binary routing'i bu boşluğu sınırlı dolduruyor; çok-modelli capability-aware routing açıkta.

### 1.3 Araştırma Sorusu
*"Akademik T2I benchmark'larından türetilen skor verisi ve prompt kategori sinyali, kullanıcıdan alınan doğal dil prompt'u için uygun T2I modelini görsel üretmeden önce önerecek bir routing sistemi kurmaya yeterli midir? Yeterliyse (i) lookup, (ii) ML-embedding ve (iii) LLM few-shot yaklaşımları doğruluk-hız-maliyet üçgeninde nasıl karşılaştırılır; (iv) ML yaklaşımı görülmemiş prompt kategorilerine anlamlı genelleme gösterir mi?"*

### 1.4 Katkılar
Dört çekirdek katkı: (1) HEIM ve PRISM-Bench değerlendirme verilerinin uyumlandırılması ve birleşik master skor tablosu; (2) prompt → kategori → Pareto-bazlı model tavsiyesi şeklinde uçtan uca routing mimarisi ve Streamlit demosu; (3) üç router yaklaşımının (lookup, ML-embedding, LLM few-shot) doğruluk-hız-maliyet üçgeninde karşılaştırmalı değerlendirmesi; (4) ML router'ın kategoriler arası genelleme kapasitesinin ölçümü.

*[Hafta 3 check-point'e bağlı, opsiyonel 5. katkı]:* Capability-aware (track-bazlı) routing ve "bottleneck-min" aggregation kavramının formal tanımı — ancak literatür taraması bu iddiayı netleştirdiyse ve check-point'te zaman varsa.

### 1.5 Tezin Yapısı
Bölüm 2 literatür; Bölüm 3 metodoloji; Bölüm 4 deney düzeni; Bölüm 5 sonuçlar; Bölüm 6 tartışma; Bölüm 7 sonuç. Tek paragraf, okuyucuya yol haritası.

---

## Bölüm 2 — Literatür Taraması (6-8 sayfa)

### 2.1 Text-to-Image Modelleri
Kısa tarihsel kronoloji: GAN dönemi (BigGAN, GigaGAN), diffusion dönemi (DALL-E 2, Stable Diffusion ailesi 1.5/2.x/3/3.5), modern transformer-tabanlı (FLUX, Qwen-Image), kapalı kaynak çağı (DALL-E 3, GPT-Image-1, Gemini 2.5 Flash-Image). Her ailenin **mimari farklılaşma** noktaları (UNet vs DiT vs MMDiT) bir tabloyla özetlenir.

### 2.2 T2I Değerlendirme Benchmark'ları
HEIM (Lee et al., NeurIPS 2023) — 26 model × 12 aspect, otomatik metrikler ağırlıklı. PRISM-Bench (Fang et al., ICLR 2026) — 19 model × 7 reasoning track × 700 prompt, GPT-4.1 + Qwen2.5-VL jürileri. T2I-CompBench++ (TPAMI 2025) compositional değerlendirme. Her benchmark'ın güçlü/zayıf yönleri ve bizim **PRISM'i ana kaynak seçme gerekçemiz** (modern modeller, tek protokol, LLM-judge) açıklanır.

### 2.3 LLM Model Routing
RouteLLM (Ong et al., 2024) — strong vs weak LLM routing, üç router (Matrix Factorization, BERT classifier, Causal LLM classifier). FrugalGPT (Chen et al., 2023) — cascade-bazlı maliyet düşürme. Bizim metodolojimiz RouteLLM'in BERT/Causal-LLM şablonlarını T2I domain'ine taşıyor; **LLM routing literatürünün T2I'a aktarılabilirliği** tartışılır.

### 2.4 T2I Model Selection / Routing
RouteT2I (Xin et al., ICCV 2025) — bizim tezle en yakın çakışan iş; ancak **iki-modelli (cloud vs edge) binary routing**. Bu tez 9-modelli heterojen havuz, capability-bazlı routing ve üç router yaklaşımının karşılaştırması ile RouteT2I'nin üzerine genişleme yapıyor. **Boşluk net:** çok-modelli, capability-aware T2I routing literatürde yok.

### 2.5 Pareto Optimizasyonu ve LLM-as-Judge
Çok-amaçlı optimizasyonun temel kavramları: Pareto frontier, dominated nokta, kalite-maliyet ekseni. LLM-as-judge metodolojisinin (Zheng et al., 2023) bias riskleri (self-preference, position bias, verbosity bias). PRISM'in **iki bağımsız jüri** kullanmasının önemi ve bizim tezde bias formal analizinin bu literatüre katkısı.

---

## Bölüm 3 — Metodoloji (8-10 sayfa)

### 3.1 Veri Kaynakları
PRISM-Bench veri yapısı: 19 model × 7 track (imagination, entity, text_rendering, style, affection, composition, long_text) × ~100 prompt/track × 2 jüri (GPT-4.1 + Qwen2.5-VL-72B). Her hücre Alignment + Aesthetic ortalaması (0-100 skala). HEIM benchmark'ı **karşılaştırma referansı** olarak kullanılır ama master tabloya dahil edilmez (model overlap: 9 hedef modelin 2'si). PartiPrompts (Yu et al., 2022) yan-validasyon ve LLM-kategorizer çapraz kontrolü için.

### 3.2 Model Havuzu
9 model: SD1.5, SDXL, SD3.5-Large, FLUX.1-schnell, FLUX.1-dev, Qwen-Image, DALL-E 2, GPT-Image-1, Gemini 2.5 Flash-Image. Çeşitlilik boyutları: açık/kapalı kaynak (6/3), eski/yeni (2022-2025), küçük/büyük (1B-20B), maliyet (0.0023-0.167 $/img). Segment ayrımı: **Premium** (3 model, $0.039+, kalite 77+), **Mid** (2 model), **Cheap** (3 model). DALL-E 2 PRISM'de yok; HEIM-only tarihsel referans olarak korunur.

### 3.3 Maliyet Hesaplaması
Kapalı kaynak modeller için resmi API fiyatları (DALL-E 2 $0.020, GPT-Image-1 [High] $0.167, Gemini 2.5 Flash $0.039 — 1024×1024 başına). Açık kaynak modeller için Replicate.com referans fiyatları (FLUX.1-schnell $0.003, FLUX.1-dev $0.030); SD ailesi ve Qwen-Image için H100 GPU saat başına $0.001525 üzerinden tahmini inference süresine göre hesap. Kaynak ve tahmin yöntemi `data/raw/model_costs.csv`'de belgeli.

### 3.4 Kategori Sistemi
PRISM 7 track'i **birincil eksen** (capability-tabanlı routing'in doğal dili). PartiPrompts 12 kategorisi (Abstract, Animals, Artifacts, Arts, Food&Beverage, Illustrations, Indoor/Outdoor Scenes, People, Produce&Plants, Vehicles, World Knowledge) **yan validasyon ekseni**: LLM-kategorizer çıktısı için tutarlılık kontrolü ve held-out test set. Her prompt'a tek track ataması yapılır (multi-label çoklu-yetenek analizi tezin "future work"u).

### 3.5 Router Yaklaşımları
Üç paralel implementasyon: **(A) Lookup table** — `(track, mode) → model` O(1) erişim, en basit baseline. **(B) ML-embedding** — SBERT prompt embedding + XGBoost classifier; LOCO-CV ile genelleme test edilir. **(C) LLM few-shot** — Claude Haiku/GPT-4o-mini'ye prompt + model profilleri + few-shot örnekleri verilip öneri istenir. Her yaklaşımın ayrı API'si: `route(prompt, mode) -> (model, gerekçe, beklenen kalite, beklenen maliyet)`.

### 3.6 Değerlendirme Protokolü
Test seti: held-out PartiPrompts'tan ~200 prompt × 3 mod (cheap, balanced, premium) = 600 routing kararı. **Metrikler:** (1) router accuracy — oracle (PRISM ground truth) ile çakışma; (2) maliyet tasarrufu — naif baseline'a göre %; (3) kalite korunması — oracle modelinkine göre %; (4) routing latency — ms; (5) generalization (LOCO-CV). **Baseline'lar:** naif-1 (her zaman en pahalı), naif-2 (her zaman en popüler), naif-3 (rastgele), naif-4 (her zaman segment overall lideri).

---

## Bölüm 4 — Deneyler (4-5 sayfa)

### 4.1 Deney Düzeni
Three-fold yapı: (i) **Routing performansı** — held-out PartiPrompts üzerinde 3 router × 3 mod karşılaştırması; (ii) **Generalization** — LOCO-CV ile ML router'ın görülmemiş kategorilere transfer performansı; (iii) **Robustness** — iki jüri (GPT-4.1 vs Qwen2.5-VL) altında router kararı kararlılığı. Donanım: Windows 11 + Python 3.9; ML eğitim Colab Pro T4. Reprodüksiyon: tüm random seed'ler (42), ML hyperparameter konfigürasyonu YAML'de.

### 4.2 Baseline Tanımları
4 naif baseline yukarıda. Ek olarak **2 oracle baseline:** (oracle-track) doğru track bilindiğinde her zaman track lideri seçilir; (oracle-prompt) prompt başına gerçek en iyi model seçilir (üst sınır). Router'ın oracle'a yakınlığı performans tavanını gösterir.

### 4.3 Metrikler
Sayısal tanımlar: accuracy = 1 - average rank distance (gerçek best vs router seçimi); cost_saving = 1 - (router_cost / naive-cost); quality_preservation = router_quality / oracle_quality; latency = router decision time (ms). Anlamlılık testleri: paired bootstrap (n=1000) ile %95 güven aralıkları.

### 4.4 Genelleme Testi (LOCO-CV)
Leave-One-Category-Out: her seferinde 11 PartiPrompts kategorisinde eğit, 1 kategoride test et; 12 fold ortalama. Hipotez: capability sinyali kategorilere transfer eder. Null hipotez: ML router LOCO-CV altında lookup baseline'ı yenemez. Beklenen sonuç: **boş çıkabilir** (Spearman 0.94 yüksek korelasyon → genelleme zayıf olabilir); negatif sonuç bile akademik bulgu.

---

## Bölüm 5 — Sonuçlar (8-10 sayfa)

### 5.1 Track-Bazlı Pareto Analizi
9 model × 7 track Pareto frontier (Şekil: `pareto_frontier_per_track.png`). Bulgular: SDXL **3/7 track'te Pareto-optimal** (entity, style, affection — overall'da dominated olmasına rağmen specialist), GPT-Image-1 **affection ve long_text'te dominated** (premium fiyatı her track'te justify edilmiyor), Qwen-Image **0/7 track'te Pareto-optimal**. Track-aware routing'in değer üreten alanlar somut olarak işaretlenir.

### 5.2 Track-Arası Korelasyon ve Havuz Çeşitliliği
Spearman korelasyon matrisi 8-model (ortalama 0.94) vs 19-model (ortalama 0.84) (Şekil: `track_correlation_heatmap.png`). En düşük korelasyon entity ↔ long_text = 0.64. Bulgu: **havuz seçimi yapay yüksek korelasyon yaratıyor**; track-bağımsızlık asıl sinyal 19-model havuzda görünür. Tezin metodolojik sonucu: model havuzu çeşitliliği routing değer önerisini doğrudan etkiler.

### 5.3 Segment-Bağımlı Routing Değeri
Her segment için track-aware vs overall-bazlı routing kalite kazancı (Şekil: `segment_track_distribution.png`, hesap: 03_routing_value.py). Premium ortalama 0.9 puan kazanç (trivial), Mid 1.6 puan, **Cheap 25 puana kadar** (text_rendering: FLUX.1-schnell 50.2 vs SDXL 25.4). Tez tezi: routing'in pratik değeri **maliyet-duyarlı modlarda** ortaya çıkar; premium kullanıcı zaten en pahalıyı seçer.

### 5.4 Premium Paradoksu
GPT-Image-1 ($0.167) vs Gemini 2.5 ($0.039) yan yana 7 track bar chart (Şekil: `premium_paradox.png`). Overall fark **0.65 puan**, fiyat farkı **4.3x**. Affection ve long_text'te Gemini önde. Bulgu: "Premium = en pahalı" naif kuralı yanlış; **rasyonel premium routing kararı GPT-Image-1'i bazı track'lerde tavsiye etmez**. Tezin pratik etki bulgusu.

### 5.5 Üç Router Yaklaşımı Karşılaştırması
Tablo: lookup, ML-embedding, LLM few-shot için accuracy, cost_saving, quality_preservation, latency, $ maliyeti. **[TBD — Hafta 2 sonu doldurulacak]** Beklenen sıralama (varsayım, ölçülmedi): lookup en hızlı, ML orta, LLM en yavaş ama yeni prompt'lara adaptiv. Cost-quality tradeoff matrisi gerçek deney sonrasında çizilecek.

### 5.6 Genelleme Sonuçları (LOCO-CV)
12-fold leave-one-category-out sonuçları. Yorum: yüksek track korelasyonu (Spearman 0.94) → category-level transfer kolay → ML router lookup'a yakın. **Pozitif sonuç:** ML router "yeni kategori" prompt'larında lookup'a yakın performans gösteriyor. **Negatif sonuç riski:** trivial olduğu için academic katkı zayıflıyor; bu durumda dürüstçe raporlanır.

### 5.7 İki Jüri Karşılaştırması
GPT-4.1 vs Qwen2.5-VL skor delta'sı per model + per track (Şekil: `judge_agreement.png`, analiz: `judge_bias_findings.md`). Bulgular: sıralama korelasyonu ρ ≥ 0.95, mutlak skor düzleminde sistematik bias (GPT-4.1 imagination'da +16 puan cömert, Qwen2.5-VL text_rendering'de +5 cömert). Top-1 anlaşmazlığı 4/7 track. Wilcoxon testi: GPT-Image-1 ve Playground'da p<0.05 sistematik fark. Tezde **stilistik bias** olarak nitelenir, tam self-preference olarak değil.

---

## Bölüm 6 — Tartışma (4-5 sayfa)

### 6.1 Bulguların Yorumu
Üç ana bulgunun birleştirilmesi: (1) routing değeri segment-bağımlı (cheap'te güçlü, premium'da trivial), (2) track-aware Pareto, overall-Pareto'dan farklı (SDXL specialist örneği), (3) jüri seçimi router kararını mutlak skor ölçeğinde etkiler. Bu üç bulgu birlikte **"basit overall-bazlı router yetersiz"** çıkarımına götürür; capability-aware mimari gerekçesi.

### 6.2 Kısıtlılıklar
**(1)** Veri kısıtı: 9 model × 700 prompt; ML router için marjinal. **(2)** Maliyet hesabı: açık kaynak için Replicate fiyatı kullanıldı; gerçek self-host maliyeti farklı olabilir. **(3)** Tek dil (İngilizce); Çince leaderboard kapsam dışı. **(4)** PRISM-Bench 2025 — ileride çıkacak modeller skorlanmamış. **(5)** Görsel üretmediğimiz için "gerçek dünya" değerlendirmesi yapılmadı; offline benchmark.

### 6.3 Self-Preference Bias ve Değerlendirici Seçimi
LLM-as-judge bias literatürü (Zheng et al., 2023) çerçevesinde bizim bulgular. GPT-4.1'in GPT-Image-1'i sistematik +5.6 puan yüksek skorlaması (p=0.047) yapısal bir endişe; ama aynı eğilim Playground'a da var (+7.0), yani saf vendor self-preference değil. Tartışma: tek-jüriye dayalı routing literatür standardı **yetersiz**; çift-jüri ortalaması veya jüri-spesifik kalibrasyon tavsiyesi.

### 6.4 Gerçek Dünya Uygulamaları
Production T2I servisleri (Fal.ai, Replicate, Stability API) için bu mimari direkt aktarılabilir. Endpoint başına ortalama maliyet düşürme **[TBD — gerçek sayılar Hafta 2 sonu deneyden gelecek]**. Edge case'ler: very-long prompts, multi-language, NSFW filtering. Bu tezin offline değerlendirmesi production deployment için bir **güvenli temel** sunar.

### 6.5 Gelecek Çalışmalar
**(1)** Capability-min routing: prompt'a multi-track etiket atayıp `min(track_score)` bazlı seçim. **(2)** Online learning: kullanıcı feedback'iyle router weight güncelleme. **(3)** Multi-modal genişletme: video generation modelleri (Sora, Runway). **(4)** Cost-quality tercih kişiselleştirmesi: Pareto üzerinde kullanıcı ağırlık öğrenme. **(5)** PRISM-Bench güncellemeleri ile yeni model entegrasyonu otomasyonu.

---

## Bölüm 7 — Sonuç (1-2 sayfa)

### 7.1 Özet
Tezin tek-paragraflık özeti: 9 T2I modelinin PRISM-Bench skorları üzerinde inşa edilen prompt-tabanlı router, üç yaklaşımla (lookup, ML, LLM) test edildi; track-bazlı Pareto frontier **[TBD %X kalite kaybıyla %Y maliyet tasarrufu — Hafta 2 sonu deneyden gelecek]** sağladı.

### 7.2 Ana Katkılar
Dört çekirdek katkının kısa tekrarı (Bölüm 1.4'ten); 5. opsiyonel katkı varsa eklenir.

### 7.3 Sınırlılıklar ve Öneriler
2 cümlelik kısıtlılık özeti + gelecek çalışma vurgusu.

---

## Arka Materyal

- **Kaynakça (BibTeX)** — yaklaşık 30-40 atıf
- **Ekler:** kod listeleri (router API'leri), genişletilmiş tablolar (per-prompt scoring), figür yüksek-çözünürlük versiyonları

---

## Sayfa Tahmini

| Bölüm | Sayfa |
|---|:---:|
| Ön materyal (özet, içindekiler) | 4-5 |
| 1. Giriş | 3-4 |
| 2. Literatür | 6-8 |
| 3. Metodoloji | 8-10 |
| 4. Deneyler | 4-5 |
| 5. Sonuçlar | 8-10 |
| 6. Tartışma | 4-5 |
| 7. Sonuç | 1-2 |
| Kaynakça + ekler | 3-5 |
| **Toplam** | **41-54** |

Üniversite minimum gereksinimi (kontrol edilecek): genelde 30-50 sayfa. Hedef aralık ✅.

---

## Öğrenilecekler Listesi (Hafta 2-3'te netleştirilecek)

Tezde geçecek ama henüz tam oturmamış teknik terimler. Tezde kullanmadan önce kullanıcı bunları içselleştirmeli, savunmada açıklayabilmeli. Hafta 2-3'te birlikte konuşulacak.

| Terim | Nerede geçiyor | Ne anlama geliyor (1 satır, kullanıcı doldurur) |
|---|---|---|
| **LOCO-CV** (Leave-One-Category-Out Cross-Validation) | Bölüm 3.5, 4.4, 5.6 | _____ |
| **Paired bootstrap** (n=1000, %95 GA) | Bölüm 4.3 | _____ |
| **MMDiT** (Multimodal Diffusion Transformer) | Bölüm 2.1 | _____ |
| **DiT** (Diffusion Transformer) | Bölüm 2.1 | _____ |
| **SBERT** (Sentence-BERT embedding) | Bölüm 3.5, 5.5 | _____ |
| **XGBoost** classifier | Bölüm 3.5, 5.5 | _____ |
| **Few-shot prompting** | Bölüm 3.5, 5.5 | _____ |
| **Spearman vs Pearson korelasyon** farkı | Bölüm 5.2, 5.7 | _____ |
| **Wilcoxon signed-rank test** | Bölüm 5.7, 6.3 | _____ |
| **Pareto frontier / dominated** | Bölüm 5.1 (ana iskelet) | _____ |
| **Self-preference bias / position bias** | Bölüm 6.3 | _____ |
| **Capability-min aggregation** (kendi önerimiz) | Bölüm 6.5, opsiyonel 5. katkı | _____ |
| **Cascade-based routing** (FrugalGPT) | Bölüm 2.3 | _____ |

**Süreç:**
1. Hafta 2 başında kullanıcı bu listeye 1-2 satır kendi cümlesiyle anlam yazar (Wikipedia / paper'a bakar, kendi diliyle özetler)
2. Tartışırız, kavram netleşir
3. Tezde rahat kullanılır

Eğer bir terim için "anlayamıyorum" veya "savunamam" diyorsan, **tezden çıkarılır**, alternatif terim/yöntem kullanılır.
