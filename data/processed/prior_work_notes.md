# Prior Work Notları — T2I Routing Literatürü

**Tarama tarihi:** 2026-04-26
**Yöntem:** Web araması (4 sorgu), arXiv/openaccess.thecvf.com fetch'leri
**Tezin pozisyonu açısından net:** Çakışan hiçbir çalışma bulunmadı. Var olanlar **dar kapsamlı** (binary routing, complexity-based) — bizim **capability profili tabanlı multi-model routing** açıkta.

---

## En Yakın Çalışma: RouteT2I (ICCV 2025)

**Künye:** Xin et al., *Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model*, ICCV 2025
**URL:** https://openaccess.thecvf.com/content/ICCV2025/papers/Xin_Adaptive_Routing_of_Text-to-Image_Generation_Requests_Between_Large_Cloud_Model_ICCV_2025_paper.pdf

### Ne yapıyor
- Prompt'a göre **iki T2I modeli** arasında dinamik seçim: "büyük cloud modeli" vs "hafif edge modeli"
- Decision model prompt özelliklerini, ağ koşullarını, edge cihaz kapasitesini değerlendirir
- Latency + cost azaltma odaklı

### Bizim tezden farkı (gap)
| Boyut | RouteT2I | Bu tez |
|---|---|---|
| Model sayısı | **2** (cloud vs edge) | 9-12 (heterojen havuz) |
| Karar boyutu | Sadece "cloud yeter mi?" (binary quality threshold) | Capability profili × maliyet × kullanıcı modu |
| Routing temeli | Prompt karmaşıklığı + sistem koşulları | Track skorları (PRISM 7 boyutu) + Pareto |
| Yöntem karşılaştırması | Tek decision model | 3 yaklaşım (lookup, ML embedding, LLM few-shot) |
| Dataset | Kendi prompt seti | HEIM + PRISM-Bench akademik benchmark'lar |

**Çıkarım:** RouteT2I ile çakışma yok; bu tez **capability-aware multi-model T2I routing** boşluğunu doldurur. Tez yazımında "RouteT2I'nin binary routing yaklaşımını N-model heterojen havuza genişletiyoruz, capability disaggregation ile prompt'un hangi yeteneklerden en çok yararlandığını dikkate alıyoruz" diye konumlandırılır.

---

## Metodoloji Şablonu: RouteLLM (LMSYS, 2024)

**Künye:** Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data*, arXiv 2406.18665
**Repo:** https://github.com/lm-sys/RouteLLM

### Ne yapıyor
- Strong vs weak LLM arasında prompt-bazlı routing
- Üç router yaklaşımı: **Matrix Factorization**, **BERT Classifier**, **Causal LLM Classifier**
- Tradeoff: kalite koruyup maliyet düşürmek
- MT-Bench %85, MMLU %45, GSM8K %35 cost reduction (GPT-4 baseline'a karşı, %95 quality preservation)

### Bizim tezden farkı
- LLM domain'i, T2I değil
- Strong/weak ikilisi (binary), çok-modelli değil
- Preference data (Chatbot Arena) eğitim verisi olarak kullanmış — bizim PRISM benchmark skorları muadili

### Bizim tezde nasıl kullanılır
- **Router B (ML-based)** için methodology şablon: SBERT embedding + classifier — RouteLLM'in BERT classifier'ı bunun T2I muadili
- **Router C (LLM-based)** few-shot prompting yaklaşımı RouteLLM'in causal LLM classifier'ından ilhamla
- Cost-quality Pareto presentation tarzı RouteLLM'den ödünç alınabilir
- **Tezde mutlaka atıf** olmalı

---

## Diğer İlgili Eserler

### T2I-CompBench / T2I-CompBench++ (NeurIPS 2023, TPAMI 2025)
**URL:** https://github.com/Karine-Huang/T2I-CompBench
- 6000-8000 compositional prompt, **3-4 ana kategori** (attribute binding, object relationships, generative numeracy, complex compositions)
- Evaluate-only benchmark, routing yapmıyor
- **Tezimiz için**: PRISM-Bench yetersiz kalırsa **kategorizasyon validasyonu** için ek kaynak. PartiPrompts'a alternatif/tamamlayıcı olabilir.

### PartiPrompts (Google, 2022)
**URL:** Parti paper, https://gweb-research-parti.web.app/parti_paper.pdf
- 1600 İngilizce prompt, **12 kategori × 11 challenge boyutu**
- Tezde **yan validasyon kaynağı** olarak halihazırda planlı

### HEIM, PRISM-Bench
- Daha önce kapsamlı incelendi, [data/processed/heim_model_inventory.md](heim_model_inventory.md), [prism_model_inventory.md](prism_model_inventory.md)

### T2I-ReasonBench (2025)
**URL:** https://arxiv.org/html/2508.17472
- Reasoning-informed T2I değerlendirmesi
- Henüz kapsamlı bakılmadı, gerekirse gün 4'te

---

## Tezin Katkı Pozisyonu (Final Form)

Mevcut katkı dört maddesi (bitirme.md'den):
1. HEIM + PRISM uyumlandırılması → **PRISM-primary master tablo, HEIM tarihsel** olarak revize
2. Uçtan uca routing mimarisi → RouteT2I'nin binary'sini genişletme olarak çerçevele
3. Üç router yaklaşımı karşılaştırması → RouteLLM şablonuyla, T2I domain'ine taşıma
4. LOCO-CV genelleme → mevcut literatürde T2I için yapılmamış; özgün katkı

**Yeni eklenebilecek 5. katkı:** **Capability-aware (multi-track) routing**, bottleneck-min aggregation. Önceki çalışmaların hiçbiri capability profili disaggregation yapmıyor — bu özgün bir konumlandırma.
