# Literatür Notları

**Tez:** T2I Modelleri için Prompt Tabanlı Otomatik Router Sistemi
**Tarama tarihi:** Nisan 2026
**Not:** Bu dosya tezin Bölüm 2 (Literatür) yazımı için hazırlık materyalidir.

---

## 1. PRISM-Bench (Fang et al., 2025) — ANA VERİ KAYNAĞI

**Künye:**
- Başlık: FLUX-Reason-6M & PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark
- Yazarlar: Fang, Yu, Duan, Huang, Bai, Cai, Wang, Liu, Liu, Li
- Kurum: CUHK, HKU, BUAA, Alibaba, SenseTime
- arXiv: 2509.09680
- Yıl: 2025
- Yer: ICLR 2026'ya kabul edildi

**Ne yapıyor:**
İki ayrı katkı sunuyor. (1) FLUX-Reason-6M: 6 milyon görüntülü dataset, T2I reasoning kapasitesi geliştirmek için. (2) PRISM-Bench: 19 modern T2I modelinin 7 farklı kapasite track'inde değerlendirildiği benchmark.

**PRISM-Bench detayları:**
- 7 track: imagination, entity, text rendering, style, affection, composition, long_text
- Her track'te 100 prompt, toplam 700 prompt
- İki bağımsız LLM jürisi: GPT-4.1 ve Qwen2.5-VL-72B
- Her görüntü 2 boyutta puanlanır: Alignment + Aesthetic
- Long_text track'i GCoT (Generation Chain-of-Thought) formatında uzun prompt'lar içerir

**Tezde kullanım:**
- Bölüm 2.2: T2I değerlendirme benchmarkları
- Bölüm 3.1: Veri kaynağı detaylandırma
- Bölüm 6.3: LLM-as-judge bias tartışması (iki jüri kullanımının önemi)

**Atıf cümlesi:**
> "PRISM-Bench (Fang et al., 2025), 19 modern T2I modelinin 7 farklı kapasite track'i üzerinde değerlendirildiği bir benchmark sunar. Her görüntü iki bağımsız LLM jürisi (GPT-4.1 ve Qwen2.5-VL-72B) tarafından prompt-uyum ve estetik boyutlarında puanlanır. Bu çalışmanın ana veri kaynağı PRISM-Bench'tir."

---

## 2. HEIM (Lee et al., 2023) — REFERANS BENCHMARK

**Künye:**
- Başlık: Holistic Evaluation of Text-to-Image Models
- Yazarlar: Lee et al. (Stanford CRFM)
- arXiv: 2311.04287
- Yıl: 2023
- Yer: NeurIPS 2023

**Ne yapıyor:**
26 T2I modelini 12 farklı boyutta (alignment, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality, efficiency, derivative works) sistematik değerlendirme.

**Tezde kullanım:**
- Bölüm 2.2: T2I benchmark literatürü içinde referans
- Tezin veri kaynağı değil, sadece referans gösterilir

**Neden HEIM yerine PRISM:**
- HEIM 2023, PRISM 2025 (modern modeller)
- HEIM'de FLUX, GPT-Image-1, Qwen-Image yok
- HEIM çok-protokollü, PRISM tek protokol (tutarlılık)

**Atıf cümlesi:**
> "HEIM (Lee et al., 2023), T2I değerlendirmede önemli bir referans benchmark olmakla birlikte modern (2024-2025) modelleri kapsamamaktadır. Bu nedenle bu çalışmada PRISM-Bench tercih edilmiştir."

---

## 3. RouteLLM (Ong et al., 2024) — METODOLOJİ ŞABLONU

**Künye:**
- Başlık: RouteLLM: Learning to Route LLMs with Preference Data
- Yazarlar: Ong, Almahairi, Wu, Chiang, Wu, Gonzalez, Kadous, Stoica
- Kurum: UC Berkeley + Anyscale (LMSYS)
- arXiv: 2406.18665
- Yıl: 2024
- GitHub: lm-sys/RouteLLM

**Ne yapıyor:**
İki LLM (güçlü ve zayıf) arasında prompt-bazlı routing yapan ilk sistematik framework. 4 router yaklaşımı sunar: Similarity-Weighted Ranking, Matrix Factorization, BERT Classifier, Causal LLM Classifier. MT-Bench'te %85, MMLU'da %45 maliyet tasarrufu raporlar (kalitenin %95'ini koruyarak).

**Önemli bulgu:**
Transfer learning kapasitesi: Router GPT-4 + Mixtral üzerinde eğitildi, Claude 3 Opus + Llama 3 8B üzerinde yeniden eğitilmeden test edildi, performans korundu. Yani router'ın öğrendiği şey model-bağımsız prompt karakteristiği.

**Tezdeki karşılığı:**
- ML router (SBERT + XGBoost) → RouteLLM'in BERT Classifier yaklaşımının T2I uyarlaması
- LLM router (few-shot prompting) → RouteLLM'in Causal LLM Classifier yaklaşımının T2I uyarlaması
- Lookup router → RouteLLM'de yok, senin orijinal eklen

**Senin tezdeki preference data karşılığı:**
RouteLLM insan tercih verisi kullanır (Chatbot Arena), senin tezde PRISM-Bench skorları kullanılır.

**Tezde kullanım:**
- Bölüm 2.3: LLM routing literatürü ana kaynağı
- Bölüm 3.5: Router yaklaşımlarının metodoloji şablonu
- Bölüm 5.5: Sonuç karşılaştırma referansı

**Atıf cümlesi:**
> "RouteLLM (Ong et al., 2024), LLM domain'inde iki model arasında prompt-bazlı routing yapan ilk sistematik framework'tür. Bu çalışmadaki ML ve LLM router yaklaşımları, RouteLLM'in BERT Classifier ve Causal LLM Classifier yaklaşımlarının T2I domain'ine uyarlanmış halleridir."

---

## 4. FrugalGPT (Chen et al., 2023) — EŞİK MANTIĞI REFERANSI

**Künye:**
- Başlık: FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance
- Yazarlar: Lingjiao Chen, Matei Zaharia, James Zou
- Kurum: Stanford
- arXiv: 2305.05176
- Yıl: 2023

**Ne yapıyor:**
Cascading routing yöntemi: prompt'u önce ucuz LLM'e gönder, çıktının kalitesini değerlendir, eşiğin altındaysa daha güçlü ve pahalı LLM'e yönlendir. Maliyet eşiği ve kalite eşiği üzerinden optimize.

**Senin tezinden ana farkı:**
- FrugalGPT: Post-hoc karar (cevap üretildikten sonra değerlendirilir)
- Senin tezin: A priori karar (cevap üretilmeden önce model seçilir)
- T2I bağlamında a priori daha verimli, çünkü her başarısız üretim $0.003-$0.167 maliyet demek

**Tezdeki kullanım:**
- Bölüm 2.3: Threshold-based routing'in akademik referansı
- Senin "düşük/orta/yüksek kalite eşiği" mod sisteminin kavramsal atası

**Atıf cümlesi:**
> "FrugalGPT (Chen et al., 2023) cascading yaklaşımı kullanır: ucuz modeli dener, çıktının kalitesini değerlendirir, yetersizse daha güçlü modele yönlendirir. Bu yöntem post-hoc karar verir. Bu tezdeki router ise a priori karar verir: prompt'u görüntü üretmeden önce analiz eder ve doğrudan uygun modele gönderir. T2I bağlamında bu yaklaşım, başarısız üretim maliyetinden kaçınır."

---

## 5. RouteT2I (Xin et al., 2025) — EN YAKIN ÇAKIŞAN İŞ ⚠️

**Künye:**
- Başlık: Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model
- Yazarlar: Xin et al.
- Yıl: 2025
- Yer: ICCV 2025

**Ne yapıyor:**
T2I generation isteklerini iki seçenek arasında dinamik olarak yönlendiren sistem: Cloud'daki büyük model (yüksek kalite, yüksek maliyet) vs Edge cihazdaki küçük model (düşük kalite, ücretsiz). Decision model prompt karmaşıklığı + ağ koşulları + edge cihaz yükünü değerlendirir.

**Senin tezinden 3 ana fark:**

| Boyut | RouteT2I | Bu tez |
|-------|----------|--------|
| Model sayısı | 2 (cloud vs edge) | 8 model heterojen havuz |
| Karar boyutu | Binary (deployment kararı) | Capability profili × maliyet × kalite eşiği |
| Yöntem çeşitliliği | Tek decision model | 3 router yaklaşımı (lookup, ML, LLM) |
| Veri kaynağı | Kendi prompt seti | Akademik benchmark (PRISM-Bench) |
| Odak | Donanım/deployment | Akademik model karşılaştırması |

**Jüri savunmasında ana cevap:**
> "RouteT2I problemi sadece cihaz maliyetleri ve iki model üzerinden çözen öncü bir çalışmadır. Bu tez ise 8 model ve 3 farklı algoritma ile problemi donanım limitlerinden çıkarıp doğrudan model kapasitelerine odaklandığı için literatürdeki bu boşluğu doldurur."

**Tezdeki kullanım:**
- Bölüm 2.4: En yakın çakışan iş, dürüstçe konumlandırma
- Bölüm 6.1: Sonuçların RouteT2I ile karşılaştırması (eğer benzer metrik varsa)

**Atıf cümlesi:**
> "RouteT2I (Xin et al., 2025), T2I generation isteklerini cloud ve edge model arasında dinamik olarak yönlendiren öncü bir çalışmadır. Bu çalışma RouteT2I'nin binary routing yaklaşımını N-modelli heterojen havuza genişletir ve capability disaggregation ile prompt'un hangi yeteneklerden en çok yararlandığını dikkate alır."

---

## Tezin Literatür Konumlandırması (Bölüm 2 ÖZET)

Bu tez üç literatür grubunun kesişiminde yer alır:

1. **T2I değerlendirme benchmarkları** (HEIM, PRISM-Bench): Veri kaynağı sağlar
2. **LLM routing** (RouteLLM, FrugalGPT): Metodoloji şablonu sağlar
3. **T2I routing** (RouteT2I): En yakın akademik komşu

**Boşluk (gap statement):** Capability-aware multi-model T2I routing, literatürde sistematik olarak çalışılmamıştır. RouteT2I binary cloud-edge ayrımı yapar; LLM routing ise farklı domain'dir. Bu tez bu boşluğu doldurur.

---

## Atıf Listesi (BibTeX için)

```bibtex
@article{fang2025prism,
  title={FLUX-Reason-6M \& PRISM-Bench: A Million-Scale Text-to-Image Reasoning Dataset and Comprehensive Benchmark},
  author={Fang, Lucas and others},
  journal={arXiv preprint arXiv:2509.09680},
  year={2025}
}

@article{lee2023heim,
  title={Holistic Evaluation of Text-to-Image Models},
  author={Lee, Tony and others},
  booktitle={NeurIPS},
  year={2023}
}

@misc{ong2024routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M. Waleed and Stoica, Ion},
  year={2024},
  eprint={2406.18665},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}

@article{chen2023frugalgpt,
  title={FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance},
  author={Chen, Lingjiao and Zaharia, Matei and Zou, James},
  journal={arXiv preprint arXiv:2305.05176},
  year={2023}
}

@inproceedings{xin2025routet2i,
  title={Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model},
  author={Xin, et al.},
  booktitle={ICCV},
  year={2025}
}
```
