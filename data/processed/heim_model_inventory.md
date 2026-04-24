# HEIM Model Envanteri ve 8 Hedef Modelle Eşleştirme

**Araştırma tarihi:** 2026-04-24
**Kaynaklar:**
- HEIM paper (arXiv 2311.04287, NeurIPS 2023) — Table 4
- [stanford-crfm/helm](https://github.com/stanford-crfm/helm) `src/helm/config/model_deployments.yaml`
- Leaderboard URL'leri SPA olduğu için WebFetch ile indirilemedi — kullanıcı tarafından tarayıcıda doğrulanmalı

## HEIM v1.0 (NeurIPS 2023 paper) Model Listesi — 26 Model

| # | Model | Organizasyon | Erişim | Params |
|---|-------|--------------|--------|--------|
| 1 | Stable Diffusion v1-4 | CompVis/LMU | Open | 1B |
| 2 | **Stable Diffusion v1-5** | Runway | Open | 1B |
| 3 | Stable Diffusion v2 base | Stability AI | Open | 1B |
| 4 | Stable Diffusion v2-1 base | Stability AI | Open | 1B |
| 5 | Dreamlike Diffusion 1.0 | Dreamlike.art | Open | 1B |
| 6 | Dreamlike Photoreal 2.0 | Dreamlike.art | Open | 1B |
| 7 | Openjourney | PromptHero | Open | 1B |
| 8 | Openjourney v4 | PromptHero | Open | 1B |
| 9 | Redshift Diffusion | nitrosocke | Open | 1B |
| 10 | Vintedois Diffusion | 22h | Open | 1B |
| 11-14 | SafeStableDiffusion (weak/medium/strong/max) | TU Darmstadt | Open | 1B |
| 15 | Promptist + SD v1-4 | Microsoft | Open | 1B |
| 16 | Lexica Search | Lexica | Open | 1B |
| 17 | **DALL-E 2** | OpenAI | Limited (API) | 3.5B |
| 18 | DALL-E mini | craiyon | Open | 0.4B |
| 19 | DALL-E mega | craiyon | Open | 2.6B |
| 20 | minDALL-E | Kakao Brain | Open | 1.3B |
| 21 | CogView2 | Tsinghua | Open | 6B |
| 22 | MultiFusion | Aleph Alpha | Limited | 13B |
| 23-25 | DeepFloyd-IF (M/L/XL) | DeepFloyd | Open | 0.4-4.3B |
| 26 | GigaGAN | Adobe | Limited | 1B |

## HEIM Sonradan Eklenmiş Modeller

**DOĞRULANDI (2026-04-24):** `model_deployments.yaml`'da görünen SDXL, SSD-1B, Segmind-Vega gibi modeller **HEIM leaderboard'da yok**. Leaderboard hâlâ paper'ın 26 modeliyle sınırlı; yaml entry'leri orphan. Kullanıcının manuel doğrulaması paper'daki listeyle birebir eşleşti.

## 8 Hedef Modelle Eşleştirme

| # | Hedef Model | HEIM Durumu | Not |
|---|-------------|-------------|-----|
| 1 | Stable Diffusion 1.5 | ✅ **Var** (v1-5) | Doğrudan skor alınabilir |
| 2 | SDXL | ❌ **Yok** | Leaderboard manuel doğrulandı — HEIM'de yok |
| 3 | Stable Diffusion 3.5 Large | ❌ **Yok** | Ekim 2024'te çıktı, HEIM'den sonra |
| 4 | FLUX.1-dev | ❌ **Yok** | Ağustos 2024'te çıktı, HEIM'den sonra |
| 5 | Qwen-Image | ❌ **Yok** | 2025, HEIM'den sonra |
| 6 | DALL-E 2 | ✅ **Var** | OpenAI, limited access ama skor paper'da |
| 7 | GPT-Image-1 | ❌ **Yok** | 2025, HEIM'den sonra |
| 8 | Gemini 2.5 Flash-Image | ❌ **Yok** | 2025, HEIM'den sonra |

## Sonuç ve Tez Üzerine Etkisi

**Net HEIM kapsaması: 2/8 (SD 1.5 + DALL-E 2).**

Kullanıcının 8 modeli arasında HEIM'den skor alabileceği sadece 2-3 model var. Geri kalan 5-6 model için **PRISM-Bench**'in zorunlu kaynak olduğu doğrulanıyor. Bu, master veri tablosunda **modellerin çoğu yalnızca bir benchmark'ta skorlanmış** demek — HEIM ve PRISM'i tek bir karşılaştırılabilir ölçek altında normalize etmek, öncesinde sanılandan **daha kritik ve riskli**.

### Olası ayarlamalar (Gün 2-3'te karar)

**A) Mevcut 8 modeli tut, köprü kur:** Ortak bir modeli (ideally SD 1.5 veya SDXL) hem HEIM hem PRISM'de bulup, iki skor dağılımının farkını regresyon ile hizala. Risk: tek köprü noktası zayıf kalibrasyon verir.

**B) Model listesini güncelle:** HEIM×PRISM kesişimini maksimize edecek şekilde modelleri seç. Olası eklemeler: SD v2-1 base (HEIM'de var), Dreamlike Photoreal, Openjourney v4 (popüler açık kaynak). Kapalı kaynak tarafında sadece DALL-E 2 + GPT-Image-1 + Gemini kalır.

**C) Dual-scored only:** Sadece her iki benchmark'ta skor sahibi modeller. Bu muhtemelen 0-2 model verir, tezi yapılamaz kılar.

**Önerilen yaklaşım:** Gün 2'de PRISM model listesi alındıktan sonra karar ver. Şu an için **A** (köprü kurma) en güçlü akademik hikâye sunar ama **B** daha pragmatik olabilir.
