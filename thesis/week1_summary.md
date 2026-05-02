# Hafta 1 Özeti

**Tarih aralığı:** 2026-04-24 — 2026-05-03
**Durum:** Tamamlandı

---

## Yapılan İşler

**Veri ve altyapı:**
- GitHub repo açıldı, klasör yapısı kuruldu
- Python environment (pandas, scikit-learn, sentence-transformers, xgboost, streamlit) hazır
- PRISM-Bench iki jüri verisi yüklendi (GPT-4.1 ve Qwen2.5-VL CSV'leri)
- 8 hedef model + 1 historical (DALL-E 2) için master tablo oluşturuldu
- Maliyet tablosu derlendi (kapalı kaynak API + açık kaynak Replicate fiyatları)

**Analizler:**
- Track-arası Spearman korelasyon analizi (8 model: 0.94, 19 model: 0.84)
- Segment-bazlı performans analizi (premium/mid/cheap)
- Track-bazlı Pareto frontier analizi (her model için Pareto-optimal track sayısı)
- İki jüri agreement analizi (Spearman ρ 0.86-0.98)
- Premium paradox bulgusu (GPT-Image-1 vs Gemini 2.5)

**Görseller:**
- `pareto_frontier_per_track.png` (7 track × Pareto)
- `track_correlation_heatmap.png` (8 vs 19 model)
- `segment_track_distribution.png` (boxplot)
- `judge_agreement.png` (iki jüri scatter)
- `premium_paradox.png` (GPT vs Gemini bar)

**Belgeler:**
- `thesis/00_outline.md` (tez outline'ı, 7 bölüm + ön/arka materyal)
- `thesis/literature_notes.md` (5 paper özet)
- `data/processed/prism_findings.md` (analiz bulguları)
- `data/processed/master_final.csv` (39 sütun × 9 satır)

---

## Ana Bulgular (Tezin omurgası)

**Bulgu 1:** Track-arası korelasyon 8 modelde 0.94, 19 modelde 0.84 → routing değeri model havuzu çeşitliliğine bağlı.

**Bulgu 2:** Segment-bağımlı routing değeri → premium'da trivial (0.9 puan), mid'de sınırlı (1.6 puan), cheap'te belirgin (text_rendering'de 24.8 puan).

**Bulgu 3:** Track-bazlı Pareto → SDXL 3/7 track'te optimal (specialist), Qwen-Image 0/7 (dominated), 3 model 7/7 (SD1.5, FLUX.1-schnell, Gemini 2.5).

**Bulgu 4:** Premium paradox → GPT-Image-1 vs Gemini 2.5 fark 0.65 puan, fiyat farkı 4.3x. Affection ve long_text'te Gemini önde.

**Bulgu 5:** İki jüri arası anlaşma → sıralamada yüksek (ρ 0.86-0.98), mutlak skor düzleminde sapmalar var. Self-preference bias değil, stilistik bias.

---

## Öğrenilen Kavramlar

- Spearman korelasyonu (sıralama benzerliği)
- Pareto-optimal / Pareto-dominated (iki boyutlu trade-off)
- SBERT (cümle embedding)
- LOCO-CV (kategori-dışı genelleme testi)
- Self-preference bias (LLM-as-judge bias çeşidi)

---

## Okunan Paperlar

- PRISM-Bench (Fang et al., 2025) — ana veri kaynağı
- HEIM (Lee et al., 2023) — referans benchmark
- RouteLLM (Ong et al., 2024) — metodoloji şablonu
- FrugalGPT (Chen et al., 2023) — eşik mantığı
- RouteT2I (Xin et al., 2025) — en yakın çakışan iş

---

## Bekleyen İşler (Hafta 2'ye taşınanlar)

- Hocayla görüşme yapılacak ([tarih — kullanıcı netleştirecek])
- PartiPrompts → PRISM track mapping kararı (Hafta 2 başında)
- Capability-min routing kararı (Gün 12 hatırlatması)
- Alt-B (LoRA fine-tuning) opsiyonel kararı (Gün 14 check-point)
