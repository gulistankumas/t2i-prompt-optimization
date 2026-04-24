# data/raw — Ham Skor Verileri

## prism_bench_gpt41_english.csv

**Kaynak:** [PRISM-Bench leaderboard](https://flux-reason-6m.github.io/#leaderboard), GPT-4.1 English jüri
**Kaç model:** 19 (PRISM'in değerlendirdiği tüm modeller)
**Kaç track:** 7 (imagination, entity, text_rendering, style, affection, composition, long_text) + overall
**Skor ölçeği:** ~0-100, her hücre o modelin o track'teki 100 prompt üzerinden ortalama "Avg" skoru (Alignment × Aesthetic birleşimi)
**İndirme tarihi:** 2026-04-24
**İndirme yöntemi:** WebFetch aracı, leaderboard HTML'inden çıkarıldı

⚠️ **Manuel doğrulama gerekli** — WebFetch çıktısı AI destekli parse. Tezde kullanmadan önce 2-3 hücre tarayıcıdan rastgele kontrol edilmeli.

## To-Do
- [ ] Qwen2.5-VL English leaderboard'unu da ayrı CSV olarak ekle (çapraz doğrulama için)
- [ ] HEIM SD 1.5 ve DALL-E 2 skorları (Gün 3)
- [ ] PRISM prompt'larını (700 adet, 7 × 100) HuggingFace'den indir
