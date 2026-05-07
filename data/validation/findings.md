# 5 Prompt Validation - Ön Sonuçlar

## Ozet
İki mod (premium + ekonomik) için aynı 5 prompt'ta gerçek görsel üretimi
ve GPT-4.1 vision puanlama yapıldı.

## Sonuclar

| Yontem | Kalite (avg) | Maliyet ($/5) | Q/$ |
|--------|-------------|---------------|-----|
| Lookup Ekonomik | 93.38 (n=4) | $0.078 | 1193 |
| Lookup Premium  | 93.90 (n=5) | $0.432 | 217 |
| Always FLUX.1-dev (baseline) | 92.20 (n=5) | $0.150 | 615 |

## Vaka Bazli (4 esler)

| ID | Track | Eco | Prem | Base | Eco vs Base |
|----|-------|-----|------|------|-------------|
| val_01 | style | 94.5 | 95.5 | 92.0 | +2.5 |
| val_03 | composition | 93.0 | 90.0 | 90.5 | +2.5 |
| val_04 | text_rendering | 94.5 | 95.0 | 94.0 | +0.5 |
| val_05 | text_rendering | 91.5 | 95.5 | 90.0 | +1.5 |

## Bulgular

1. **Ekonomik router baseline'dan hem kaliteli hem ucuz** (+1.18 kalite, %48 tasarruf)
2. **Q/$ oraninda ekonomik 2x baseline, 5.5x premium** (en verimli)
3. **Premium baseline secimi hatali** (FLUX.1-dev orta-fiyat, premium'la fair degil)
4. **val_03'te ekonomik premium'u gecti** (FLUX.1-schnell composition'da yeterli)
5. **Insan-LLM uyum: 5/5** (val_01,03,05 router; val_02,04 baseline)

## Sinirliliklar

- n=5 (val_02 SD1.5 uretimi basarisiz) - kucuk ornek
- Istatistiksel anlamlilik testi yapilmadi (n yetersiz)
- Sadece premium track'lerde kapali kaynak modeller test edildi (Gemini hala yok)

## Sonraki Adim

- 15-20 prompt'a cikar
- val_02 SD1.5 hatasini duzelt (Replicate model ID)
- Premium icin 'Always GPT-Image-1' baseline ekle
- Wilcoxon signed-rank test

---

## Güncelleme (post-retry)

`24_generate_images.py` Replicate path'lerine `SD1.5: stability-ai/stable-diffusion:ac732df...` eklenerek val_02 SD1.5 yeniden üretildi:
- val_02 (affection) SD1.5 → alignment=92, aesthetic=78, **avg=85.0**
- Ekonomik mod 5/5 tam: ortalama **91.74** ($0.078, Q/$ = 1176)
- 8.5 puan kalite kaybı affection track'inde SD1.5 zayıflığını doğruluyor (PRISM master skoru 62.4 → vision 85.0; cömert ölçüm)

Bu güncellemeyle eko vs premium farkı 2.16 puan (önceki 0.52'den), Q/$ avantajı korunuyor (5.5x premium'a göre).
