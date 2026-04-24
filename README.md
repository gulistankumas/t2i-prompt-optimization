# T2I Prompt-Based Model Router

Bitirme tezi: T2I modelleri için prompt tabanlı otomatik router sistemi.

Kullanıcı bir prompt girer, sistem 8 T2I modelinden hangisinin en iyi kalite-maliyet dengesini sunacağını **görsel üretmeden önce** tahmin eder ve önerir. Skor verisi HEIM (Stanford, 2023) ve PRISM-Bench (2025) benchmark'larından alınır.

Detaylı plan: [bitirme.md](bitirme.md).

## Kurulum

```bash
python -m venv .venv
# Windows bash:
source .venv/Scripts/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Klasör Yapısı

```
data/raw/          # Ham HEIM / PRISM skor indirmeleri
data/processed/    # Temizlenmiş, birleştirilmiş master veri
notebooks/         # Keşifsel analiz
src/data_loader/   # Veri çekme ve normalize
src/router/        # 3 router implementasyonu (lookup, ML, LLM)
src/evaluation/    # Metrik ve baseline karşılaştırma
src/utils/         # Ortak yardımcılar
demo/              # Streamlit web app
thesis/            # Tez dosyaları
figures/           # Grafikler
external/          # HEIM/PRISM gibi dış repo klonları (gitignored)
```

## Çıktılar

1. Master veri seti (HEIM + PRISM)
2. Üç router implementasyonu
3. Streamlit demo
4. Analiz grafikleri
5. Tez (~40-50 sayfa)
