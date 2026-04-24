# Bitirme Tezi: T2I Model Router Projesi

## Proje Künyesi

**Tez Başlığı:** T2I Modelleri için Prompt Tabanlı Otomatik Router Sistemi

**Alternatif Başlık (Alt-B yapılırsa):** T2I Modelleri için Prompt Tabanlı Otomatik Router Sistemi ve Model-Spesifik Prompt Adaptasyonu

**Süre:** 30 gün (uzatma yok)

**Kritik Check-Point:** Gün 14 sonu

---

## Tek Cümlelik Özet

Kullanıcı bir prompt girer, sistem o prompt için 8 T2I modelinden hangisinin en iyi kalite-maliyet dengesini sunacağını **görsel üretmeden önce** tahmin eder ve önerir.

---

## Araştırma Sorusu

Prompt metnine bakarak, görsel üretilmeden önce, hangi T2I modelinin en iyi sonucu en uygun maliyetle vereceği tahmin edilebilir mi?

## Katkı İddiası

HEIM ve PRISM-Bench gibi çoklu-model değerlendirme dataset'leri mevcut, ancak bunlar statik model sıralamaları sunar. Bu tezde prompt'a özgü, dinamik model seçim sistemi geliştirilmiştir.

---

## Proje Bileşenleri

### 1. Veri Kaynakları

| Kaynak | İçerik | Rol |
|--------|--------|-----|
| HEIM (Stanford, 2023) | 26 T2I modeli, 62 senaryo, 12 değerlendirme boyutu | Ana dataset |
| PRISM-Bench (2025) | FLUX, Qwen-Image, GPT-Image-1, Gemini dahil modern modeller, 7 track | Modern modelleri ekleme |
| PartiPrompts (Google) | 1600 İngilizce prompt, 12 kategori | Kategori iskeleti |

**Neden hazır dataset:** Görsel üretim maliyetli ve 1 ayda mümkün değil. Bu akademik dataset'ler zaten görselleri üretip değerlendirmiş, skorları kullanacağız.

### 2. Model Havuzu (8 Model)

**Açık Kaynak (5):**
- Stable Diffusion 1.5 (eski nesil, ucuz, hızlı - baseline)
- SDXL (2023-2024 standardı, orta segment)
- Stable Diffusion 3.5 Large (modern Stability AI modeli)
- FLUX.1-dev (açık kaynak lideri, fotogerçekçi)
- Qwen-Image (metin rendering ve kompozisyon)

**Kapalı Kaynak (3):**
- DALL-E 2 (tarihsel öncü, HEIM'de insan skorları var)
- GPT-Image-1 (OpenAI'nin son modeli, PRISM'de lider)
- Gemini 2.5 Flash-Image (Google modeli, duygu aktarımı)

**Çeşitlilik boyutları:** açık/kapalı, eski/yeni, ucuz/pahalı

### 3. Prompt Sınıflandırma

**12 Kategori (PartiPrompts'tan):** artifacts, animals, indoor scenes, produce/plants, abstract, arts, food, vehicles, illustrations, outdoor scenes, people, world knowledge

**Yöntem:** LLM API ile (Claude Haiku veya GPT-4o-mini). Prompt gönderilir, kategori döner. Ucuz (<$0.001/prompt), hızlı (1-2 saniye).

### 4. Router (3 Yaklaşım Karşılaştırılacak)

**A) Lookup Table (basit):**
Önceden hesaplanmış tablo. (kategori, mod) → en iyi model. O(1) erişim.

**B) ML-based (orta):**
SBERT embedding + XGBoost/MLP. Kullanıcı modu ile birlikte → model önerisi.

**C) LLM-based (modern):**
Few-shot prompting ile LLM'e prompt + model profilleri ver, öneri al.

**Karşılaştırma metrikleri:** doğruluk, hız, maliyet

### 5. Maliyet-Kalite Optimizasyonu

**Kalite skoru:** HEIM + PRISM ortalaması (0-100)

**Maliyet skoru:**
- Açık kaynak: inference süresi × parameter count
- Kapalı kaynak: API fiyatı

**Pareto Frontier:** Dominated olmayan modeller bulunur.

**Kullanıcı modları:**
- Ucuz mod: Pareto'nun düşük maliyet ucundan
- Dengeli mod: Ortadan (kalite/maliyet oranı en yüksek)
- Premium mod: Pareto'nun yüksek kalite ucundan

### 6. Demo (Streamlit Web App)

**Akış:**
1. Kullanıcı prompt girer
2. Mod seçer (ucuz/dengeli/premium)
3. Router kategori tespit eder ve model önerir
4. Neden o model seçildi gösterilir
5. HEIM/PRISM'den örnek görsel gösterilir
6. Maliyet tasarrufu tablosu

---

## Baseline Karşılaştırmaları

Router'ın üstünlüğünü kanıtlamak için:
- **Naif-1:** Hep en popüler modeli seç
- **Naif-2:** Hep en pahalı modeli seç
- **Naif-3:** Rastgele seç

## Değerlendirme Metrikleri

- Router doğruluğu (%)
- Maliyet tasarrufu (%)
- Kalite korunması (%)
- Kategori bazlı performans

---

## Opsiyonel Genişletme: Alternatif B (Prompt Rewriter)

**Ne yapar:** Router modeli seçtikten sonra, o modele özel optimize edilmiş prompt üretir. FLUX seçildiyse doğal dil uzun cümleye, SD 1.5 seçildiyse modifier-stacked formata çevirir.

**Nasıl yapılır:**
- Her model için 300-500 (ham_prompt, optimize_prompt) çifti sentetik üret (GPT-4o-mini ile)
- Qwen2.5-3B veya Llama-3.2-3B'yi LoRA ile fine-tune et
- Model conditioning ile tek model tüm T2I modellerine adaptasyon yapar

**Karar noktası:** Gün 14 check-point sonunda karar verilecek. Router tam bitmişse ve analizler hazırsa yapılacak, değilse unutulacak.

---

## 30 Günlük Günlük Plan

### HAFTA 1 – Veri ve Altyapı

**Gün 1**
- Hocayla randevu al
- GitHub proje repo'su aç, klasör yapısını kur
- Python environment kur, kütüphaneleri yükle
- HEIM GitHub repo'sunu klonla, dokümantasyonu oku
- 8 modelinden hangilerinin HEIM'de olduğunu tespit et

**Gün 2**
- HEIM skor verilerini indir (sadece skorlar, görseller değil)
- Veri çekme script'ini yaz
- Jupyter notebook'ta HEIM verisini keşfet
- Eksik/tutarsız verileri not et

**Gün 3**
- PRISM-Bench GitHub'ını klonla
- PRISM skorlarını indir (7 track için)
- Loader script'i yaz
- DiffusionDB verisi yerini/formatını not et (Alt-B hazırlığı)

**Gün 4**
- HEIM ve PRISM verilerini birleştirmek için master tablo tasarla
- Skorları normalize et (0-100 aralığına)
- 8 modelin master CSV'sini oluştur
- Kapalı kaynak modellerin API fiyatlarını topla

**Gün 5**
- PartiPrompts dataset'ini indir
- HEIM+PRISM prompt'larını PartiPrompts kategorilerine eşle
- LLM API ile otomatik kategorizasyon
- Örneklem doğrulaması (100 prompt elle kontrol)

**Gün 6**
- Master tablo v2: kategori kolonu eklenmiş hali
- Veri kalite kontrolü, eksik satırları temizle
- Tez outline yaz (her bölüm için 1 paragraflık açıklama)
- Hocayla ilk haftalık toplantı

**Gün 7**
- Literatür taraması: HEIM, PRISM-Bench paper'larını oku
- RouteLLM, ImageReward, HPSv2 paper'larını yüzeysel oku
- Atıfları kaydet (Notion/Zotero)
- Hafta 1 özeti, hafta 2'nin günlük planı

### HAFTA 2 – Router Geliştirme

**Gün 8**
- Keşifsel analiz notebook'u
- Her (model, kategori) çifti için ortalama kalite skorları
- İlk heatmap: modeller × kategoriler
- Bulguları not et

**Gün 9**
- Pareto frontier hesaplama
- Kategori bazında ayrı Pareto grafikleri
- matplotlib/seaborn ile profesyonel grafikler
- Figures klasörüne kaydet

**Gün 10**
- Router v1: Lookup table yaklaşımı
- (kategori, mod) → en iyi model tablosu
- Unit testler

**Gün 11**
- Router v2: ML-based yaklaşım
- Prompt → SBERT embedding
- XGBoost veya basit MLP
- Train/val/test split (%70/15/15)

**Gün 12**
- Router v3: LLM-based yaklaşım
- Claude/GPT API ile few-shot prompting
- Üç yaklaşımın hız ve maliyet karşılaştırması

**Gün 13**
- Router değerlendirme
- Test seti üzerinde 3 router'ı çalıştır
- Baseline'larla karşılaştır
- Metrikler: doğruluk, maliyet tasarrufu, kalite korunması

**Gün 14 – CHECK-POINT**
- Tüm sonuçları topla
- Hocayla haftalık toplantı
- Öz-değerlendirme: Her şey %100 bitti mi?
  - Evet → Alt-B'ye hazır ol
  - Hayır → Alt-B unutulacak, orijinal plana devam

### HAFTA 3 – İki Senaryo

#### SENARYO A: Check-point geçildi, Alt-B yapılıyor

**Gün 15**
- Alt-B veri hazırlığı
- Her model için 300-500 (base_prompt, optimize_prompt) çifti üret
- GPT-4o-mini API ile sentetik üretim
- JSON format hazırla

**Gün 16**
- LoRA fine-tuning (Colab Pro veya Kaggle)
- Model: Qwen2.5-3B veya Llama-3.2-3B
- Framework: Unsloth
- Eğitim arka planda çalışırken giriş bölümü yaz

**Gün 17**
- Fine-tune değerlendirme: 50 test prompt
- Ham vs optimize prompt karşılaştırması
- Streamlit demo iskeletini kur
- Router + rewriter entegrasyonu

**Gün 18**
- Demo finalize
- Bug fix, test
- Ekran kaydı al

**Gün 19**
- Tez yazımı: Giriş bölümü (3-4 sayfa)

**Gün 20**
- Tez yazımı: Literatür bölümü (6-8 sayfa)

**Gün 21**
- Tez yazımı: Metodoloji 1. kısım (4-5 sayfa)

#### SENARYO B: Check-point geçilmedi, Alt-B yok

**Gün 15**
- Streamlit demo iskeleti
- Sayfa yapısı, sidebar, ana panel

**Gün 16**
- Router entegrasyonu
- Her router yaklaşımı için tab
- Cache mekanizması

**Gün 17**
- Görselleştirme: router kararı, bar chart, maliyet tablosu
- Örnek görsel gösterimi

**Gün 18**
- Demo finalize, bug fix, test
- Ekran kaydı al
- README yaz

**Gün 19**
- Tez yazımı: Giriş bölümü

**Gün 20**
- Tez yazımı: Literatür bölümü

**Gün 21**
- Tez yazımı: Metodoloji bölümü

### HAFTA 4 – Yazım ve Teslim

**Gün 22**
- Tez yazımı: Metodoloji 2. kısım
- Diyagramlar (pipeline görseli)

**Gün 23**
- Tez yazımı: Deneyler bölümü (4-5 sayfa)
- Deney düzeni, baseline tanımları, metrikler

**Gün 24**
- Tez yazımı: Sonuçlar bölümü (6-8 sayfa)
- Router performans karşılaştırmaları
- Kategori bazlı analiz
- Pareto sonuçları

**Gün 25**
- Tez yazımı: Tartışma + Sonuç (5-6 sayfa)
- Bulgu yorumları, kısıtlılıklar, gelecek çalışmalar
- Türkçe ve İngilizce özet

**Gün 26**
- Kaynakça tamamla, atıfları kontrol
- BibTeX düzenle

**Gün 27**
- Tezi baştan sona 1. okuma
- Mantık tutarsızlıkları, tekrarlar

**Gün 28**
- Tezi baştan sona 2. okuma
- Format kontrol
- Danışmandan son feedback (mümkünse)

**Gün 29**
- Son düzeltmeler
- Savunma sunumu hazırla (15-20 dakika)
- Prova yap

**Gün 30 – Teslim**
- Son kontrol
- Yedekle (GitHub + Drive + USB)
- Resmi teslim

---

## Nihai Çıktılar

1. Temizlenmiş master veri seti (HEIM + PRISM birleşimi)
2. Python paketi (3 router implementasyonu)
3. Streamlit demo (çalışan web uygulaması)
4. Analiz grafikleri (heatmap, Pareto frontier, kategori performans)
5. Tez (~40-50 sayfa)
6. Savunma sunumu (15-20 dakika)
7. GitHub repo (reproducible)
8. Opsiyonel: Fine-tuned prompt rewriter (Alt-B başarılı olursa)

---

## Zayıf Noktalar ve Hazır Cevaplar

**Z1: Veri sınırlı (100-150 ortak prompt)**
Cevap: Küçük ama akademik benchmark'lardan alındığı için kalite yüksek, sonuçların yönü güvenilir.

**Z2: Router karmaşıklık değer mi?**
Cevap: ML yaklaşımı yeni kategorilere genelleme yapabilir, lookup table sabit kalır.

**Z3: Gerçek dünya testi yok**
Cevap: Pilot deployment gelecek çalışma, bu tez offline değerlendirme.

**Z4: HEIM 2023, bazı modeller eski**
Cevap: Tarihsel karşılaştırma ve HEIM'in insan skor zenginliği için.

---

## Çalışma Prensipleri

- Her gün sonu GitHub commit + push
- Her gün sonu kısa özet notu
- Her hafta sonu danışmanla check-in
- 2 saatten fazla debug'da takılma, sor
- Mükemmel yerine çalışan iş çıkar
- Check-point'te kendine dürüst ol

---

## Risk Yönetimi

**Her hafta sonu soru:** Planın %80+'ı yapıldı mı?
- Evet → Devam
- %60-80 → Hafta sonu telafi
- <%60 → Danışmanla konuş, daralt

**Kırılma senaryosu:** Fine-tuning başarısız → Few-shot prompt engineering'e 1 günde dön → Yine olmazsa sadece router ile devam.

---

## Klasör Yapısı

thesis-t2i-router/
- data/raw/ (ham veri)
- data/processed/ (temizlenmiş veri)
- notebooks/ (keşifsel analiz)
- src/data_loader/
- src/router/
- src/evaluation/
- src/utils/
- demo/ (Streamlit app)
- thesis/ (tez dosyaları)
- figures/ (grafikler)
- README.md