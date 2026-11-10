# BIST veri hattı

Her iş günü kapanıştan sonra (18:45 TSİ) ve sabah 07:30'da yedek olarak çalışır. BIST hisselerinin
günlük fiyat geçmişini çeker, teknik göstergeleri hesaplar ve sonuçları `data/` klasörüne yazar.
Sabah raporu bu dosyaları okur.

- `data/latest.json` — tüm göstergeler, kurulumlar ve seviyeler (raporun okuduğu dosya)
- `data/latest.md` — aynı verinin okunabilir tablo hali
- `hisseler.txt` — taranacak hisseler; satır ekleyip silerek değiştirebilirsin

Elle çalıştırmak için: **Actions → BIST teknik tarama → Run workflow**.

## Kurallar

| Kod | Koşul |
| --- | --- |
| L1 | Kapanış > SMA50 > SMA200, RSI 45-65, fiyat SMA20/SMA50'ye %2 yakın ve günü yükselişle kapatmış |
| L2 | Kapanış > SMA200, RSI son 3 günde 30 altına inip bugün 30 üstüne çıkmış |
| L3 | Kapanış önceki 20 günün zirvesinin üstünde, hacim 20 gün ortalamasının ≥1.5 katı, RSI < 75 |
| S1 | Kapanış < SMA50 < SMA200, RSI 35-55, fiyat SMA20/SMA50'ye %2 yakın ve günü düşüşle kapatmış |
| S2 | RSI > 75 ve MACD histogramı daralıyor veya negatif uyumsuzluk var |
| S3 | Kapanış önceki 20 günün dibinin altında, hacim ≥1.5 kat, RSI > 25 |

Seviyeler: giriş = kapanış ±0.5 ATR; stop = swing dip/tepe ±0.25 ATR ile 1.5 ATR'den uzak olanı (en fazla 2 ATR);
H1 = 1.5R, H2 = 2.5R. `rr_engele_kadar`, en yakın direnç (long) veya destek (short) seviyesine kadar olan
mesafenin riske oranıdır; 1.5'in altındaysa hedeften önce bir engel var demektir.

Veri kaynağı: Yahoo Finance; gelmeyen hisse için İş Yatırım yedek olarak denenir.
Bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.
