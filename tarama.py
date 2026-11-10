"""BIST teknik tarama.

Her hisse için ~400 günlük günlük fiyat geçmişini çeker, göstergeleri hesaplar,
L1-L3 / S1-S3 kurallarını uygular ve sonuçları data/ klasörüne yazar.

Veri kaynağı: Yahoo Finance (yfinance). Bir hisse Yahoo'dan gelmezse
İş Yatırım'ın herkese açık fiyat servisi yedek olarak denenir.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

KOK = Path(__file__).resolve().parent
VERI = KOK / "data"
TR = timezone(timedelta(hours=3))

MAKRO = {
    "XU100.IS": "BIST 100",
    "XU030.IS": "BIST 30",
    "USDTRY=X": "USD/TRY",
    "EURTRY=X": "EUR/TRY",
    "BZ=F": "Brent petrol",
    "GC=F": "Altın (ons)",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^VIX": "VIX",
}


# --------------------------------------------------------------------------- veri

def tamamlanmis_gunler(df: pd.DataFrame) -> pd.DataFrame:
    """Seans bitmeden (18:30 TSİ öncesi) çalışılırsa bugünün yarım mumunu çıkarır."""
    simdi = datetime.now(TR)
    if len(df) and df.index[-1].date() >= simdi.date() and simdi.hour * 60 + simdi.minute < 18 * 60 + 30:
        return df.iloc[:-1]
    return df


def hisse_listesi() -> list[str]:
    satirlar = (KOK / "hisseler.txt").read_text(encoding="utf-8").splitlines()
    return [s.strip().upper() for s in satirlar if s.strip() and not s.startswith("#")]


def yahoo_cek(semboller: list[str]) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    sonuc: dict[str, pd.DataFrame] = {}
    for deneme in range(3):
        eksik = [s for s in semboller if s not in sonuc]
        if not eksik:
            break
        try:
            ham = yf.download(
                eksik, period="2y", interval="1d", group_by="ticker",
                auto_adjust=True, progress=False, threads=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Yahoo hatası (deneme {deneme + 1}): {e}", file=sys.stderr)
            time.sleep(10)
            continue
        for s in eksik:
            try:
                df = ham[s] if isinstance(ham.columns, pd.MultiIndex) else ham
                df = df.dropna(subset=["Close"])
                if s.endswith(".IS"):
                    df = tamamlanmis_gunler(df)
                if len(df) >= 60:
                    sonuc[s] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
            except KeyError:
                pass
        time.sleep(5)
    return sonuc


def isyatirim_cek(kod: str) -> pd.DataFrame | None:
    """Yedek kaynak: İş Yatırım günlük fiyat servisi (kapanış, en yüksek, en düşük, hacim)."""
    bitis = datetime.now(TR)
    baslangic = bitis - timedelta(days=730)
    url = (
        "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
        f"?hisse={kod}&startdate={baslangic:%d-%m-%Y}&enddate={bitis:%d-%m-%Y}"
    )
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        kayitlar = r.json().get("value", [])
    except Exception as e:  # noqa: BLE001
        print(f"İş Yatırım hatası {kod}: {e}", file=sys.stderr)
        return None
    if not kayitlar:
        return None
    df = pd.DataFrame(kayitlar)
    gerekli = {"HGDG_TARIH", "HGDG_KAPANIS", "HGDG_MAX", "HGDG_MIN", "HGDG_HACIM"}
    if not gerekli.issubset(df.columns):
        return None
    df = pd.DataFrame({
        "Open": pd.to_numeric(df.get("HGDG_AOF", df["HGDG_KAPANIS"]), errors="coerce"),
        "High": pd.to_numeric(df["HGDG_MAX"], errors="coerce"),
        "Low": pd.to_numeric(df["HGDG_MIN"], errors="coerce"),
        "Close": pd.to_numeric(df["HGDG_KAPANIS"], errors="coerce"),
        "Volume": pd.to_numeric(df["HGDG_HACIM"], errors="coerce"),
    }).set_index(pd.to_datetime(df["HGDG_TARIH"], dayfirst=True, errors="coerce"))
    df = tamamlanmis_gunler(df.dropna(subset=["Close"]).sort_index())
    return df if len(df) >= 60 else None


# ------------------------------------------------------------------- göstergeler

def wilder(seri: pd.Series, n: int) -> pd.Series:
    return seri.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def gostergeler(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l = d["Close"], d["High"], d["Low"]
    for n in (20, 50, 200):
        d[f"SMA{n}"] = c.rolling(n).mean()
    fark = c.diff()
    rs = wilder(fark.clip(lower=0), 14) / wilder(-fark.clip(upper=0), 14)
    d["RSI"] = 100 - 100 / (1 + rs)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = wilder(tr, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    d["MACD"] = macd
    d["MACD_SINYAL"] = macd.ewm(span=9, adjust=False).mean()
    d["MACD_HIST"] = d["MACD"] - d["MACD_SINYAL"]
    # Kırılım karşılaştırması için bugünü hariç tutan önceki 20 gün
    d["ZIRVE20"] = h.shift(1).rolling(20).max()
    d["DIP20"] = l.shift(1).rolling(20).min()
    d["HACIM_ORT20"] = d["Volume"].shift(1).rolling(20).mean()
    d["ZIRVE52H"] = h.rolling(250, min_periods=120).max()
    d["DIP52H"] = l.rolling(250, min_periods=120).min()
    return d


def swing_noktalari(df: pd.DataFrame, pencere: int = 3, geri: int = 40) -> tuple[float | None, float | None]:
    """Son `geri` gün içindeki en yakın swing dip ve swing tepe (iki yanında `pencere` mum)."""
    son = df.tail(geri)
    h, l = son["High"].values, son["Low"].values
    dip = tepe = None
    for i in range(len(son) - pencere - 1, pencere - 1, -1):
        if dip is None and l[i] == l[i - pencere:i + pencere + 1].min():
            dip = float(l[i])
        if tepe is None and h[i] == h[i - pencere:i + pencere + 1].max():
            tepe = float(h[i])
        if dip is not None and tepe is not None:
            break
    return dip, tepe


def yuvarla(x, n=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), n)


def seviyeler(yon: str, giris: float, atr: float, swing: float | None,
              direnc_destek: float | None) -> dict:
    if yon == "long":
        atr_stop = giris - 1.5 * atr
        swing_stop = swing - 0.25 * atr if swing is not None and swing < giris else atr_stop
        stop = max(min(atr_stop, swing_stop), giris - 2 * atr)
        risk = giris - stop
        h1, h2 = giris + 1.5 * risk, giris + 2.5 * risk
        engel_rr = (direnc_destek - giris) / risk if direnc_destek and direnc_destek > giris else None
    else:
        atr_stop = giris + 1.5 * atr
        swing_stop = swing + 0.25 * atr if swing is not None and swing > giris else atr_stop
        stop = min(max(atr_stop, swing_stop), giris + 2 * atr)
        risk = stop - giris
        h1, h2 = giris - 1.5 * risk, giris - 2.5 * risk
        engel_rr = (giris - direnc_destek) / risk if direnc_destek and direnc_destek < giris else None
    return {
        "giris_alt": yuvarla(giris - 0.5 * atr), "giris_ust": yuvarla(giris + 0.5 * atr),
        "stop": yuvarla(stop), "h1": yuvarla(h1), "h2": yuvarla(h2),
        "risk_yuzde": yuvarla(risk / giris * 100), "rr_h1": 1.5, "rr_h2": 2.5,
        # En yakın direnç (long) / destek (short) H1'den önce geliyorsa bu oran 1.5'in altına düşer
        "rr_engele_kadar": yuvarla(engel_rr),
    }


def kurallar(d: pd.DataFrame) -> dict:
    s, o = d.iloc[-1], d.iloc[-2]
    c, rsi = s["Close"], s["RSI"]
    sma20, sma50, sma200 = s["SMA20"], s["SMA50"], s["SMA200"]
    hacim_kat = s["Volume"] / s["HACIM_ORT20"] if s["HACIM_ORT20"] else np.nan
    rsi3 = d["RSI"].tail(4)
    hist = d["MACD_HIST"].tail(3).values
    yakin = lambda a, b: abs(a - b) / b <= 0.02  # noqa: E731

    tutunma = s["Low"] <= max(sma20, sma50) * 1.02 and c > s["Open"] and c >= min(sma20, sma50)
    red = s["High"] >= min(sma20, sma50) * 0.98 and c < s["Open"] and c <= max(sma20, sma50)

    # Negatif uyumsuzluk: son 10 gün zirvesi önceki 10-30 gün zirvesinden yüksek, RSI zirvesi düşük
    yakin_p, onceki_p = d.tail(10), d.iloc[-30:-10]
    neg_uyum = bool(
        yakin_p["High"].max() > onceki_p["High"].max()
        and yakin_p["RSI"].max() < onceki_p["RSI"].max()
    )
    daralma = bool(hist[-1] > 0 and hist[-1] < hist[-2] < hist[-3])

    return {
        "L1": bool(c > sma50 > sma200 and 45 <= rsi <= 65 and (yakin(c, sma20) or yakin(c, sma50)) and tutunma),
        "L2": bool(c > sma200 and rsi3.iloc[:-1].min() < 30 and rsi > 30),
        "L3": bool(c > s["ZIRVE20"] and hacim_kat >= 1.5 and rsi < 75),
        "S1": bool(c < sma50 < sma200 and 35 <= rsi <= 55 and (yakin(c, sma20) or yakin(c, sma50)) and red),
        "S2": bool(rsi > 75 and (daralma or neg_uyum)),
        "S3": bool(c < s["DIP20"] and hacim_kat >= 1.5 and rsi > 25),
        "_hacim_kat": hacim_kat, "_neg_uyum": neg_uyum, "_daralma": daralma,
        "_rsi_onceki": o["RSI"],
    }


def izleme_notu(d: pd.DataFrame, k: dict) -> list[str]:
    """Kurala yaklaşan ama tetiklenmeyen durumlar."""
    s = d.iloc[-1]
    notlar = []
    if s["Close"] > s["SMA50"] > s["SMA200"] and 45 <= s["RSI"] <= 65 and abs(s["Close"] - s["SMA20"]) / s["SMA20"] <= 0.04:
        notlar.append("L1'e yakın: yükseliş trendinde SMA20'ye %4 mesafede")
    if s["Close"] > s["SMA200"] and 30 <= s["RSI"] <= 35:
        notlar.append("L2'ye yakın: RSI 30-35 arasında, SMA200 üstünde")
    if s["ZIRVE20"] and 0 < (s["ZIRVE20"] - s["Close"]) / s["Close"] <= 0.02:
        notlar.append("L3'e yakın: 20 günlük zirveye %2'den az mesafe")
    if s["DIP20"] and 0 < (s["Close"] - s["DIP20"]) / s["Close"] <= 0.02:
        notlar.append("S3'e yakın: 20 günlük dibe %2'den az mesafe")
    if 70 <= s["RSI"] <= 75:
        notlar.append("S2'ye yakın: RSI 70-75 arasında")
    return notlar


def hisse_analiz(kod: str, df: pd.DataFrame) -> dict:
    d = gostergeler(df)
    s = d.iloc[-1]
    if len(d) < 210 or pd.isna(s["SMA200"]):
        return {"kod": kod, "durum": "yetersiz_gecmis", "gun_sayisi": len(d)}
    k = kurallar(d)
    dip, tepe = swing_noktalari(d)
    kurulumlar = []
    for kural in ("L1", "L2", "L3", "S1", "S2", "S3"):
        if not k[kural]:
            continue
        yon = "long" if kural.startswith("L") else "short"
        engel = None
        if yon == "long":
            adaylar = [x for x in (s["ZIRVE20"], s["ZIRVE52H"], tepe) if x and x > s["Close"]]
            engel = min(adaylar) if adaylar else None
        else:
            adaylar = [x for x in (s["DIP20"], s["DIP52H"], dip) if x and x < s["Close"]]
            engel = max(adaylar) if adaylar else None
        kurulumlar.append({
            "kural": kural, "yon": yon,
            **seviyeler(yon, s["Close"], s["ATR"], dip if yon == "long" else tepe, engel),
        })
    onceki_kapanis = d["Close"].iloc[-2]
    return {
        "kod": kod, "durum": "ok",
        "tarih": d.index[-1].strftime("%Y-%m-%d"),
        "kapanis": yuvarla(s["Close"]), "degisim_yuzde": yuvarla((s["Close"] / onceki_kapanis - 1) * 100),
        "sma20": yuvarla(s["SMA20"]), "sma50": yuvarla(s["SMA50"]), "sma200": yuvarla(s["SMA200"]),
        "rsi": yuvarla(s["RSI"], 1), "rsi_onceki": yuvarla(k["_rsi_onceki"], 1),
        "atr": yuvarla(s["ATR"]), "atr_yuzde": yuvarla(s["ATR"] / s["Close"] * 100),
        "macd": yuvarla(s["MACD"], 3), "macd_sinyal": yuvarla(s["MACD_SINYAL"], 3),
        "macd_hist": yuvarla(s["MACD_HIST"], 3),
        "macd_hist_daraliyor": k["_daralma"], "negatif_uyumsuzluk": k["_neg_uyum"],
        "zirve20": yuvarla(s["ZIRVE20"]), "dip20": yuvarla(s["DIP20"]),
        "zirve52h": yuvarla(s["ZIRVE52H"]), "dip52h": yuvarla(s["DIP52H"]),
        "hacim": yuvarla(s["Volume"], 0), "hacim_ort20": yuvarla(s["HACIM_ORT20"], 0),
        "hacim_kat": yuvarla(k["_hacim_kat"]),
        "swing_dip": yuvarla(dip), "swing_tepe": yuvarla(tepe),
        "trend": ("yukselis" if s["Close"] > s["SMA50"] > s["SMA200"]
                  else "dusus" if s["Close"] < s["SMA50"] < s["SMA200"] else "karisik"),
        "kurulumlar": kurulumlar,
        "izleme": izleme_notu(d, k) if not kurulumlar else [],
    }


def makro_analiz(veri: dict[str, pd.DataFrame]) -> list[dict]:
    sonuc = []
    for sembol, ad in MAKRO.items():
        df = veri.get(sembol)
        if df is None or len(df) < 2:
            sonuc.append({"sembol": sembol, "ad": ad, "durum": "veri_yok"})
            continue
        c = df["Close"]
        satir = {
            "sembol": sembol, "ad": ad, "durum": "ok", "tarih": df.index[-1].strftime("%Y-%m-%d"),
            "kapanis": yuvarla(c.iloc[-1], 4), "degisim_yuzde": yuvarla((c.iloc[-1] / c.iloc[-2] - 1) * 100),
        }
        if len(c) >= 200:
            satir["sma50"] = yuvarla(c.rolling(50).mean().iloc[-1], 2)
            satir["sma200"] = yuvarla(c.rolling(200).mean().iloc[-1], 2)
            satir["sma200_ustunde"] = bool(c.iloc[-1] > satir["sma200"])
        sonuc.append(satir)
    return sonuc


# ------------------------------------------------------------------------ çıktı

def markdown_yaz(rapor: dict) -> str:
    satirlar = [f"# BIST teknik tarama — {rapor['olusturma']}", ""]
    satirlar.append(f"Analiz edilen: {rapor['ozet']['analiz_edilen']} / {rapor['ozet']['toplam']} hisse"
                    f" · Kaynak: {rapor['ozet']['kaynaklar']}")
    satirlar.append(f"Referans işlem günü: {rapor['referans_gun']}")
    if rapor["ozet"]["gecikmeli"]:
        satirlar.append(f"Gecikmeli (referans günden eski veri): {', '.join(rapor['ozet']['gecikmeli'])}")
    if rapor["ozet"]["eksik"]:
        satirlar.append(f"Eksik: {', '.join(rapor['ozet']['eksik'])}")
    satirlar += ["", "## Makro", "", "| Gösterge | Kapanış | % | SMA200 üstü |", "| --- | --- | --- | --- |"]
    for m in rapor["makro"]:
        if m["durum"] == "ok":
            ust = {True: "evet", False: "hayır"}.get(m.get("sma200_ustunde"), "-")
            satirlar.append(f"| {m['ad']} | {m['kapanis']} | {m['degisim_yuzde']} | {ust} |")
    satirlar += ["", "## Kurulumlar", "",
                 "| Hisse | Kural | Kapanış | Giriş | Stop | H1 | H2 | Risk % | R/R engele | RSI | Hacim x |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h in rapor["hisseler"]:
        for k in h.get("kurulumlar", []):
            satirlar.append(
                f"| {h['kod']} | {k['kural']} | {h['kapanis']} | {k['giris_alt']}-{k['giris_ust']} | {k['stop']} | "
                f"{k['h1']} | {k['h2']} | {k['risk_yuzde']} | {k['rr_engele_kadar'] or '-'} | {h['rsi']} | {h['hacim_kat']} |")
    satirlar += ["", "## Tüm hisseler", "",
                 "| Hisse | Kapanış | % | Trend | RSI | SMA20 | SMA50 | SMA200 | ATR % | Hacim x | İzleme |",
                 "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for h in rapor["hisseler"]:
        if h["durum"] != "ok":
            satirlar.append(f"| {h['kod']} | {h['durum']} | | | | | | | | | |")
            continue
        satirlar.append(
            f"| {h['kod']} | {h['kapanis']} | {h['degisim_yuzde']} | {h['trend']} | {h['rsi']} | {h['sma20']} | "
            f"{h['sma50']} | {h['sma200']} | {h['atr_yuzde']} | {h['hacim_kat']} | {'; '.join(h['izleme'])} |")
    return "\n".join(satirlar) + "\n"


def main() -> int:
    kodlar = hisse_listesi()
    semboller = [f"{k}.IS" for k in kodlar] + list(MAKRO)
    print(f"{len(kodlar)} hisse ve {len(MAKRO)} makro gösterge çekiliyor...")
    veri = yahoo_cek(semboller)

    # Referans gün: BIST 100'ün son tamamlanmış günü. Hisse verisi bundan gerideyse yedek kaynağa bakılır.
    endeks = veri.get("XU100.IS")
    referans = endeks.index[-1].date() if endeks is not None and len(endeks) else None
    print(f"Referans işlem günü (BIST 100): {referans}")

    kaynak: dict[str, str] = {}
    hisseler = []
    for kod in kodlar:
        df = veri.get(f"{kod}.IS")
        kaynak[kod] = "yahoo"
        son = df.index[-1].date() if df is not None and len(df) else None
        if df is None or len(df) < 210 or (referans and son < referans):
            yedek = isyatirim_cek(kod)
            yedek_son = yedek.index[-1].date() if yedek is not None else None
            if yedek is not None and len(yedek) >= 210 and (son is None or yedek_son > son or len(df) < 210):
                df, kaynak[kod] = yedek, "isyatirim"
            print(f"{kod}: yahoo={son} isyatirim={yedek_son} -> {kaynak[kod]}")
        if df is None:
            hisseler.append({"kod": kod, "durum": "veri_yok"})
            continue
        try:
            sonuc = hisse_analiz(kod, df)
            sonuc["kaynak"] = kaynak[kod]
            if sonuc["durum"] == "ok":
                sonuc["gecikmeli"] = bool(referans and df.index[-1].date() < referans)
            hisseler.append(sonuc)
        except Exception as e:  # noqa: BLE001
            print(f"Analiz hatası {kod}: {e}", file=sys.stderr)
            hisseler.append({"kod": kod, "durum": f"hata: {e}"})

    tamam = [h for h in hisseler if h["durum"] == "ok"]
    rapor = {
        "olusturma": datetime.now(TR).strftime("%Y-%m-%d %H:%M TSİ"),
        "veri_tarihi": max((h["tarih"] for h in tamam), default=None),
        "referans_gun": str(referans) if referans else None,
        "ozet": {
            "gecikmeli": [h["kod"] for h in tamam if h.get("gecikmeli")],
            "toplam": len(kodlar), "analiz_edilen": len(tamam),
            "eksik": [h["kod"] for h in hisseler if h["durum"] != "ok"],
            "kaynaklar": ", ".join(sorted({h["kaynak"] for h in tamam})) or "-",
        },
        "aciklama": {
            "kurallar": "L1-L3 long, S1-S3 short; tanımlar README.md'de.",
            "seviyeler": "Giriş = kapanış ±0.5 ATR. Stop: swing dip/tepe ±0.25 ATR ile 1.5 ATR'den uzak olanı, en fazla 2 ATR. "
                         "H1 = 1.5R, H2 = 2.5R. rr_engele_kadar = en yakın direnç/destek mesafesi / risk.",
        },
        "makro": makro_analiz(veri),
        "hisseler": hisseler,
    }
    VERI.mkdir(exist_ok=True)
    (VERI / "latest.json").write_text(json.dumps(rapor, ensure_ascii=False, indent=1), encoding="utf-8")
    (VERI / "latest.md").write_text(markdown_yaz(rapor), encoding="utf-8")
    print(f"Tamam: {len(tamam)}/{len(kodlar)} hisse analiz edildi. Eksik: {rapor['ozet']['eksik']}")
    # Hiç veri gelmediyse iş başarısız sayılsın ki GitHub e-posta ile haber versin
    return 0 if tamam else 1


if __name__ == "__main__":
    sys.exit(main())
