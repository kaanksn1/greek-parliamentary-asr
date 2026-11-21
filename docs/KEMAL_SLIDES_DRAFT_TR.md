# Kemal — 2 slayt taslağı (Ozan'ın sunumu için)

Ozan'a girdi olarak hazırlandı; bütün sunumu üstlenmez. Sayılar
`results/benchmark/*/score.json` ve `results/training/*/run.json` dosyalarından
alındı (21 Kasım, dondurulmuş testin tamamı: 8.679 klip). Ayrıntılı analiz:
[KEMAL_BASELINES.md](KEMAL_BASELINES.md) → "Results and analysis".

---

## Slayt 1 — HMM/CTC ailesi: dört baseline

| Sistem | Nereden | WER % | CER % |
|---|---|---:|---:|
| wav2vec 2.0 (`lighteternal/wav2vec2-large-xlsr-53-greek`) | hazır Yunanca model, ek eğitim yok | 57,81 | 20,30 |
| MFCC + GMM-HMM | standart yöntem, kendi kodumuz ve eğitimimiz (CPU, 25 EM turu) | 89,15 | 49,67 |
| HuBERT + CTC (mHuBERT-147) | bizim CTC eğitimimiz (1.500 güncelleme) | 99,89 | 82,18 |
| Plain CTC (conv + BiLSTM) | sıfırdan bizim eğitimimiz (2.300 güncelleme) | 99,95 | 91,37 |

- Hepsi standart baseline ailesi; GMM-HMM, CTC ve HuBERT-CTC'de kredi kendi
  uygulama/eğitim emeğimiz için.
- HuBERT: Yunanca ASR başlıklı tek hazır model lisanssız ve model kartsız
  olduğu için seçilmedi; Yunancayı kapsayan mHuBERT-147'ye kendi CTC başlığımızı eğittik.

**Konuşmacı notu:** Dört sistemden en iyisi hazır wav2vec; hataları çoğunlukla
aynı okunan harflerin yanlış yazımı (ο/ω, ι/η), bu yüzden WER yüksek ama CER
düşük. Kendi eğittiklerimiz arasında en iyisi klasik GMM-HMM: CER'de attention
ve RNN-T'den daha iyi, güven aralıkları çakışmıyor; hazır Whisper-small ile
güven aralıkları çakışıyor, yani "denk" diyoruz, "daha iyi" demiyoruz.

---

## Slayt 2 — Neden iki CTC modeli başarısız, ne öğrendik?

- **Yetersiz eğitim:** dev kaybı son adımda hâlâ düşüyordu (HuBERT-CTC
  10,5 → 3,1 → 2,5); en iyi checkpoint son adımdı. Teslim bütçesinde 2–3 epoch.
- **Plain CTC her klip için tek bir "kelime" üretiyor** (`ταααααα…`): 8.679 klipte
  8.679 hipotez kelimesi; hataların %91'i silme. Model harf dağılımını öğrendi,
  kelime sınırlarını öğrenemedi.
- **GMM-HMM yakınsadı** (log-olabilirlik −108,8 → −100,5); WER'in yüksek olması
  kelime sözlüğü olmayan harf tabanlı kod çözücüden geliyor.
- **Veri gözlemi:** birkaç test referansı sesinden çok kısa (medyan 14,1 harf/sn;
  5 klip 4'ün altında). Örn. 6,5 saniyelik klibin referansı tek kelime
  (`διαστάσεις`), modeller bütün cümleyi yazıyor. Test kuralı gereği
  düzeltilmedi, puanlamada kaldı.

**Konuşmacı notu:** CTC sonuçları "yöntem çalışmıyor" demek değil; aynı yöntem
yeterli güncelleme ve veriyle çalışır. Bizim teslim süremizde iki CTC modeli
öğrenmenin başında kaldı ve bunu dev kaybı eğrisiyle gösterebiliyoruz. Tek seed
ve resmî, konuşmacıdan bağımsız olmayan bölme kullanıldı.
