# Adnan — 1-2 slayt taslağı (Ozan'ın sunumu için)

Bu taslak Ozan'a teslim edilecek bir girdidir; bütün sunumu veya ortak
karşılaştırmayı üstlenmez. Sayılar yalnız depodaki kodla üretilmiş gerçek
koşulardan gelecek; aşağıdaki sayı alanları hâlâ **[PENDING — Serdar'ın
merkezî koşumu]** olarak işaretli ve Serdar'ın gerçek GPU koşumu/Ozan'ın ortak
scoring'i tamamlanana kadar doldurulmamalı.

---

## Slayt 1 — Attention (hybrid CTC/attention) ve RNN-T

**Mimari**
- Attention: Listen, Attend and Spell (Chan ve ark., 2016) + Watanabe ve ark.
  (2017) hybrid CTC/attention yardımcı kaybı. Düz PyTorch/torchaudio.
- RNN-T: Graves (2012) transducer; unidirectional pyramidal LSTM encoder,
  LSTM prediction network, feed-forward joint network. Kayıp:
  `greek_asr.models.rnnt_loss` (saf PyTorch/autograd, log-uzayda forward
  algoritması) -- `torchaudio.functional.rnnt_loss`'un derlenmiş CUDA/ROCm
  kernel'i Serdar'ın RX 7800 XT'sinde bulunamadığı için değiştirildi (bkz.
  `docs/reviews/ADNAN_0437631_GPU.json`, bulgu 3).
- İkisi de **baseline ailesi**dir (bkz. `docs/BASELINES.md`); kredi kendi
  uygulama/eğitim emeğimiz içindir.

**Durum**
- Kod, config (`configs/sequence/attention.json`,
  `configs/sequence/transducer.json`) ve 28 testin tamamı CPU'da geçiyor
  (sentetik tensörler + 3 gerçek train-pilot klip).
- Gerçek eğitim/dev seçimi/test sonucu: **[PENDING — Samed'in veri teslimi +
  Serdar'ın GPU koşumu]**.
- WER / CER (dev, final test): **[PENDING]**.

**Konuşmacı notu:** Bu iki model "baseline" sayılır çünkü mimarileri standart;
kredi aldığımız kısım sıfırdan kendi yazıp eğitmemiz. Şu an hiçbir sayı yok
çünkü hem veri (Samed) hem merkezî GPU koşumu (Serdar) bekliyor. Yeni RNN-T
loss backend'i (saf PyTorch) CPU'da torchaudio'nun kendi kaybına karşı
değer+gradyan eşitliğiyle doğrulandı, ama gerçek GPU/ROCm hızı hâlâ
ölçülmedi -- bu ayrı bir açık madde.

---

## Slayt 2 — Whisper fine-tuning katkısı

**Katkı tasarımı**
- Başlangıç: `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d`
  (Serdar'ın eğitimsiz baseline'ı ayrı/değişmeden kalır).
- İki koşul, tek eğitim kodu: HParl-only vs. HParl + CSS10 Greek (yalnız
  eğitimde). Aynı seed/checkpoint/tokenizer/dev/test/scoring; eşlenmiş
  effective batch ve optimizer update sayısı (yalnız veri bileşimi değişiyor).

**Durum**
- Kod + 2 config + ilgili testler CPU'da geçiyor (gerçek pinned tokenizer +
  gerçek optimizer adımı, küçük rastgele model ile).
- `max_steps` kasıtlı olarak boş: Serdar'ın GPU pilot ölçümü (adım süresi,
  VRAM) + Samed'in gerçek veri saatleri olmadan rastgele bir sınır seçilmedi.
- HParl-only vs. HParl+CSS10 WER/CER karşılaştırması: **[PENDING]**.

**Konuşmacı notu:** Burada iddia ettiğimiz "bonus" yok; sadece iki koşulu adil
şekilde karşılaştırmak için bütçeyi (adım sayısı, effective batch) önceden
eşitledik ki fark gerçekten veri bileşiminden gelsin, ekstra eğitim
süresinden değil. Karışık veri daha kötü çıkarsa da aynı kurallarla
raporlanacak.
