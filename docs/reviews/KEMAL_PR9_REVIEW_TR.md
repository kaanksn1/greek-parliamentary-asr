# Kemal PR #9 — kod, runtime ve HuBERT kararı

19 Kasım 2026. İncelenen commit:
**`680b875d533b94da378ddf940b66b9a97cf55baa`**,
[PR #9](https://github.com/kaanksn1/greek-parliamentary-asr/pull/9).
Model implementasyonları, testler ve Kemal'in kişisel katkı dosyası kendi
commit'leriyle korunur. Owner yalnız merkezi doğrulama, kaynak seçimi,
entegrasyon ve belgeleri tamamladı.

**Sonuç:** dört baseline ailesinin kod paketinde incelenen yolda yeni bir
engelleyici hata bulunmadı. **164/164 test, 0 atlanan, 44,536 saniye** ile mevcut
ROCm ortamında bağımsız olarak geçti. Gerçek GMM-HMM CPU, plain CTC GPU,
lisanslı mHuBERT-147 + CTC GPU ve yayımlanmış wav2vec GPU yolları da çalıştı.
Bu, tam eğitim/nihai benchmark sonucu değildir.

## HuBERT için kesinleşen kaynak seçimi

1. `ntinouldinho/hubert_greek_6gram`
   @ `e89272763ec7320f5cf57638dfdedc3ecaa1ce86` **ana benchmark'a seçilmedi**.
   Exact revision'ın model API kaydında lisans alanı boş; README/model kartı
   veya LICENSE dosyası yok. Eğitim verisi de açıklanmıyor. Bu bulgu
   "Yunanca transkripsiyon üretemez" demek değildir; Kemal'in CPU kanıtları
   korunur. Eksik hak/provenance kaydını varsayımla tamamlamıyoruz.
2. Seçilen yol: `utter-project/mHuBERT-147`
   @ `7ad3fc0bc5106c58c9c13526abccad527150d135` üzerine **kendi Yunanca CTC
   eğitimimiz**. Resmî metadata Yunancayı (`el`) ön eğitim dilleri arasında
   listeliyor; [model kartı](https://huggingface.co/utter-project/mHuBERT-147)
   lisansı **CC-BY-NC-SA-4.0** olarak açıklıyor.
3. Bu encoder tek başına ASR checkpoint'i değildir. Tam eğitim tamamlanınca
   sonuç "grubumuz tarafından eğitilmiş standart HuBERT-CTC baseline'ı" diye
   raporlanacak; hazır Yunanca HuBERT çıkarımı diye etiketlenmeyecek. Standart
   aileyi kendimiz eğitmek, hocanın açıklamasına göre o aileyi yeni mimari yapmaz.
4. Kendi kodumuzun MIT lisansı model ağırlıklarını yeniden lisanslamaz.
   Model kaynağı/atıf, değişiklikler ve dağıtılan türev ağırlıkların lisans
   koşulları korunacak. Lisansın resmî açıklaması:
   [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Exact-revision birincil metadata snapshot'ı:
[KEMAL_HUBERT_SOURCES_2026-11-19.json](KEMAL_HUBERT_SOURCES_2026-11-19.json).
Karar `configs/kemal_model_sources.json`, README, baseline envanteri ve yöntem
belgesine işlendi. Yayımlanmış lisansı belirsiz adayın adapter'ı tarihsel
araştırma kodu olarak korunur; planlanan benchmark komutlarında kullanılmaz.

## Bağımsız gerçek runtime doğrulaması

[Tam makine çıktısı](KEMAL_680b875d_RUNTIME.json), owner betiği:
`scripts/probe_kemal_submission.py`.

| Yol | Gerçek kontrol |
|---|---|
| MFCC + GMM-HMM, CPU | Üç train klibinde üç küçük Viterbi/mixture iterasyonu; save/load ve tam ID'li predict başarılı |
| Plain CTC, GPU | Standart boyuttaki Conv/BiLSTM, iki optimizer update; bütün mevcut gradient'ler sonlu, parametre değişti; checkpoint/predict başarılı |
| mHuBERT-147 + CTC, GPU | Gerçek lisanslı encoder ve yeni Greek CTC başlığı; transformer açıkken iki update, sonlu gradient ve parametre değişimi; checkpoint/predict başarılı |
| Yayımlanmış wav2vec, GPU | Tam pinli gerçek checkpoint, audio-only ortak runner; üç train ID'sine eksiksiz tahmin |

Ortam: RX 7800 XT, Python 3.12.3, PyTorch ROCm 2.9.1, TorchAudio ROCm 2.9.0,
Transformers 4.57.6, NumPy 1.26.4; tam sürümler JSON ve
`requirements-rocm.lock.txt` içinde. Yeni paket kurulmadı. Desteklenen GPU
operasyonları kullanıldı; NVIDIA attention hızlandırmasının bulunmaması
CPU fallback veya model başarısızlığı diye sayılmadı.

Yalnız üç özgün **train** pilotu, toplam 6,61 saniye kullanıldı. Geçici dev
rolü de bu train pilotlarından oluşan bir kod testi fixture'ıdır; resmî dev
değildir ve gerçek ayar seçimi olarak kullanılmaz. Kaynak ses/ref dosyaları
değişmedi. Training fixture hedeflerinde annotation etiketi yalnız bellekte
hazırlanır. Geçici eğitilmiş ağırlıklar silindi; lisanslı yayımlanmış encoder
ve wav2vec ağırlıkları tekrar kullanım için yerel cache'te kalır.

Kemal'in iki yayımlanmış checkpoint için commit ettiği CPU kayıtlarının bütün
kod hash'leri ve source-lock hash'leri `680b875d` checkout'ıyla eşleşti.
Owner lisansı belirsiz yayımlanmış HuBERT'i yeniden indirmedi/çalıştırmadı.
Bu kayıtlar tanılama/araştırma kanıtıdır, ana sonuç tablosu değildir.

## Entegrasyon ve kalan gerçek işler

Owner ortak inference kaydına, yayımlanmış HuBERT adapter'ının kullandığı
`wav2vec.py` helper dosyasının hash'ini de ekledi. Model matematiği, mimari,
bağımlılık pinleri, sayısal eğitim bütçeleri ve tarihsel source lock değişmedi.

Bu owner entegrasyonu `df8a300dd22f6f24a3698c78811acd0703187839` ile Samed'in
`4db1a4aa61eb78acbb531b705c45e3bbd090854e` veri kodunun geçici birleşiminde
**196/196 test, 0 atlanan, 46,585 saniye** ile geçti. Yalnız ortak
`contributions/README.md` envanteri çakıştı; dosyaların gerçek varlık durumuyla
çözüldü. Kişisel raporlar ve model/veri kaynakları değiştirilmedi. Büyük LFS
sesleri indirilmedi; bu kontrol tam ses teslimi/hash doğrulaması değildir.

GMM-HMM/CTC infeasible train örnekleri açık ID/gerekçe ile raporlanıyor;
CTC blank/tekrar ve valid-length decoding testleri mevcut. Dev annotation
hazırlığı ve geçici dev WER politikası belgeli; Ozan'ın son ortak scoring
politikası seçildikten sonra gerçek dev tuning o politikayla yapılmalı.

`*_measure.json` dosyalarının temsil edici tam koşuları **henüz yapılmadı**.
GMM ölçüm config'i 2 saatlik train grubuna, CTC/HuBERT ölçüm config'leri gerçek
train/dev manifestlerine ihtiyaç duyuyor. Ana depodaki üç kısa klip bu verinin
yerine geçmez. Sonraki iş: kullanılan gerçek train sesini edinip temsil edici
süre/VRAM/RAM ölçümü, bütçe seçimi ve Samed'in seçilmiş ham/temiz train sesini
depoya teslimi. GMM-HMM/CTC/HuBERT tam eğitimleri, gerçek dev seçimi ve final
test karşılaştırması bundan sonradır. `max_steps:null` alanları doldurulmadı.

Nihai test verisi/ref'leri değişmedi; resmî dev/test üzerinde çıkarım veya
WER/CER benchmark yapılmadı. Kısa diagnostiklerden epoch maliyeti, RTFx veya
istatistiksel yeterlilik çıkarılmadı.
