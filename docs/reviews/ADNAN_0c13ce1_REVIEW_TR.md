# Adnan PR #5 — düzeltmelerin doğrulanması, 19 Kasım 2026

İncelenen commit: **`0c13ce1ac51b561dc9b1d284a46bcbebda512ad9`**,
[PR #5](https://github.com/kaanksn1/greek-parliamentary-asr/pull/5).
Önceki inceleme `0437631` içindi; o sürümün hataları yeni sürümde hâlâ varmış
gibi değerlendirilmedi. Bu inceleme teammate model kaynaklarını değiştirmedi.

**Sonuç: önceki ana eğitim engelleri kapanmış.** Yeni sürümün 114/114 testi
geçti. Attention, RNN-T ve Whisper'ın gönderilen üç eğitim-pilotu üzerinde
GPU forward/backward/optimizer kontrolleri başarılı. Attention/RNN-T gerçek
train/checkpoint/predict CLI yolları da GPU'da doğrulandı. Tam veri eğitim
bütçesi, tam koşular ve son benchmark henüz yapılmadı. Aşağıdaki P2 kod ve
belge düzeltmeleri kalıyor; mevcut sonlu float32 pilot yolunun çalışmadığı
iddia edilmiyor. PR #5 bu incelemede birleştirilmedi.

## Gerçek doğrulama ve kanıtlar

Ortam: RX 7800 XT, WSL Python 3.12.3,
PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`,
TorchAudio `2.9.0+rocm7.2.1.gite3c6ee2b`, Transformers `4.57.6`, NumPy `1.26.4`.
Mevcut izole runtime kullanıldı; paket kurulmadı. Tam pinler
`requirements-rocm.lock.txt` içinde. Test komutu:
`python -m unittest discover -s tests -v` — 114 test, 33,911 saniye, başarılı.
Samed'in `4db1a4aa61eb78acbb531b705c45e3bbd090854e` dalıyla geçici birleşme
çakışmasızdı; birleşik **146/146 test 35,992 saniyede geçti**. Merge commit'i
üretilmedi. Büyük LFS sesleri bu entegrasyon kontrolünde indirilmedi;
bu sonuç tam ses teslimi/hash doğrulaması değildir.

| Gönderilen model pilotu | Sonlu loss | Bütün mevcut gradient'ler sonlu | Gözlenen parametre güncellemesi |
|---|---|---|---|
| Attention / hybrid CTC-attention | Evet | Evet | Evet |
| RNN-T / yeni pure-PyTorch loss | Evet | Evet | Evet |
| Whisper fine-tuning | Evet | Evet | Evet |

- [GPU kanıtı](ADNAN_0c13ce1_GPU.json): gönderilen pilot fonksiyonları
  değiştirilmeden çağrıldı. Owner betiği optimizer adımını gözlemleyerek
  gradient sonluluğunu ve ilk gradient'li parametrenin değişimini kaydetti.
  PCM32 dört örnek doğru olarak dört örneğe çözüldü; kaynak hash'i değişmedi.
  Whisper decoder girdisi `[50258, 50281, 50359, 50363, ...]`, çift başlangıç yok.
  Gönderilen guard sentetik 31 saniyelik girdiyi açıkça reddetti. JSON'daki
  feature-extractor-only deneyi guard'ı kullanmayan upstream davranışı gösterir;
  artık modelin 31 saniyeyi sessizce kestiği anlamına gelmez.
- [CLI kanıtı](ADNAN_0c13ce1_ENTRYPOINTS.json): her iki gerçek model CLI'ı GPU'da
  2 optimizer güncellemesi / 6 microbatch tamamladı (accumulation=3), best
  checkpoint'ten yükleyip üç istenen train ID'sine eksiksiz tahmin üretti.
  Bu küçük sayılar yalnız diagnostik bütçedir; gerçek eğitim config'lerine
  aktarılmadı. Geçici ağırlıklar kontrolün sonunda kaldırıldı.
- [Bağımsız RNN-T matematik kanıtı](ADNAN_0c13ce1_RNNT_MATH.json): 16 küçük
  lattice şekli × batch 2 için bütün hizalamalar ayrı bir oracle ile sayıldı.
  Değişken uzunluklar, boş hedef, U>T ve sıfır olmayan blank ID dahil.
  Float64 loss/gradient farkları `1e-10` toleransının altında; hepsi başarılı.
  Bu algoritma doğrulaması GPU hız ölçümü değildir.

CLI diagnostik fixture'ının `dev` rolü, gönderilen unit testlerde olduğu gibi
**özgün train pilotundan** yapılmıştır. Resmî development seti değildir,
başarı/ayar seçimi iddiası taşımaz. Gerçek test veya dev çıkarımı yapılmadı;
gerçek veri dosyaları veya referansları değiştirilmedi. Üç kaynak train klibi
toplam 6,61 saniyedir. Bir-adımlık kontrollerden epoch maliyeti, RTFx,
istatistiksel yeterlilik veya tam eğitim kapasitesi çıkarılmadı.

Owner'ın eski inceleme betiği yeni zorunlu `decoder_start_token_id` argümanına
uyarlandı. Bu bir owner-harness uyarlamasıdır, yeni model hatası değildir.
Yeni JSON'lar hem gönderilen kodu hem çalışan diagnostik betiğini hash'ler.

## Doğrulanan önceki düzeltmeler

- PCM16/PCM32 dtype ve ölçekleri doğru; desteklenmeyen format açık hata veriyor.
- Attention maskesi doğru cihazda. Gerçek GPU forward/backward artık başarılı.
- Attention/RNN-T train, dev-role fixture ve predict yollarında cihaz aktarımı
  var; GPU isteği sessizce CPU'ya düşürülmüyor.
- RNN-T yeni loss'u gerçekten çağırıyor ve `rnnt_loss_backend` alanına yazıyor.
- Whisper yalnız gerçek decoder-start token'ını label başından çıkarıyor;
  dil/task prefix'i, EOS ve attention-mask tabanlı padding korunmuş.
- Whisper train/predict yolları >30 saniyeyi feature extraction öncesi reddediyor.
- Attention/RNN-T `max_steps` artık optimizer güncellemesi; accumulation sonraki
  epoch'a taşınıyor. Gerçek CLI kontrolü bu yolu da çalıştırdı.
- HParl-only/mixed Whisper config'leri başlangıç modeli/revision, seed, ortak
  dev ve optimizer bütçesi bakımından eşleşiyor. Bütçe hâlâ bilinçli `null`.
- Resume desteğinin bulunmadığı artık açık yazılmış. Katkı ve baseline ayrımı
  korunmuş; bare encoder veya yanlış dil modeliyle satır doldurulmamış.

## Kalan P2 kod düzeltmeleri

1. **NaN atlandığında Whisper accumulation fazı bozulabiliyor.**
   [whisper_finetune.py:306–312](https://github.com/kaanksn1/greek-parliamentary-asr/blob/0c13ce1ac51b561dc9b1d284a46bcbebda512ad9/greek_asr/models/whisper_finetune.py#L306)
   sonlu olmayan loss'ta gradient'leri siliyor, fakat accumulation konumunu
   sıfırlamıyor. Bir sonraki update beklenenden az microbatch içerebilir.
   Tüm loss'lar sonlu değilse başarılı update bütçesi hiç dolmayıp döngü sürebilir.
   En basit çözüm açık hata ile durmak; atlama korunacaksa cycle sayacı doğru
   sıfırlanmalı, deneme/ardışık hata sınırı ve anlamlı regresyon testi olmalı.
   Bu durum gerçek pilotta görülmedi; normal sonlu yolun bozuk olduğu söylenmiyor.
2. **RNN-T'nin ignored-padding sözleşmesi kodla uyuşmuyor.**
   [rnnt_loss.py:64–65](https://github.com/kaanksn1/greek-parliamentary-asr/blob/0c13ce1ac51b561dc9b1d284a46bcbebda512ad9/greek_asr/models/rnnt_loss.py#L64)
   padding maskesinden önce bütün hedef indekslerini `gather` ediyor.
   Gerçek hedef uzunluğunun dışındaki `-100` padding out-of-bounds hatası veriyor.
   Yukarıdaki matematik JSON'u bunu ayrı kaydeder. **Mevcut collate 0 ile pad
   ediyor**, dolayısıyla mevcut koşu bundan etkilenmiyor. Ya maskeden önce
   güvenli indeks kullanılsın ya da yalnız geçerli vocabulary ID'lerinin padding
   olabileceği sözleşmesi açıkça yazılsın; "padding hiç okunmaz" iddiası kalkmalı.

## README/config/belgelerde eşleştirilecek bilgiler

- `configs/sequence/transducer.json` hâlâ `loss_backend` olarak
  `torchaudio.functional.rnnt_loss` yazıyor. Gerçek backend
  `pure_pytorch_logspace`. `docs/SEQUENCE_MODELS.md:147` de eski backend'i söylüyor.
- `rnnt_loss.py:14–18` ve `SEQUENCE_MODELS.md:63–69` içindeki "donanım
  doğrulaması gerekmez" iddiası düzeltilmeli. Standart ops veya meta-device
  testi gerçek GPU kontrolünü kaldırmaz. Bu incelemenin gerçek RX 7800 XT
  kontrolü şimdi başarılı; başka donanım ve tam eğitim boyutu ayrıca ölçülür.
- `whisper_finetune.py` giriş açıklaması owner baseline'ın uzun sesi sessizce
  kestiğini, `SEQUENCE_MODELS.md:192–196` owner'ın attention mask vermediğini
  söylüyor. Gerçek `greek_asr/pretrained/whisper.py:51–60` açıkça
  `truncation=False`, attention mask ve long-audio timestamps kullanıyor.
- `SEQUENCE_MODELS.md:165–167`, train ve predict'in ikisinin de model source
  lock doğruladığını söylüyor. Predict gerçekte checkpoint config'ini okuyup
  loader'ı çağırıyor; `_validate_revision` çağrısı yok. Ya download öncesi
  doğrulama ve ilgili predict testi eklensin ya da açıklama gerçek davranışı
  doğru anlatsın. Mevcut seçilmiş checkpoint'in yanlış olduğu saptanmadı.
- `SEQUENCE_MODELS.md:313–324`, kodun dev WER yazdığı ve dev loss'un scorer
  normalizasyonuyla yeniden hesaplandığı izlenimini veriyor. Gerçekte best
  checkpoint teacher-forced ham dev loss ile seçiliyor; WER raporlanmıyor.
  Bu seçme ölçütü açıkça belgelenmeli; Ozan'ın ortak WER/CER normalizasyonu
  ayrı scoring aşamasında kalmalı. Yeni bir model-specific scorer gerekmiyor.
- `requirements-sequence.txt` yorumunda CPU ve ROCm paketlerinin major/minor
  sürümlerinin aynı olduğu söyleniyor; listelenen sürümler aynı değil.
  Bu **uyumsuzluk kanıtı değildir**: mevcut ROCm ortamında testler ve GPU
  kontrolleri geçti. Yalnız açıklama ve ortamların ayrımı doğru yazılmalı.
- PR açıklamasındaki "GPU doğrulanmadı", "split metadata yok" gibi ifadeler
  sürüm/tarih bağlamıyla güncellenmeli. PR #2 henüz birleşmedi; main'de
  manifest yokluğu, Samed'in kendi dalında metadata bulunmadığı anlamına gelmez.

## Adnan'a kısa takip promptu

> `0c13ce1` sürümünü kontrol ettik: 114 test geçti; üç model de RX 7800 XT'de
> sonlu loss/gradient ve gerçek parameter update verdi. Attention ve RNN-T'nin
> train/checkpoint/predict CLI'ları da GPU'da başarılı. Önceki ana engeller kapandı.
>
> Aynı PR'da şu son düzeltmeleri yap:
> 1. Whisper nonfinite-loss yolunu fail-fast yap veya accumulation konumunu
>    doğru sıfırlayıp deneme sınırı koy; anlamlı hata-yolu testi ekle.
> 2. RNN-T padded hedeflerinin maskeden önce gather edilmesi ile "padding
>    okunmaz" sözleşmesini uyumlu yap; mevcut collate'ın 0 kullandığını koru.
> 3. Transducer config'indeki ve mimari belgesindeki eski loss backend'ini düzelt.
> 4. Owner Whisper adapter'ı, attention mask/uzun ses, GPU doğrulaması,
>    CPU/ROCm sürümleri ve dev loss/WER hakkındaki yanlış ifadeleri düzelt.
> 5. Whisper predict source-lock kontrolünü açıklamayla eşleştir; kontrol
>    ekliyorsan yanlış revision'ın loader çağrılmadan reddedildiğini test et.
>
> Kendi katkı dosyanı/PR açıklamanı yalnız gerçek iş ve bu kanıtlara göre güncelle;
> merkezî GPU koşularını kendi bilgisayarında yapılmış gibi yazma. Testi,
> kaynak lock'unu veya Samed'in verisini değiştirme. Rastgele eğitim bütçesi
> seçme. Yeni commit SHA ve test sonucunu gönder. Tam veri boyutunda pilot ve
> bütün tam koşular Serdar'da kalacak; bu başarı henüz benchmark sonucu değil.

## Yeniden üretim

Geçerli Linux checkout ve mevcut izole ROCm ortamında, yeni çıktı yollarıyla:

```bash
python -m unittest discover -s tests -v
HSA_ENABLE_DXG_DETECTION=1 python scripts/probe_sequence_submission.py --submission-root /path/to/submission --device cuda --output outputs/reviews/gpu-new.json
HSA_ENABLE_DXG_DETECTION=1 python scripts/probe_sequence_entrypoints.py --submission-root /path/to/submission --device cuda --output outputs/reviews/cli-new.json
python scripts/verify_rnnt_submission.py --submission-root /path/to/submission --output outputs/reviews/math-new.json
```

İlk iki diagnostik gerçekten küçük training step'leri çalıştırır; salt import
kontrolü değildir. Matematik betiği yalnız sentetik CPU tensor'ları kullanır.
Hiçbiri nihai test protokolüne veya tam benchmark'a izin vermez.
