# PR #2 ve PR #5 — ikinci inceleme, 18 Kasım 2026

**Sonuç:** Samed'in önceki metadata/raporlama düzeltmeleri doğrulandı; gerçek
ses teslimi tamamlanmadı. Adnan'ın test sayısı doğru, fakat teslim yalnız veri
beklemiyor: ses çözme, GPU eğitimi ve Whisper hedeflerinde engelleyici hatalar
var. Bu incelemede iki PR da main'e birleştirilmedi. Geçici bir kopyada birlikte
test edilmeleri kabul/merge anlamına gelmez.

| İncelenen teslim | Sabit commit | Bağımsız doğrulama |
|---|---|---|
| [Samed / PR #2](https://github.com/kaanksn1/greek-parliamentary-asr/pull/2) | `e5aa488c08929c5d55f115ddbacb52120b9e1ea3` | 76/76 test; metadata/rapor/freeze kontrolü |
| [Adnan / PR #5](https://github.com/kaanksn1/greek-parliamentary-asr/pull/5) | `043763190b0cc35759cf58d2abe36955c48102dd` | 93/93 test; üç eğitim pilotuyla gerçek GPU kontrolü |
| İki PR'ın geçici birleşimi | Yukarıdaki iki commit; merge commit'i üretilmedi | Çakışmasız birleşme; 125/125 test, 34,042 saniye |

Ortak testler iki dalda da bulunduğundan 76 + 93 = 169 ayrı test yoktur.
Komut: `python -m unittest discover -s tests -v`. Adnan ve birleşik testler,
geçerli Linux `.git` dizini olan yerel WSL kopyalarında çalıştırıldı. Windows
worktree'sinin Windows yoluna işaret eden `.git` dosyasını WSL Git'in
çözememesi model koduna ait bir hata olarak sayılmadı.

Kullanıcının 18 Kasım yanıtına göre teslim **21 Kasım 2026 Cumartesi akşamı**,
`Europe/Istanbul`; kesin saat verilmedi. Buradan keyfî bir eğitim adımı/saat
bütçesi türetilmedi.

## Samed: doğrulanan düzeltmeler

[Makine çıktısı](SAMED_e5aa488.json),
[ilk inceleme](SAMED_PR2_REVIEW_TR.md). İlk inceleme eski commit'e aittir;
aşağıdaki düzeltmeler o tarihsel raporu geçersiz bir olay kaydı yapmaz.

- Özgün HParl Kaldi dosyaları ve CSS10 `transcript.txt` artık commit edilmiş:
  16 kaynak dosyasının byte hash'leri doğrulandı.
- Test manifesti, referansları ve freeze JSON'u ilk `d28c97f` commit'inden beri
  değişmemiş. Metadata kontrolleri başarılı. **Tam test sesi hash kontrolü
  yapılmış sayılmıyor**, çünkü bu ses dosyaları checkout'ta yok.
- Literal test cümlesi sayısı ham eğitimde 14 örnek / 14 ayrı metin;
  iki temiz eğitim koşulunda da 6 / 6. Normalize edilmiş sonuç artık ayrı
  açıklanıyor: 22 örnek / 21 ayrı metin.
- README resmî oturum bölünmesini ve konuşmacı örtüşmesini açıkça belirtiyor.
  Train/test ortak 76 konuşmacı, testte train konuşmacısına ait 7.777 örnek;
  oturum kesişimi yok. Bu sonuç görülmemiş konuşmacıya genelleme ölçümü değil.
  Hocanın Mail 1 / How 1 resmî split istisnası kullanılıyor; test yeniden bölünmedi.
- CPS kuralı revize: yalnız >60 karakter/saniye olan 22 kayıt çıkarılıyor;
  40–60 arası yedi kayıt inceleme kuyruğunda korunuyor. Bu bir sezgisel risk
  filtresi olarak anlatılıyor. İnsan tarafından doğrulanmış karar sayısı sıfır.
- Temizlik günlüğü ve özet hash'leri tutarlı; rapor yeniden üretildiğinde
  commit edilmiş belgeyle byte byte aynı. 17 aynı-ses tekrarı da çıkarılmış.
- Temiz eğitim manifestleri: HParl 76.302 örnek / yaklaşık 99,487 saat;
  HParl+CSS10 78.146 / 103,625 saat. Bunlar teslimin metadata değerleridir;
  sahibin tüm ses baytlarını bağımsız yeniden denetlediği iddiası değildir.

**Açık P1: gerçek sesler depoda değil.** Ham envanterin işaret ettiği 93.977
yoldan hiçbiri inceleme checkout'ında bulunmuyor; eski üç eğitim pilotu ayrı
yolda duruyor. Özgün metadata eklenmesi ses tesliminin yerini tutmaz.
`configs/data_delivery.json` bunu artık doğru biçimde bekleyen iş olarak yazıyor.
Kayıtlı dosya boyutlarından tam kullanılan ses yaklaşık 14,08 GiB; yalnız
test+dev+CSS10 yaklaşık 3,399 GiB. Plandaki 10 GiB bir varsayımdır, hesabın
gerçekte kalan kotası doğrulanmış değildir. Ücretli depolama alınmadı.

Serdar/Samed, gerçek kota ve ölçülmüş eğitim bütçesine göre repo içi teslimi
tamamlamalı. Gerekirse yalnız gerçekten kullanılacak eğitim alt kümesi
train tarafından, sabit seed ve kayıtlı grup seçimiyle belirlenir; donmuş test
korunur. Release/Drive bağlantısı veya indirme betiği tek başına kabul edilmez.
LFS kullanılırsa pointer'ların sunucuda erişilebilir gerçek blob'ları olmalı;
temiz clone'da `verify-freeze` **`--skip-audio` olmadan** ve tam veri doğrulaması
başarılı olmalı. Ham ve temiz manifest aynı değişmemiş sesi paylaşabilir;
temiz metinler, günlük ve lineage yine teslim edilir.

## Adnan: eğitimi engelleyen bulgular

[Gerçek GPU/sentetik sözleşme çıktısı](ADNAN_0437631_GPU.json).
İnceleme ortamı: RX 7800 XT, WSL Python 3.12.3,
PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`,
TorchAudio `2.9.0+rocm7.2.1.gite3c6ee2b`, Transformers `4.57.6`, NumPy `1.26.4`.
Tam kurulum pinleri `requirements-rocm.lock.txt` içinde. Adnan'ın CPU
requirements dosyası bu mevcut GPU ortamının üzerine kurulmadı.

1. **P1 — CSS10 PCM32 yanlış çözülüyor.**
   [common.py:90–101](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/common.py#L90)
   header'dan sample width okusa da bütün payload'ı `int16` yorumluyor.
   Samed'in `items_css10_el.jsonl.gz` denetiminde CSS10'un 1.844 kaydının
   tamamı 4-byte sample genişliğinde. Geçerli sentetik PCM32 dosyasında dört
   örnek yerine sekiz örnek döndü; genlik ve süre yanlış. Bu hata karışık veri
   fine-tuning deneyini doğrudan bozar; kaynak sesi yeniden yazmak çözüm değil.

2. **P1 — attention GPU pilotu cihaz uyuşmazlığında duruyor.**
   [common.py:203–208](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/common.py#L203)
   maskeyi CPU `torch.arange` ile kuruyor, uzunluklar GPU'da.
   Gönderilen pilot fonksiyonu değiştirilmeden çalıştırıldığında
   `Expected all tensors to be on the same device ... cuda:0 and cpu` verdi.
   CPU testinin geçmesi bu yolu doğrulamıyor.

3. **P1 — RNN-T loss mevcut AMD ortamında GPU kernel'i bulamıyor.**
   [transducer.py:200–204](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/transducer.py#L200)
   gerçek pilotta `torchaudio::rnnt_loss ... 'CUDA' backend` hatası verdi.
   ROCm PyTorch burada da `cuda` cihaz/dispatch adını kullanır. Fonksiyon CPU'da
   var; bu bulgu "RNN-T kaldırılmış" veya "her ortamda çalışmaz" demek değil.
   Kullanılacak çalışan loss backend'i/build'i ya da açıkça belgelenip ölçülen
   başka bir yürütme yolu gerekli. Autograd kayıtları GPU kernel kanıtı değil.

4. **P1 — tam attention/RNN-T train ve predict komutları CPU'da kalıyor.**
   [attention.py:292–315](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/attention.py#L292),
   [transducer.py:213–234](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/transducer.py#L213):
   model/batch GPU'ya taşınmıyor; CLI'da cihaz seçimi de yok. Pilotun ayrı
   `.to(device)` çağrısı bu tam eğitim yolunu düzeltmiyor. Aynı eksiklik predict
   için de var. Dev, checkpoint yükleme ve çözümleme de aynı cihaz sözleşmesine
   dahil olmalı.

5. **P1 — Whisper decoder başlangıcı iki kez ekleniyor.**
   [whisper_finetune.py:85–100](https://github.com/kaanksn1/greek-parliamentary-asr/blob/043763190b0cc35759cf58d2abe36955c48102dd/greek_asr/models/whisper_finetune.py#L85)
   label başında `50258` bırakıyor; modelin otomatik sağa kaydırması yine
   `50258` ekliyor. Sabitlenmiş gerçek tokenizer/config ile decoder girişinin
   başı `[50258, 50258, 50281, 50359, 50363, ...]` olarak doğrulandı.
   [Hugging Face'in standart collator tarifi](https://github.com/huggingface/blog/blob/main/fine-tune-whisper.md)
   bu yüzden label başındaki gerçek `decoder_start_token_id` değerini çıkarır.
   Dil/task prefix'i, gerçek EOS ve padding maskesi korunmalı; tokenizer'ın
   farklı olabilen `bos_token_id` alanıyla kör karşılaştırma yapılmamalı.

Whisper pilotunda bir optimizer adımı GPU'da tamamlandı, loss sonluydu;
yaklaşık 9,59 GB peak allocated bellek kaydedildi. Bu, yukarıdaki label
hatasını veya tam veri eğitimini doğrulamıyor. Üç eski train klibi toplam
6,61 saniye; resmî dev/test çıkarımı ve WER/CER yapılmadı. Tam eğitim yok,
eğitilmiş checkpoint saklanmadı. JSON'da dış duvar saati ile pilotun içeride
ölçtüğü adım süresi ayrı; kısa pilottan epoch/RTFx sonucu çıkarılmıyor.

## Tam koşudan önce kapatılacak diğer noktalar

- **P2 — uzun sesi sessiz kesme.** Whisper feature extractor varsayılanıyla
  31 saniyelik iki sentetik sinyalin farklı son saniyesi tamamen kayboldu;
  özellikler aynı çıktı. Ancak mevcut temiz HParl train maksimum 15,1 s,
  mixed train 26,4 s, dev 14,62 s: PCM32 hatası düzeltildikten sonra bu manifestler
  için >30 s kayıt yok. Bu yüzden mevcut veride ayrıca uzun-ses hatası varmış
  gibi sunulmuyor. `<=30 s` açık girdi kontrolü ve test yeterli olabilir;
  inference uzun ses destekleyecekse tam ses işlenmeli. Standart owner Whisper
  adapter'ı zaten uzun sesi sessizce kesmez; iki yolun aynı olduğu iddiası düzeltilmeli.
- **P2 — accumulation adım sayacı.** Attention/RNN-T `step` değerini her
  microbatch'te artırıyor, optimizer yalnız accumulation tamamlanınca çalışıyor.
  `max_steps` bölünmüyorsa son gradient uygulanmıyor. Varsayılan accumulation=1
  yolunun hatalı olduğu iddia edilmiyor. Ölçülecek tam bütçe için optimizer
  adımı/microbatch ayrı raporlanmalı ve artık-batch davranışı test edilmeli.
  Whisper bu ayrımı zaten yapıyor; bu eleştiri ona ait değil.
- Gerçek koşu kayıtları cihazı, tam ortamı, veri/çıktı/checkpoint kimliğini ve
  seed'i taşımalı. Resume iddiası varsa NumPy/GPU RNG durumu da saklanıp
  gerçekten sınanmalı; yalnız CPU RNG dosyası tam GPU resume kanıtı değil.
- `max_steps:null` bırakılması bilinçli ve doğru bir bekleme koşulu. Ölçüm
  olmadan rastgele sayı yazılmamalı. CPU requirements dosyası ile gerçek ROCm
  runtime'ı açıkça ayrılmalı; sürüm numarasına bakıp test edilmiş uyumsuzluk
  varmış gibi yazılmamalı.

## Adnan'a verilecek düzeltme promptu

> PR #5'i `0437631` üzerinden bağımsız inceledik. 93/93 testin doğru;
> Samed'in `e5aa488` dalıyla geçici birleşimde de 125/125 geçti. Yine de gerçek
> RX 7800 XT kontrolünde aşağıdaki engeller çıktı. Deadline kullanıcıya göre
> 21 Kasım Cumartesi akşamı; kesin saat bilinmiyor. Aynı kendi branch/PR'ında
> düzelt ve kendi hesabından commit et. İnceleme raporundaki sabit commit
> bağlantıları ile `ADNAN_0437631_GPU.json` somut yeniden üretim kanıtıdır.
>
> 1. `common.load_waveform` için doğru PCM sample-width çözme yap. HParl PCM16,
>    CSS10 auditindeki 1.844 dosyanın hepsi PCM32. Width=4'ü int16 yorumlama.
>    Kaynak dosyayı değiştirme; 16/32-bit örnek sayısı, genlik, resampling ve
>    byte hash koruma testleri ekle. Desteklenmeyen format açık hata vermeli.
> 2. Attention `padding_mask` ve model içindeki yeni tensor'ları doğru cihaza
>    bağla. CPU ve GPU'da farklı uzunluklu örneklerle attention maskesi ve sonlu
>    forward/backward doğrulansın. Bir satırlık pilot düzeltmesiyle yetinme.
> 3. Attention ve RNN-T'nin **gerçek train/predict API ve CLI'ına** cihaz
>    seçimini ekle. Model, batch, loss/decode, dev ve checkpoint yükleme aynı
>    sözleşmeye uysun; GPU istenip yoksa sessiz CPU'ya dönmesin. Çalışan CLI
>    komutlarını belgede güncelle. Serdar gerçek GPU doğrulamasını merkezde yapacak.
> 4. RNN-T'nin mevcut ROCm'de eksik `torchaudio::rnnt_loss` GPU kernel'i için
>    çalışan backend/reçete sun. CPU testi başarı diye gösterme. GPU encoder
>    + CPU loss gibi bir yol seçilirse açık config/rapor olsun; loss'tan GPU
>    parametrelerine sonlu gradient ve gerçek parameter update kanıtlanıp süre
>    ölçülsün. Standard RNN-T yerine CTC geçirip aynı model diye raporlama.
>    Mevcut owner ortamını veya eski proje ortamını bozacak paket kurulumu yapma.
> 5. Whisper label başındaki gerçek `model.config.decoder_start_token_id` için
>    otomatik decoder shift'ini hesaba kat. Yalnız padding -100, gerçek EOS ve
>    el/transcribe prefix'i korunmalı. Sabit tokenizer ile label + shift sonrası
>    decoder girdisinde başlangıcın bir kez bulunduğunu sınayan test yaz.
> 6. Whisper için uzun-ses davranışını açıklaştır. Mevcut doğru çözülmüş train/dev
>    30 saniyenin altında; bu varsayımı doğrula ve >30 s için açık hata veya tam
>    destek sağla. Testi değiştirme, test kayıtlarını silme. Standart owner
>    inference ile açıklamadaki yanlış eşitliği düzelt.
> 7. Attention/RNN-T optimizer-step ve microbatch sayılarını ayır; accumulation>1
>    ve bölünmeyen bütçeyi test et. Rastgele `max_steps` seçme. Whisper iki
>    koşulda aynı başlangıç checkpoint/revision, dev, optimizer, sampling ve
>    ölçülmüş optimizer-step bütçesini korusun; gerçek CSS oranını raporla.
> 8. Hata regresyonları ve mevcut bütün testleri çalıştır; README/SEQUENCE_MODELS
>    ve kendi gerçek katkı dosyanı düzelt. Kodun hazır olması, full training'in
>    yapıldığı veya baseline sonuçlarının elde edildiği anlamına gelmesin.
>
> Freeze manifestini/referanslarını, final-test guard'larını ve kaynak lock
> baytlarını değiştirme. Resmî dev model seçimi içindir; test çıkarımı final
> protokol commit'inden sonra tek nihai karşılaştırma aşamasında kalacak.
> Push sonrası commit SHA, test özeti, değişen komutlar, bağımlılık/backend
> tercihi ve kalan engelleri gönder. Serdar aynı submitted fonksiyonlar ve
> gerçek train/predict yolunda GPU kontrolünü tekrarlayacak. Sonra kullanılacak
> gerçek veri üzerinde süre/bellek pilotuyla tam eğitim bütçesi seçilecek.

Samed'e kısa takip: Önceki rapor düzeltmeleri doğrulandı. Şimdi kullanılacak
gerçek seslerin repo içi teslimi, kota bilgisi ve temiz clone doğrulaması
gerekiyor. Özgün test ve referanslar aynı kalacak. Tam veri gelmeden metadata
testini tam ses doğrulaması olarak sunma. Depolama seçimini Serdar'la, ücretli
işlem yapmadan ve gerçek eğitim alt kümesi kararıyla birlikte tamamla.

## İncelemeyi yeniden çalıştırma

```bash
python scripts/review_data_submission.py --data-root /path/to/samed-checkout --output outputs/reviews/samed-new.json
HSA_ENABLE_DXG_DETECTION=1 python scripts/probe_sequence_submission.py --submission-root /path/to/adnan-linux-clone --device cuda --output outputs/reviews/adnan-new.json
```

İkinci komut model ağırlığı indirebilir ve üç **train** pilotunda tek-adımlık
gradient/optimizer kontrolleri çalıştırır; salt import kontrolü değildir.
ASR ağırlıkları veya benchmark sonuçları bu inceleme klasörüne eklenmez.
İlgili kodu değiştirmeden, geçerli Git checkout ve sabit ROCm ortamında çalıştır.
