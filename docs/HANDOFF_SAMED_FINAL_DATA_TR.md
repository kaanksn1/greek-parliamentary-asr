# Samed — son veri düzeltmesi ve eğitim sesi teslimi

20 Kasım 2026. Bu görev veri hazırlama, kaynak denetimi ve veri teslimidir.
Tam model koşuları Kaan'da; ortak son değerlendirme ve sunum Ozan'dadır.
Teslim cumartesi akşamı; hocanın kesin saatini bilmiyoruz.

## 1. Güncel dal ve kesin eğitim listesi

Kaan'ın veri/bütçe inceleme değişikliklerini içeren güncel `feature/p3-data`
dalından devam et. Yerel kendi değişikliklerini silme veya geçmişi yeniden yazma.
İnceleme PR'ı birleşmeden çalışacaksan aynı içerik `feature/p3-data-budget`
dalındadır; kendi dalına normal merge ile al, başka üyenin commitlerini kendi
adına yeniden üretme.

```bash
git fetch origin
git switch feature/p3-data
git pull --ff-only origin feature/p3-data
```

Kesin kaynak: `configs/training_data_selection.json`.
Seçim senin `subset` fonksiyonunla `--hours 30.5 --seed 42` kullanılarak yapıldı.
Tam oturumlarla gerçek temiz süre **30.883408 saat**, **23.602 kayıt**, **208
bilinen konuşmacı**, **18 oturum**. Ham tarafta aynı oturumların **23.615** kaydı
var; temizlikte çıkarılan **13** kaydın WAV'ı da ham teslimin parçasıdır.

Dosyalar:

- Temiz HParl: `data/manifests/subsets/train_hparl_30.5h_seed42.jsonl`
- Referans: `data/manifests/subsets/references/train_hparl_30.5h_seed42.jsonl`
- Karışım: `data/manifests/subsets/train_hparl_30.5h_seed42_css10.jsonl`
- Karışım referansı: `data/manifests/subsets/references/train_hparl_30.5h_seed42_css10.jsonl`
- Ham seçili veri: `data/manifests/subsets/raw/train_hparl_30.5h_seed42.jsonl`
- Ham referans: `data/manifests/subsets/raw/references/train_hparl_30.5h_seed42.jsonl`
- **Tek tek yüklenecek WAV listesi ve hashleri:**
  `data/manifests/subsets/train_hparl_30.5h_seed42.delivery.jsonl`

Oturumlar: `olomeleia_13, 15, 2, 20, 21, 23, 29, 32, 33, 35, 36, 37, 40,
45, 48, 49, 5, 9`. Tam adlar ve dosya SHA-256 değerleri JSON'dadır.
Bu listenin dışından yeni bir ses altkümesi seçme.

CSS10'un mevcut 1.844 klibi korunursa karışım **25.446 kayıt / 35.020678 saat**.
Bu sayı, aşağıdaki düzeltilmiş denetim sonrasında değişirse gerçek yeni sayıyı
kodla üretip açıkça raporla. Aynı 18 HParl kaynak oturumunu koru.

Depolama kararı: seçili **ham** HParl sesleri 3.31564 GiB civarı; mevcut
test/dev/CSS10 ile bütün sesler **6.714255 GiB**. Kendi checkpointlerimiz ve
pay için 2.6 GiB ayrıldı; toplam yaklaşık **9.314255 GiB**.
Sıradaki tam oturumu eklemek ses+bu payı 10 GiB üstüne çıkarıyor. Aday öneklerin
gerçek boyutları `docs/reviews/budget_2026-11-20/training_candidates.json` içindedir.
Kullanıcı toplam **10 GiB ücretsiz sınırını** onayladı; ücretli alan açma.
LFS bandwidth ayrı bir sınırdır: gereksiz temiz-klon indirmelerini tekrarlama,
mevcut yerel ses/cache ile çalış ve son uzaktan erişilebilirlik kontrolü için
kalan bandwidth'i hesaba kat. Hesabın gerçek kalan kotası API'den görülemedi.

## 2. Öncelikli hata: CSS10 aslında IEEE float32

Kaan LFS seslerini gerçekten indirdi ve 1.844 CSS10 dosyasının tamamının kaynak
hashini doğruladı. Hepsi **RIFF format tag 3 / IEEE float32 / mono 22050 Hz**.
4 bayt örnek genişliği, integer PCM32 olduğu anlamına gelmiyor.

Şu komut Python 3.12 ortamında mevcut veri koduyla hata veriyor:

```bash
python -m greek_asr.data verify --only css10_el_train.jsonl
# DATA ERROR: undecodable WAV (unknown format: 3): ...Paramythi_horis_onoma_0000.wav
```

Senin gerçek veri koşunun Python 3.15'te olması farkı açıklıyor: `wave`, IEEE
float desteğini [3.15'te ekledi](https://docs.python.org/3.15/library/wave.html).
Python 3.12'de aynı destek yok. Ayrıca 3.15'te dosyayı açabilmek de mevcut
`signal_stats` sonucunu doğru yapmıyor: `greek_asr/data/audit.py`, `width == 4`
için bütün baytları `<i4` olarak yorumluyor. Gerçek float değerleri `<f4` okunmalı.
İlk klipte doğru RMS yaklaşık **-27.86 dBFS**, mevcut integer yorumuyla **-6.09
dBFS**. CSS10 sinyal istatistikleri ve bunlara dayalı flag/review sonuçları
yeniden üretilmeli. Önceki raporu doğru denetim olarak sunamayız.

Kanıt ve tekrar üretme:

```bash
python scripts/inspect_css10_delivery.py --output outputs/css10_encoding_check.json
```

Kaydedilmiş çıktı:
`docs/reviews/budget_2026-11-20/css10_encoding_diagnosis.json`.

Kaan ortak model girişini düzeltti: `greek_asr/audio.py`, `infer.py` doğrulaması
ve `models/common.py` ince bağdaştırıcısı artık gerçek PCM16/PCM32/float32
ayrımını yapıyor. Mimari değişmedi, kaynak WAV'lar yeniden yazılmadı. Gerçek
26.4 saniyelik CSS10 ile Whisper GPU adımı ve regresyon testleri geçti.

Senin düzeltmen:

1. `data/common.py` içindeki probe/doğrulamayı Python **3.12** ile çalışır hale
   getir. Ortak `greek_asr.audio.read_wav_bytes/read_wav` yardımcılarını kullanabilir,
   mevcut veri kontratındaki payload/hash alanlarını koruyabilirsin.
2. Probe sonucunda örnek kodlamasını taşı; `audit.signal_stats` float ile integer
   örnekleri format tag'e göre ayrı çözsün. Float değerleri 2^31'e bölme,
   integer gibi reinterpret etme, otomatik gain/peak normalization uygulama.
   NaN/Inf, bozuk/truncated WAV ve LFS pointer görünür hata versin.
3. Mevcut **ham float32 WAV'ları olduğu gibi koru**. Sesleri PCM16'ya çevirip ham
   dosyanın üzerine yazma; doğru decode ve 16 kHz resample model girişinde bellekte.
4. Float fixture ile bilinen genlik/frame sayısı testi, integer PCM32 ile aynı
   genişliğe rağmen farklı encoding testi, float istatistik testi ve gerçek
   CSS10 hash/header kontrolü ekle. Yalnız dosya varlığına bakan test yeterli değil.
5. Audit/clean raporlarını, flagleri, review kuyruğunu ve işlem günlüklerini
   doğru float sinyaliyle yeniden üret. Sürümü ve Python/NumPy ortamını belirt.
   Sinyal denetimini tamamlamadan `locally_audited=true` yazma.

## 3. Yalnız kullanılan ham veriyle yeniden üretilebilir temizlik

Depoda teslim edilecek train, yukarıdaki 18 oturumdur. Tam 99.49 saatlik
manifesti kullanılabilir eğitim verisi gibi göstermeyelim. Tam kaynak envanteri
kalabilir; README'de bunun erişilebilir/çalıştırılan train'den ayrı bir kaynak
aday envanteri olduğunu açıkça yaz.

Audit -> clean -> seçili manifest/karışım -> verify zinciri, repodaki **seçili
ham** veri + değişmeyen dev/test + CSS10 ile çalışmalı. 99 saatlik eksik WAV'ları
okumaya zorlayan varsayılanlar varsa seçili kapsamı destekleyen küçük bir CLI
seçeneği ekle; örneğin `--selection configs/training_data_selection.json`.
Bu seçenek şu an mevcut CLI'de yok; eklersen gerçek komutunu README'ye yaz.
Kullanıcının manuel dosya yüklemesi veya dışarıdan train indirmesi gerekmesin.

- Kaynak oturum listesi ve raw ID/audio hashleri sabit kalsın.
- Gerçek temizleme kararları ham -> temiz ID/ref eşleşmesiyle izlenebilsin.
- Temiz seste değişiklik yoksa aynı raw WAV yolunu kullan; audioyu iki kez yükleme.
- Yeni audit'ten sonra seçim/karışım manifesti hashleri değişirse planı ve
  ilgili lineage raporunu kodla yenile; Kaan'a değişen dosyaları listele.
- Dev ve testte yeni split yapma. CSS10 sadece training'dedir.
- Manuel inceleme yaptığında gerçekten dinlediğin kayıtları ve kararlarını
  logla; dinlenmeyenleri tamamlanmış gösterme. Önce seçili train/CSS10'un
  şüpheli örneklerine bak. Belirsiz metni başka model çıktısıyla gold'a çevirme.

## 4. Sesleri repoya yükleme ve kabul koşulları

1. Delivery JSONL'deki **23.615** ham WAV'ın tamamını orijinal yol/byte/hash ile
   Git LFS'ye ekle. Sadece temiz 23.602 klibi yüklemek yeterli değil; temizleme
   ile çıkarılan ham örnekler de yeniden üretim için gerekli.
2. `git lfs ls-files` ve dosya hashleriyle pointer oid'lerinin gerçek ses
   SHA-256 değerleriyle aynı olduğunu kontrol et. Yalnız pointer commit edip
   blob'u yüklememek teslim sayılmaz. Kota nedeniyle upload başarısızsa hazır
   yazma; eksik dosya/byte sayısını bildir. Drive/Release/indirme betiği tek
   başına depo içi veri tesliminin yerini tutmaz.
3. Veri doğrulaması seçili raw train, temiz HParl, karışım, CSS10, dev ve testin
   gerçek seslerini kapsasın. `--skip-audio` kullanma. Mevcut `verify` yalnız
   belirli sabit isimleri kabul ediyor; seçili manifest yollarını veya selection
   planını doğrulayacak şekilde güncelle ve kullandığın gerçek komutu kaydet.
4. `python -m greek_asr.data verify-freeze` ses açıkken geçsin.
   `verify-source-metadata` de geçsin. Testin ilk dondurulan hashleri değişmesin:
   freeze `9a680104427fa0e202229167f5eb09bf61c713ebce29c9e369413a9f230675c3`;
   test manifesti `b0842bfbcbffa173585adb2debf35fb359748e0bd693da5a8dfabdb69de6f65a`.
   Teste temizlik, filtre, ekleme/çıkarma, yeniden etiketleme veya ASR koşusu yok.
5. Raporlarda kullanılan train/dev/test için konuşmacı, kayıt, saat ve oturum
   sayılarını üret. Seçili mevcut train hedeflerinde birebir test-cümlesi
   eşleşmesi **3**, ayrıca normalize diagnostik eşleşme **9**; bunlar farklı
   tanımlardır. Metin temizliği değişirse sayıları yeniden hesapla. Resmi split
   speaker-disjoint değildir: seçili train ile **49** test konuşmacısı ortak,
   **4.782** test kaydı bu seslere aittir; oturum örtüşmesi **0**. Bunu gizleme,
   unseen-speaker sonucu deme; hocanın izin verdiği resmi split istisnasını açıkla.
6. `python -m unittest discover -s tests -v` çalıştır. Python 3.12 + sabitlenmiş
   paketlerle veri komutlarını da gerçek ses üzerinde çalıştır. GPU gerekmez;
   model eğitimi, baseline sonucu ve son test değerlendirmesi bu görevin parçası değil.
7. `README.md`, `docs/DATA_PIPELINE_TR.md`, üretilen `DATA_REPORT_TR.md`,
   `configs/data_delivery.json`, `configs/data_source.json` ve kendi
   `contributions/SAMED_ORTAKAYA_210901024_SmtOrtakaya.md` raporunu gerçek son
   duruma göre güncelle. Başka üyelerin kişisel raporlarını yazma.
8. Kendi hesabından commit/push yap; PR #2'yi güncelle. Son SHA, test özeti,
   gerçek sesli doğrulama çıktıları, ham/temiz sayılar/saatler, yüklenen byte ve
   blob sayısı ile değişen manifest hashlerini Kaan'a gönder. Dış kaynakların
   lisans/atıf bilgilerini koru.

Kaan ses teslimini doğruladıktan sonra ölçülmüş konfigürasyonlarla merkezî
eğitimlere başlayacak. Son test için model/checkpoint/ayar/scoring protokolü
ayrıca commit edilecek; erken veri freeze'ı bu son test izni değildir.
