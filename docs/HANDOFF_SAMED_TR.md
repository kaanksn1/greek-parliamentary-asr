# Samed'e devir — veri edinme, testin dondurulması ve gerçek temizlik

Hazırlanma tarihi: 18 Kasım 2026. Bu belge Samed'in kendi çalışma oturumuna
kopyalayabileceği ayrıntılı görev promptudur; işin yapıldığına dair bir rapor
değildir. Samed'in görevi aşağıdaki ilk veri aşamasıyla başlayabilir. Ortak
veri sözleşmesi ve repo bağlantısı güncel README'den kontrol edilmelidir.

**İlk atamanın tarihsel kaydı:** Samed daha sonra PR #2'yi teslim etti.
O sürümün incelemesi ve güncel düzeltme promptu
[SAMED_PR2_REVIEW_TR.md](reviews/SAMED_PR2_REVIEW_TR.md) içindedir. Yeni
çalışmaya ilk görevi tekrarlayarak veya testi yeniden dondurarak başlanmaz;
mevcut freeze korunarak inceleme maddeleri tamamlanır.

18 Kasım devir güncellemesi: public repo ve ortak arayüz hazır; güncel hafif
test sayısı 44. Serdar'ın dört hazır modeli de yalnız mevcut üç train klibinde
GPU kontrolünden geçti. Bu, tam veri edinme/temizleme, testin dondurulması veya
benchmark sonucu değildir. Aşağıdaki tam veri işi Samed'in kendi teslimidir.

---

## Kopyalanacak görev promptu

Ben **Samed Ortakaya**, öğrenci numaram **210901024**, GitHub hesabım
**SmtOrtakaya**. Anonymous grubunun NLP Proje 3 veri sorumlusuyum. Konumuz
**Yunanca parlamento konuşmalarında otomatik konuşma tanıma ve alan verisiyle
fine-tuning**. Benim adıma yalnız aşağıdaki veri görevini tamamla. Başka ekip
üyesinin modelini, deney analizini, sunumunu veya kişisel katkı raporunu yazma.

### 1. Önce projeyi ve mevcut durumu oku

Çalışacağın public repo:
**https://github.com/kaanksn1/greek-parliamentary-asr**.
GitHub hesabına gönderilen write davetini kendi hesabınla kabul et. Eski
BANKING77 veya RAGTruth reposunda çalışmadığını doğrula. Güncel repoyu klonla,
`main` durumunu kontrol et ve kendi `feature/p3-data` branch'inde çalış.
Var olan yerel değişiklikleri silme veya başka bir üyenin branch'ini ezme.

Yeni bir checkout için başlangıç komutları:

```text
git clone https://github.com/kaanksn1/greek-parliamentary-asr.git
cd greek-parliamentary-asr
git switch -c feature/p3-data
```

Branch zaten varsa tekrar oluşturmaya çalışma; mevcut durumunu kontrol et.
İlk push henüz görünmüyorsa boş repodan ayrı bir başlangıç geçmişi üretmek
yerine Serdar'ın ilk `main` commit'ini al. Kendi hesabının Git author
kimliğini kontrol et; başka üyenin adını veya e-postasını kopyalama.

Sırasıyla oku:

1. `AGENTS.md` ve `docs/TEAM.md`.
2. `README.md`, `data/README.md`, `DATA_LICENSE.md`.
3. `docs/TEACHER_CLARIFICATION_TR.md` ve `docs/PROTOCOL_DRAFT.md`.
4. `docs/SECONDARY_SOURCE_TR.md`, `configs/data_source.json`.
5. `docs/DATA_CONTRACT.md`: ortak manifest/ID/yol/hash sözleşmesi.
6. Mevcut pilotun `scripts/fetch_train_pilot.py`,
   `scripts/audit_train_pilot.py`, `greek_asr/pilot_data.py` kodu ve ilgili
   testleri. Yeni veri hazırlamayı mevcut pilotun bozulmasına yol açmadan yap.

Sözleşme dosyası bu checkout'ta yoksa ilk iş güncel `main`/Serdar entegrasyon
branch'ini kontrol et. Kendi farklı şemanı bağımlı modellerden habersiz
kesinleştirme; gerçek verinin zorunlu kıldığı bir değişikliği somut örnekle
Serdar ve ilgili model/değerlendirme sahiplerine bildir.

Devir öncesinde zaten yapılmış olan başlangıç işi:

- HParl'in resmî metadata'sı ve ZIP dizini kontrol edildi; 92.133 WAV girdisi
  görüldü. Tam korpusun kalite ve süre denetimi tamamlanmadı.
- Yalnız resmî **train** bölümünden seed 42 ile seçilmiş üç özgün WAV,
  transkript ve kaynak satırları `data/raw/pilot/` içinde. Toplamları
  **6,61 saniye**, 16 kHz mono PCM16.
- Pilot ses hash/CRC, kaynak ID, transkript ve konuşmacı eşleşmesi denetlendi.
  Üçüncü referansta `<spoken_noise>` var; korunup inceleme için işaretlendi.
- Ortak WER/CER scorer'ı ve 15 başlangıç veri/değerlendirme testi vardı.
  JiWER 4.0.0 ile 500 kurgusal fixture karşılaştırması yapıldı. Bunlar ASR
  başarı sonucu değildir. Güncel kodda eklenen test sayısını ayrıca kontrol et.
- Serdar'ın bilgisayarında bu üç **train** klibinde Whisper-small çıkarımı
  çalıştı. Bu bir çalışma kontrolüdür; fine-tuning veya benchmark değildir.
- Tam veri edinme, gerçek temizlik, testin fiilî dondurulması ve temizlenmiş
  kullanılacak saat hesabı bu devir başlangıcında tamamlanmış değildir.

Başlangıç altyapısını kendi yeni yaptığın iş olarak katkı dosyana yazma.

### 2. Görev sınırı ve ekip çalışma biçimi

Benim kapsamım: kaynakların edinilmesi, kaynak/lisans kaydı, ham/temiz veri,
resmî splitlerin korunması, erken test dondurma, veri kalite denetimi,
yeniden üretilebilir train/dev temizleme ve karışık eğitim manifestleri.

Serdar GitHub/entegrasyon, ortak runner ve merkezî koşuları; Kemal HMM/CTC
ailesini; Adnan attention/transducer/fine-tuning'i; Ozan ortak değerlendirme
ve sunumu yönetiyor. **CPU dahil bütün tam model eğitimleri ve model tahmin
koşuları Serdar'ın bilgisayarında yapılacak.** Ben kendi ortamımda veri
hazırlayabilir, ses/veri testlerini ve küçük veri işlev kontrollerini
çalıştırabilirim; bağımsız ASR benchmark veya model seçimi yapmayacağım.

Kendi veri modüllerim, hazırlama scriptlerim, configlerim, veri testlerim ve
veri raporum için sorumluluk al. Ortak scorer'ı veya başka üyenin modelini
gereksiz yere yeniden yazma. Web uygulaması, veritabanı, release pipeline,
genel amaçlı policy checker gibi projeye hizmet etmeyen işler ekleme.

### 3. Kullanılacak kaynaklar ve sürelerin anlamı

**Ana kaynak: HParl 1.0.0 — Hellenic Parliamentary Speech Corpus.**

- Katalog: https://inventory.clarin.gr/corpus/1602
- Kalıcı kimlik: http://hdl.handle.net/11500/CLARIN-EL-0000-0000-7665-A
- Lisans: CC BY-NC 4.0. Üretici/atıf/lisans kaydını ve bizim değişikliklerimizi
  tut; projenin MIT kod lisansını veri lisansı yerine koyma.
- Kaynak metadata'sı ve GET başlıkları `configs/data_source.json` içinde.
  `scripts/fetch_train_pilot.py` mevcut erişim örneğidir; bu scriptin 64 MiB
  sınırı onu tam veri indirme aracı yapmaz.
- Kaynağın resmî Kaldi splitlerini ve oturum bilgilerini kullan. Rastgele
  80/10/10 bölen bir üçüncü taraf aynayı resmî split sayma.

**Ek kaynak: CSS10 Greek — Kaggle dağıtımı v1. Yalnız eğitim için.**

- Üretici: https://github.com/Kyubyong/css10
- Dağıtım: https://www.kaggle.com/datasets/bryanpark/greek-single-speaker-speech-dataset
- Katalog referansı: `bryanpark/greek-single-speaker-speech-dataset`.
- Metadata'da sürüm 1, veri kimliği 30096, CC0: Public Domain yazıyor.
  Arşivle gelen lisans ve atıfları da kaydet. Kod deposunun Apache-2.0
  lisansını seslerin lisansı diye yazma; `DATA_LICENSE.md` ayrımını koru.
- Bir LibriVox sesli kitabını okuyan tek konuşmacı var. Üretici ses ve kitap
  metnini insan eliyle eşleştirmiş; bu bizim yeni anotasyonumuz değil.
  Orijinal ve üretici-normalize metin alanlarını birlikte sakla.
- CSS10'u HParl dev veya testine ekleme. Farklı alan ve tek konuşmacı
  sınırlamasını raporla; katkısının yararlı olduğu henüz gösterilmiş değil.

Kaynakların bildirdiği planlama süreleri:

| Kaynak/bölüm | Üreticinin bildirdiği süre |
|---|---:|
| HParl train | 99:31:41 |
| HParl dev | 09:03:33 |
| HParl test | 11:12:28 |
| HParl toplam | 119:47:42 |
| CSS10 Greek | 04:08:14 |
| HParl train + CSS10 ham eğitim adayı | 103:39:55 |
| İki korpusun bütün kaynak havuzu | 123:55:56, yaklaşık 124 saat |

Bu rakamları yerelde ölçülmüş, temizlenmiş veya kullanılmış saat diye sunma.
Dev/test saatlerini eğitim saati hesabına katma. Yerel gerçek süreleri ses
dosyalarından hesapla ve kaynak bildiriminden sapmayı açıkla. Hoca sabit bir
minimum saat vermedi: istatistiksel sonuca yetecek, donanım/zaman/maliyet
bütçesinde çalışabilecek en büyük kullanılabilir veri bekleniyor. Keyfî
20/25 saat seçme. Ölçülen veri kalitesini ve Serdar'ın gerçek koşum bütçesini
esas alan kararı kaydet; sınıf tablosundaki `available` değerini `used` diye
kendiliğinden değiştirme.

### 4. Öncelik: özgün testi erkenden dondur

**Train/dev temizliğinden ve kaynak karıştırmadan önce bunu tamamla.**
Hocanın açık talimatı: “Test setini önceden dondurun ve asla bir işlem
uygulamayın. Training ve dev splitine istediğinizi yapabilirsiniz.”

1. Özgün HParl sürümünü, test dosya listesini, resmî ID/oturum eşlemesini,
   ses ve referans dosyalarını kaynaktan edin. Edinme ve hash çıkarma, test
   başarımı analizi değildir; bu aşamada ASR tahmini veya WER/CER üretme.
2. Testin kaynak ses ve transkript baytlarını değiştirmeden koru. Kaldi
   metadata'sının kaynak kopyasını da sakla. Ortak manifesti üretirken
   `reference` içeriğini özgün kaynak satırından aynen çıkar; normalize etme.
3. Kaynak sürümü, dosya byte sayısı/SHA-256, ID listesi ve manifest kimliğini
   içeren dondurma kaydı üret. Sözleşmede tanımlı alanları ve yolları kullan.
   Kimliklerin deterministik ve başka kaynaklardan ayrışmış olmasını sağla.
4. Test dondurma kodu, dosyaları ve kontrolünü ayrı bir commit'te kaydet;
   commit kimliğini Serdar'a ilet ve PR üzerinden `main`'e entegrasyonunu
   sağla. Sonraki train/dev işlerinin bu kaydı değiştirmediğini anlamlı bir
   kontrolle doğrula.
5. Testten bozuk/uzun/etiketli/başarısız kayıt **çıkarma**, kayıt **ekleme**,
   yeniden split etme, metni düzeltme, ses temizleme, kalıcı resampling,
   trimming veya yeniden etiketleme yapma. Bir sorun varsa özgün testte
   bırak ve durumunu kaydet. Test dosyaları üzerindeki değişikliği sessizce
   yeni hash çıkararak “yeni freeze” diye meşrulaştırma.
6. Test içeriğinden temizlik eşiği, sözlük, tokenizer, normalizasyon veya
   model seçme. Train/test çakışması tespit edilirse korunacak taraf testtir;
   gerekiyorsa train/dev tarafını düzelt ve gerekçesini kaydet.

Erken **veri dondurma** ile daha sonra model sahipleri/Ozan/Serdar'ın yapacağı
**son deney protokolü dondurma** farklıdır. İkinci aşama test çıkarımından
önce model, checkpoint, decoder ve scoring kararlarını commit ile sabitler.
Bu veri görevi son test koşumu için izin veya hazır sonuç anlamına gelmez.

### 5. Ham veri edinme ve teslim

- Kullanılan sürümü, indirme URL'sini, tarihini, dosya listesini, boyutunu,
  SHA-256 değerlerini, lisans ve kaynak satırlarını kaydet. İndirme kesilirse
  bozuk dosyayı tamamlanmış sayma; var olan farklı baytları sessizce ezme.
- ZIP/TAR içinden çıkarırken hedef dışına çıkan `..`/mutlak yolları ve
  beklenmeyen linkleri reddet. Aynı kaynak dosyanın yinelenmesini ve
  metadata'da olup sesi olmayan kayıtları görünür hata olarak raporla.
- Makineye özel `C:\Users\...` veya `/mnt/c/Users/...` yolunu paylaşılan
  manifest/configlere gömme. Repo köküne göre bağıl, taşınabilir yolları
  ortak sözleşmeye göre kullan; WSL dönüşümünü çalıştırma tarafında bırak.
- Kullanıcı adı, Kaggle anahtarı, GitHub token'ı, cache veya yerel venv'i
  commit etme. Lisanslı veriyi elde etme şartlarına uy; erişim sorunu varsa
  hatayı somut biçimde bildir, başka kaynağı habersizce ikame etme.
- Hocaya göre **ham ve temiz verinin kendisi public repoda bulunmalı**.
  Sadece URL, Google Drive bağlantısı, “scripti çalıştırın” açıklaması veya
  `.gitignore` yeterli değil. Ses büyüklüğü nedeniyle Serdar'ın belirlediği
  Git/Git LFS teslim yöntemini uygula ve temiz bir clone'da gerçek ses
  baytlarının edinilebildiğini doğrula. LFS pointer'ını indirilmiş ses sanma.
- HParl'in yaklaşık 11,9 GB özgün ZIP'ini tek dosya olarak Git'e/LFS'e
  yüklemeye çalışma. GitHub'ın güncel dosya/kota sınırlarını ve bu reponun
  kullanılabilir LFS kotasını Serdar'la kontrol et; bireysel WAV'lar veya
  uygun boyutlu parçalar ve kaynak eşlemesiyle gerçek veri teslimini çöz.
  Git LFS aracının kurulu olması bütün seslerin LFS sunucusuna yüklendiği
  anlamına gelmez. Kullanıcı ücretli depolama satın almayı onaylamadı;
  bu görevi ek ücret/paket satın alarak tamamlamaya çalışma.
- Bütün korpusun veya seçilen alt kümenin “ham veri” tanımını raporda açıkla:
  hangi kaynak kayıtları edinildi, hangileri kullanıldı, hangileri dışarıda
  kaldı ve neden? Şeffaflık için elenen kayıtların ham kökenini kaybetme.

### 6. Train/dev denetimi ve gerçek temizlik

Önce kaynak envanterini çıkar, sonra sorunlara göre kuralları belirle.
İki korpusu birleştirmek, dosyayı JSONL yapmak, etiketi yeniden adlandırmak
veya split oluşturmak tek başına temizlik değildir.

İncelenecek gerçek sorunlar:

- Decode edilemeyen, sıfır örnekli, eksik veya metadata'sıyla uyuşmayan ses.
- Kaynakta var olan sample-rate/channel/bit-depth çeşitliliği; çok uzun/kısa
  parçalar; aşırı sessizlik/clipping gibi kalite işaretleri. Eşik seçimini
  train/dev'de gerekçelendir; otomatik işareti kendiliğinden yanlış veri sayma.
- Aynı baytlı ses tekrarları; farklı ID altında aynı kayıt; gerekiyorsa
  yeniden kodlanmış tekrar ihtimali. Yöntemin hangi tekrarları yakaladığını
  ve yakalayamadığını yaz.
- Yinelenen metin ile yinelenen sesi ayır. Aynı cümlenin farklı kişilerce
  söylenmesi otomatik olarak hatalı tekrar değildir.
- Ses–metin hizalama/eksik kelime/yanlış eşleme şüphesi. İnsan kaynaklı
  referansı ve kaynağını koru. Duyarak güvenilir biçimde doğrulanamayan
  Yunanca metni tahmin ederek düzeltme; inceleme gerektirdiğini belirt.
- HParl'deki `<spoken_noise>` ve başka anotasyon işaretleri. Train/dev'de
  nasıl ele alındığını açıkça belgeleyen politika oluştur; test referansına
  otomatik olarak uygulama. Nihai scoring politikası Ozan'ın alanındadır.
- Kaynak/splitler arasında ID, ses, metin, oturum ve varsa konuşmacı
  örtüşmesi. Resmî splitin speaker-disjoint olduğunu varsayma. Konuşmacı
  veya oturum kökeni kaynakta yoksa uydurmak yerine `null`/bilinmiyor olarak
  sözleşmeye uygun kaydet. ID deseninden çıkarım yapılıyorsa kanıtını yaz.

Her gerçek işlem için makinece okunur bir günlük tut: kayıt kimliği, kaynak,
split, işlem türü, gerekçe, kullanılan kural/sürüm, önce/sonra dosya veya
metin hash'i, önce/sonra süre ve varsa doğrulama kanıtı. Ham kaynak asla
üzerine yazılmaz. Sadece raporlamak yeterli olan bir sorun için gereksiz
değişiklik yapma. Hiç düzeltme bulunmadıysa bulmuş gibi iş/bonus üretme.

ASR model çıktısını doğrulanmış gold olarak kopyalama. Kaynak insan
eşleştirmesini koru; yeni bir insan doğrulaması yapılırsa kim, hangi örnek,
nasıl ve hangi güvenilirlikle bilgisini gerçekten kaydet.

### 7. Ortak çıktı ve iki eğitim koşulu

`docs/DATA_CONTRACT.md` içindeki kesin şemaya bağlı kal. Son model kodu
hazırlanınca şema uyuşmazlığı yaşamamak için örnek manifesti erken paylaş.

Kesin ana çıktı yolları:

- Ses manifestleri: `data/manifests/train.jsonl`, `dev.jsonl`, `test.jsonl`.
- Ayrı referanslar: `data/manifests/references/train.jsonl`, `dev.jsonl`,
  `test.jsonl`; her satır yalnız `id` ve `reference` içerir.
- Erken test kaydı: `data/manifests/test.freeze.json`.

Ses manifestinin sözleşme alanları; son satırdaki kayıt/konuşmacı bilgisi
mevcutsa korunur, kaynakta bulunmuyorsa `null` kullanılır:

| Alan | Anlamı |
|---|---|
| `id` | `hparl:<özgün_id>` veya `css10_el:<özgün_id>`; tüm kaynaklarda benzersiz |
| `source` | `hparl` veya `css10_el` |
| `original_id` | Kaynakta var olan özgün kimlik |
| `source_split` | Özgün bölüm: `train`, `dev`, `test` veya gerçekten bölünmemiş kaynakta `unsplit` |
| `split` | Projede kullanıldığı bölüm: `train`, `dev` veya `test` |
| `audio_path` | Proje köküne göre bağıl ses yolu; makineye özel mutlak yol değil |
| `audio_sha256` | Referans verilen gerçek ses dosyasının SHA-256 değeri |
| `sampling_rate`, `channels`, `num_frames` | Dosyadan ölçülen gerçek ses özellikleri |
| `duration_seconds` | Pozitif, sonlu ve gerçek sesle uyumlu süre |
| `source_recording_id`, `speaker_id` | Kaynakta doğrulanabilen değerler; yoksa `null` |

Kaldi'nin `validation`/geliştirme adını ortak sözleşmedeki `dev` adına
eşlerken kaynak dosyasını ve özgün bölüm bilgisini provenance kaydında koru.
CSS10 için kaynak gerçekten split içermiyorsa `source_split=unsplit`,
projedeki kullanım `split=train` olur. Eğitim için temiz ses türetilirse
ham → temiz ilişkiyi ve iki dosyanın hash'ini ayrıca kaydet. Test manifesti
ise özgün, değiştirilmemiş test sesini gösterir. `reference` alanını ses
manifestine katıp model girdisine sızdırma; ayrı referans dosyasını kullan.

`test.freeze.json` kaynak/sürüm, test ID listesi ve sayısı, manifest/referans
dosyası SHA-256 değerleri, kayıt bazında ses ve özgün referans satırı kimliği,
özgün ID/oturum eşlemesi ve dondurma tarihini taşımalı. Sözleşmedeki tam
anahtar düzenini kullan; Serdar'ın doğrulayıcısıyla uyumunu kontrol et.

Serdar'ın mevcut ortak çıkarım CLI'ı `greek_asr.infer` modülüdür:

```text
python -m greek_asr.infer --manifest data/manifests/train.jsonl --backend whisper --split train --output outputs/NAME
```

Komut ve bağımlılıklar için güncel README'yi de kontrol et. Bu arayüz örneği veri
görevinde tam model koşusu başlatman için bir talimat değildir. Runner'a
referans dosyası verilmez; ses yolu/hash'leri kullanılır ve her ID için boş
metin dahil bir tahmin satırı üretilir. Veri manifesti başarısız kayıtların
sessizce atlanmasını gerektiren bir yapı oluşturmamalı.

Üretilecek mantıksal çıktılar:

1. Her kaynağın özgün dosyaları ve edinme/provenance kaydı.
2. Bütün HParl resmî splitlerinin ham manifestleri ve ayrı erken test freeze
   kaydı; test referansı kaynakla aynı ve sabit kalır.
3. HParl clean train ve clean dev manifestleri; her değişiklik günlükte.
4. CSS10 Greek clean train manifesti; kaynak orijinal/normalize metni korunur.
5. **HParl-only train** ve **HParl + CSS10 train** olmak üzere iki seçim
   manifesti. Ortak HParl dev ve özgün test değişmez. Karışım oranı ve
   eğitim adımı politikasını Adnan/Serdar belirler; veri sahibi gizlice
   tekrar örnekleme veya farklı deney bütçesi dayatmaz.
6. Scorer için sözleşmedeki `id, reference` referans JSONL'leri. Bunları
   elle düzenleyip manifestten ayrışan ikinci bir gold kaynağı oluşturma.
7. Önce/sonra toplamları ve cleaning günlüğünden kodla üretilen veri raporu.

Kaynaklar arası ID çakışmasını önleyen ortak önekleri kullan. Mevcut
`train-pilot-0001` benzeri tarihsel pilot ID'lerini yeni korpusun tam manifest
kimliği sanma veya eski runtime kayıtlarını bozacak şekilde yeniden yazma.
Model tahmin dosyasının standardı `{"id":"...","text":"..."}`;
referans ayrı tutulur ve model girdisine verilmez.

### 8. Kod, bağımlılık ve test teslimi

Görevin yalnız rapor değil; **ham veriden temiz veriye yeniden çalıştırılabilir
bir pipeline** üretmek. Kaynak okuma, audit, test freeze ve temizleme/manifest
oluşturma komutlarını README/veri raporunda tam yaz. Kesin komut adlarını
uygulama sırasında seç; mevcut olmayan bir script çalışıyormuş gibi belgeye
yazma. Önerilen alanlar `greek_asr/data/`, veri hazırlama/audit scriptleri,
`data/` altındaki manifest/loglar ve `tests/test_data_*.py` dosyalarıdır.

Ek paket gerekiyorsa Serdar'la bağımlılık uyumunu koruyarak kesin sürümünü
belirt (`paket==x.y.z`). Ortak ASR ortamını gereksiz yere baştan kurma.
Seed kullanan her seçimde seed'i ve seçilen ID listesini kaydet. Aynı ham
girdi ve configle yeniden çalıştırma aynı temiz veri/manifesti üretmeli;
farklı sonuç varsa nondeterministik kısmı açıklayıp düzelt.

Anlamlı veri testleri en az şunları kapsasın:

- Ham bir örneğin kaynak ID, transkript, ses hash'i ve süresinin doğru
  manifestte görünmesi; Unicode Yunanca metnin korunması.
- Eksik ses/metadata, tekrar ID, bozuk ses ve hatalı hash'in görünür hatası.
- Somut temizlik fixture'ının doğru düzeltme/eleme ve gerekçe kaydını üretmesi;
  sağlam örneğin korunması. Elenmiş kayıtların süre/sayı toplamına etkisi.
- CSS10'un yalnız train'e girmesi; HParl resmî test ID kümesinin tam eşitliği.
- Train/dev hazırlama tekrar çalıştırıldığında test manifesti, kaynak
  referansı ve ses hash'lerinin **hiç değişmemesi**; test değiştirilirse
  kontrolün başarısız olması.
- Aynı config/seed ile tekrar çalıştırmada manifest ve log tutarlılığı;
  HParl-only ile mixed manifestleri arasında beklenen kaynak farkı.
- Arşiv çıkarma uygulanıyorsa hedef dışı path'in reddedilmesi; manifest
  yollarının başka bir clone'da makineye özel konum gerektirmemesi.

Mevcut testleri de çalıştır:

```text
python -m unittest discover -s tests -v
```

Tam korpusu her unit testte yeniden indirme. Küçük açık fixture'larla
regresyonu kontrol et; tam veri edinme/audit için ayrı yeniden çalıştırma
komutları ve gerçek çalışma kaydını teslim et. Sadece yardımcı fonksiyon
veya “dosya var” testiyle veri pipeline'ı test edilmiş sayma.

### 9. Veri raporu ve kişisel teslim

Örneğin `docs/DATA_REPORT_TR.md` içinde aşağıdakileri kod çıktılarıyla yaz:

- Kaynak, sürüm, lisans, insan transkript kökeni ve kaynak dosya hash'leri.
- Kaynak × split bazında ham/temiz kayıt sayısı ve süre; süre hesap yöntemi.
- Sorun/kural bazında işaretlenen, düzeltilen, elenen ve korunan kayıt
  sayıları ve süreleri. Aynı kayıt birden çok kurala giriyorsa toplamları
  çifte saymadan açıkla.
- Tekrar/örtüşme bulguları; HParl oturum/splitlerinin nasıl korunduğu.
- Test freeze commit'i/manifest/hash kimliği ve sonraki değişmezlik kontrolü.
- En az birkaç gerçek sorun örneği; kaynağı, işlem gerekçesi ve önce/sonra
  farkı. Güvenilir düzeltmesi olmayan örneği düzeltilmiş gibi sunma.
- CSS10 tek konuşmacı/sesli kitap ve HParl tutanak–ses farkı sınırlamaları.
- Kullanılabilecek temiz eğitim miktarı ve Serdar'a iletilen boyut/süre
  raporu; model başarımı veya kesin bonus puanı iddiası yapma.
- Her script ve dosyanın ne yaptığı, tam komutlar ve bilinen eksikler.

1–2 veri slaytı taslağı ve Ozan için kısa konuşmacı notu hazırla. Bunlar
sunumun tamamını senin hazırlaman gerektiği anlamına gelmez.

Kendi gerçek yaptığın işleri şu dosyaya yaz:

```text
contributions/SAMED_ORTAKAYA_210901024_SmtOrtakaya.md
```

`contributors/`, ortak `CONTRIBUTIONS.md` veya kullanıcı adı placeholder'ı
kullanma. Dosyayı sadece fiilen yaptığın iş, kendi commit/PR bağlantıları,
çalıştırdığın test/komutlar ve bulgularla doldur. Başka üye adına dosya
oluşturma veya commit atma. Git author kimliğini kendi hesabınla doğrula.

### 10. Serdar'a PR ile teslim edilecek paket

1. Kendi `feature/p3-data` branch'inin PR'ı ve gerçek commit'lerin.
2. Önce test freeze kaydı; ardından temizleme/pipeline commit'leri.
3. Ham/temiz veri ve lisans/provenance dosyaları; clean clone'da ses
   baytlarını ve hash'leri doğrulama komutu.
4. HParl-only, mixed, dev ve dondurulmuş test manifestleri; ortak sözleşmeyle
   uyum; testin değişmediğini doğrulayan sonuç.
5. Tekrarlanabilir edinme, audit, hazırlama komutları; config/seed ve kesin
   bağımlılık bilgisi. Serdar'ın kendi makinesinde çalıştırabileceği yollar.
6. Veri test sonuçları, veri raporu ve koddan üretilmiş önce/sonra sayılar.
7. Kendi katkı dosyan, 1–2 slaytın ve konuşmacı notların.
8. Bilinen eksikler ve çözülememiş sorunlar. Başarısız adıma başarılı denmez;
   kırık/eksik veri sessizce düşürülmez, uydurma transcript/sonuç üretilmez.

PR açıklamasında hangi dosyaların üretildiğini, nasıl tekrar üretileceğini,
ne kadar ham/temiz ses olduğunu, testin nasıl sabitlendiğini ve doğrulamanın
sonucunu somut yaz. Tam model koşularını Serdar'a bırak. Kaynak paylaşımı,
donanım bütçesi veya ortak şema yüzünden ilerleyemeyen bir adım varsa
önce etkilenmeyen işi tamamla, sonra hangi somut bilgi/değişikliğin
gerektiğini belirt; problemi gizleyerek tamamlandı deme.

---

Bu promptta adı geçen planlanan veri çıktıları henüz tamamlanmış teslimler
değildir. Gerçek dosya yollarında `docs/DATA_CONTRACT.md`, kapsamda
`docs/TEAM.md`, test kısıtında `docs/TEACHER_CLARIFICATION_TR.md` esas alınır.
