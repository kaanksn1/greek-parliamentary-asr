# Anonymous — Proje 3 görev dağılımı

17 Kasım 2026 tarihli plan. Önceki beş kişilik ekip ve çalışma düzeni korunur:
**CPU baseline'lar dahil bütün tam eğitim ve model tahmin koşuları Serdar'ın
RX 7800 XT / 32 GB RAM bilgisayarında çalışır; sunumu Ozan yapar.** Diğer
üyeler kendi kodunu, küçük kontrollerini, konfigürasyonlarını ve sonuç analizini
kendi hesabından teslim eder. Merkezî koşum, kodun yazarlığını değiştirmez.

Bu belge bir görev atamasıdır; aşağıdaki uygulamaların tamamlandığı anlamına
gelmez. 18 Kasım'da kullanıcı Serdar'ın kendi işlerini yapmayı, public repo
açmayı, dört üyeyi davet etmeyi ve Samed'e ayrıntılı prompt hazırlamayı atadı.
Serdar'ın entegrasyon/standart çıkarım/ortam işi başladı; Samed'in veri
temizliği ve diğer üyelerin modelleri kendi sahiplerinde kalır.
Mevcut ortak scorer, üç eğitim klibi ve runtime kontrolü başlangıç altyapısıdır;
üyelerin henüz yapmadığı kişisel çalışma olarak yazılmaz.

Sonraki kullanıcı kararı: Samed düzeltilmiş veri teslimini pushlayıp Serdar'ın
incelemesiyle kullanılabilir hale getirdikten sonra sıradaki model geliştiricisi
Adnan olacak. [Ayrıntılı Adnan promptu](HANDOFF_ADNAN_TR.md). Adnan kendi kodunu
hazırlamak için Kemal'in modellerinin bitmesini beklemez; bütün tam koşular
Serdar'da ve ortak son test aşaması bütün gereklilikler tamamlandıktan sonradır.
Bu sıra değişikliği Kemal/Ozan'ın sahipliğini veya iş yükü paylarını değiştirmez.

18 Kasım 2026 iş yükü tahmini: aşağıdaki yüzdeler planlanan **aktif insan
emeğinin yaklaşık dağılımıdır**; gerçekleşmiş kişisel katkı veya not ağırlığı
değildir. Kodlama, hata düzeltme, veri inceleme, analiz, entegrasyon ve sunum
hazırlığını kapsar; bilgisayarın kendi başına çalıştığı süreyi kapsamaz.
Koşum hazırlama, izleme ve hataya müdahale aktif iş sayılır. Görev sahipliği
değiştirilmedi; yüzdeler toplamı %100'dür.

| Kişi | GitHub | Tahmini iş yükü | Ana sorumluluk | Önerilen branch |
|---|---|---:|---|---|
| 1 — Serdar Kaan Kesen | `kaanksn1` | %25 | GitHub/entegrasyon, ortam, merkezî koşular; Whisper, Canary, Parakeet, Qwen3-ASR standart çıkarımı | `feature/p3-integration` |
| 2 — Samed Ortakaya | `SmtOrtakaya` | %17 | HParl + CSS10 Greek, veri denetimi, gerçek temizlik ve split/manifestler | `feature/p3-data` |
| 3 — Kemal Arif Karabacıoğlu | `kemalarif` | %23 | MFCC + GMM-HMM, plain CTC; wav2vec 2.0 ve HuBERT ASR | `feature/p3-ctc-hmm` |
| 4 — Adnan Demir | `adnan-demir` | %23 | Attention encoder–decoder (hybrid CTC/attention dahil), RNN-T; kendi Whisper fine-tuning deneyimiz | `feature/p3-sequence-models` |
| 5 — Ozan Serhan Denizhan | `serhandenizhan` | %12 | Ortak değerlendirme, karşılaştırma, grafikler ve sunum | `feature/p3-evaluation` |

Serdar'ın payı farklı ASR ortamları ve entegrasyondan; Kemal ve Adnan'ın payı
birden fazla modelin eğitim reçetesi ve hata düzeltmesinden kaynaklanır.
HuBERT için head eğitimi veya ROCm uyumluluğu ek emek gerektirirse tahmin
gerçek çalışma kayıtlarına göre güncellenir. Bu yüzdeler kişisel katkı
dosyalarına yapılmış iş gibi aktarılmaz.

## 1 — Serdar: entegrasyon ve merkezî koşum

- Ayrı ASR ortamını, tam sabitlenmiş paket sürümlerini, checkpoint revision'larını
  ve tekrar çalıştırma komutlarını hazırlar. Mevcut kısa GPU kontrolünü tam
  eğitim kapasitesi ölçümü olarak sunmaz.
- Whisper, Canary, Parakeet ve Qwen3-ASR için standart, ek eğitim yapılmamış
  çıkarım komutlarını ve ortak tahmin biçimine dönüştürmeyi hazırlar. ROCm'de
  çalışmayan arka ucu hata kaydıyla raporlar ve çözümünü araştırır; zorunlu
  baseline'ı sessizce listeden çıkarmaz.
- Üyelerin çalışır kod/config paketlerini inceler; bütün tam train, dev ve
  son test tahminlerini sırayla çalıştırır. Hatalı model kodunu ilgili sahibi
  düzeltir. Her koşuda kod sürümü, seed, veri kimliği, komut, süre, bellek,
  log, çıktı ve checkpoint bilgisini saklar.
- Samed'in erken test dondurma manifestini doğrular; Ozan'la ortak scoring
  protokolünün son sürümünü test çıkarımından önce commit eder.
- PR entegrasyonu, README'nin kodla uyumu, kod LICENSE'ı, veri lisanslarının
  korunması ve son public depo tesliminden sorumludur.

## 2 — Samed: veri, temizlik ve kaynak kaydı

- HParl 1.0.0 ile CSS10 Greek v1'in kaynak, lisans, özgün transkript, ses ve
  süre bilgilerini kaydeder. Ham/temiz verinin gerçekten depoda teslim edilme
  biçimini Serdar'la belirler; dış bağlantıyı veri teslimi saymaz.
- **Train/dev temizliği ve kaynak karıştırmadan önce** özgün HParl test
  üyelerini, ses ve referanslarını hash'leriyle sabitler. Testten kayıt
  çıkarmaz, eklemez, temizlemez, yeniden bölmez veya yeniden etiketlemez.
  Bu dondurma henüz yapılmış değildir.
- Train/dev üzerinde bozuk/boş ses, tekrarlar, hizalama, metin–ses uyumsuzluğu
  ve `<spoken_noise>` gibi gerçek sorunları araştırır. Bulunmayan bir sorunu
  varmış gibi yazmaz. Her değişikliğin önce/sonra halini, gerekçesini, kayıt
  sayısını ve süresini günlükte tutar. Biçim dönüştürmeyi temizlik saymaz.
- Resmî HParl oturum/split kimliklerini korur; konuşmacı, kayıt, metin ve ses
  örtüşmelerini raporlar. Testle çakışma bulunursa çözüm train/dev tarafındadır.
  Kaynakta bulunmayan speaker/session alanlarını uydurmaz.
- CSS10 yalnız eğitime girer. Tek konuşmacılı sesli kitabın parlamento
  konuşmalarından farklı olduğunu ve hazır insan eşleştirmesinin bizim yeni
  anotasyonumuz olmadığını açıklar. Model çıktısını doğrulanmış gold saymaz.
- Ham/temiz saat ve kayıt sayısı raporu, yeniden üretilebilir hazırlama kodu,
  anlamlı veri testleri, örnek sorunlar ve 1–2 slayt teslim eder.

## 3 — Kemal: HMM ve CTC ailesi

- Standart MFCC + GMM-HMM için özellik, sözlük/lexicon, decoder ve eğitim
  reçetesini; ayrıca plain supervised encoder + CTC için eğitim/tahmin kodunu
  hazırlar. Bunlar iki ayrı baseline satırıdır.
- Yunanca ASR head'li wav2vec 2.0 checkpoint'inin standart çıkarımını ortak
  çıktıya bağlar. HuBERT için Yunancaya uygun ASR head'li checkpoint'i
  doğrular; bulunamazsa standart HuBERT-CTC head eğitimi reçetesini hazırlar
  ve bütçesini ölçüme sunar. Bare encoder/duygu sınıflandırıcısını ASR saymaz.
- Standart toolkit kullanılabilir; kullanılan sürüm/reçete, kendi uyarlaması,
  eğitim kodu ve tam komutlar depoda bulunur. Sıfırdan yeni mimari yazması
  beklenmez. Hazır wav2vec çıkarımı plain CTC eğitimini otomatik karşılamaz.
- Her model için küçük işlev kontrolü, dev ile sınırlı ayar seçimi, merkezî
  koşum paketi, dönen sonuçların analizi, hata örnekleri ve 1–2 slayt hazırlar.

## 4 — Adnan: sıralı modeller ve fine-tuning katkısı

- Standart attention encoder–decoder (hybrid CTC/attention bu aileye dahil)
  ile RNN-T transducer için ayrı, belgeli
  eğitim/tahmin reçeteleri ve konfigürasyonlar hazırlar. Kaynağı belirtilmiş
  standart toolkit kullanabilir; özel mimari geliştirme zorunluluğu yoktur.
- Ayrı katkı deneyinde Whisper'ın seçilen başlangıç checkpoint'ini fine-tune
  eder. Ek eğitimsiz Whisper baseline'ı Serdar'ın standard çıkarımında kalır.
  Standart Parakeet çıktısını kendi supervised transducer teslimi saymaz.
- Aynı eğitim kodunda HParl-only ve HParl + CSS10 koşullarını hazırlar.
  Başlangıç modeli, ortak HParl dev/test ve scoring aynı kalır; adım/örnekleme/
  süre bütçesini kaydeder. Ek veri etkisini ek eğitim süresiyle karıştırmamak
  için karşılaştırma bütçesini önceden belirler. Son seçim yalnız dev ile yapılır.
- Küçük forward/backward/çıktı kontrolleri, merkezî koşum komutları, model
  hatalarının düzeltmeleri, kendi sonuç/hata analizi ve 1–2 slayt teslim eder.

## 5 — Ozan: değerlendirme ve sunum

- Mevcut scorer'ı inceler; corpus WER ana metriği, CER destek metriği ve
  gerekçelerini veri/görev üzerinden açıklar. Yunanca, boş çıktı, ID eksikliği
  ve toplu edit sayıları kontrollerini sürdürür. Mevcut kodu kendi sıfırdan
  yazdığı katkı olarak göstermeden kendi gerçek değişikliklerini kaydeder.
- Nihai scoring politikasını train/dev ile kararlaştırır. Yeni Mail 2'nin
  istediği tek metin normalizasyonunu her modelin referans/hipotezine aynı
  uygular; kaynak test dosyaları ve referansları değişmez. Testten ayar/model
  veya normalizasyon seçmez.
- Bütün modellerin aynı test ID/hash listesinde eksiksiz tahmin verdiğini
  kontrol eder. Boş tahminleri hata olarak puanlar; başarısız örnekleri düşürüp
  modeli daha başarılı göstermez.
- Commit edilmiş tahminlerden tek komutla sonuç tablosu, 2–3 anlamlı grafik
  ve mümkünse kayıt/oturum bağımlılığını koruyan güven aralığı üretir. Hazır
  baseline ile kendi eğitimimizi ayrı etiketler. Tabloya elle sayı yazmaz.
- Üyelerin yöntem/hata açıklamalarını birleştirir, sınırlamaları yazar, slaytları
  toplar ve sunumu yapar. Her üye kendi bölümünü ve kodunu açıklayabilmelidir.

## Ortak teslim ve çalışma sırası

Önerilen dosya sahipliği: `greek_asr/data/` ve hazırlama scriptleri Samed;
`greek_asr/models/{gmm_hmm,ctc,wav2vec,hubert}*` Kemal;
`greek_asr/models/{attention,transducer,whisper_finetune}*` Adnan;
ortak runner/ortam ile `greek_asr/pretrained/` altındaki Whisper/Canary/
Parakeet/Qwen adapter'ları Serdar; scoring/karşılaştırma/grafikler Ozan.
Bu yollar öneridir; dosyaların mevcut olduğu iddiası değildir. Ortak CLI,
manifest veya bağımlılık değişiklikleri ilgili sahiplerle PR'da uzlaştırılır.

Her üye kendi branch/PR'ında kendi kodunu, uygun testini, yöntem ve bulgu
metnini, hata örneklerini, 1–2 slaytını ve konuşmacı notlarını teslim eder.
Kişisel rapor yalnız yapılan gerçek işi anlatır ve
`contributions/<NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md` biçimindedir.
Herkes kendi GitHub hesabından commit eder; başka üye adına rapor/commit
üretilmez. Bu plan kişisel katkı raporu değildir.

Merkezî koşum paketi: commit/PR, sabit bağımlılıklar, veri manifesti, seed,
tek çalıştırma komutu, beklenen çıktılar ve küçük kontrol sonucu. Tahminler
`{"id":"source:utterance-id","text":"..."}` JSONL biçiminde, gold metin
modele verilmeden üretilir. Referans dosyası ayrı `id, reference` içerir.
Veri kimlikleri kaynaklar arasında benzersizdir. Ses manifesti kaynak/split,
göreli ses yolu, hash, süre ve varsa özgün kayıt/konuşmacı kimliğini taşır.

| Aşama | Teslim ve bağımlılık |
|---|---|
| 1 | Dil/veri rezervasyonu; özgün testi erken dondurma; veri ve scoring sözleşmesi |
| 2 | Train/dev denetimi ve gerçek temizlik; model sahiplerinden standart reçeteler; ölçülmüş donanım/veri bütçesi |
| 3 | Merkezî baseline train/dev koşuları; kendi fine-tuning deneyi; her üyeden sonuç/hata analizi |
| 4 | Dev ile seçim; model/decoder/scoring ayarlarının commit ile sabitlenmesi; ardından aynı test üzerinde son karşılaştırma |
| 5 | Koddan üretilen tablolar, ham/temiz veri teslimi, README/katkı/lisans kontrolü ve sunum |

Teslim tarihi henüz verilmediği için gün/tarih uydurulmadı. Yeni Mail 2 kesin
[baseline envanterini](BASELINES.md) netleştirdi; Yunancayı destekleyen hiçbir
aile opsiyonel olarak gizlenmez. Hoca onayı +5'in kazanıldığı
anlamına gelmez: gerçek temizleme ve yeniden üretilebilir deney gerekir.
Sunum dışında ek uygulama/backend/demo işi planlanmadı.

18 Kasım veri PR incelemesi: Samed'in kendi kodu/katkısı değiştirilmeden 73 test,
rapor üretimi ve freeze geçmişi kontrol edildi. Tam ses teslimi ile yeni
speaker/literal cümle raporu ve diğer düzeltmeler
[inceleme raporunda](reviews/SAMED_PR2_REVIEW_TR.md). PR #2 henüz birleştirilmedi.
Resmî HParl bölümünün speaker-disjoint olmadığı README'de açıklanmalı; yeni
kendi split'imiz yapılırsa speaker/recording grupları sabit seed ile atanır.
