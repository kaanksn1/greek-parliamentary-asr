# Hoca yönergeleri ve son depo denetimi

21 Kasım 2026 denetimi; başlangıç public main sürümü `0db2307179a977213615cdc17ba1ab0ce9368e9e`.
Kapsam, kullanıcının paylaştığı genel proje yönergesi, önceki projelerdeki puan kesintisi duyurusu, P3 baseline/bonus maili, doğrudan hocanın üç maddelik yanıtı ve daha sonra paylaşılan split/baseline mailleridir. Bu belge bir not veya bonus garantisi değildir.

## Açık kalan zorunlu ve doğrulanmamış işler

| Madde | Denetim sonucu | Gerekli işlem / sahip |
|---|---|---|
| Her üyeye kişisel katkı dosyası | **Eksik:** beş üyeden dördünün dosyası var; Serhan'ınki yok | Serhan gerçek yaptığı işi `contributions/OZAN_SERHAN_DENIZHAN_220911758_serhandenizhan.md` dosyasına yazıp kendi hesabından commit etmeli. Boş şablon veya başkasının adına hazırlanmış rapor sayılmaz. |
| Her üyenin kendi GitHub hesabından commit'i | **Eksik:** incelenen main geçmişinde `serhandenizhan` hesabına bağlı commit yok | Serhan kendi gerçek sunum/analiz katkısını kendi branch/PR'ıyla, geçerli depo son tarihinden önce göndermeli. Mevcut dört üyenin gerçek commit'leri korunur. |
| Sunum ve son ödev yüklemesi | **Üye teslimi bekliyor:** kullanıcı 21 Kasım'da son yüklemeyi Serhan'ın yapacağını belirtti; denetleyen ajan yükleme yapmadı | Genel yönergede PPTX ve tam GitHub bağlantısı birlikte isteniyor. Serhan gerçek Blackboard koşulunu/son saati kontrol edip prime submitter ile tek teslimi koordine etmeli; sunumu ertelemek depo son tarihini ertelemez. |
| LFS aylık indirme hakkı | **Doğrulanamadı:** mevcut ses/seçili ağırlıklar 8.954381 GiB, anonim örnek nesneler erişilebilir; hesabın kalan bandwidth miktarı okunamadı | Hesap sahibi Billing and licensing / Usage ekranından kalan hakkı kontrol etmeli. Depolamaya sığmak, yeni tam clone için yeterli indirme hakkı olduğunu kanıtlamaz. Ücretli kota açılmadı. |
| Diğer gruplarla dil/veri çakışması | Kullanıcı Yunanca + HParl/CSS10 satırını sınıf tablosuna yazdığını teyit etti; tüm sınıfın güncel seçimleri bağımsız doğrulanmadı | Aynı eğitim verisi/dili kullanan yeni grup olmadığını ortak tabloda son kez kontrol edin. Rezervasyon teyidi `configs/data_source.json` içindedir. |

GitHub Free ve Pro için depolama ile indirme kotası ayrı ayrı 10 GiB'dir; başkalarının LFS indirmesi de depo sahibinin kullanımına yazılır. Sıfır bütçede kotanın dolması erişimi engelleyebilir. Buradaki risk **mevcut bir indirme hatası** iddiası değildir: anonim kontrol geçti, kalan hesap kotası bilinmiyor. Kaynak: [GitHub LFS faturalandırma belgesi](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).

Public clone doğrulaması gerçek WAV ve ağırlıklarla, zaten yüklenmiş nesnelerin yerel LFS önbelleği kullanılarak yapıldı. Yeniden yaklaşık 9 GiB indirilmedi; bu nedenle bu kontrolü yeni bir kullanıcının tüm payload'ı internetten indirdiğinin kanıtı diye sunmuyoruz.

21 Kasım sonraki erişim kontrolünde bir train WAV'ı ve küçük seçili GMM ağırlığı gerçekten anonim indirildi; toplam 1.786.223 baytın ikisi de tam SHA-256 eşleşmesi verdi. [İndirme kanıtı](reviews/final_benchmark_2026-11-21/small_public_download.json). Bu, güncel LFS payload erişiminin çalıştığını gösterir; kalan aylık hakkı veya tam clone'u doğrulamaz. Kullanıcının özel CSV raporunda yalnız depolama SKU'su vardı, bandwidth miktarı belirtilmiyordu; rapor depoya konmadı. README, kod/sonuç/sunum incelemesi için LFS payload indirmeden clone yolunu ve gerçek yürütme için seçmeli/tam indirme komutlarını açıklıyor. Ham/temiz veri teslimi değiştirilmedi.

## Önceki projelerdeki kesintiler

| Hocanın uyarısı | Mevcut kanıt / durum |
|---|---|
| Bütün vanilla ders baseline'ları ve eğitim kodu | **Tamam:** 10 gerekli aile, ayrıca iki kendi Whisper koşulumuz. [Envanter](BASELINES.md), model kaynakları, eğitim kodu ve tüm 12 sonuç satırı depoda. Kendi eğitilen standart modeller yine baseline diye işaretli. |
| `contributions/NAME_ID_GITHUB.md` | Dört doğru kişisel dosya mevcut; beşinci ve ilgili kişisel commit yukarıdaki açık iş. Ortak README kişisel dosyaların yerine sayılmıyor. |
| Ham ve temiz veri gerçekten depoda | **Tamam:** seçili ham HParl (çıkarılanlar dahil), dev/test/CSS10 özgün WAV'ları LFS'de; ham/temiz manifestler, referanslar ve kaynak metaverisi Git'te. Temiz veri aynı korunmuş WAV'ları temiz hedef/manifestlerle seçer. Kullanılmayan kaynak kataloğu kullanılan eğitim verisi gibi gösterilmez. |
| Tek hesabın tüm işi yüklemesi | Dört üyenin GitHub hesabıyla eşleşen gerçek geçmişi mevcut; Serhan henüz yok. [Public geçmiş kanıtı](reviews/final_benchmark_2026-11-21/teacher_audit.json). |
| README'de açıklanmayan dosya/çöp dosya | **Kontrol edildi:** kökteki her yol README tablosunda; her script [haritada](REPOSITORY_MAP.md). İzlenen `__pycache__`, `.pyc`, sanal ortam veya kısmi çıktı yok. Model ağırlıkları ve yardımcı ajan yönergesinin amacı açık. |
| Kendi koduna LICENSE | **Tamam:** kökte MIT `LICENSE`; ses/model lisansları ayrıca `DATA_LICENSE.md` ve `checkpoints/README.md` içinde. Dataset lisansı kod lisansı yerine kullanılmıyor. |
| Metrik gerekçesi | **Tamam:** kelime düzeyindeki WER, Yunanca yazım/çekim hatalarını tamamlayan CER, corpus S/D/I, boş hipotez ve hız kapsamı açıklı. [HF değerlendirme bölümü](https://huggingface.co/learn/audio-course/en/chapter5/evaluation) ile kaynaklandırılmış. |
| ML'ye hizmet etmeyen mühendislik | Repo ASR veri/eğitim/benchmark iş akışına odaklı; backend, veritabanı, release sistemi yok. Yerel GitHub/izleme araçları model katkısı diye sunulmuyor. |

## P3, rigor ve tekrar üretim

| Gereklilik | Kanıt / yorum |
|---|---|
| İngilizce dışı, farklı dil; güvenilir gerçek referans | Yunanca HParl + yalnız eğitimde CSS10. Kaynak referansları korunuyor; model çıktıları gold yapılmıyor. Yunanca seçimi tek başına düşük kaynak bonusu iddiası değildir. |
| İki korpus ve domain bonusu | Hoca planı ilke olarak uygun buldu. Parlamentodaki değerlendirme aynı; CSS10 yalnız train. Gerçek temizleme/izlenebilirlik var; birleştirme veya yeniden biçimleme tek başına temizlik diye sunulmuyor. +5 puan kazanılmış gibi yazılmıyor. |
| İstatistiksel olarak anlamlı, donanım bütçesine uygun veri | 30.883408 saat temiz HParl + 4.137269 saat CSS10; dev 9.059189, test 11.207889 saat. Seçim ve yedi eğitim bütçesi gerçek CPU/GPU süre/bellek ölçümleri ve ücretsiz alan sınırıyla gerekçeli. Üç kliplik pilot tam eğitim yerine sayılmıyor. |
| Gerçek temizleme | Seçili ham/temiz ID farkı günlükteki 13 çıkarma kararıyla eşleşiyor: dokuz `cps_extreme`, dört `duplicate_audio`. 15.370 metin politikası girdisi ayrı sayılıyor; insan dinleme kararı 0 ve inceleme kuyruğu 176 açıkça yazılı. [Üretilmiş veri raporu](DATA_REPORT_TR.md). |
| Split kuralı | HParl'ın belgeli resmî oturum bölmesi korunuyor. Hocanın resmî split istisnası kullanılıyor; **speaker-disjoint değil**. README 208/64/82 tanımlı HParl train/dev/test konuşmacısını, kayıt ve saatleri veriyor. CSS10'un kodlanmış ID'si yok ve tamamı yalnız train. |
| Örtüşme raporu | Seçili train ile testte ortak 49 konuşmacı, bu seslere ait 4.782 test klibi; ortak oturum/ses hash'i 0. Literal test cümlesi 3, ortak final normalizasyonla 9. Bunlar [koddan üretiliyor](../results/data_integrity.json), yeni konuşmacı başarısı diye sunulmuyor. |
| Testi önceden dondurma | Dondurma kimliği `9a680104…`; özgün test ses/manifest/referansları temizlikten önce korunmuş. Son test başı/sonu sesli doğrulama kayıtları mevcut. Testten kayıt silme, yeni etiket veya yeniden bölme yok. |
| Dev seçimi, sonra tek final test aşaması | Tam dev kaybı checkpoint'i seçiyor; kaynak/batch/GMM ayarı 65 dev klibinde sabitleniyor. Final protokol `0ff3196c` commit'inde test başlamadan önce commit edildi. Başarısız altyapı denemeleri ve aynı ayarla yeniden başlatmalar loglarda korunuyor; testten ayar seçilmedi. |
| Dile uygun model ve gerçek ASR head | Model kartı/revision bağlantıları final protokolde. Whisper multilingual, Parakeet v3. HuBERT sonucu bizim eğitimimizle oluşmuş gerçek Yunanca CTC head'ini içeriyor; lisanssız aday veya bare encoder sonuç diye kullanılmıyor. |
| Aynı test ve aynı scoring | Her sistemin 8.679 özgün ID'si, referansı ve final normalizasyonu aynı. Politikada test dosyası yeniden yazılmaz; scoring sırasında ortak bellek dönüşümü uygulanır. Eksik/kısmi sistemden skor üretilmez. |
| Baseline / katkı ayrımı | Hazır Qwen/Canary/Parakeet/Whisper ve yayımlanmış fine-tuned wav2vec baseline. Kendi eğitim/uygulama emeği ayrıca belirtilir; matched HParl-only/mixed Whisper deneyi ayrı satırlardadır. |
| Tam checkpoint ve revision | Immutable upstream kimlikleri, kaynak/scorer hashleri, seçili ağırlık hashleri, decoder/device/batch final protokol ve model lock'unda; seçili yedi gerçek ağırlık LFS'de. |
| Her sayının committed koddan gelmesi | Eğitim tabloları orijinal `run.json` kayıtlarından, final tablolar/şekiller kayıtlı tahmin/skorlardan üretilir. Arşiv doğrulayıcı 12 sistemin S/D/I, WER/CER, bootstrap ve CSV değerlerini tekrar hesaplar. [Son kabul kaydı](reviews/final_benchmark_2026-11-21/results_validation.json). |
| Ortamlar ve seed | Gereksinimler `==` veya kesin wheel URL'siyle sabit; ayrı Transformers/NeMo lock'ları. Seed 42 ve gerçek runtime kayıtları mevcut. Her iki ölçülmüş ortamda `pip check` geçti. |
| İşe yönelik testler | 240 test, sıfır atlama: veri/format/temizleme/freeze, model padding/device/loss/decoder ve evaluation kontrolleri. [Orijinal kayıt](reviews/final_benchmark_2026-11-21/frozen_runtime_tests.log). Bu, başarısız modellerin yakınsadığını iddia etmez. |
| Raw → temiz → eğitim → final tablo | [Tekrar üretim rehberi](FINAL_BENCHMARK.md), seçili veri komutları ve script envanteri depoda; Colab'a özel yükleme veya Drive bağlantısı gerekmiyor. Exact upstream model cache'i kendi seçili ağırlık/veri tesliminin yerine geçmiyor. |
| Public depo / tam bağlantı | Anonim depo ve LFS örnek erişimi geçti. Teslim URL'si `https://github.com/kaanksn1/greek-parliamentary-asr`. Son yükleme ve teslim zamanını Serhan/prime submitter kontrol eder; ajan LMS teslimi yaptığını iddia etmez. |

## Kalite ve yorum sınırları

- Plain CTC ve HuBERT-CTC WER'leri yaklaşık %100; attention/RNN-T/GMM sonuçları da zayıf. Kaydedilmiş çıktılar özellikle CTC'de çok kısa/yinelenen parçalara çöküyor. Tam çalışmış olmak yakınsama kanıtı değildir. Gerçek kötü sonuçlar tutulur; kesin neden gösterilmeden decoder hatası veya başarılı eğitim iddiası kurulmaz. Final test görüldükten sonra bunları ayarlamak bu protokolün adil değerlendirmesini bozar.
- Matched Whisper farkı aynı hesap bütçesi ve tek seed altındaki veri bileşimi deneyidir. Speaker-bootstrap aralığı test konuşmacılarındaki örnekleme belirsizliğini gösterir; eğitim seed'leri arası varyansı ölçmez. Resmî splitteki ortak konuşmacılar nedeniyle sonuçlar görülmemiş konuşmacı genellemesi değildir.
- İnceleme kuyruğuna insan dinlemesi yapılmadı. Otomatik flag'ler doğrulanmış yanlış etiket sayılmaz; yeni insan anotasyonu veya kapsamlı manuel temizlik katkısı iddiası kurulmaz.
- Bu sınırlamalar saklanarak tam not hedeflenemez. Kapsamlı benchmark ve doğru raporlama, güçlü/zayıf sonuçları birlikte görünür tutar; nihai Correctness/Rigor/novelty değerlendirmesi hocanındır.

## Bu denetimde giderilen belge sorunları

Önceki baseline listesi tamamlanan eğitim/testleri hâlâ bekliyor gösteriyordu; güncel envanterle değiştirildi. Tam kaynak envanteri ile teslim edilen seçili train'i karıştıran veri raporu, gerçek seçili manifest ve temizleme günlüğünden yeniden üretilir hale getirildi. Eski görev/pilot/taslak belgelerine tarihsel durum notu ve güncel sonuca yönlendirme eklendi. README'den her script'in amacı bulunabilir; bütün yerel Markdown bağlantıları tekrar kontrol edilir.

Kişisel katkı raporları kendi yazarlarına aittir; bu düzeltmeler bir üyeye yapmadığı işi yazmaz. GMM LM betimlemesindeki eski bigram/trigram farkı [erratum](FINAL_PROTOCOL_ERRATA.md) ile açık tutulur; dondurulmuş protokol, model algoritmaları, test, kaynak sesler ve seçili ağırlıklar bu denetimde değiştirilmez.

## Serhan'ın teslim kontrolü

1. Gerçek sunum/analiz çalışmasını, doğru adlı kişisel dosyasını ve kendi commit'lerini PR ile zamanında tamamla; prime submitter ile tek yüklemeyi koordine et.
2. Blackboard'daki güncel PPTX + GitHub biçimini ve kesin son saati kontrol et. Linki anonim pencerede aç; LFS erişim/kota durumunu da kontrol et.
3. Genel sunum yönergesi: teknik derinlik %50, akıcılık %25, tempo %25; slayttan okuma akıcılık puanını düşürür. Bildirilen tempo kuralında iki dakikanın altı en yüksek kademedir, 3:30'da kesilme/0 riski vardır; P3'e özel daha yeni bir süre verilmişse onu uygula.
4. Tüm üyeler sunum/sonraki tartışma oturumuna katılmalı ve MEDU yoklamasını kontrol etmeli. Son tabloları [sunum devrindeki](HANDOFF_SERHAN_FINAL_TR.md) kaynaklardan al; pilot sayılarını final gibi kullanma.
