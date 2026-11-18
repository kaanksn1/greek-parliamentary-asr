# Adnan'a devir — attention, RNN-T ve Whisper fine-tuning

18 Kasım 2026 kullanıcı kararı: Samed düzeltmelerini tamamlayıp veri teslimini
pushladıktan sonra sıradaki model geliştiricisi Adnan olacak. Görev sahipliği
değişmedi; Adnan Kemal'in modellerinin bitmesini beklemeden kendi kodunu
hazırlayabilir. Tam model koşuları yine Serdar'ın bilgisayarında yapılır.

Hazırlama anında Samed'in PR #2'si hâlâ `4e4a80d` sürümünde ve inceleme
eksikleri açıktır. Aşağıdaki prompt bu eski sürümü tamamlanmış saymaz: gerçek
eğitim için Samed'in güncel, kabul edilmiş veri teslimi gerekir. Veri beklerken
yalnız bağımsız model kodu ve küçük sentetik kontroller ilerleyebilir.

## Adnan'ın kendi kod asistanına vereceği prompt

```text
Ben Adnan Demir, öğrenci numaram 220911784, GitHub hesabım adnan-demir.
Anonymous grubunun NLP Proje 3 model sorumlusuyum.

Repo: https://github.com/kaanksn1/greek-parliamentary-asr
Kendi branch'im: feature/p3-sequence-models

Samed'in veri tesliminden sonra kendi üç görevimi tamamla:
A) Standart attention encoder-decoder ASR; hybrid CTC/attention bu aileye dahil.
B) Standart RNN-T transducer ASR.
C) Aynı Whisper başlangıç checkpoint'inden HParl-only ve HParl + CSS10
   fine-tuning koşullarını hazırlamak.

Bu mesaj bu model görevlerini açıkça Adnan kapsamında atar. Başka üyelerin
veri pipeline'ını, GMM-HMM/CTC/wav2vec/HuBERT modellerini, ortak scorer'ını,
sunumunu veya kişisel katkı raporlarını benim adıma yapma.

1. GÜNCEL REPO VE VERİYİ DEVRAL

- Yerel değişiklikleri kontrol et ve koru. Güncel main'den kendi branch'ime
  geç; branch varsa mevcut çalışmasını incele. Başkasının kimliğiyle commit,
  force-push, git clean veya veri silme yapma.
- Şunları oku: AGENTS.md, README.md, docs/TEAM.md, docs/BASELINES.md,
  docs/DATA_CONTRACT.md, docs/PROTOCOL_DRAFT.md,
  docs/TEACHER_UPDATES_2026-11-18_TR.md, docs/RUNTIME_ENVIRONMENT.md,
  docs/NEMO_RUNTIME.md ve configs/model_sources.lock.json.
- Samed'in son DATA_REPORT_TR.md, DATA_PIPELINE_TR.md, veri incelemesi,
  data/manifests/conditions.json ve test.freeze.json dosyalarını oku.
- Eski 4e4a80d incelemesini veri teslimi kabul edilmiş gibi kullanma.
  Güncel veri PR'ının Serdar tarafından kabul edilip main'e alındığını veya
  Serdar'ın açıkça onayladığı veri commit'ini doğrula. Başkasının PR'ını
  kendin tamamlandı sayıp merge etme.
- Kullanılan kod/veri commit'ini, koşul manifestlerini ve hash'lerini kaydet.
  Saat ve kayıt sayılarını eski prompttan kopyalama; son dosyalardan hesapla.
- Gerçek ses ve özgün metadata mevcutsa Samed'in verify-freeze ve verify
  komutlarını çalıştır. LFS pointer'ı gerçek ses sanma. Dosyalar eksikse
  bunu açıkça bildir; model kodu/sentetik testlerle devam edebilirsin fakat
  gerçek veri koşusu başarılı olmuş gibi yazma ve kendi veri pipeline'ını kurma.

Ortak dosyalar:
- data/manifests/train.jsonl + references/train.jsonl: HParl-only train.
- data/manifests/train_hparl_css10.jsonl + references/train_hparl_css10.jsonl:
  karışık train.
- data/manifests/dev.jsonl + references/dev.jsonl: ortak HParl dev.
- data/manifests/test.jsonl + references/test.jsonl + test.freeze.json:
  korunacak özgün test; bu görevde model çıkarımı/puanlama yapılmayacak.

2. SAHİPLİK, TEST VE ÇALIŞTIRMA SINIRLARI

- Bütün tam eğitimler ve model tahmin koşuları, CPU baseline'lar dahil,
  Serdar'ın RX 7800 XT / 32 GB RAM bilgisayarında çalışır. Kendi makinemde
  küçük sentetik CPU testleri yapılabilir. Gerçek GPU pilotunu da Serdar'a
  eksiksiz komut/config olarak teslim et.
- Şimdi global final-test protokolü oluşturma veya test sonucu üretme.
  Dev ayar/checkpoint seçimidir. Final test ancak bütün seçimler ve scoring
  Serdar/Ozan tarafından commit ile sabitlendikten sonra ortak aşamada açılır.
- Samed'in özgün testini yeniden dondurma, yeniden bölme, filtreleme,
  kırpma, yeniden etiketleme veya referanslarını normalize ederek kaydetme.
- Resmî HParl bölümünde konuşmacılar örtüşüyor. Mailin resmî split seçeneğini
  kullanıyoruz; speaker-disjoint/yeni konuşmacı sonucu iddiası yazma.
- CSS10 yalnız train'dedir. Dev/teste karıştırma. Model çıktısını gold sayma.
- Kod hazırlamak için Kemal'i bekleme. Baseline koşuları ve kendi fine-tuning
  koşularının sırasını Serdar'ın ortak deney planına göre ayır; eksik diğer
  baseline'ları tamamlandı sayma.

3. STANDART ATTENTION ASR

- Standart, belgeli bir supervised attention encoder-decoder reçetesi kur.
  Hybrid CTC/attention seçebilirsin; seçim, kaynak ve loss ağırlıkları açık olsun.
- Standart attention/RNN-T'yi kendimiz eğitsek de bunlar baseline aileleri
  olarak raporlanır; gerçek uygulama/eğitim emeği ayrıca katkıda belirtilir.
  Rastgele başlangıç ağırlığını anlamlı eğitimsiz ASR sonucu diye sunma.
- Ses encoder'ı, attention/decoder, girdi özellikleri, tokenizer/vocab,
  padding/mask ve decoding yolunu açıkça tanımla. Bu bir ASR modeli olmalı.
- Standart toolkit kullanılabilir. Resmî kaynak/reçete, tam sürüm veya commit,
  mimari/config, eğitim ve tahmin komutları depoda bulunmalı. Sıfırdan genel
  bir ASR framework'ü veya gereksiz backend geliştirme.
- Kaynaksız özel mimariyi vanilla baseline diye sunma. Hazır Whisper çıkarımı
  bu ayrı supervised model teslimimin yerine geçmez. Hybrid yardımcı CTC
  başlığı da Kemal'in ayrı plain CTC teslimini tamamladığım anlamına gelmez.
- Tokenizer/vocab öğrenilecekse yalnız train'den öğren. Sabit pretrained
  tokenizer kullanılıyorsa kaynağı/revision'ı belli olsun. Dev/test üzerinde
  vocab veya normalizasyon fit etme.
- Teacher forcing eğitim/dev-loss hesabında kullanılabilir; tahmin üretirken
  gold transkripti prompt veya decoder başlangıç metni olarak verme.

4. STANDART RNN-T

- Acoustic encoder, prediction network, joint network, blank sembolü,
  RNN-T loss ve gerçek transducer decoding yolunu kur ve belgele.
- CTC loss kullanan bir sistemi RNN-T diye adlandırma. Hazır Parakeet TDT
  çıkarımı da kendi RNN-T eğitim reçetemin yerine geçmez.
- Loss backend'ini ve lisansını resmî kaynaklardan doğrula. Serdar'ın AMD
  ortamında CUDA-only bağımlılık varsayma: NeMo/Parakeet inference başarısı,
  RNN-T eğitim loss'unun çalıştığını ispatlamaz.
- Önce çok küçük bir örnekte gerçek loss/backward ve gradyan kontrolü hazırla.
  T/U uzunlukları, blank ID ve dtype/device sözleşmesini kontrol et.
- Uyum sorunu çıkarsa hatayı ve uygulanabilir çözümü somutlaştır. Sessizce
  modele CTC takma, RNN-T'yi listeden çıkarma veya tam koşuyu CPU'ya düşürme.
  Ücretli bulut/servis kullanma. GPU doğrulaması Serdar'ın koşumunda yapılır.

5. WHISPER FINE-TUNING KATKISI

- Başlangıç: multilingual openai/whisper-small.
- Revision: 973afd24965f72e36ca33b3055d56a652f456b4d.
  Bunu configs/model_sources.lock.json ile doğrula; hareketli main kullanma.
- Owner'ın ek eğitimsiz Whisper baseline'ını ve kaynak ağırlıklarını koru.
  Yeni eğitim checkpoint'lerini ayrı çıktı yollarına kaydet.
- Tek eğitim kodu, iki ayrı config kullan:
  1) HParl-only.
  2) Aynı HParl + yalnız eğitimde CSS10 Greek.
- Aynı başlangıç ağırlığı/tokenizer, seed, ortak HParl dev/test ve scoring
  kullan. Yunanca transcription ayarını doğru kur; .en checkpoint veya
  çeviri görevi kullanma.
- Veri bileşiminin etkisini ek optimizasyonla karıştırma. Eşlenmiş eğitim
  bütçesini önceden seç: örneğin aynı optimizer update sayısı ve effective
  batch. Eğitimde görülen örnek/ses süresi, CSS10 örnekleme oranı ve seçilen
  checkpoint adımını da raporla. Yalnız aynı epoch sayısı yeterli gerekçe değil.
- İki koşulda aynı checkpoint seçim kuralını kullan; seçim yalnız dev'den.
  Bir koşul için test skoruna bakarak epoch, karışım oranı veya beam seçme.
- Varsayımsal iyileşme veya bonus yazma. Karışık veri daha kötü çıkarsa aynı
  kurallarla raporlanır; sonucu iyi göstermek için koşul değiştirilmez.

6. ORTAK VERİ VE SCORING SINIRLARI

- Manifest ile ayrı referans dosyasını ID üzerinden eksiksiz eşleştir.
  Eksik/tekrar ID, bozuk ses veya çözülemeyen hedefi sessizce atlama.
- HParl 16 kHz, CSS10 22.05 kHz olabilir. Modelin istediği örnekleme hızına
  yalnız yükleme sırasında deterministik dönüşüm yap; raw/frozen dosyaları
  değiştirme. Pad/attention mask ve gerçek uzunlukları doğru taşı.
- Eğitim etiketlerindeki padding loss'ta yok sayılsın. Whisper'da BOS/EOS,
  language/task tokenları ve -100 maskelerini test et; gerçek EOS'u padding
  sanıp silme. Hedefleri sessizce kesme veya Yunanca karakterleri düşürme.
- Ozan'ın ortak scorer/normalizasyonunu yeniden yazma veya modele göre
  değiştirme. <spoken_noise>, noktalama, harf büyüklüğü ve vurgu politikası
  train/dev üzerinden ortak kararla versiyonlanır; frozen referanslar korunur.
  Politika kesinleşmemişse gerçek checkpoint seçimi için bağımlılığı açıkça
  kaydet; testten veya tek modelin daha iyi skorundan kural türetme.
- Tahmin üretiminin girdisi ses ve önceden belirlenmiş model ayarlarıdır.
  Gold metin modele verilmez. Gerçek boş çıktı korunur; başarısız işlem boş
  metinle başarılı sonuç gibi gösterilmez.
- Tahmin JSONL: her giriş ID'si için tam bir {"id":"...","text":"..."} satırı.
  Başarısız/kısmi koşular başarılı predictions dosyasından ayrı işaretlenir.
- Uzun ses/hedef stratejisini train/dev'de açıkça belirle. Sonradan zor örnek
  veya test klibi düşürerek sistemi daha başarılı gösterme.

7. BÜTÇE, ORTAM VE KOŞU KAYDI

- Samed'in son kabul edilmiş veri sürümünü kullan. Keyfî küçük saat sınırı
  belirleme. Serdar için gerçek modelle ölçülecek pilot komutunu hazırla:
  forward/backward, sonlu loss/gradyan, VRAM, step süresi ve tahmini epoch
  maliyeti. Üç klip inference smoke'u eğitim kapasitesi ölçümü değildir.
- Batch/gradient accumulation, gradient checkpointing ve precision ayarları
  gerçek pilotta doğrulansın. AMP kullanılıyorsa taşma/NaN kontrolü olsun.
- Bütçe alt küme gerektirirse Serdar'ın kabul ettiği, seed'li grup/oturum
  seçimiyle manifest üretimini veri sahibiyle koordine et. Yeni split yapma;
  dev/testi değiştirme. İki Whisper koşulunda aynı HParl alt kümesini kullan.
- Mevcut ortamları yerinde bozma. Yeni gereksinimleri ayrı, sürümleri sabit
  dependency dosyası ve kurulum notuyla öner. CUDA wheel'ını ROCm Torch'un
  yerine sessizce kurma. Inference ortamı eğitim için henüz doğrulanmış sayılmaz.
- Her gerçek koşuda kod commit'i, config/veri/referans/tokenizer hash'leri,
  model revision'ı, seed, trainable parametreler, optimizer/scheduler, LR,
  effective batch, precision, gerçek update sayısı, süre ve ortam kaydedilsin.
- Checkpoint'te model/config, optimizer/scheduler, global step ve devam için
  gerekli RNG bilgileri korunsun. Best ve last ayırt edilsin. Dev seçim
  kaydı bulunsun; mevcut checkpoint veya sonuçlar üstüne yazılmasın.

8. KOD VE DOĞRULAMA TESLİMİ

Önerilen yeni alanlar (mevcut dosyaymış gibi davranma; çalışır halde oluştur):
- greek_asr/models/attention.py
- greek_asr/models/transducer.py
- greek_asr/models/whisper_finetune.py
- configs/sequence/ altında yöntem ve iki Whisper koşulu config'leri.
- Kendi train/predict CLI'larım, küçük anlamlı testlerim ve yöntem dokümanlarım.

Dosya adları değişebilir; README'de var olan gerçek komutları yaz.
Owner'ın greek_asr.infer ve pretrained adapter'larını ezme. Paylaşılan arayüz
değişikliği gerekirse Serdar'la uzlaştır; kendi CLI'ın aynı çıktı sözleşmesini
üretebilir. Test çıkarımı varsayılan kapalı olsun; global final protokol
olmadan --split test çalışmasın. Bu görevde fake final protokol oluşturma.

Gerekli anlamlı kontroller:
- Değişken uzunluk/padding, ayrı reference-ID join ve Unicode/Yunanca kapsamı.
- Tiny attention forward/backward ve gerçekten güncellenen parametreler.
- Tiny RNN-T loss/backward, doğru uzunluk/blank ve bilinen küçük loss örneği.
- Whisper label/padding/prefix kontrolü ve fine-tuning'de gerçek optimizer adımı.
- Checkpoint kaydet/yükle sonrası tutarlı evaluation davranışı.
- Audio-only tahmin, gold sızıntısı olmaması, tüm ID'ler ve gerçek boş çıktı.
- Test erişiminin protokolsüz engellenmesi ve kaynak dosyaların değişmemesi.
- İki Whisper koşulunda yalnız planlanan veri farkı ve eşlenmiş bütçe.

Küçük sentetik kontroller CPU'da çalışabilir; GPU gerektiren kontrol
çalışmadıysa çalıştı deme. Tüm uygun repo testlerini de çalıştır. Yalnız
config varlığını kontrol eden testlerle model matematiği doğrulandı sayılmaz.

9. SERDAR'A KOŞUM PAKETİ, PR VE SONRAKİ ANALİZ

İlk teslim: çalışır kod, konfigürasyonlar, bağımlılıklar, test çıktısı ve her
model/koşul için tek gerçek çalıştırma komutu. Komutla birlikte çalışma
dizini, veri/checkpoint girişleri, beklenen çıktı dosyaları, cihaz ve henüz
ölçülmediyse pilotta ölçülecek bellek/süre alanlarını açıkla.

Tam koşumdan sonra Serdar'ın döndürdüğü gerçek loglarla kendi modelimin
hatalarını düzelt; kendi yöntem/sonuç açıklamamı ve dev hata örneklerimi yaz.
Henüz koşulmamış deneyler için sayı, checkpoint, hata örneği veya tablo uydurma.
Gerçek modellerin 1-2 slayt taslağını ve konuşmacı notlarını Ozan'a teslim
edilecek dosya olarak hazırla; bütün sunumu veya ortak karşılaştırmayı üstlenme.

Kendi katkı dosyam:
contributions/ADNAN_DEMIR_220911784_adnan-demir.md

Yalnız gerçek yaptığım kod/test/analiz/commit'leri yaz. Başlangıç altyapısını
ve başkasının modelini kendi katkım diye gösterme. Açık kaynak reçete/kodu
uyarladıysam kaynağını ve gereken lisans/atıfları koru.

Kendi hesabımla kendi branch'ime commit/push yap ve PR aç. PR açıklamasında:
- Seçilen standart mimari/reçete ve kaynaklar.
- Çalışan kod/testler ve gerçek doğrulama komutları.
- Samed'in kabul edilmiş veri sürümü ve hash'leri.
- Serdar'ın çalıştıracağı eksiksiz komutlar.
- Tamamlananlar, merkezî koşum bekleyenler ve gerçek teknik engeller.

Görevi yalnız plan yazarak bırakma: kendi kapsamımdaki uygulanabilir kod,
config, test ve çalıştırma paketini tamamla. Merkezi GPU koşusu gerektiren
aşamaları açıkça bekliyor olarak bırak; onları yapılmış gibi işaretleme.
```
