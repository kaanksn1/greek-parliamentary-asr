# PR #5 — son P2 düzeltmeleri tamamlandı

19 Kasım 2026. Kullanıcı kalan düzeltmeleri Serdar'a açıkça atadı.
Adnan'ın `0c13ce1` model commit'leri ve kişisel katkı dosyası korundu;
owner'ın sınırlı tamamlama commit'i:
**`5a8df80f02584652de59618b081bba583028a627`**.
Bu belge [önceki incelemenin](ADNAN_0c13ce1_REVIEW_TR.md) kapanış kaydıdır;
önceki bulgular kendi sürümleri için tarihsel olarak korunur.

Tamamlananlar:

- Whisper train/dev loss sonlu değilse eğitim hemen durur; bekleyen
  gradient'ler temizlenir, `run.json` durum/hata/faz/ID/adım bilgileriyle
  `failed` olarak yazılır. NaN atlayıp eksik accumulation ile update yapmak
  veya sonsuz yeniden denemek kaldırıldı. Nonfinite dev loss best checkpoint
  seçemez. Normal sonlu koşu `complete` olur.
- Tahmin, loader veya ağırlık indirmesi öncesinde seçilmiş
  `openai/whisper-small` kaydını ve revision'ı source lock'a karşı doğrular.
  Lock'ta doğru pinlenmiş başka bir model de Whisper yerine kabul edilmez.
- RNN-T gerçek hedef uzunluğunun dışındaki padding indekslerini gather
  öncesi güvenli hale getirir. Geçersiz gerçek hedef etiketleri gizlenmez.
  Kayıp ve gradient'lerin padding sentinel'ından bağımsızlığı test edildi.
- Transducer config'indeki backend, dev loss/WER ayrımı, owner Whisper
  attention mask/long-form davranışı, CPU/ROCm ortam açıklamaları ve GPU
  doğrulama bilgileri gerçek kodla eşleştirildi. Eğitim bütçeleri `null`
  kaldı; sayısal hyperparameter veya checkpoint source lock değiştirilmedi.

## Gerçek doğrulama

- Model/ortak suite: **119/119 test başarılı**. Nonfinite train/dev,
  yanlış revision/yanlış aile ve ignored-padding regresyonları dahil.
- Samed `4db1a4aa61eb78acbb531b705c45e3bbd090854e` ile geçici kod
  birleşimi: **151/151 test, 33,164 saniye, başarılı**. Büyük LFS sesleri
  indirilmedi; bu bir tam ses teslimi/hash doğrulaması değildir.
- [Son commit'in GPU kontrolleri](SEQUENCE_P2_5a8df80f_GPU.json): üç modelde
  sonlu loss ve bütün mevcut gradient'ler, gerçek parametre güncellemesi.
- [Gerçek attention/RNN-T CLI kontrolleri](SEQUENCE_P2_5a8df80f_ENTRYPOINTS.json):
  GPU train, checkpoint'ten yükleme ve predict başarılı. Accumulation=3 ile
  iki optimizer update/altı microbatch; üç train ID'si eksiksiz. Bu küçük
  bütçe yalnız diagnostik içindir.
- [Bağımsız RNN-T oracle](SEQUENCE_P2_5a8df80f_MATH.json): 16 şekil × batch 2
  için loss/gradient karşılaştırmaları başarılı; `-100` ignored-padding
  sözleşmesi artık başarılı. Float64 toleransı `1e-10`.

GPU: RX 7800 XT; mevcut ayrı ROCm runtime, tam pinler
`requirements-rocm.lock.txt` içinde. Paket kurulmadı. Kaynaklar ve çalışan
diagnostik betikleri JSON kanıtlarında hash'lenir. Komutlar önceki incelemenin
"Yeniden üretim" bölümündeki üç owner diagnostik betiğidir; yalnız
`--submission-root` ve yeni output yolları bu commit'e çevrilmiştir.

Gerçek eğitim/dev seçimi/benchmark yapılmadı. Kontroller yalnız üç özgün
train-pilot klibi, unit fixture'ları ve sentetik matematik örnekleri kullandı.
CLI fixture'ındaki dev rolü de train pilotudur; resmî development seti değildir.
Geçici model ağırlıkları kaldırıldı. Test kaynakları/ref'leri değiştirilmedi,
resmî dev/test üzerinde çıkarım yapılmadı. Tam eğitim için gerçek kullanılan
train sesinin teslimi ve temsil edici süre/bellek ölçümü hâlâ gerekir.
