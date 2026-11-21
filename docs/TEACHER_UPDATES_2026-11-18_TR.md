# İki yeni P3 mailinin projeye uygulanması

> **21 Kas?m durum notu:** a?a??daki kaynak se?imi/pilot/bekleme ifadeleri belgenin tarihindeki durumu korur. Kullan?c? s?n?f tablosu kayd?n? teyit etti; se?ili veri, yedi e?itim ve 12 final sistem tamamland?. G?ncel kan?t ve a??k i?ler: [son denetim](TEACHER_FINAL_AUDIT_TR.md), [veri raporu](DATA_REPORT_TR.md), [final benchmark](FINAL_BENCHMARK.md). Hocan?n kurallar? bu durum notuyla de?i?tirilmez.

Kullanıcı iki maili 18 Kasım 2026'da paylaştı. Bu, maillerin gönderim tarihi
iddiası değildir. Aşağıdaki metin gerekliliklerin özetidir; eski onay yanıtı
[TEACHER_CLARIFICATION_TR.md](TEACHER_CLARIFICATION_TR.md) içinde korunur.

## 1. Bölme ve sızıntı

- Genel kural konuşmacı bazlı bölmedir. Aynı konuşmacının klipleri bölümler
  arasında dağıtılmaz. Uzun bir kayıttan kesilen klipler tek kayıt sayılır.
- Mailin **How 1** maddesi belgelenmiş resmî train/dev/test bölümünün
  kullanılabileceğini açıkça söylüyor. README hangi bölümün kullanıldığını ve
  konuşmacıların ayrı olup olmadığını açıklar. Dolayısıyla resmî bölümü kullanmak
  ile konuşmacıdan bağımsız değerlendirme yapmak aynı iddia değildir.
- Kendi bölümümüz yapılırsa konuşmacı ve mevcut kayıt/oturum ilişkileri önce
  gruplanır, gruplar sabit seed'li ve commit edilmiş kodla atanır. Konuşmacı ID'si
  yoksa gerçek kaynak kayıt/dosya gibi en yakın gruplama birimi ve sınırlaması
  açıklanır. Rastgele klip bölmesi yapılmaz.
- Her bölüm için konuşmacı sayısı, örnek sayısı ve saat raporlanır. Eksik
  konuşmacı ID'leri sayılır; bilinmeyenler yeni konuşmacıymış gibi uydurulmaz.
- Test referanslarından kaçının **kelimesi kelimesine** eğitim hedeflerinde
  bulunduğu raporlanır. Test örneği sayısı ile benzersiz cümle sayısı ayrı
  tutulur. Etiket silme/casefold gibi işlemlerle bulunan örtüşmeler ayrıca
  tanımlanır; bunlar birebir eşitlik yerine sunulmaz.
- Dev ayar/checkpoint/epoch/beam seçimidir. Test, bütün ayarlar ve scoring
  sabitlendikten sonraki son değerlendirme aşamasında kullanılır; testten
  model, filtre, normalizasyon veya decoder seçilmez.

### HParl için kararın sınırı

Samed'in PR #2'si resmî HParl oturum bölümünü koruyor. İncelenen
`4e4a80d638a5e9fb177ead6d75f46112ef4e165e` sürümü konuşmacıdan bağımsız değildir:
ilgili sayılar ve yeniden üretim [inceleme raporunda](reviews/SAMED_PR2_REVIEW_TR.md).
Bu rapor bulgusu Mail 1'deki resmî bölüm istisnasıyla birlikte README'ye yazılmalı;
"yeni konuşmacılara genelleme" sonucu olarak sunulmamalıdır. Örtüşmeyi koşulsuz
"sızıntı sorunu yok" diye nitelemek doğru değildir.

Yeni mail, önceden dondurulan özgün testin sessizce değiştirilmesi için gerekçe
değildir. Resmî bölüm seçeneğini açıkça kullanırız; daha sonra farklı bir
değerlendirme tasarımı seçilirse bunun yeni protokol olduğu açıkça belirtilir,
eski testin hash'leri veya geçmiş sonuçlar yeniden yazılmaz. Bu inceleme
sırasında hiçbir test dosyası değiştirilmedi ve model testte çalıştırılmadı.

## 2. Baseline listesi ve uygulanabilirlik

Kesin liste [BASELINES.md](BASELINES.md) içinde sahipleriyle eşleştirildi:

1. MFCC veya log-mel + GMM-HMM.
2. CTC encoder.
3. Attention encoder–decoder; hybrid CTC/attention bu aileye dahil.
4. RNN-T transducer.
5. Dile fine-tune edilmiş wav2vec 2.0 + CTC head.
6. Dile fine-tune edilmiş HuBERT + CTC head.
7. Multilingual Whisper; `.en` kullanılmaz.
8. `nvidia/canary-1b-v2`.
9. `nvidia/parakeet-tdt-0.6b-v3`; v2 İngilizcedir.
10. `Qwen/Qwen3-ASR-1.7B` veya `Qwen/Qwen3-ASR-0.6B`.

Dilimizi destekleyenlerin tamamı çalıştırılır. Bir yöntem/checkpoint
uygulanamıyorsa üreticinin model kartıyla dil/head kısıtı README'de açıklanır;
yanlış dil ayarında sonuç yazılmaz. Greek HuBERT ASR checkpoint'i hâlâ
doğrulanmayı bekliyor: yalnız araştırılmamış olması "desteklenmiyor" kanıtı
değildir. Bare encoder veya duygu sınıflandırma head'i transkripsiyon değildir.

Başkası tarafından fine-tune edilmiş hazır ASR checkpoint'i de baseline'dır.
Grubun kendi eğitimi/fine-tuning'i ayrıca katkıdır. Kendi GMM-HMM/CTC eğitimimiz
standart baseline olmaya devam eder; eğitim/uygulama emeği için ayrıca kredi
alınabilir. Hazır Qwen/Canary/Parakeet çalıştırmak yeni model katkısı değildir.

## 3. Aynı karşılaştırma ve scoring

- Bütün sistemler aynı test kimliklerini ve aynı kaynak referansları kullanır.
- Tek, versiyonlanmış metin normalizasyonu/scoring kuralı bütün referans ve
  hipotezlere aynı şekilde uygulanır. Kural train/dev ile seçilip son testten
  önce commit edilir. Mail 2'nin bu açık gereği, **puanlama sırasında bellekte**
  uygulanan ortak normalizasyonu, dondurulmuş kaynak dosyalarını değiştirmekten
  ayırır. Kaynak test referanslarına yeni metin yazılmaz.
- `<spoken_noise>` gibi anotasyon işaretlerinin ele alınışı dev üzerinde
  kararlaştırılmalı; her model ve iki eğitim koşulu için aynı olmalıdır.
  Şu anki scorer geçicidir. Nihai politika Ozan'ın görevidir.
- Her pretrained modelin tam adı ve immutable revision'ı kaydedilir. Baseline
  ve kendi eğitilmiş sonuçlarımız ayrı satır/tablolarda açıkça etiketlenir.
- Metrik gerekçesi [Hugging Face Evaluation bölümüne](https://huggingface.co/learn/audio-course/en/chapter5/evaluation)
  dayandırılır: corpus WER kelime hatalarını, CER yazım/karakter farklarını
  tamamlayıcı olarak ölçer. Corpus edit sayıları toplanır. Ek hız metriği
  seçilirse donanım/batch/zamanlama tanımı açıklanır; üç kısa runtime klibi
  tam veri hız benchmark'ı olarak sunulmaz.

Bu iki mail ham/temiz veriyi depoda teslim etme veya gerçek üye katkılarını
koruma gereğini kaldırmıyor. Samed'in ses teslimi ve rapor düzeltmeleri PR #2
incelemesinde ayrı açık maddelerdir.
