# Hoca yanıtı ve deney planına etkisi

> **21 Kasım durum notu:** aşağıdaki kaynak seçimi/pilot/bekleme ifadeleri belgenin tarihindeki durumu korur. Kullanıcı sınıf tablosu kaydını teyit etti; seçili veri, yedi eğitim ve 12 final sistem tamamlandı. Güncel kanıt ve açık işler: [son denetim](TEACHER_FINAL_AUDIT_TR.md), [veri raporu](DATA_REPORT_TR.md), [final benchmark](FINAL_BENCHMARK.md). Hocanın kuralları bu durum notuyla değiştirilmez.

Kullanıcı bu yanıtı 17 Kasım 2026'da paylaştı. Ekran görüntüsünde mailin
gönderim tarihi görünmüyor; bu tarih mailin gönderim tarihi olarak sunulmaz.

Sorulan ilk konu, HParl ve ikinci bir Yunanca korpusu anlamlı biçimde temizleyip
birleştirmenin custom veri seti bonusu kapsamında değerlendirilmesiydi.
Yanıt: **“Uygundur.”** Bu, önerilen planın uygunluğuna onaydır. Gerçek temizleme
işini ve katkı kanıtlarını tamamlamadan bonusun kazanıldığını veya +5 puanın
kesinleştiğini söylemeyiz. Kaynaklar ve türetme işlemleri açıkça belirtilecek.

İkinci soru, ek kaynağı yalnız eğitimde kullanıp değerlendirmeyi parlamento
konuşmalarında tutmanın alan odaklı ASR bonusuna uygunluğuydu. Yanıt:

> Olur. Test setini önceden dondurun ve asla bir işlem uygulamayın.
> Training ve dev splitine istediğinizi yapabilirsiniz.

Buna göre iki ayrı aşama var:

1. **Önce test verisi sabitlenecek:** kaynak sürümü, ID/oturum listesi, özgün
   ses ve referans dosyalarının hash'leri kaydedilecek. Testte temizleme,
   filtreleme, kayıt ekleme/çıkarma, yeniden bölümleme ve referans düzeltme
   yapılmayacak. İkinci kaynak test kümesine karıştırılmayacak. Bu aşama
   Samed'in PR #2'sinde metadata olarak mevcut; tam kaynak ses teslimi ve
   Serdar'ın bağımsız byte doğrulaması henüz tamamlanmış değildir.
2. **Sonra son deney protokolü sabitlenecek:** model/checkpoint, eğitim ayarları,
   seçim kararı ve puanlama kuralı train/dev üzerinden belirlenip test
   çıkarımı/puanlamasından önce commit edilecek. Testten yeni karar üretilmeyecek.

Veri temizliği ile sonuçların puanlanması farklı işlemlerdir. Test sesine veya
kaynak referansına değişiklik yazılmayacak. Sonradan paylaşılan ikinci yeni
mail, bütün sistemler için tek normalizasyon/scoring kuralını açıkça istiyor.
Bu kural train/dev ile belirlenip puanlama sırasında referans/hipoteze aynı
şekilde bellekte uygulanır; frozen kaynak dosyaları yeniden yazılmaz.
Mevcut geçici NFC/casefold/boşluk normalizer'ı nihai politika değildir.

Üçüncü soru minimum ses süresiydi. Hoca sabit bir saat sayısı vermiyor;
uygulamaya göre istatistiksel sonuca yetecek ve grubun donanım/maliyet bütçesinde
çalışabilecek **en büyük veri setini** bekliyor. Dolayısıyla “minimum saat
yanıtını bekliyoruz” maddesi kapanmıştır; keyfî bir küçük saat sınırı seçmek
yerine veri bütçesini ölçerek gerekçelendireceğiz.

Veri bütçesi için train/dev'de gerçek süre, VRAM, epoch maliyeti, veri kalitesi
ve değerlendirme belirsizliği incelenecek. Üç kliplik çalışma kontrolü bu bütçeyi
belirlemeye veya istatistiksel yeterlilik iddiasına yetmez. Tek tek kelimeleri
bağımsız örnek saymak yerine oturum/konuşmacı bağımlılıkları da dikkate alınacak.

Sonraki kaynak planında CSS10 Greek'in Kaggle v1 dağıtımı, üreticinin CC0
katalog kaydıyla ikinci eğitim kaynağı seçildi; yerel edinme/denetim bekliyor.
Bu seçim hocanın mailinde isim verilerek onaylanmış bir kaynak gibi sunulmaz.
[Kaynak kararı](SECONDARY_SOURCE_TR.md) ve [görev dağılımı](TEAM.md).

Yeni veri tesliminde PR #2 kaynak/temiz manifestler ve bir erken freeze
getiriyor; bağımsız inceleme hash/commit tutarlılığını doğruladı. Tam ses ve
Kaldi dosyaları henüz public teslimde yok. Ayrıntı
[inceleme raporunda](reviews/SAMED_PR2_REVIEW_TR.md).

Hâlâ açık işler: tam ses teslimi ve bağımsız byte doğrulaması, veri incelemesinin
düzeltmeleri, sınıf rezervasyonu, teslim tarihi, çalışan standart reçeteler ve
donanım ölçümlerine dayalı nihai veri/eğitim bütçesi. Kesin baseline listesi,
resmî split istisnası ve speaker raporu gereği
[iki yeni mail özetinde](TEACHER_UPDATES_2026-11-18_TR.md) kayıtlı.
