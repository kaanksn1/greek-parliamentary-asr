# Hoca yanıtı ve deney planına etkisi

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
   henüz fiilen tamamlanmış değildir.
2. **Sonra son deney protokolü sabitlenecek:** model/checkpoint, eğitim ayarları,
   seçim kararı ve puanlama kuralı train/dev üzerinden belirlenip test
   çıkarımı/puanlamasından önce commit edilecek. Testten yeni karar üretilmeyecek.

Veri temizliği ile sonuçların puanlanması farklı işlemlerdir. Mevcut geçici
NFC/casefold/boşluk normalizer'ı, hocanın test dönüşümlerine ayrıca izin verdiği
anlamına gelmez. Test sesine veya kaynak referansına değişiklik yazılmayacak;
nihai scoring tanımı train/dev üzerinde kesinleştirilirken bu kısıt korunacak.
Teste ek dönüşüm uygulama konusunda bir istisna varsayılmayacak.

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

Hâlâ açık işler: iki kaynağın tam edinme/denetimi, sınıf rezervasyonu, teslim
tarihi, Week 9 slaytlarının tam kapsamı, testin gerçek dondurma kaydı ve
donanım ölçümlerine dayalı nihai veri/eğitim bütçesi.
