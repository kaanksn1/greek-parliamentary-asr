# İkinci kaynak kararı — CSS10 Greek

> **Superseded planning record:** pending/acquisition/scoring statements below are historical. Current data: [selected report](DATA_REPORT_TR.md); protocol: [final protocol](../configs/final_test_protocol.json); completed results: [final table](../results/benchmark/SUMMARY.md).

17 Kasım 2026 planı: HParl'e **yalnız eğitimde kullanılmak üzere CSS10 Greek**
eklenecek. İkinci korpus henüz indirilmedi; temizleme, yerel süre ölçümü,
karma veriyle eğitim ve bu ek kaynağın yararını gösteren bir deney yapılmadı.
Bu belge kaynak seçimini kaydeder; tamamlanmış veri veya bonus iddiası değildir.

## Kaynak kimliği ve sürüm

[Üreticilerin CSS10 deposu](https://github.com/Kyubyong/css10), Greek satırında
**04:08:14** süreyi, **Rapunzelina** adlı tek okuyucuyu ve
*Παραμύθι χωρίς όνομα / Tale Without Name* sesli kitabını bildirir. Depodaki
Greek veri bağlantısı aşağıdaki Kaggle kaydına gider.

| Alan | Kayıt |
|---|---|
| Veri adı | CSS10 Greek — Greek Single Speaker Speech Dataset |
| Katalog referansı | `bryanpark/greek-single-speaker-speech-dataset` |
| Kaggle veri kimliği | `30096` |
| Planlanan sabit sürüm | Kaggle `currentVersionNumber = 1` |
| Katalog boyut alanı | `totalBytes = 1314687746` |
| Katalog görünürlüğü | `isPrivate = false` |
| Katalog lisans alanı | `licenseName = CC0: Public Domain` |

Bu alanlar 17 Kasım 2026 tarihinde okunan
[resmî Kaggle API kaydından](https://www.kaggle.com/api/v1/datasets/view/bryanpark/greek-single-speaker-speech-dataset)
gelir. [Kaggle veri sayfası](https://www.kaggle.com/datasets/bryanpark/greek-single-speaker-speech-dataset)
açıklaması `transcript.txt` satır düzenini
`ses_yolu|özgün_metin|normalize_metin|süre` olarak verir. Buradaki **v1**, Kaggle
kaydı sürümüdür; CSS10 kod deposunun ayrıca doğrulanmış bir sürüm etiketi
olduğu anlamına gelmez. Veri edinilirken indirilen arşivin sürümü, dosya listesi
ve hash değerleri kaydedilecek; henüz arşiv içeriği yerelde doğrulanmadı.

## Referans metinlerin kökeni

[CSS10 makalesi §3.3](https://arxiv.org/html/1903.11269#S3.SS3), bölünmüş
ses parçalarıyla kitap metninin uzmanlar tarafından eşleştirildiğini açıklar.
§3.3.1'de ayrıca kısaltmaların ve sayıların yazıyla açılması gibi metin
normalizasyonları anlatılır. Bu kaynak, ASR modelinin ürettiği doğrulanmamış
transkriptlere dayanmıyor. Buna karşılık üreticinin insan eşleştirmesi, bizim
her parçayı yeniden dinleyip hatasız bulduğumuz anlamına gelmez. Özgün ve
üretici tarafından normalize edilmiş metin alanları korunacak; üreticinin daha
önce yaptığı işlemler kendi temizleme katkımız olarak sayılmayacak.

## Lisans kaydı

Kaggle katalog metası veri için **CC0: Public Domain** bildirir. CSS10
[kod deposunun LICENSE dosyası](https://github.com/Kyubyong/css10/blob/master/LICENSE)
ise **Apache-2.0** içerir; kod deposundaki bu lisans seslerin lisansı yerine
yazılmayacak. [LibriVox'un kaynak eser sayfası](https://librivox.org/paramythi-horis-onoma-by-penelope-delta/)
kayıtları ABD'de public domain olarak tanımlar ve ABD dışı statünün kontrol
edilmesini ister. Aynı sayfadaki **4:12:45**, tam sesli kitabın süresidir;
CSS10'un bildirdiği **4:08:14** ile karıştırılmayacak. Veri edinme ve yayımlama
aşamasında teslim edilen lisans/atıf dosyaları da kayda eklenecek. Projemizin
kod lisansı iki korpusun lisanslarının yerine geçmez; HParl'in CC BY-NC 4.0
koşulları kendi verileri üzerinde korunur.

## Sürelerin doğru yorumu

HParl'in [kaynak makalesindeki](https://arxiv.org/html/2301.00304) split süreleri
ile CSS10'un üretici bildiriminden hesaplanan **planlama toplamları**:

| Kaynak / aday bölüm | Bildirilen süre |
|---|---:|
| HParl train | 99:31:41 |
| CSS10 Greek, yalnız train adayı | 04:08:14 |
| Birleşik ham eğitim adayı | **103:39:55** |
| HParl dev | 09:03:33 |
| HParl test | 11:12:28 |
| HParl toplam | 119:47:42 |
| İki kaynağın toplamı | **123:55:56 ≈ 123.93 saat** |

Bu süreler yerel dosyalar üzerinden ölçülmüş süreler veya temizleme sonrası
kullanılacak saatler değildir. İki kaynağın toplamı eğitim saati olarak
yazılmayacak. Kullanılabilir eğitim miktarı gerçek veri denetimi ve donanım,
süre, maliyet ölçümlerinden sonra raporlanacak. Hocanın sabit bir minimum saat
şartı yok; istatistiksel olarak anlamlı ve uygulanabilir en büyük veriyi
kullanma gerekçesi ayrıca gösterilecek.

## Deneyde kullanım ve sınırlama

CSS10 Greek tek konuşmacılı sesli kitap okumasıdır. Parlamento konuşmalarının
konuşmacı, söyleyiş ve kelime çeşitliliğini temsil etmez; alan odaklı test
verisi olarak sunulmayacak. Ek kaynağın HParl başarımını artıracağı henüz
bilinmiyor. Bu nedenle HParl-only kontrolü ve HParl + CSS10 eğitimi, aynı
başlangıç checkpoint'i ve açıkça eşlenmiş optimizasyon bütçesiyle karşılaştırılacak.
Yalnız ek veri nedeniyle daha fazla güncelleme yapan bir koşulu, veri
çeşitliliğinin etkisi olarak yorumlamayacağız. Karışım oranı, epoch, diğer
ayarlar ve son checkpoint **HParl dev** üzerinden seçilecek.

HParl'in özgün testi önce kimlikleri ve kaynak hash değerleriyle dondurulacak;
bu işlem henüz tamamlanmadı. Test sesine veya referansına temizleme, filtreleme,
ekleme/çıkarma, yeniden bölme ya da yeniden etiketleme uygulanmayacak. CSS10
HParl dev/testine eklenmeyecek. Son model ve puanlama protokolü de test
çalıştırılmadan önce ayrıca sabitlenecek; test sonucu model seçmeyecek.

İki hazır korpusu birleştirmek veya dosya biçimini çevirmek tek başına
temizleme değildir. Train verisinde gerçekten saptanan tekrarlar, bozuk sesler,
ses–metin uyumsuzlukları ve benzeri sorunlar için yapılan her işlem; kaynak
kimliği, gerekçe ve önce/sonra kayıt-süre sayılarıyla günlüğe yazılacak. Gerçek
bir düzeltme saptanmamışsa yapılmış gibi raporlanmayacak. Hocanın planı uygun
bulması, bu çalışmalar tamamlanmadan custom veri bonusunun kazanıldığı
anlamına gelmez; [hoca yanıtı](TEACHER_CLARIFICATION_TR.md) esas alınır.

## Common Voice neden seçilmedi?

Common Voice Greek insan doğrulamalı bir alternatif olsa da Mozilla'nın
[resmî açıklaması](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/)
dağıtımın yalnız Mozilla Data Collective üzerinden yapıldığını belirtir;
[Greek 26.0 kartında](https://mozilladatacollective.com/datasets/cmqiny85c00xunq07lvutf0he)
yeniden barındırma/paylaşma sınırlaması da bulunur. Bu erişim ve dağıtım
koşullarını hocanın ham/temiz verinin repoda bulunması beklentisiyle uyumlu
biçimde kesinleştirmeden, onu bu planın ikinci kaynağı olarak seçmiyoruz.
Bu tercih güncel katalog koşullarına dayanan bir proje kararıdır; geçmişte
edinilmiş bütün sürümlerin hukuki statüsü hakkında genel bir yargı değildir.
