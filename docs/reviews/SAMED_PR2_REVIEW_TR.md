# Samed PR #2 — veri teslimi incelemesi

İncelenen sürüm: **`4e4a80d638a5e9fb177ead6d75f46112ef4e165e`**, branch
`feature/p3-data`, [PR #2](https://github.com/kaanksn1/greek-parliamentary-asr/pull/2).
İnceleme: 18 Kasım 2026. PR bu incelemede main'e birleştirilmedi.
Yeni iki hoca maili [ayrı gereklilik notunda](../TEACHER_UPDATES_2026-11-18_TR.md).

## Doğrulanan işler

- Commit'ler Samed'in GitHub hesabına bağlı. Kişisel katkı dosyası doğru adla var.
- Python 3.12.3 / NumPy 1.26.4 ortamında **73/73 test geçti**. Yeni veri
  testleri NumPy gerektiriyor; eski stdlib-only `python -S` komutu tüm 73 test
  için uygun değildir. Model paketi kurulması gerekmiyor.
- `greek_asr.data.report.build_report` ile rapor yeniden üretildi; commit
  edilmiş `docs/DATA_REPORT_TR.md` ile bayt bayt aynı çıktı.
- Test dondurma commit'i `d28c97f` temizlik commit'lerinden önce. Test
  manifesti, referansları ve freeze dosyası o commit'ten beri değişmemiş.
  Güncel manifest/referans hash'leri freeze içindekilerle tutarlı.
- Önceden gerçekten edinilmiş üç pilotun ses hash'leri ve özgün referansları
  Samed'in ham train envanteriyle eşleşiyor.
- HParl-only ve HParl + CSS10 ayrı koşullar. CSS10 yalnız eğitimde; dev/test
  ortak. Ham/temiz sayıları ve temizlik günlüğü mevcut.

Sınır: tam ses ve özgün Kaldi dosyaları bu checkout'ta bulunmadığından tam
audio-hash doğrulamasını veya baştan sona gerçek veri pipeline'ını bağımsız
çalıştırdığım iddia edilmiyor. Testler küçük fixture verisi kullanıyor.

## Düzeltme 1 — P1: ses teslimi henüz yok

Raw manifestlerde HParl train/dev/test için sırasıyla 76.341 / 7.113 / 8.679,
CSS10 için 1.844 kayıt var. Bu dört envanterin işaret ettiği **93.977 ses
yolunun hiçbiri** incelenen Git checkout'ında yok. Yalnız eski üç pilot WAV
depoda duruyor; bunlar tam manifestlerin işaret ettiği dosyalar değil.
Özgün test Kaldi `text`, `utt2spk`, `wav.scp`, `spk2utt`, `spk2gender`
dosyaları da eksik. Dolayısıyla `verify-freeze` / `verify` tam teslimde
çalıştırılamıyor; yalnız metadata/hash tutarlılığı doğrulandı.

[DATA_PIPELINE_TR.md bölüm 7](https://github.com/kaanksn1/greek-parliamentary-asr/blob/4e4a80d638a5e9fb177ead6d75f46112ef4e165e/docs/DATA_PIPELINE_TR.md#L218)
train sesini `.git/info/exclude` ile dışarıda tutup GitHub Releases'e koymayı
öneriyor. Hocanın "ham ve temiz veri repoda; bağlantı veya indirme script'i
yetmez" koşulunun bu yöntemle otomatik sağlandığını varsayamayız. Gerçek
depolama/kota kararı ve dosyaların ulaşılabilirliği Serdar/Samed birlikte
tamamlanmalı; ücretli işlem yapılmamalı. Git LFS kullanılırsa pointer değil,
sunucudan inen gerçek dosyalarla temiz clone doğrulanmalı. Ses değişmediğinde
ham ve temiz manifestlerin aynı ses baytlarını paylaşması ayrıca kopya üretmeyi
gerektirmez; temiz referanslar ve lineage yine depoda kalmalı.

Kabul koşulu: kullanılan ham/temiz veri ile özgün test metadata'sı erişilebilir
olmalı; temiz clone'da tam `verify-freeze` ve `verify` başarılı olmalı.

## Düzeltme 2 — P2: birebir test cümlesi sayısı eksik

[audit.py:31](https://github.com/kaanksn1/greek-parliamentary-asr/blob/4e4a80d638a5e9fb177ead6d75f46112ef4e165e/greek_asr/data/audit.py#L31)
etiket silme, NFC, casefold ve boşluk birleştirme yapıyor;
[audit.py:286](https://github.com/kaanksn1/greek-parliamentary-asr/blob/4e4a80d638a5e9fb177ead6d75f46112ef4e165e/greek_asr/data/audit.py#L286)
sonra set kesişiminin büyüklüğünü sayıyor. Raporun `21` değeri normalize
edilmiş **benzersiz metin** sayısıdır; Mail 1'in istediği kelimesi kelimesine
eşleşen test örneği sayısı değildir.

Hiçbir normalizasyon uygulamadan, mevcut kaynak test referanslarıyla yeniden
hesaplanan sonuç:

| Karşılaştırılan eğitim hedefi | Eşleşen test örneği | Benzersiz test metni |
|---|---:|---:|
| Ham HParl train | 14 | 14 |
| Temiz HParl train | 6 | 6 |
| Temiz HParl + CSS10 train | 6 | 6 |

Asıl eğitim koşullarının değerleri rapora kodla eklenmeli. Normalize edilmiş
örtüşme korunacaksa dönüşümleri doğru adla ayrı açıklanmalı. Test dosyasını
değiştirmek veya bu cümleleri testten silmek gerekmez.

Kabul koşulu: ham ve kullanılan temiz koşullar için literal örnek sayısı +
benzersiz metin sayısı; tekrar eden iki test örneğinin tek benzersiz metin
olabileceğini ve etiket/case farkını gösteren anlamlı veri testi.

## Düzeltme 3 — P2: konuşmacı durumu README'de açık değil

Resmî HParl oturum bölümü korunmuş. Manifestlerden hesaplanan ham durum:

| Bölüm | Konuşmacı | Örnek | Saat | Oturum |
|---|---:|---:|---:|---:|
| train | 376 | 76.341 | 99,528 | 43 |
| dev | 64 | 7.113 | 9,059 | 3 |
| test | 82 | 8.679 | 11,208 | 3 |

Train/dev ortak konuşmacı 59, train/test 76, dev/test 16. Oturum kesişimleri
üç çiftte de sıfır. Testteki **7.777 örneğin konuşmacısı train'de de bulunuyor**.
Bu değerler kayıtlı metadata'dan, sesler yeniden dinlenmeden hesaplandı.

Mail 1 **How 1**, belgelenmiş resmî bölümün kullanılmasına izin veriyor ve
README'de konuşmacıların ayrı olup olmadığının yazılmasını istiyor. Bu yüzden
burada "mutlaka testi yeniden böl" sonucu çıkarmıyoruz. Bunun yerine resmî
oturum bölümünün **speaker-disjoint olmadığını**, bu istisnayla kullanıldığını
ve sonucun yeni konuşmacıya genellemeyi ölçmediğini README açıkça söylemeli.
[report.py:149](https://github.com/kaanksn1/greek-parliamentary-asr/blob/4e4a80d638a5e9fb177ead6d75f46112ef4e165e/greek_asr/data/report.py#L149)
içindeki koşulsuz "sızıntı düzeltmesi gerektirmez" ifadesi yerine bu sınırlama
yazılmalı. Testi değiştirmeden önce yeni bir tasarımın gereği ayrıca belirlenir;
bu inceleme hiçbir yeniden bölümleme yapmadı.

Kullanılan temiz train 76.296 örnek / 99,480 saat / 376 bilinen konuşmacı;
karışık train 78.139 örnek / 103,617 saat. CSS10'un 1.843 temiz kaydının
`speaker_id` ve `source_recording_id` alanları boş. Üretici tek okuyucu ve tek
kitabı bildirdiği için bu bilginin kaynaklı grup kimliğine taşınması veya
bilinmeyen ID sayısının ayrıca yazılması gerekir; bilinmeyenleri yok sayıp
karışık train'de kesin 376 konuşmacı varmış gibi sunmayın.

## Düzeltme 4 — P2: 40 karakter/sn kesin hata kanıtı değildir

V2 politika bu eşik nedeniyle HParl'den 28, CSS10'dan 1 kayıt çıkarmış.
İnsan inceleme kararı sayısı sıfır. Bazı kayıtlar saniyeler içinde binlerce
karakter hedefi taşıyor ve ciddi hizalama sorunu adayı; buna karşılık eşiğin
hemen üstündeki tüm kayıtların yanlış olduğuna dair bağımsız kanıt yok.

Örneğin `css10_el:Paramythi_horis_onoma_0856` için 3,402 saniye, 138 boşluk
dışı karakter ve 40,5612 karakter/sn kayıtlı. Silinen toplam yedi kayıt
40–60 karakter/sn aralığında. Bu kayıtları dinlemeden "fiziksel olarak
imkânsız" veya "doğrulanmış yanlış etiket" diye nitelemek fazla kesin.
CTC'nin zaman adımı/etiket uzunluğu koşulu modele ve tokenizer'a bağlıdır;
tek bir karakter/sn eşiği her ASR mimarisinde geçersiz loss ispatı değildir.

Kural otomatik kalite sezgiseli olarak açıklanabilir. Sınır kayıtları gerçek
incelemeye alınmalı veya belirsizlikte korunmalı; kullanılan filtre ve gerekçesi
versionlanmalı. İnsan dinlemesi yapılmadıysa yapıldı diye raporlanmamalı.
Yeni karar yalnız train/CSS10'u etkiler; frozen test değişmez.

## Düzeltme 5 — P2: README çalışma durumunu ters anlatıyor

[README veri bölümü](https://github.com/kaanksn1/greek-parliamentary-asr/blob/4e4a80d638a5e9fb177ead6d75f46112ef4e165e/README.md#L89)
"full corpus has not yet been downloaded, frozen or cleaned" diyor; aynı
PR'da gerçek edinme, freeze ve temizlik raporu var. Doğru ayrım: Samed'in
makinesindeki gerçek koşum ve metadata teslimi var; repodaki tam ses teslimi
ve bizim bağımsız tam-byte doğrulamamız henüz yok. Ayrıca `docs/DATA_REPORT_TR.md`
dosya envanterinde açıklanmalı. Raporun sayıları elle düzeltilmez; üreten kod
değiştirilip rapor tekrar üretilir.

## Mail 2'nin model planına etkisi

Sahipli baseline matrisi duyurulan aileleri zaten kapsıyor. Dört owner
checkpoint'i doğru multilingual sürümler ve revision'lar ile kayıtlı;
eğitimsiz GPU smoke'ları model katkısı değildir. Adnan'ın transducer görevi
RNN-T olarak somutlanmalı, attention ailesinin hybrid CTC/attention'ı
kapsadığı yazılmalı. Greek HuBERT + ASR head desteği hâlâ teyit bekliyor;
bare encoder sonucu veya gerekçesiz atlama kabul edilmez. Dil desteği olmayan
bir checkpoint ancak model kartı bağlantılı gerekçeyle uygulanamaz yazılabilir.
Tek normalize/scoring kuralı dev ile seçilip bütün sistemlere aynı uygulanır.

## Yeniden üretme ve kapsam

Owner inceleme script'i yalnız metadata okur; Samed'in pipeline'ını veya
temizlik kararlarını onun yerine uygulamaz:

```bash
git fetch origin feature/p3-data
git worktree add --detach ../samed-review 4e4a80d638a5e9fb177ead6d75f46112ef4e165e
python scripts/review_data_submission.py --data-root ../samed-review --output outputs/samed_review.json
```

Makineyle üretilen kanıt [SAMED_4e4a80d.json](SAMED_4e4a80d.json) içinde.
Test komutu, `requirements-data.txt` kurulu Python 3.12 ortamında:
`python -m unittest discover -s tests -v` (review checkout'ından).

Bu inceleme Samed'in kaynak kodunu/kişisel katkısını değiştirmedi, yeni model
eğitmedi, testte çıkarım/puanlama yapmadı ve kimseye mesaj göndermedi.

## Samed'e verilebilecek düzeltme promptu

```text
Ben Samed Ortakaya (SmtOrtakaya), PR #2 / feature/p3-data üzerinde kendi veri
teslimimi tamamlıyorum. Mevcut freeze'i ve başka üyelerin işlerini değiştirme.
Önce docs/reviews/SAMED_PR2_REVIEW_TR.md ve iki yeni hoca maili özetini oku.

1. Aynı resmî HParl testini koru. d28c97f'teki test manifesti, referansları ve
   freeze dosyası aynen kalmalı. Yeni bir test veya clip-shuffle üretme.
2. audit/report koduna literal cümle örtüşmesi ekle: kullanılan train koşulu
   ile frozen kaynak test referansları arasında dönüşümsüz string eşitliği;
   eşleşen test örnek sayısı ve benzersiz metin sayısı ayrı. Raw=14, mevcut
   temiz HParl-only=6, mevcut mixed=6; kendi kodunla doğrula, elle tablo yazma.
   Mevcut normalize edilmiş 21 değerinin dönüşümlerini ayrı tanımla.
3. README'de resmî oturum bölümünü kullandığımızı, konuşmacıdan bağımsız
   olmadığını ve Mail 1 How 1 istisnasını açıkla. Speaker/utterance/hour
   değerlerini kullanılan splitler için raporla. CSS10'un bilinmeyen grup
   alanlarını kaynaklı tek okuyucu/kitap bilgisiyle çöz veya sayıca açıkla.
4. >40 karakter/sn kuralını kesin fiziksel hata diye sunma. Özellikle sınır
   kayıtları gerçek incelemeye al veya belirsizlikte koru; karar ve kanıtları
   kaydet. Dinlenmeyen kaydı dinlendi yazma. Politika değişirse train/CSS10
   audit/clean/report'u yeniden üret, kaynak raw ve frozen test sabit kalsın.
5. Ham/temiz ses ve özgün Kaldi metadata teslimini Serdar'la tamamla. Kota/
   yöntem doğrulanmadan ücretli işlem yapma. Yalnız indirme script'ini veya
   Release linkini hocanın repo şartını sağlamış gibi gösterme. Gerçek sesle
   temiz clone'da verify-freeze ve verify çalışsın.
6. README'deki indirilmedi/dondurulmadı/temizlenmedi ifadesini gerçek yerel
   koşum ile eksik public ses teslimini ayıracak şekilde düzelt. Veri raporunu
   elle yazma; generator'dan üret. Sadece kendi gerçek katkı dosyanı güncelle.
7. Yeni koşullar için anlamlı regression testlerini ekle; requirements-data
   kurulu ortamda tüm testleri çalıştır. Aynı branch'ten kendi hesabınla pushla
   ve PR açıklamasında tamamlananlarla bekleyenleri ayrı yaz.

Diğer üyelerin model/scoring kodunu değiştirme ve tam model koşusu çalıştırma.
Model koşuları Serdar'ın bilgisayarında, nihai scoring politikası Ozan'da.
```
