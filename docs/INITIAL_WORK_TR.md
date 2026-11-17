# Başlangıç durumu — 17 Kasım 2026

Çalışma konusu: **Yunanca parlamento konuşmalarında ASR ve alan verisiyle
fine-tuning**. Dil/dataset rezervasyonu henüz sınıf tablosuna işlenmiş sayılmıyor.

17 Kasım plan güncellemesi: beş kişilik görevler [TEAM.md](TEAM.md) içinde
atandı. İkinci eğitim kaynağı CSS10 Greek v1 seçildi; üreticinin bildirdiği
4:08:14 ile HParl toplamı 123:55:56 kaynak havuzu ediyor. Bu, yerelde ölçülmüş
veya temizlenip kullanılan süre değildir. [Kaynak incelemesi](SECONDARY_SOURCE_TR.md).
Kullanıcının isteğiyle bu aşama görev/sınıf tablosu planlamasıyla biter; yeni
uygulama, tam veri indirme veya eğitim bu güncellemenin parçası değildir.

Tamamlanan somut işler:

- HParl'in resmî katalog metadata'sı ve özgün ZIP dizini okundu: arşivde
  92.133 WAV var. Tam arşiv indirilmedi.
- Seed 42 ile yalnız özgün eğitim bölümünden seçilen üç WAV ve insan kaynaklı
  özgün transkript satırları indirildi. Toplam 6,61 saniye; üçü de 16 kHz,
  mono PCM16. Kaynak seçimi ve dosya hash'leri veriyle birlikte saklandı.
- WAV SHA-256/CRC32, başlık/süre, kaynak ID, özgün transkript ve konuşmacı
  eşleşmesi kontrol edildi. Dosyalar ve referanslar değiştirilmedi.
- Üçüncü referansta `<spoken_noise>` bulundu. Bu etiketi sıradan sözcük gibi
  puanlamak yanıltıcı olabileceğinden, denetim bunu inceleme gerektiren veri
  olarak işaretliyor. Henüz temizleme veya kesin etiket politikası uygulanmadı.
- Corpus WER/CER, tam tahmin-ID kümesi, girdi/kod hash'leri ve açık test
  kullanım argümanını içeren ortak değerlendirme başlangıcı yazıldı.
- 15 veri/değerlendirme testi geçti. 500 deterministik kurgusal örnek korpusta
  WER/CER, JiWER 4.0.0 ile aynı çıktı. Bunlar model başarı sonuçları değil.
- Mevcut WSL/ROCm ortamında RX 7800 XT üzerinde forward/backward ve
  revision'ı sabit Whisper-small'ın üç eğitim kaydındaki çıkarımı çalıştı.
  Yunanca çıktılar ve gerçek koşum kaydı `results/runtime_smoke/` altında.
  Bunlar benchmark, model seçimi veya fine-tuning kazanımı olarak sunulmuyor.

Yeniden çalıştırma kodu: `scripts/fetch_train_pilot.py`,
`scripts/audit_train_pilot.py`, `scripts/verify_scoring.py`,
`scripts/smoke_whisper_train.py`. Son scriptin mevcut yerel GPU ortamını
kullanması, tam bağımsız eğitim ortamının kurulmuş olduğu anlamına gelmez.

Sıradaki işler:

1. Sınıf tablosunda Anonymous için Greek/el + HParl rezervasyonunu doğrulamak;
   9. hafta slaytları ve son tarihi netleştirmek. Hocanın yeni yanıtına göre
   sabit minimum saat yok; istatistiksel yeterlilik ve donanım bütçesi esas.
2. Eğitim tarafında daha geniş, kaynak/oturum bilgisi korunmuş veri denetimi;
   gerçek sorunları saymak ve `<spoken_noise>` gibi işaretler için açık politika.
3. Tam veri bütçesi, ham/temiz verinin public depoda teslim biçimi ve bağımsız
   ASR eğitim ortamını hazırlamak.
4. Bütün uygulanabilir standart baseline'ları tamamlarken hazır sürümleri
   kendi fine-tuning sonuçlarımızdan ayrı tutmak; ayarları development ile seçmek.
5. Resmî testten önce son protokolü commit etmek; aynı kohort ve scoring ile
   nihai karşılaştırmayı üretmek.

Yeni hoca yanıtı iki-korpus planını uygun buldu. Train/dev temizleme ve
karışık-kaynak deneylerinden önce özgün test verisinin sabitlenmesi ayrı bir
ön koşuldur; bu işlem henüz tamamlanmadı. [Karar kaydı](TEACHER_CLARIFICATION_TR.md).

Mevcut üç klip, final proje veri miktarı veya korpus kalitesinin temsilî
ölçümü değildir. Henüz tam baseline eğitimi, fine-tuning veya test değerlendirmesi
yapılmadı.
