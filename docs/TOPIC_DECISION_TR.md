# Konu kararı — 16 Kasım 2026

Çalışma konusu: **Yunanca parlamento konuşmalarının otomatik transkripsiyonu
ve alan verisiyle modelin ince ayarı**.

Soru: Standart ASR modelleri bu alanda nasıl performans gösteriyor ve aynı
başlangıç modelinin kendi parliamentary-audio fine-tuning çalışmamızdan sonraki
performansı, aynı değerlendirme kuralları altında nasıl değişiyor?

İlk veri kaynağı [HParl'in resmî kaydı](https://inventory.clarin.gr/corpus/1602).
Üretici yaklaşık 120 saat, Kaldi biçimi ve CC BY-NC 4.0 bildiriyor. Makale
transkriptlerin parlamento görevlileri tarafından üretildiğini, uzun kayıtların
otomatik hizalandığını ve tutanak/ses arasında bazı uyumsuzluklar bulunduğunu
açıklıyor. Bu nedenle referans kökeni güçlü bir başlangıç olsa da hizalamayı
hatasız veya tüm kaydı yeniden insan doğrulamasından geçmiş saymıyoruz.
[Kaynak makale](https://arxiv.org/abs/2301.00304).

Dil seçimi gerekçeleri:

- Kullanıcının güncel tablosunda Macarca/VoxPopuli ve İzlandaca/Samrómur dolu;
  bu diller elendi. VoxPopuli'nin başka bir dilini de veri çakışması riskinden
  dolayı başlangıç kaynağı seçmiyoruz.
- Yunanca, Qwen3-ASR ile Canary/Parakeet'in açıklanan dil listelerinde mevcut.
  Whisper ve Greek ASR head'li wav2vec2 için de somut aday var. Dil desteği,
  bu makinelerde çalıştırmanın doğrulandığı anlamına gelmez.
- Parlamento/kanun yapımı bağlamı, hocanın alan odaklı ASR bonusu için
  gerekçelendirilebilir bir aday. Makale Yunancayı düşük kaynaklı ASR bağlamında
  inceliyor; hocanın bonus değerlendirmesi ve puan miktarı garanti değil.

Diğer adaylarda görülen engeller: İzlandaca dolu; Romence ROMPAR'da public veri
lisansı doğrulanamadı ve duyulmayan kelime sonları bağlamdan tamamlanıyor;
CDP-eval erişim/ND lisans ve yalnız test verisi sorunu taşıyor. Bazı Estonca
modeller önerilen parlament verisini zaten eğitimde kullanmış. Baskça/Galce
güçlü düşük kaynak adayları ama üç büyük baseline ailesinin dil desteği daha
sınırlı. Bu nedenlerle ilk uygulama yönü Yunanca olarak belirlendi.

Eleme kaynakları: [ROMPAR makalesi](https://arxiv.org/html/2606.15984v1),
[ROMPAR veri kartı](https://huggingface.co/datasets/avramandrei/rompar),
[SpeeD/CDP-eval](https://speed.pub.ro/downloads/speech-datasets/),
[TalTech veri kaynağı](https://cs.taltech.ee/staff/tanel.alumae/data/est-pub-asr-data/).

Hazır HParl'i indirmek kendi özgün korpusumuzu toplamak anlamına gelmez.
Gerçek temizleme katkısını veriyi inceledikten sonra ölçüp raporlayacağız;
hazır veriyle otomatik tam Data novelty puanı beklemiyoruz.

Sınıf tablosuna girilecek taslak:

| Groups | Language (Dialect) | dataset | audio duration | notes from professor |
|---|---|---|---|---|
| Anonymous | Greek / Modern Greek (el) | HParl + CSS10 Greek (Kaggle v1) | ~124 h available (~120 h HParl + 4.14 h CSS10 Greek); final used hours after cleaning/profiling | |

Bu satır henüz sınıf tablosuna yazılmış sayılmıyor. Dil/veri rezervasyonu ve
teslim tarihi netleşmeli; tam eğitim bütçesi donanım ölçümleriyle belirlenmeli.
`notes from professor` alanı hocanın notları için boş bırakılır. Süre alanındaki
`available`, kaynakların bildirdiği havuzdur; `used` veya temizlenmiş saat
iddiası değildir. Kullanıcı bu aşamadan sonra başlamamızı istediği için yeni
uygulama, tam veri indirme ve eğitim bekler.

**17 Kasım'da paylaşılan hoca yanıtıyla güncelleme:** anlamlı temizlemeyle iki
korpustan özel benchmark oluşturma planı ve ek kaynağı eğitimde kullanıp
parlamento testini koruma yaklaşımı uygun bulundu. Test önce dondurulacak ve
değiştirilmeyecek; train/dev'de işlemler yapılabilir. Sabit minimum saat yok;
istatistiksel olarak anlamlı, donanım/maliyet sınırında kullanılabilecek en
büyük veri bekleniyor. Bu yanıtın kendisi ikinci korpusu belirlemiyor veya
bonusun kazanıldığı anlamına gelmiyor. [Ayrıntılı kayıt](TEACHER_CLARIFICATION_TR.md).

**17 Kasım görev ve kaynak planı:** ikinci kaynak CSS10 Greek seçildi.
Üreticinin tablosu 04:08:14, tek konuşmacı ve LibriVox sesli kitap kaynağı
bildiriyor; resmî Kaggle dağıtım metadata'sı v1/CC0 olarak listeliyor.
HParl'in 119:47:42 toplamıyla kaynak havuzu 123:55:56 ediyor. HParl eğitim
bölümü + CSS10 ham eğitim adayı 103:39:55; dev/test bu eğitim hesabına girmez.
Yerel temizlenmiş ve kullanılan süre henüz yok. CSS10 eğitimde kalacak;
parlamento dev/test'i ek kaynakla karışmayacak. Kaynak seçimi ve tek konuşmacı/
alan farkı sınırlamaları [SECONDARY_SOURCE_TR.md](SECONDARY_SOURCE_TR.md),
beş kişilik sahiplik [TEAM.md](TEAM.md) içinde kayıtlıdır.
