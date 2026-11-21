# Serhan için son sunum teslimi

Eğitimler tamamlandı; final protokolü `0ff3196c` commit'inde, test tahmininden
önce kilitlendi. Sahiplik: sunumu ve kendi katkı dosyanı kendi GitHub hesabından
ayrı branch/PR ile teslim et. Başkasının yazdığı modeli veya merkezî koşuyu kendi
çalışman olarak yazma. Bu belge bir görev tarifidir; tamamlanmış katkı iddiası değildir.

## Kullanılacak gerçek kaynaklar

- `results/benchmark/SUMMARY.md`, `results.csv`, `summary.json`: koddan üretilen
  nihai tablo; yalnız `complete` satırlarını sonuç diye sun. `pending` satırına
  tahminî sayı yazma. Tüm 12 sistem bitince paketin durumu `complete` olur.
- `results/benchmark/figures/`: WER/CER ve hız grafikleri, PNG/SVG.
- Her sistemin `score.json`: hata sayıları ve speaker-bootstrap güven aralıkları.
- Her sistemin `error_examples.json`: aynı kayıtlar üzerinden gerçek hata
  örnekleri. Otomatik seçimi insan değerlendirmesi veya doğrulanmış yeni gold
  olarak gösterme. Gerekirse tahmini orijinal referansla yan yana göster.
- `results/PRESENTATION_BRIEF_TR.md`: güncel sonuç paketi ve sınırlamalar.
- `results/data_integrity.json`: gerçek split/konuşmacı/saat/örtüşme sayıları.
- `configs/final_test_protocol.json`: tam checkpoint adları, sürümler, ayarlar.
- `results/training/`: orijinal eğitim kayıtları. `README.md` ve
  `docs/FINAL_BENCHMARK.md`: yöntem, tekrar üretim ve sınırlamalar.

## İçerik sırası

1. **Soru:** Yunanca parlamento konuşmalarını transkribe etmek; ek edebî okuma
   verisinin aynı bütçede Whisper uyarlamasına etkisi.
2. **Veri:** HParl 30.883408 saat / 23.602 temiz eğitim kaydı; CSS10 ek 4.137269
   saat / 1.844 kayıt; karışım 35.020678 saat / 25.446 kayıt. Test 8.679 kayıt /
   11.207889 saat; dev 7.113 kayıt / 9.059189 saat. Bu sayılar manifestlerden
   üretilmiş raporda bulunur. HParl train/dev/test 208/64/82 tanımlı konuşmacı.
   CSS10 üreticiye göre tek okuyucudur; manifestte konuşmacı/oturum ID'si yoktur,
   tamamı yalnız eğitimdedir. Yeni kimlik uydurma.
3. **Gerçek temizlik:** seçili 23.615 ham eğitim kaydından dokuz aşırı karakter
   hızı ve dört yinelenen ses kaydı çıkarıldı; ham sesler de repoda. İki korpusu
   birleştirmek veya dosya biçimi değiştirmek tek başına temizlik değildir.
   CSS10 IEEE float32 doğru okunur, sesler değiştirilmez; insan dinleme kararı 0.
4. **Adil benchmark:** bütün gerekli standart aileler, toplam 10 baseline +
   iki kendi Whisper koşulumuz; aynı değişmeyen test ve tek normalizasyon.
   MFCC+GMM-HMM, plain CTC, hybrid attention, RNN-T, Greek wav2vec-CTC,
   bizim eğitilmiş HuBERT-CTC, multilingual Whisper-small, Canary v2,
   multilingual Parakeet v3, Qwen3-ASR-0.6B. Gelişmiş hazır modeli çalıştırmak
   model katkımız değildir. Tam checkpoint adını kullan.
5. **Katkı deneyi:** başlangıç checkpoint'i, seed 42, effective batch 16,
   öğrenme oranı, warmup ve 1.600 eğitim güncellemesi aynı; fark eğitim bileşimi.
   İkisinde de tam dev kaybıyla 500. adım ağırlığı seçilmiş. Model seçimi testten
   yapılmadı. Baseline ve kendi eğitim satırlarını ayır.
6. **Sonuçlar:** koddan üretilen WER/CER tablosu, güven aralıkları, iki Whisper
   koşulu arasındaki eşleştirilmiş fark ve gerçek örnek. İyileşme yoksa açıkça
   söyle. Güven aralığı sıfırı kapsıyorsa kesin üstünlük iddiası kurma.
7. **Sınırlamalar:** resmî split oturuma göre ayrıdır, konuşmacıya göre ayrık
   değildir. 49 test konuşmacısı eğitimde de var; 4.782 test kaydı bu seslere
   ait, oturum örtüşmesi 0. Literal metin örtüşmesi 3; final normalizasyonda 9
   test cümlesi. Bunu yeni konuşmacı başarısı diye sunma. Tek seed, sınırlı
   eğitim, farklı CPU/GPU batch'leri ve edebî/parlamento alan farkını belirt.
8. **Tekrar üretim:** public repo; gerçek ham/temiz veri ve seçili ağırlıklar
   LFS'de; sabit sürümler, script, protokol, çıktılar. Hocanın custom/domain
   boyutlarını uygun bulması tamamlanmış +5 puan garantisi değildir.

Sunum süresini dersin gerçek kuralına göre ayarla; bu sırayı gereksiz metinle
doldurma. Sonuç sayılarını dev/pilot sonuçlarıyla değiştirme. Mevcut test
çalışmasını yeniden çalıştırma, ayar değiştirme veya testten model seçme.

## Kendi teslimin

Sunum kaynağını ve dışa aktarılmış dosyayı `presentation/` altında tut;
eklenen bu kök klasörü README'de açıkla. Sunum dosyasında sonuçların hangi
nihai commit'ten alındığını belirt. Kendi yaptığın sunum/analizi, gerçek
commit'lerini ve kontrollerini şu dosyada kendi hesabından yaz:

`contributions/OZAN_SERHAN_DENIZHAN_220911758_serhandenizhan.md`

Branch: `feature/p3-presentation` veya kendi açık branch'in. `main` güncel
halinden çalış; başka üyelerin kodunu/kayıtlarını silme. PR açıklamasında
eklenen dosyaları, sonuç paketi commit'ini ve yaptığın gözden geçirmeyi yaz.
README'deki katkı envanterinde yalnız gerçekten tamamlanan durumunu güncelle.
