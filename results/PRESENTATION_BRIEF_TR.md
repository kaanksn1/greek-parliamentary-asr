# Serhan için gerçek sonuç paketi

Durum: 12/12 sistem tamamlandı; son sonuçlar.

Tablolar: results/benchmark/results.csv ve SUMMARY.md. Grafikler: results/benchmark/figures/.
Her modelin score.json dosyasında ham hata sayıları ve speaker-bootstrap aralıkları; error_examples.json dosyasında otomatik seçilmiş gerçek hata örnekleri var.
Veri: 30.883408 saat HParl; ek 4.137269 saat CSS10; karışım 35.020678 saat. Test: 8,679 değişmeyen kayıt.
Baseline ve katkı satırlarını ayır. Qwen/Canary/Parakeet çalıştırmak kendi model katkımız değildir.
İki Whisper koşulunda fark yalnız eğitim bileşimi; aynı 1,600 güncelleme/effective batch 16. İkisinde de tam dev kaybıyla seçilen ağırlık 500. adımdan. İyileşme iddiasını gerçek delta ve güven aralığına göre kur.
Resmî split konuşmacı ayrık değildir; bunu sunumda belirt. Sınırlamalar ve başarısız/çok zayıf modelleri gizleme.
Kendi sunum/analiz katkını contributions/OZAN_SERHAN_DENIZHAN_220911758_serhandenizhan.md içinde kendi hesabınla kaydet; kendi branch/PR üzerinden sunumu ekle.

## Tamamlanan katkı deneyinin gerçek sonuçları

HParl-only WER: %23.32; HParl+CSS10 WER: %22.47.
Karışım eksi HParl farkı: -0.85 yüzde puan; eşleştirilmiş %95 konuşmacı-bootstrap aralığı [-1.93, -0.06].
Bu, tek eğitim seed’i ve bu resmî test kohortu için gözlenen farktır; farklı seed/konuşmacı/alanlarda garanti değildir.
Çok zayıf CTC sonuçlarını ve Whisper baseline hata örneklerini gizleme; bunlar da aynı testin gerçek sonuçlarıdır.

Kayıtlı uzatılmış eğitim — hubert_ctc_extended: WER %99.89 → %30.78, CER %82.18 → %9.54 (seçilen adım 8,000); ayrı satırdır, orijinal satır korunur.
Kayıtlı uzatılmış eğitim — plain_ctc_extended: WER %99.95 → %100.69, CER %91.37 → %78.74 (seçilen adım 15,000); ayrı satırdır, orijinal satır korunur.