# HParl eğitimlerini geciktirmeden başlatma

> **21 November completion note:** all selected data delivery and seven training allocations are complete; all twelve final systems were evaluated. Current evidence: [training](../results/training/SUMMARY.md), [benchmark](../results/benchmark/SUMMARY.md), [reproduction guide](FINAL_BENCHMARK.md). Earlier pilot-stage pending statements below describe their dated review stage, not current delivery.

20 Kasım'da kullanıcı Samed'i beklemeden eğitime geçmemizi istedi. Kaan aynı
kilitli 18 HParl oturumunun 23.615 ham WAV'ını özgün CLARIN ZIP'inden sınırlı
HTTP aralıklarıyla aldı; her dosya kaynak CRC/byte/SHA-256 kontrolünden geçti.
Dil, split, referans, ham ses veya seçilen oturum listesi değişmez.

`scripts/verify_hparl_training.py`, geçici bir hard-link staging kökünde
Samed'in **değişmeyen** `data.audit.run_audit` ve `data.clean.run_clean`
uygulamalarını yalnız teslim edilen ham HParl train/dev üzerinde yeniden
çalıştırır. Donmuş test sadece bütünlük/leakage kontrolü için hashlenir.
Üretilen temiz train ve dev ses/ref manifestleri kilitli sürümle byte-byte
aynı olmalı. Fark varsa eğitim başlamaz. Kaynak depodaki test dosyaları
yazılmak üzere açılmaz; staging orijinal WAV'lara hard link verir.

Başarılı rapor `configs/hparl_training_acceptance.json` olarak commit edilir.
Bu kabul HParl-only koşular içindir; CSS10 denetimini kabul etmiş sayılmaz.
Kabul `d98abe995f85ff634704ec9f60b26ca671a21a29` commitinde sabitlendi ve
gerçek eğitimler 20 Kasım 03:06 Türkiye saatinde başlatıldı. Çalışan native
checkout bu committe kalır; paralel Git LFS teslim/doküman commitleri çalışan
kod veya ayarları değiştirmez. Bu saat teslim saati veya bitiş tahmini değildir.
Ham seslerin Git LFS ile depoya teslimi eğitimle paralel tamamlanabilir;
son teslimde kullanılan bütün ham/temiz veriler gerçekten depoda bulunmalı.

Komutlar, native Linux checkout ve sabitlenmiş Python 3.12 ROCm ortamındadır:

```bash
python scripts/acquire_selected_hparl.py --output outputs/hparl_training/acquisition.json
python scripts/verify_hparl_training.py --staging outputs/hparl_training/repro --output outputs/hparl_training/acceptance.json
# İncelenen acceptance raporu configs/hparl_training_acceptance.json olarak commit edildikten sonra:
HSA_ENABLE_DXG_DETECTION=1 MIOPEN_FIND_MODE=FAST python scripts/run_primary_training_queue.py --output outputs/hparl_primary_20261120
```

`run_primary_training_queue.py`, GMM-HMM'yi CPU'da iki thread ile; plain CTC,
HParl-only Whisper, attention, HuBERT ve RNN-T'yi GPU'da sırayla çalıştırır.
Başka bir eğitim koşusunun çıktısını ezmez. Her işte kabul/veri/config ve kod
kimliği, PID, gerçek optimizer adımı, süre ve hata kaydı tutulur.
`run_training_job.py` mimariyi, loss'u veya optimizer kuralını değiştirmez;
mevcut gradient clipping'e nonfinite hata kontrolü ve gerçek adım sayacı ekler.
GMM feature çıkarım ilerlemesini de loglar. Rastgele seed ve ölçülmüş bütçeler
configlerden gelir. Yaklaşık epoch hedefleri yakınsama iddiası değildir.

Eğitim süresince Windows'taki WSL sürecinin uyumaması için geçici bir
system-awake isteği kullanıldı; bu yerel yardımcı betik model veya veri işi
olmadığı için depodan çıkarıldı. Eğitim kayıtları ve sonuçlar etkilenmez.

Samed'in kalan görevi CSS10 float-aware veri probe/audit/clean düzeltmesi,
karışımın doğrulanması ve ilgili raporlardır. İkinci Whisper koşulu ancak bu
ayrı kabulden sonra başlar; optimizasyon bütçesi ilk koşulla aynı tutulur.
Ortak nihai scoring protokolü ve son test çıkarımı sonraki ayrı aşamadır.
