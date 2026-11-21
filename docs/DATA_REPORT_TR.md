# Kullanılan veri ve gerçek teslim raporu

Bu rapor `python scripts/report_benchmark_data.py` ile gerçek seçili manifestler ve temizleme günlüklerinden üretilir; sayıları elle değiştirmeyin.
Kapsam: eğitimde gerçekten kullanılan seçili HParl, eğitim-only CSS10 ve değişmeyen resmî dev/test. Tam kaynak kataloğunun kullanılmayan ~99 saatlik train envanteri bu teslimin ses kapsamı değildir.

| Kohort | Kayıt | Kodlanmış konuşmacı | Oturum/kayıt ID | Saat |
|---|---:|---:|---:|---:|
| raw_hparl_train | 23615 | 208 | 18 | 30.896592 |
| hparl_train | 23602 | 208 | 18 | 30.883408 |
| mixed_train | 25446 | 208 | 18 | 35.020678 |
| css10_train | 1844 | 0 | 0 | 4.137269 |
| dev | 7113 | 64 | 3 | 9.059189 |
| test | 8679 | 82 | 3 | 11.207889 |

Producer single-reader corpus; original manifests encode no speaker/session IDs. All CSS10 is training-only; no invented IDs or cross-source identity claims.
CSS10 üzerinde yeni train/dev/test bölmesi yapılmadı; kaynağın tamamı yalnız train kullanılır. HParl için üreticinin resmî oturum bölmesi korunur.

## Gerçek temizlik ve izlenebilirlik

Seçili ham HParl ile temiz HParl arasındaki 13 çıkarılan ID, kayıttaki drop kararlarıyla birebir eşleşir.

| Çıkarma kuralı | Kayıt |
|---|---:|
| cps_extreme | 9 |
| duplicate_audio | 4 |

Metin politikası günlük girdisi: 15370. Bunlar annotation tag/karakter/harf/boşluk uyumlamasını kaydeder; formatlama tek başına gerçek veri temizliği veya yeni gold anotasyon sayılmaz.
İnsan inceleme kararı: 0; inceleme kuyruğu: 176 kayıt. İncelenmemiş flag'ler doğrulanmış etiket hatası diye sunulmaz.
Orijinal WAV baytları korunur. Temiz veri, aynı gerçek WAV dosyalarını seçen temiz manifest ve referanslardan oluşur. CSS10 float32 doğru format tag ile okunur; gain/ses normalizasyonu yoktur.
Tam nedenler, before/after metin/hash ve kanıt: `data/clean/selected_train_hparl_30.5h_seed42/cleaning_log.jsonl`. Günlük SHA-256: `986f3135156b397fb9c04ec0797095462dc39fb1f0270ad01ffceae25411caa0`.

## Resmî split ve örtüşmeler

| Train koşulu → test | Ortak konuşmacı | Bu seslere ait test kaydı | Ortak oturum | Ortak ses hash | Literal cümle | Final normalize cümle |
|---|---:|---:|---:|---:|---:|---:|
| hparl_train | 49 | 4782 | 0 | 0 | 3 | 9 |
| mixed_train | 49 | 4782 | 0 | 0 | 3 | 9 |

Bu benchmark speaker-disjoint değildir ve yeni konuşmacılara genelleme diye sunulamaz. Dokümante resmî split istisnası kullanılır; yeni rastgele klip bölmesi yapılmamıştır. Literal ve normalize eşleşme ayrı ölçümlerdir.
Test referans/sesleri değiştirilmez, zor kayıt çıkarılmaz. Her sistem aynı orijinal test IDsini ve tek dondurulmuş scoring politikasını kullanır.

## Depo teslimi ve doğrulama

Seçili ham HParl WAVlarının tamamı (çıkarılanlar dahil), dev/test/CSS10 WAVları Git LFS ile depodadır. Ham metaveri ve ham/temiz manifest/referanslar normal Git içindedir. Drive veya indirme betiği veri teslimi yerine kullanılmaz.
Özgün ses + seçili inference ağırlıkları: 8.954381 GiB; 10 GiB depolama sınırı içinde. Aylık indirme kotası ayrı bir hesap sınırıdır; bu rapor kalan bandwidth'i ölçmez.
Temiz public clone, mevcut yüklenmiş LFS nesne önbelleğiyle gerçek ses üzerinde doğrulandı (bant genişliği için yeniden indirilmedi). Kanıt: `docs/reviews/final_benchmark_2026-11-21/clean_clone_verification.json`. Anonim örnek nesne erişimi: aynı klasörde `public_delivery.json`.
Kaynak kimliği/lisans: `DATA_LICENSE.md`, `configs/data_source.json`. Dondurulmuş seçim: `configs/training_data_selection.json`. Makine-okunur tüm sayılar ve manifest hashleri: `results/data_integrity.json`.

## Tekrar üretim

```bash
python -m greek_asr.data verify-source-metadata
python -m greek_asr.data audit --selection configs/training_data_selection.json
python -m greek_asr.data clean --selection configs/training_data_selection.json
python -m greek_asr.data verify --selection configs/training_data_selection.json
python -m greek_asr.data verify-freeze
python scripts/report_benchmark_data.py
```

Tarihsel `python -m greek_asr.data report` komutu tam-kaynak/envanter raporudur ve bu dosyaya yazar; mevcut benchmark için son komut olarak yukarıdaki seçili-kohort rapor üreticisini çalıştırın. Eski envanter boyutlarını teslim edilmiş eğitim saati olarak kullanmayın.
