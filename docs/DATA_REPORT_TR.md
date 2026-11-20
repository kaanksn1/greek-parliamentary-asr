# Veri raporu — HParl + CSS10 Greek (Samed, veri sorumlusu)

Bu dosya `python -m greek_asr.data report` komutuyla pipeline çıktılarından üretilir; içindeki sayılar elle yazılmamıştır. Model başarımı veya bonus iddiası içermez.

**Güncel kapsam:** sahibin kilitli eğitim seçimi `configs/training_data_selection.json` (`6033ef269a32a811…`, 18 tam resmî train oturumu, seed 42), değişmeyen resmî dev, dondurulmuş resmî test ve yalnız eğitimde CSS10. Tam ~99 saatlik HParl train manifesti (`data/manifests/raw/hparl_train.jsonl`, `data/manifests/train.jsonl`) yalnız **kaynak aday envanteridir**: sesleri repoda teslim edilmedi ve eğitimde kullanılmaz. Denetim/temizlik/karışım/doğrulama zinciri `--selection` ile yalnız teslim edilen seçili ham sesleri okur.

## 1. Kaynaklar, sürümler ve lisanslar

| Kaynak | Sürüm | Lisans | Arşiv SHA-256 | Bayt | Edinme (UTC) |
|---|---|---|---|---|---|
| HParl | 1.0.0 | CC-BY-NC-4.0 | `676ba16a70c44001d364ba190610c477e900df9356cb86f081df6da19b8b1108` | 11887118613 | 2026-11-18T09:13:45+00:00 |
| CSS10 Greek (Kaggle) | 1 | CC0: Public Domain | `cb2a923fa2d5092cab7e2509be4724b38a86fb5b8a18a519299865bc91411f17` | 1287643166 | 2026-11-18T09:32:54+00:00 |

HParl transkriptleri üreticinin insan transkriptleridir; CSS10 metinleri üreticinin kitap metniyle insan eliyle eşleştirmesidir. Bu projede yeni insan anotasyonu yapılmadı (aşağıdaki insan inceleme kararları hariç). Lisans ayrıntısı: `DATA_LICENSE.md`.

## 2. Testin erken dondurulması

| Alan | Değer |
|---|---|
| Dondurma zamanı (UTC) | 2026-11-18T09:22:24+00:00 |
| Kayıt sayısı | 8679 |
| Toplam süre | 11.21 sa (11:12:28) |
| Konuşmacı / oturum | 82 / 3 |
| ID listesi SHA-256 | `bf1fc6b715c5613c737b8388ebc9468030250954ba22d45170c67b5c67f76ad5` |
| `test.jsonl` SHA-256 | `b0842bfbcbffa173585adb2debf35fb359748e0bd693da5a8dfabdb69de6f65a` |
| `references/test.jsonl` SHA-256 | `f3d6b377959b8a8183e4b45d1ee1e2db693f5f53fb1467b194f7c258bae2e5f9` |
| `test.freeze.json` SHA-256 | `9a680104427fa0e202229167f5eb09bf61c713ebce29c9e369413a9f230675c3` |
| Temizlik öncesi/sonrası doğrulama | ok / ok |
| Kaynakta not edilen (değiştirilmeyen) durumlar | 0 |

Test sesleri ve özgün transkript satırları değiştirilmedi; kayıt eklenmedi/çıkarılmadı, yeniden bölünmedi, normalize edilmedi. CSS10 teste girmedi. `python -m greek_asr.data verify-freeze` her hash'i yeniden hesaplar; temizlik komutu bu kontrolü başta ve sonda çalıştırır.

## 3. Ham veri envanteri (yerel ölçüm)

| Bölüm | Transkript satırı | Sesli kayıt | Ölçülen süre | Üretici bildirimi | Fark | Konuşmacı | Oturum | Kaynak sorunları |
|---|---|---|---|---|---|---|---|---|
| train | 76341 | 76341 | 99.53 sa (99:31:41) | 99:31:41 | +0 sn | 376 | 43 | yok |
| dev | 7113 | 7113 | 9.06 sa (9:03:33) | 09:03:33 | +0 sn | 64 | 3 | yok |
| test | 8679 | 8679 | 11.21 sa (11:12:28) | 11:12:28 | +0 sn | 82 | 3 | yok |
| CSS10 | 1844 | 1844 | 4.14 sa (4:08:14) | 04:08:14 | +0 sn | 1 (üretici bildirimi) | — | yok |

Süre yöntemi: her WAV başlığındaki `num_frames / sampling_rate`. Oturum kimliği yöntemi: `derived_session_from_utterance_id(<speaker>-<session>_<start>_<end>_<n>_<m>)`. Fark sütunu yerel ölçüm ile üretici bildirimi arasındaki farktır; açıklaması aşağıdaki sorunlar ve yuvarlama ile birlikte okunmalıdır.

CSS10 örnekleme hızları: {'22050': 1844}; arşivde transkriptte geçmeyen 1844 WAV dosyası var (kullanılmadı, silinmedi).

## 4. Denetim bulguları (seçili train, dev, CSS10)

Denetim şeması `p3-data-audit-v2`, politika `p3-data-clean-v3`. Ses okuma: greek_asr.audio RIFF reader; samples decoded by format tag: PCM16/PCM32 integer full-scale, IEEE float32 as stored (no gain, no peak normalization); NaN/Inf, truncated WAV and LFS pointers fail. Ortam: Python 3.12.10, NumPy 1.26.4, Windows. Test sesleri yalnız sızıntı kontrolü için hash'lendi; testten hiçbir istatistik eşik seçiminde kullanılmadı.

Bayraklar adaydır; tek başına hatalı veri kanıtı değildir. Bir kayıt birden çok bayrak taşıyabilir.

| Ölçü / bayrak | CSS10 Greek (yalnız train) | HParl dev | HParl train (seçili 18 oturum) |
|---|---|---|---|
| Kayıt / saat | 1844 / 4.137 | 7113 / 9.059 | 23615 / 30.897 |
| Kodlama | ieee_float32le: 1844 | pcm_s16le: 7113 | pcm_s16le: 23615 |
| RMS dBFS p05 / p50 / p95 | -29.199 / -27.67 / -26.395 | -26.809 / -22.561 / -18.539 | -30.964 / -23.945 / -19.54 |
| Tepe dBFS p50 / max | -7.84 / -4.694 | -6.983 / -2 | -8.073 / -2.616 |
| `annotation_tag` | 0 | 4267 (312.5 dk) | 13536 (1002.0 dk) |
| `cps_boundary` | 1 (0.1 dk) | 0 | 1 (0.1 dk) |
| `cps_extreme` | 0 | 0 | 9 (0.7 dk) |
| `cps_high` | 1 (0.1 dk) | 10 (0.4 dk) | 16 (0.9 dk) |
| `cps_low` | 22 (3.5 dk) | 0 | 0 |
| `duplicate_audio` | 0 | 0 | 4 (0.1 dk) |
| `duplicate_suspect` | 0 | 1 (0.0 dk) | 3 (0.1 dk) |
| `long_edge_silence` | 1 (0.2 dk) | 0 | 2 (0.2 dk) |
| `non_greek_characters` | 1839 (247.9 dk) | 31 (2.8 dk) | 98 (10.3 dk) |
| `silence_heavy` | 0 | 0 | 1 (0.0 dk) |

Aynı baytlı ses grupları: 4 grup / 8 kayıt; aynı PCM içerikli gruplar: 4 (seçili train + dev + CSS10 + test hash'leri). SHA-256 birebir tekrarları ve yalnız başlığı farklı kopyaları yakalar; yeniden kodlanmış veya kesilmiş tekrarlar yalnız `duplicate_suspect` sezgiseliyle dinlemeye bırakılır.

**Puanlama için önemli:** kaynak referanslarında `<spoken_noise>` gibi işaretler var (yalnız okundu, değiştirilmedi): dev: 4267/7113 referans (%60.0), 5235 işaret; test: 5122/8679 referans (%59.0), 6219 işaret. Train hedeflerinden çıkarıldıkları için modeller bunları üretmez; referans ve hipotezde aynı kuralla ele alınmaları ortak puanlama politikasında dev üzerinde kararlaştırılmalıdır.

## 4a. CSS10 IEEE float32 düzeltmesi: eski ve yeni denetim

1.844 CSS10 WAV'ının tamamı RIFF format tag 3 (IEEE float32), mono, 22050 Hz'dir. Önceki denetim (`p3-data-audit-v1`) 4 baytlık örnekleri integer PCM32 sayıp float baytlarını 2^31'e bölüyordu; RMS, dBFS ve bunlara bağlı bayraklar bu yüzden **yanlıştı**. Düzeltmede format tag'e göre çözülür, float değerler olduğu gibi kullanılır; kazanç veya tepe normalizasyonu yok, özgün WAV'lar değiştirilmedi.

Eski denetim: `data/audit/audit_summary.json` `8eef12d9305402c8…` — INVALID signal statistics: IEEE float32 sample bytes were read as integer PCM32 (audit schema v1). Kept only as history.

| CSS10 bayrağı / ölçü | Eski (integer yorum, geçersiz) | Yeni (float32) |
|---|---|---|
| `cps_boundary` | 1 | 1 |
| `cps_high` | 1 | 1 |
| `cps_low` | 22 | 22 |
| `long_edge_silence` | 0 | 1 |
| `non_greek_characters` | 1839 | 1839 |
| RMS dBFS min / p50 / max | -7.215 / -6.105 / -5.992 | -36.603 / -27.67 / -22.294 |
| Tepe dBFS min / p50 / max | -4.998 / -4.998 / -4.998 | -18.022 / -7.84 / -4.694 |

Kayıt bazında bayrak değişimleri (ilk birkaç ID; tam liste lineage dosyasında, ilk 50):

| Bayrak | Eski | Yeni | Eklenen örnek | Kalkan örnek |
|---|---|---|---|---|
| `long_edge_silence` | 0 | 1 | `css10_el:Paramythi_horis_onoma_1738` | — |

Temiz CSS10: 1844 ham → 1844 kayıt (4.137269 saat); çıkarma kuralları: yok. Temiz CSS10 manifesti/referansı önceki sürümle **bayt-özdeş** kaldı; yol değişmedi.

## 4b. Bölüm, konuşmacı, oturum ve test cümlesi örtüşmesi

Kullanılan bölüm **HParl 1.0.0'ın resmî oturum bazlı train/dev/test bölümüdür** ve konuşmacıdan bağımsız (speaker-disjoint) **değildir**. Hocanın izin verdiği belgelenmiş resmî bölüm istisnası (How 1) kullanıldı; dondurulmuş test yeniden bölünmedi. Sonuçlar yeni konuşmacıya genelleme (unseen-speaker) ölçümü değildir. Sayılar manifestlerden hesaplanır; eksik kimlik uydurulmaz, ayrı sayılır.

| Küme | Kaynak | Kayıt | Saat | Bilinen konuşmacı | Konuşmacı ID'si boş | Oturum/kayıt | Oturum ID'si boş |
|---|---|---|---|---|---|---|---|
| temiz CSS10 (yalnız train) | css10_el | 1844 | 4.137269 | 0 | 1844 | 0 | 1844 |
| temiz seçili HParl train | hparl | 23602 | 30.883408 | 208 | 0 | 18 | 0 |
| karışım HParl + CSS10 | css10_el | 1844 | 4.137269 | 0 | 1844 | 0 | 1844 |
| karışım HParl + CSS10 | hparl | 23602 | 30.883408 | 208 | 0 | 18 | 0 |
| ham seçili HParl train | hparl | 23615 | 30.896592 | 208 | 0 | 18 | 0 |
| dev (değişmedi) | hparl | 7113 | 9.059189 | 64 | 0 | 3 | 0 |
| test (dondurulmuş) | hparl | 8679 | 11.207889 | 82 | 0 | 3 | 0 |

CSS10 Greek: üretici tek okuyucu (Rapunzelina) ve tek kitap (Penelope Delta, 'Paramythi horis onoma') bildiriyor (kaynak: docs/SECONDARY_SOURCE_TR.md, Kyubyong/css10); örnek bazında konuşmacı veya kayıt kimliği verilmediği için speaker_id ve source_recording_id boş bırakıldı ve bilinmeyen olarak sayıldı.

| Küme | Testle ortak konuşmacı | Konuşmacısı bu kümede olan test örneği | Testle ortak oturum | Dev ile ortak konuşmacı | Dev ile ortak oturum |
|---|---|---|---|---|---|
| temiz CSS10 (yalnız train) | 0 | 0 | 0 | 0 | 0 |
| temiz seçili HParl train | 49 | 4782 | 0 | 41 | 0 |
| karışım HParl + CSS10 | 49 | 4782 | 0 | 41 | 0 |
| ham seçili HParl train | 49 | 4782 | 0 | 41 | 0 |
| dev | 16 | — | 0 | — | — |

**Test cümlesinin eğitim hedefinde bulunması.** *Literal eşleşme:* dondurulmuş test referans string'i hiçbir dönüşüm uygulanmadan (etiket silme, büyük/küçük harf, Unicode veya boşluk düzeltmesi yok) eğitim hedeflerinden en az birine birebir eşittir; eşleşen test örneği ve benzersiz metin ayrı sayılır. *Normalize eşleşme* yalnız tanısal bir ölçüdür, puanlama kuralı değildir: iki tarafta `<...>` işaretleri silinir, Unicode NFC, casefold ve boşluk birleştirme uygulanır, benzersiz metinler kesiştirilir. İkisi farklı tanımlardır ve birbirinin yerine sunulmaz.

| Eğitim hedefi | Literal test örneği | Literal benzersiz metin | Normalize test örneği | Normalize benzersiz metin | Referans dosyası |
|---|---|---|---|---|---|
| temiz CSS10 (yalnız train) | 0 | 0 | 0 | 0 | `data/manifests/references/css10_el_train.jsonl` `8167c13e9002537f…` |
| temiz seçili HParl train | 3 | 3 | 9 | 9 | `data/manifests/subsets/references/train_hparl_30.5h_seed42.jsonl` `bcf07eb981fa2136…` |
| karışım HParl + CSS10 | 3 | 3 | 9 | 9 | `data/manifests/subsets/references/train_hparl_30.5h_seed42_css10.jsonl` `28b0f955f1a3cbbd…` |
| ham seçili HParl train | 5 | 5 | 9 | 9 | `data/manifests/subsets/raw/references/train_hparl_30.5h_seed42.jsonl` `56ac531ff4869c02…` |

Bağlam (eğitimde kullanılmaz): tam kaynak aday train envanteri 76341 kayıt / 99.528 saat / 376 konuşmacı / 43 oturum. Hiçbir eşleşen cümle testten silinmedi; test değişmedi. Makine okunur kayıt: `data/clean/selected_train_hparl_30.5h_seed42/cohort_summary.json`.

## 5. Temizlik sonucu (seçili kapsam)

Bir kayıt birden çok çıkarma kuralına uyuyorsa yalnız config'teki ilk kurala yazılır.

| Bölüm | Ham kayıt | Ham süre | Temiz kayıt | Temiz süre | Çıkarılan | Çıkarılan süre | Metni değişen |
|---|---|---|---|---|---|---|---|
| HParl train (seçili 18 oturum) | 23615 | 30.90 sa (30:53:48) | 23602 | 30.88 sa (30:53:00) | 13 | 0.79 dk | 13527 |
| HParl dev | 7113 | 9.06 sa (9:03:33) | 7113 | 9.06 sa (9:03:33) | 0 | 0.00 dk | 0 |
| CSS10 Greek (yalnız train) | 1844 | 4.14 sa (4:08:14) | 1844 | 4.14 sa (4:08:14) | 0 | 0.00 dk | 1843 |

| Bölüm | İşlem | Kural | Kayıt | Süre |
|---|---|---|---|---|
| HParl train (seçili 18 oturum) | çıkarma | `cps_extreme` | 9 | 41.67 sn |
| HParl train (seçili 18 oturum) | çıkarma | `duplicate_audio` | 4 | 5.79 sn |
| HParl train (seçili 18 oturum) | metin | `strip_annotation_tags` | 13527 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `cps_boundary` | 1 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `cps_high` | 7 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `duplicate_suspect` | 3 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `long_edge_silence` | 2 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `non_greek_characters` | 98 | — |
| HParl train (seçili 18 oturum) | inceleme (korundu) | `silence_heavy` | 1 | — |
| HParl dev | inceleme (korundu) | `cps_high` | 10 | — |
| HParl dev | inceleme (korundu) | `duplicate_suspect` | 1 | — |
| HParl dev | inceleme (korundu) | `non_greek_characters` | 31 | — |
| CSS10 Greek (yalnız train) | metin | `collapse_whitespace` | 1839 | — |
| CSS10 Greek (yalnız train) | metin | `lowercase` | 1770 | — |
| CSS10 Greek (yalnız train) | metin | `remove_characters_absent_from_hparl_train` | 1839 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_boundary` | 1 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_high` | 1 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_low` | 22 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `long_edge_silence` | 1 | — |

Günlük: `data/clean/selected_train_hparl_30.5h_seed42/cleaning_log.jsonl` (15383 satır, `986f3135156b397f…`). İnceleme kuyruğu: `data/clean/selected_train_hparl_30.5h_seed42/review_queue.jsonl` (176 kayıt; CSS10 Greek (yalnız train) 23, HParl dev 42, HParl train (seçili 18 oturum) 111). Uygulanan insan dinleme kararı: 0 — bu turda hiçbir kayıt dinlenmiş sayılmaz.

CSS10 metin uyumlaması: üreticinin normalize metni alındı, NFC, küçük harf = True (seçili temiz HParl hedeflerinde büyük harf oranı 0.0), seçili temiz HParl hedeflerinde geçmeyen harf dışı karakterler boşluğa çevrildi (korunan: `yok`). Özgün metin `data/manifests/raw/css10_el_texts.jsonl` içinde değişmeden durur.

### Konuşma hızı kuralı

Saniyede > 60 boşluk dışı karakter: train/CSS10'da hizalama sorunu **adayı** olarak çıkarılır (dinlenmedi; sezgisel). 40–60 aralığı korunur ve inceleme kuyruğuna yazılır. Dev'de bu kuralla çıkarma yok.

| Bölüm | Sınır kaydı (korundu) | kar/sn | Süre (sn) | Karakter |
|---|---|---|---|---|
| HParl train (seçili 18 oturum) | `hparl:gewrgios_tsipras-olomeleia_9_00187500_00190500_2_1` | 47.47 | 4.34 | 206 |
| CSS10 Greek (yalnız train) | `css10_el:Paramythi_horis_onoma_0856` | 40.56 | 3.40 | 138 |

## 5b. Kilitli dosyalar ve CSS10/karışım lineage

Makine okunur kabul raporu: `configs/css10_mixture_lineage.json` (durum `css10_reaudited_mixture_ready`). Kilitli HParl seçimi, dev ve dondurulmuş test yalnız karşılaştırıldı; hiçbiri yeniden yazılmadı.

| Dosya | Yol | SHA-256 | Sonuç |
|---|---|---|---|
| Seçili temiz HParl | data/manifests/subsets/train_hparl_30.5h_seed42.jsonl | `f9731a5679c90e55…` | bayt-özdeş yeniden üretildi |
| Dev | data/manifests/dev.jsonl | `316fbcfbfaf42a3f…` | bayt-özdeş yeniden üretildi |
| Dondurulmuş test | data/manifests/test.jsonl | `b0842bfbcbffa173…` | freeze ok → ok |

| Çıktı | Önceki | Güncel | Durum |
|---|---|---|---|
| CSS10 manifest | `data/manifests/css10_el_train.jsonl` `8deab17bc1dd5be9…` | `data/manifests/css10_el_train.jsonl` `8deab17bc1dd5be9…` | aynı |
| CSS10 references | `data/manifests/references/css10_el_train.jsonl` `8167c13e9002537f…` | `data/manifests/references/css10_el_train.jsonl` `8167c13e9002537f…` | aynı |
| Karışım manifest | `data/manifests/subsets/train_hparl_30.5h_seed42_css10.jsonl` `309d9909f60408e2…` | `data/manifests/subsets/train_hparl_30.5h_seed42_css10.jsonl` `309d9909f60408e2…` | aynı |
| Karışım references | `data/manifests/subsets/references/train_hparl_30.5h_seed42_css10.jsonl` `28b0f955f1a3cbbd…` | `data/manifests/subsets/references/train_hparl_30.5h_seed42_css10.jsonl` `28b0f955f1a3cbbd…` | aynı |

Karışım kuralı: kilitli temiz HParl seçimi + güncel temiz CSS10, ID'ye göre sıralı. Güncel karışım 25446 kayıt / 35.020678 saat (css10_el: 1844 kayıt / 4.137269 sa, hparl: 23602 kayıt / 30.883408 sa). Karışım oranı/örnekleme ve eğitim bütçesi model konfigürasyonunda sahip tarafından belirlenir.

## 6. Gerçek örnekler (günlükten)

| Bölüm | İşlem | Kural | ID | Önce | Sonra | Kanıt |
|---|---|---|---|---|---|---|
| HParl train (seçili 18 oturum) | modify_text | `strip_annotation_tags` | `hparl:afrodith_stampoulh-olomeleia_32_00625000_00628000_1_1` | <spoken_noise> τη συγκρότηση ενιαίου χώρου τριτοβάθμιας εκπαίδευσης και έρευνας | τη συγκρότηση ενιαίου χώρου τριτοβάθμιας εκπαίδευσης και έρευνας | — |
| HParl train (seçili 18 oturum) | modify_text | `strip_annotation_tags` | `hparl:afrodith_stampoulh-olomeleia_32_00627500_00630500_1_1` | <spoken_noise> τεχνολογικές εφαρμογές και αφ ετέρου δημιουργεί ισχυρά | τεχνολογικές εφαρμογές και αφ ετέρου δημιουργεί ισχυρά | — |
| HParl train (seçili 18 oturum) | drop | `cps_extreme` | `hparl:anastasia_xristodoulopoylou-olomeleia_45_00012500_00015500_1_1` | — | — | {'chars': 4071, 'chars_per_second': 1109.2643, 'duration_se… |
| HParl train (seçili 18 oturum) | drop | `cps_extreme` | `hparl:anastasia_xristodoulopoylou-olomeleia_45_00142500_00145500_1_1` | — | — | {'chars': 500, 'chars_per_second': 136.9863, 'duration_seco… |
| HParl train (seçili 18 oturum) | drop | `duplicate_audio` | `hparl:dhmhtrios_kammenos-olomeleia_36_00835000_00838000_1_1` | — | — | {'duplicate_of': 'hparl:dhmhtrios_kammenos-olomeleia_36_008… |
| HParl train (seçili 18 oturum) | drop | `duplicate_audio` | `hparl:domna_maria_mixahlidou-olomeleia_2_00887500_00890500_1_1` | — | — | {'duplicate_of': 'hparl:domna_maria_mixahlidou-olomeleia_2_… |
| CSS10 Greek (yalnız train) | modify_text | `lowercase+remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0000` | Όταν κατάλαβε ο γερο-Βασιλιάς Συνετός πως μετρήθηκαν πια οι μέρες του, φώναξε το γιο του,… | όταν κατάλαβε ο γερο βασιλιάς συνετός πως μετρήθηκαν πια οι μέρες του φώναξε το γιο του τ… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase+remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0001` | - Φθάνουν, γιε μου, τα παιχνίδια και οι διασκεδάσεις. Ήλθε η ώρα να παντρευθείς και να πά… | φθάνουν γιε μου τα παιχνίδια και οι διασκεδάσεις ήλθε η ώρα να παντρευθείς και να πάρεις … | — |
| CSS10 Greek (yalnız train) | modify_text | `remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0006` | κάστρα γερά και γεμάτα στρατιώτες τειχογύριζαν το βασίλειο το λαμπρό παλάτι, χτισμένο ψηλ… | κάστρα γερά και γεμάτα στρατιώτες τειχογύριζαν το βασίλειο το λαμπρό παλάτι χτισμένο ψηλά… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase` | `css10_el:Paramythi_horis_onoma_0007` | δέσποζε τη χώρα όπου ζούσαν με άνεση οι πολίτες δρόμοι φαρδείς και καλοστρωμένοι ένωναν τ… | δέσποζε τη χώρα όπου ζούσαν με άνεση οι πολίτες δρόμοι φαρδείς και καλοστρωμένοι ένωναν τ… | — |
| CSS10 Greek (yalnız train) | modify_text | `remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0009` | από πάνω από τον ψηλό πύργο του παλατιού του, έβλεπε απέραντα χωράφια σπαρμένα, ρεματιές … | από πάνω από τον ψηλό πύργο του παλατιού του έβλεπε απέραντα χωράφια σπαρμένα ρεματιές κα… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase` | `css10_el:Paramythi_horis_onoma_0466` | Και με μεγαλόπρεπη κίνηση σήκωσε την περγαμηνή και ξεσκέπασε | και με μεγαλόπρεπη κίνηση σήκωσε την περγαμηνή και ξεσκέπασε | — |

## 7. Model sahiplerine teslim edilen eğitim/değerlendirme dosyaları

| Dosya | Manifest | Kayıt | Saat | SHA-256 |
|---|---|---|---|---|
| temiz seçili HParl train | `data/manifests/subsets/train_hparl_30.5h_seed42.jsonl` | hparl: 23602 | 30.883408 | `f9731a5679c90e55…` |
| karışım HParl + CSS10 | `data/manifests/subsets/train_hparl_30.5h_seed42_css10.jsonl` | css10_el: 1844, hparl: 23602 | 35.020678 | `309d9909f60408e2…` |
| ham seçili HParl train | `data/manifests/subsets/raw/train_hparl_30.5h_seed42.jsonl` | hparl: 23615 | 30.896592 | `3991a65229c95fdf…` |
| dev | `data/manifests/dev.jsonl` | 7113 | 9.059189 | `316fbcfbfaf42a3f…` |
| dondurulmuş test | `data/manifests/test.jsonl` | 8679 | 11.207889 | `b0842bfbcbffa173…` |

Temiz manifestler ham seçili satırların aynısını (aynı özgün WAV yolu/hash'i) kullanır; temizlik için ses kopyası üretilmedi. CSS10 22050 Hz float32 kalır; model giriş hattı bellekte 16 kHz'e çevirir.

## 8. Sınırlamalar ve bilinen eksikler

- Otomatik bayraklar dinlenmeden hata sayılmadı; korunup inceleme kuyruğuna yazıldı. Bu turda insan dinleme kararı yok.
- Ses–metin hizası ASR modeliyle kontrol edilmedi; model çıktısı gold sayılmaz. Konuşma hızı kuralı sezgiseldir.
- Resmî bölüm konuşmacıdan bağımsız değildir (bölüm 4b).
- Yeniden kodlanmış/kesilmiş tekrarlar SHA-256 ile yakalanamaz.
- Seçili kapsamda train içi tekrar/şüphe kontrolleri yalnız seçili 18 oturumu görür; kilitli temiz HParl dosyası tam envanterden üretilmişti ve seçili kapsam onu bayt-özdeş yeniden üretti (bölüm 5b).
- CSS10 tek okuyuculu sesli kitaptır; parlamento konuşmasını temsil etmez ve yalnız train'dedir. Eski integer yorumlu CSS10 sinyal istatistikleri geçersizdir ve yalnız tarihsel kayıttır.
- `<spoken_noise>` işaretleri yalnız train hedeflerinden çıkarıldı; dev/test referansları kaynakla aynıdır.

## 9. Verinin repoda teslimi

Durum `configs/data_delivery.json` karar kaydından okunur; `pending` teslim edilmiş sayılmaz. Seçili ham HParl train seslerinin LFS teslimini proje sahibi yürütür; temiz manifestler aynı ham WAV'ları paylaşır.

| Grup (kaynak arşiv listesi) | Dosya | GiB | Yöntem | Durum |
|---|---|---|---|---|
| `css10_source_metadata` | 1 | 0.001 | plain Git, `-text` | in_repo |
| `css10_unreferenced_metadata` | 1 | 0.001 | — | teslim edilmez |
| `css10_unreferenced_wav` | 1844 | 1.224 | — | teslim edilmez |
| `css10_used_wav` | 1844 | 1.224 | Git LFS | in_repo |
| `hparl_dev_wav` | 7113 | 0.972 | Git LFS | in_repo |
| `hparl_source_metadata` | 15 | 0.038 | plain Git, `-text` so bytes are never converted | in_repo |
| `hparl_test_wav` | 8679 | 1.203 | Git LFS | in_repo |
| `hparl_train_wav` | 76341 | 10.681 | Git LFS, subset actually used | pending |

Seçili ham train sesi: 23615 WAV, 3560140964 bayt; test + dev + CSS10 + seçili train toplam ≈ 6.714255 GiB (onaylı sınır 10.0 GiB, ücretli alan yok).

Bu çıktıları üreten ortam: Python 3.12.10, NumPy 1.26.4, Windows.

## 10. Yeniden üretme komutları (yalnız teslim edilen sesle)

```bash
python -m greek_asr.data verify-source-metadata
python -m greek_asr.data verify-freeze                       # test sesleri dahil
python -m greek_asr.data audit --selection configs/training_data_selection.json
python -m greek_asr.data clean --selection configs/training_data_selection.json    # CSS10 + karışım + lineage
python -m greek_asr.data verify --selection configs/training_data_selection.json   # seçili ham/temiz, CSS10, karışım, dev + freeze
python -m greek_asr.data verify --only css10_el_train.jsonl dev.jsonl
python -m greek_asr.data report
python -m unittest discover -s tests -v
```

Kaynaktan baştan üretim (`hparl-download`, `prepare-freeze`, `prepare-clean`) tam arşiv gerektirir; ayrıntı `docs/DATA_PIPELINE_TR.md`.

## 11. Sunum için veri slaytı taslağı

- Resmî HParl bölümü (speaker-disjoint değil): seçili train 23615 ham → 23602 temiz kayıt, 30.883 saat, 208 konuşmacı, 18 oturum; testin 49 konuşmacısı train'de de var, oturum örtüşmesi 0.
- Test erken donduruldu (8679 kayıt, 11.21 sa (11:12:28)); hiçbir işlem yok, hash'lerle doğrulandı.
- CSS10 (yalnız train) float32 olarak doğru denetlendi: 1844 kayıt, 4.137 saat; karışım 25446 kayıt / 35.021 saat.
- Literal test cümlesi eşleşmesi: temiz seçili HParl train 3, karışım HParl + CSS10 3, ham seçili HParl train 5 (normalize ölçü ayrı).
