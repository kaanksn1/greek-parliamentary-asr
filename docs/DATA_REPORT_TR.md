# Veri raporu — HParl + CSS10 Greek (Samed, veri sorumlusu)

Bu dosya `python -m greek_asr.data report` komutuyla pipeline çıktılarından üretilir; içindeki sayılar elle yazılmamıştır. Model başarımı veya bonus iddiası içermez.

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

## 4. Denetim bulguları (train/dev/CSS10)

Bayraklar adaydır; tek başına hatalı veri kanıtı değildir. Bir kayıt birden çok bayrak taşıyabilir.

| Bayrak | CSS10 Greek (yalnız train) | HParl dev | HParl train |
|---|---|---|---|
| `annotation_tag` | 0 | 4267 (312.5 dk) | 44487 (3289.8 dk) |
| `cps_boundary` | 1 (0.1 dk) | 0 | 6 (0.4 dk) |
| `cps_extreme` | 0 | 0 | 22 (2.0 dk) |
| `cps_high` | 1 (0.1 dk) | 9 (0.4 dk) | 62 (3.4 dk) |
| `cps_low` | 22 (3.5 dk) | 0 | 0 |
| `duplicate_audio` | 0 | 0 | 17 (0.4 dk) |
| `duplicate_suspect` | 0 | 1 (0.0 dk) | 14 (0.4 dk) |
| `long_edge_silence` | 0 | 0 | 7 (0.8 dk) |
| `non_greek_characters` | 1839 (247.9 dk) | 31 (2.8 dk) | 323 (32.4 dk) |
| `silence_heavy` | 0 | 0 | 6 (0.3 dk) |

Aynı baytlı ses grupları: 17 grup / 34 kayıt (test dahil karşılaştırma). Aynı PCM içerikli gruplar: 17. Yöntem SHA-256 ile birebir tekrarları ve yalnız WAV başlığı farklı kopyaları yakalar; yeniden kodlanmış, kesilmiş veya ses düzeyi değişmiş tekrarları yakalamaz. Bunlar için aynı konuşmacı + aynı metin + ±tolerans süre eşleşmesi `duplicate_suspect` olarak işaretlenir ve dinlemeye bırakılır.

**Puanlama için önemli:** kaynak referanslarında `<spoken_noise>` gibi işaretler sık geçiyor (yalnız okundu, değiştirilmedi): dev: 4267/7113 referans (%60.0), 5235 işaret; test: 5122/8679 referans (%59.0), 6219 işaret. Train hedeflerinden çıkarıldıkları için modeller bu işaretleri üretmeyecek. İşaretler puanlamada referansta bırakılırsa her biri silme hatası sayılır ve WER yapay olarak yükselir. Bu işaretlerin referans ve hipotezde aynı kuralla ele alınması Ozan'ın puanlama politikasında, dev üzerinde ve test dosyalarına yazılmadan kararlaştırılmalıdır.

## 4b. Bölüm, konuşmacı ve test cümlesi örtüşmesi

Kullanılan bölüm: **HParl 1.0.0'ın resmî oturum bazlı train/dev/test bölümü**. Bu bölüm konuşmacıdan bağımsız (speaker-disjoint) **değildir**: aynı milletvekilleri farklı oturumlarda birden çok bölümde konuşuyor. Hocanın mailindeki How 1 seçeneği belgelenmiş resmî bölümün kullanılmasına izin veriyor; dondurulmuş test bu nedenle yeniden bölünmedi. Sonuçlar yeni konuşmacılara genelleme ölçümü olarak sunulmamalıdır; ölçülen şey, test konuşmacılarının train'de de bulunabildiği yeni oturumlardaki başarımdır (ortak konuşmacı sayıları aşağıda). Sayılar manifestlerden koddan hesaplanır (ses okunmaz).

| Küme | Örnek | Saat | Bilinen konuşmacı | Konuşmacı ID'si boş örnek | Oturum |
|---|---|---|---|---|---|
| ham HParl train | 76341 | 99.528 | 376 | 0 | 43 |
| HParl dev | 7113 | 9.059 | 64 | 0 | 3 |
| HParl test (dondurulmuş) | 8679 | 11.208 | 82 | 0 | 3 |
| temiz dev (kullanılan) | 7113 | 9.059 | 64 | 0 | 3 |
| eğitim koşulu `clean_hparl_only` | 76302 | 99.487 | 376 | 0 | 43 |
| eğitim koşulu `clean_hparl_css10` | 78146 | 103.625 | 376 | 1844 | 43 |

Konuşmacı ID'si boş örnekler bilinen konuşmacı sayısına katılmaz; sayıları ayrı sütunda yazılır. CSS10 Greek: üretici tek okuyucu (Rapunzelina) ve tek kitap (Penelope Delta, 'Paramythi horis onoma') bildiriyor (kaynak: docs/SECONDARY_SOURCE_TR.md, Kyubyong/css10); örnek bazında konuşmacı veya kayıt kimliği verilmediği için speaker_id ve source_recording_id boş bırakıldı ve bilinmeyen olarak sayıldı.

| Bölüm çifti | Ortak konuşmacı | Ortak oturum |
|---|---|---|
| train ∩ dev | 59 | 0 |
| train ∩ test | 76 | 0 |
| dev ∩ test | 16 | 0 |

Testteki 8679 örneğin 7777 tanesinin konuşmacısı train'de de var (76 ortak konuşmacı).

**Test cümlelerinin eğitimde birebir bulunması.** Literal eşitlik: dondurulmuş test referans string'i hiçbir dönüşüm uygulanmadan (etiket silme, büyük/küçük harf, Unicode veya boşluk düzeltmesi yok) eğitim koşulunun en az bir hedef string'ine eşitse eşleşme sayılır. Aynı cümle birden çok test örneğinde geçebildiği için örnek sayısı ve benzersiz metin sayısı ayrı verilir.

| Eğitim hedefi | Literal eşleşen test örneği | Literal benzersiz metin | Normalize eşleşen test örneği | Normalize benzersiz metin | Karşılaştırılan referans dosyası |
|---|---|---|---|---|---|
| `raw_hparl_train` | 14 | 14 | 22 | 21 | `data/manifests/raw/references/hparl_train.jsonl` `d897e088f73b500d…` |
| `clean_hparl_only` | 6 | 6 | 22 | 21 | `data/manifests/references/train.jsonl` `0e231d017d84eddc…` |
| `clean_hparl_css10` | 6 | 6 | 22 | 21 | `data/manifests/references/train_hparl_css10.jsonl` `f92365155047a9d5…` |

Test referansları: `data/manifests/references/test.jsonl` `f3d6b377959b8a8183e4b45d1ee1e2db693f5f53fb1467b194f7c258bae2e5f9` (8679 örnek, 8678 benzersiz metin). Normalize sütunlar ayrı bir ölçüdür: iki tarafta da `<...>` işaretleri silinir, Unicode NFC, casefold ve boşluk birleştirme uygulanır; literal eşitlik yerine sunulmaz. Hiçbir eşleşen cümle testten silinmedi; dondurulmuş test değişmedi. Makine okunur kayıt: `data/clean/cohort_summary.json`.

Denetim adımındaki ek kontroller (ham bölümler arası; `distinct_texts_after_tag_strip_nfc_casefold_space` normalize benzersiz metin kesişimidir, literal değildir):

| Tür | train∩dev | train∩test | dev∩test |
|---|---|---|---|
| original_ids | 0 | 0 | 0 |
| distinct_texts_after_tag_strip_nfc_casefold_space | 15 | 21 | 2 |

## 5. Temizlik sonucu

Bir kayıt birden çok çıkarma kuralına uyuyorsa yalnız config'teki ilk kurala yazılır (çifte sayım yok).

| Bölüm | Ham kayıt | Ham süre | Temiz kayıt | Temiz süre | Çıkarılan | Çıkarılan süre | Metni değişen |
|---|---|---|---|---|---|---|---|
| HParl train | 76341 | 99.53 sa (99:31:41) | 76302 | 99.49 sa (99:29:14) | 39 | 2.5 dk | 44456 |
| HParl dev | 7113 | 9.06 sa (9:03:33) | 7113 | 9.06 sa (9:03:33) | 0 | 0.0 dk | 0 |
| CSS10 Greek (yalnız train) | 1844 | 4.14 sa (4:08:14) | 1844 | 4.14 sa (4:08:14) | 0 | 0.0 dk | 1843 |

| Bölüm | İşlem | Kural | Kayıt | Süre |
|---|---|---|---|---|
| HParl train | çıkarma | `cps_extreme` | 22 | 2.0 dk |
| HParl train | çıkarma | `duplicate_audio` | 17 | 0.4 dk |
| HParl train | metin | `strip_annotation_tags` | 44456 | — |
| HParl train | inceleme (korundu) | `cps_boundary` | 6 | — |
| HParl train | inceleme (korundu) | `cps_high` | 40 | — |
| HParl train | inceleme (korundu) | `duplicate_suspect` | 14 | — |
| HParl train | inceleme (korundu) | `long_edge_silence` | 7 | — |
| HParl train | inceleme (korundu) | `non_greek_characters` | 322 | — |
| HParl train | inceleme (korundu) | `silence_heavy` | 6 | — |
| HParl dev | inceleme (korundu) | `cps_high` | 9 | — |
| HParl dev | inceleme (korundu) | `duplicate_suspect` | 1 | — |
| HParl dev | inceleme (korundu) | `non_greek_characters` | 31 | — |
| CSS10 Greek (yalnız train) | metin | `collapse_whitespace` | 1839 | — |
| CSS10 Greek (yalnız train) | metin | `lowercase` | 1770 | — |
| CSS10 Greek (yalnız train) | metin | `remove_characters_absent_from_hparl_train` | 1839 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_boundary` | 1 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_high` | 1 | — |
| CSS10 Greek (yalnız train) | inceleme (korundu) | `cps_low` | 22 | — |

Günlük: `data/clean/cleaning_log.jsonl` (46338 satır, SHA-256 `5fa67ba3e71c617aa83938a9ae1f9aa93ef2eab46bed0dd70d0c740b50fbfae9`). İnceleme kuyruğu: `data/clean/review_queue.jsonl` (451 kayıt). Uygulanan insan inceleme kararı: 0.

CSS10 metin uyumlaması: üreticinin normalize metni (sayılar yazıyla) alındı, NFC uygulandı, küçük harfe çevirme = True (HParl train büyük harf oranı 0.0), HParl train referanslarında hiç geçmeyen harf dışı karakterler boşluğa çevrildi. HParl train'de korunan harf dışı karakterler: `yok`. Özgün ve normalize metin `data/manifests/raw/css10_el_texts.jsonl` içinde değişmeden durur.

### Konuşma hızı kuralı (otomatik kalite sezgiseli)

Saniye başına boşluk dışı karakter > 60 olan train/CSS10 kayıtları hizalama sorunu **adayı** olarak eğitimden çıkarılır (train p99 değerinin yaklaşık üç katı; bu kayıtlarda birkaç saniyelik sese yüzlerce–binlerce karakter düşüyor). Bu bir sezgiseldir: kayıtlar dinlenmedi, etiketlerin yanlış olduğu kanıtlanmadı. 40–60 aralığı belirsiz kabul edilir; bu kayıtlar eğitimde **korunur** ve inceleme kuyruğuna yazılır. Dev'de hiçbir kayıt bu kuralla çıkarılmaz. Bir modelin bu hedeflerle eğitilip eğitilemeyeceği model/tokenizer'a bağlıdır; kural bunun genel ispatı olarak sunulmaz. Çıkarılan: CSS10 Greek (yalnız train) 0, HParl dev 0, HParl train 22.

| Bölüm | Sınır kaydı (korundu) | kar/sn | Süre (sn) | Karakter | Durum |
|---|---|---|---|---|---|
| HParl train | `hparl:apostolos_karanastashs-olomeleia_10_04697500_04700500_4_1` | 58.18 | 4.40 | 256 | korundu, inceleme bekliyor |
| HParl train | `hparl:efh_axtsioglou-olomeleia_42_01817500_01820500_3_1` | 60.00 | 4.70 | 282 | korundu, inceleme bekliyor |
| HParl train | `hparl:elenh_stamatakh-olomeleia_27_05582500_05585500_2_1` | 48.52 | 2.70 | 131 | korundu, inceleme bekliyor |
| HParl train | `hparl:gewrgios_tsipras-olomeleia_9_00187500_00190500_2_1` | 47.47 | 4.34 | 206 | korundu, inceleme bekliyor |
| HParl train | `hparl:panos_kammenos_ellhnwn-olomeleia_27_04762500_04765500_2_1` | 46.00 | 6.00 | 276 | korundu, inceleme bekliyor |
| HParl train | `hparl:xrhstos_xatzhsavvas-olomeleia_10_03515000_03518000_1_2` | 56.00 | 2.50 | 140 | korundu, inceleme bekliyor |
| CSS10 Greek (yalnız train) | `css10_el:Paramythi_horis_onoma_0856` | 40.56 | 3.40 | 138 | korundu, inceleme bekliyor |

## 6. Gerçek örnekler (günlükten)

| Bölüm | İşlem | Kural | ID | Önce | Sonra | Kanıt |
|---|---|---|---|---|---|---|
| HParl train | modify_text | `strip_annotation_tags` | `hparl:afrodith_stampoulh-olomeleia_12_05115000_05118000_1_1` | <spoken_noise> και συνάδελφοι θα προσπαθήσω να μη γίνω και εγώ κακή | και συνάδελφοι θα προσπαθήσω να μη γίνω και εγώ κακή | — |
| HParl train | modify_text | `strip_annotation_tags` | `hparl:afrodith_stampoulh-olomeleia_12_05117500_05120500_1_1` | <spoken_noise> οικονομολόγος διότι δεν είμαι οικονομολόγος όπως κάποιοι συνάδελφοι προηγο… | οικονομολόγος διότι δεν είμαι οικονομολόγος όπως κάποιοι συνάδελφοι προηγουμένως γίνανε κ… | — |
| HParl train | drop | `cps_extreme` | `hparl:aleksandros_flampourarhs-olomeleia_42_01207500_01210500_3_1` | — | — | {'chars': 4619, 'chars_per_second': 549.881, 'duration_seco… |
| HParl train | drop | `cps_extreme` | `hparl:anastasia_xristodoulopoylou-olomeleia_45_00012500_00015500_1_1` | — | — | {'chars': 4071, 'chars_per_second': 1109.2643, 'duration_se… |
| HParl train | drop | `duplicate_audio` | `hparl:dhmhtrios_kammenos-olomeleia_36_00835000_00838000_1_1` | — | — | {'duplicate_of': 'hparl:dhmhtrios_kammenos-olomeleia_36_008… |
| HParl train | drop | `duplicate_audio` | `hparl:dhmhtrios_koutsoumpas-olomeleia_31_01647500_01650500_1_1` | — | — | {'duplicate_of': 'hparl:dhmhtrios_koutsoumpas-olomeleia_31_… |
| CSS10 Greek (yalnız train) | modify_text | `lowercase+remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0000` | Όταν κατάλαβε ο γερο-Βασιλιάς Συνετός πως μετρήθηκαν πια οι μέρες του, φώναξε το γιο του,… | όταν κατάλαβε ο γερο βασιλιάς συνετός πως μετρήθηκαν πια οι μέρες του φώναξε το γιο του τ… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase+remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0001` | - Φθάνουν, γιε μου, τα παιχνίδια και οι διασκεδάσεις. Ήλθε η ώρα να παντρευθείς και να πά… | φθάνουν γιε μου τα παιχνίδια και οι διασκεδάσεις ήλθε η ώρα να παντρευθείς και να πάρεις … | — |
| CSS10 Greek (yalnız train) | modify_text | `remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0006` | κάστρα γερά και γεμάτα στρατιώτες τειχογύριζαν το βασίλειο το λαμπρό παλάτι, χτισμένο ψηλ… | κάστρα γερά και γεμάτα στρατιώτες τειχογύριζαν το βασίλειο το λαμπρό παλάτι χτισμένο ψηλά… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase` | `css10_el:Paramythi_horis_onoma_0007` | δέσποζε τη χώρα όπου ζούσαν με άνεση οι πολίτες δρόμοι φαρδείς και καλοστρωμένοι ένωναν τ… | δέσποζε τη χώρα όπου ζούσαν με άνεση οι πολίτες δρόμοι φαρδείς και καλοστρωμένοι ένωναν τ… | — |
| CSS10 Greek (yalnız train) | modify_text | `remove_characters_absent_from_hparl_train+collapse_whitespace` | `css10_el:Paramythi_horis_onoma_0009` | από πάνω από τον ψηλό πύργο του παλατιού του, έβλεπε απέραντα χωράφια σπαρμένα, ρεματιές … | από πάνω από τον ψηλό πύργο του παλατιού του έβλεπε απέραντα χωράφια σπαρμένα ρεματιές κα… | — |
| CSS10 Greek (yalnız train) | modify_text | `lowercase` | `css10_el:Paramythi_horis_onoma_0466` | Και με μεγαλόπρεπη κίνηση σήκωσε την περγαμηνή και ξεσκέπασε | και με μεγαλόπρεπη κίνηση σήκωσε την περγαμηνή και ξεσκέπασε | — |

Önce/sonra metin yalnız metni değişen kayıtlarda günlükte tutulur; çıkarılan kayıtlarda metin hash'i ve ses hash'i vardır. Dinleyerek doğrulanmayan Yunanca metin tahminle düzeltilmedi.

## 7. Model sahiplerine teslim edilen koşullar

| Koşul | Manifest | Kayıt | Süre | Örnekleme hızı | SHA-256 |
|---|---|---|---|---|---|
| hparl_only | `data/manifests/train.jsonl` | hparl: 76302 | 99.49 sa (99:29:14) | 16000: 76302 | `85f5482c69323488…` |
| hparl_css10 | `data/manifests/train_hparl_css10.jsonl` | css10_el: 1844, hparl: 76302 | 103.62 sa (103:37:28) | 16000: 76302, 22050: 1844 | `24bb49c5fef5f3c4…` |
| ortak dev | `data/manifests/dev.jsonl` | hparl: 7113 | 9.06 sa (9:03:33) | 16000: 7113 | `316fbcfbfaf42a3f…` |
| dondurulmuş test | `data/manifests/test.jsonl` | 8679 | 11.21 sa (11:12:28) | — | `b0842bfbcbffa173…` |

Karışım oranı, örnekleme, adım/epoch sayısı ve iki koşulun eşlenmiş eğitim bütçesi veri sorumlusu tarafından belirlenmez; Adnan/Serdar HParl dev üzerinde seçer ve kendi koşu kayıtlarına yazar. CSS10 sesi özgün örnekleme hızında kalır; model girdi hattı yükleme sırasında yeniden örnekler.

## 8. Sınırlamalar ve bilinen eksikler

- Otomatik bayraklar (sessizlik, kırpılma, konuşma hızı aykırılığı, şüpheli tekrar) dinlenmeden hatalı sayılmadı; kayıtlar korunup inceleme kuyruğuna yazıldı.
- Ses–metin hizası ASR modeliyle kontrol edilmedi; model çıktısı doğrulanmış gold sayılmaz. Konuşma hızı kuralı otomatik bir sezgiseldir; çıkarılan kayıtların yanlış etiketli olduğu dinlenerek doğrulanmadı.
- Resmî bölüm konuşmacıdan bağımsız değildir (bölüm 4b); sonuçlar yeni konuşmacılara genelleme ölçümü değildir.
- İnsan inceleme kararı uygulanmadıysa (bölüm 5) hiçbir kayıt dinlenmiş sayılmaz.
- Yeniden kodlanmış/kesilmiş tekrarlar SHA-256 ile yakalanamaz; yalnız şüphe sezgiseli vardır.
- CSS10 tek konuşmacılı sesli kitap okumasıdır; parlamento konuşmasını temsil etmez ve yararı henüz gösterilmedi. Sadece train'e girdi.
- HParl'de transkript–ses farkları kaynağın kendi segmentasyonundan gelebilir; düzeltme yapılmadı.
- `<spoken_noise>` gibi işaretler yalnız train hedef metninden çıkarıldı; dev ve test referansları kaynakla aynıdır. Nihai puanlamada bu işaretlerin ele alınışı Ozan'ın puanlama politikasındadır.
- Oturum kimliği kaynakta ayrı alan değilse utterance ID deseninden türetilir; yöntem ve başarısız türetme sayısı envanterde yazılıdır.

## 9. Verinin repoda teslimi

Boyutlar kaydedilmiş arşiv listelerinden (`archive_members.tsv`) hesaplanır; durum `configs/data_delivery.json` karar kaydından okunur. Ham ve temiz manifestler aynı ses baytlarını paylaşır (train/dev için ses türetilmedi), bu yüzden ikinci bir ses kopyası yoktur. Yalnız indirme script'i veya Release bağlantısı teslim sayılmaz.

| Grup | Dosya | GiB | Yöntem | Durum |
|---|---|---|---|---|
| `css10_source_metadata` | 1 | 0.001 | plain Git, `-text` | in_repo |
| `css10_unreferenced_metadata` | 1 | 0.001 | — | teslim edilmez (manifestte yok) |
| `css10_unreferenced_wav` | 1844 | 1.224 | — | teslim edilmez (manifestte yok) |
| `css10_used_wav` | 1844 | 1.224 | Git LFS | in_repo |
| `hparl_dev_wav` | 7113 | 0.972 | Git LFS | in_repo |
| `hparl_source_metadata` | 15 | 0.038 | plain Git, `-text` so bytes are never converted | in_repo |
| `hparl_test_wav` | 8679 | 1.203 | Git LFS | in_repo |
| `hparl_train_wav` | 76341 | 10.681 | Git LFS, subset actually used | pending |

Ücretsiz Git LFS depolama: 10 GiB. Test + dev + CSS10 sesi kümülatif 3.399 GiB (sığıyor); tam train eklenince 14.080 GiB (sığmıyor). `pending`/`blocked` durumundaki gruplar teslim edilmiş sayılmaz. Kaynak metin dosyalarının bayt eşitliği temiz clone'da `python -m greek_asr.data verify-source-metadata` ile kontrol edilir.

Bu çıktıları üreten ortam: Python 3.15.2, NumPy 2.5.5, Windows.

## 10. Yeniden üretme komutları

```bash
python -m greek_asr.data hparl-download            # ~11.9 GB, kaldığı yerden devam eder
python -m greek_asr.data prepare-freeze            # extract + ham manifest + test freeze + doğrulama
python -m greek_asr.data prepare-clean --css10-archive <kaggle.zip> --transcript el/transcript.txt
python -m greek_asr.data split-summary             # konuşmacı/oturum ve literal örtüşme (ses gerekmez)
python -m greek_asr.data verify-source-metadata    # Kaldi/CSS10 kaynak metin dosyalarının bayt kontrolü
python -m greek_asr.data delivery-plan             # gerçek teslim boyutları
python -m greek_asr.data verify                    # tüm manifestleri sözleşme ve sesle doğrular
python -m unittest discover -s tests -v
```

Ayrıntılı açıklama: `docs/DATA_PIPELINE_TR.md`.

## 11. Sunum için veri slaytı taslağı ve konuşmacı notu

**Slayt 1 — Veri ve test dondurma**

- HParl 1.0.0 (CC BY-NC 4.0): resmî oturum bazlı train/dev/test korundu; bölüm konuşmacıdan bağımsız değil (testteki 82 konuşmacının 76 tanesi train'de de var).
- Test erkenden donduruldu: 8679 kayıt, 11.21 sa (11:12:28); her ses ve referans SHA-256 ile kayıtlı, hiçbir işlem yok.
- Ek kaynak CSS10 Greek yalnız eğitimde.

**Slayt 2 — Temizlik**

- HParl train: 76341 → 76302 kayıt (99.53 sa (99:31:41) → 99.49 sa (99:29:14)); 44456 transkriptte işaret temizliği.
- Çıkarma kuralları: cps_extreme: 22, duplicate_audio: 17.
- Karışık eğitim koşulu: 103.62 sa (103:37:28); dev ve test iki koşulda da aynı.
- Testte literal olarak eğitimde de bulunan örnek: raw_hparl_train 14, clean_hparl_only 6, clean_hparl_css10 6.

**Konuşmacı notu (Ozan için):** Testi her şeyden önce dondurduk ve sonra hiç dokunmadık; bunu hash'lerle kanıtlıyoruz. Temizliği yalnız train/dev'de, kural ve gerekçesiyle yaptık; her değişiklik günlükte. Otomatik bayrakları hata saymadık, dinlemeye bıraktık. Resmî bölüm konuşmacıdan bağımsız olmadığı için sonuçları yeni konuşmacıya genelleme diye sunmuyoruz. CSS10'un katkısı ayrı deneyle ölçülecek.
