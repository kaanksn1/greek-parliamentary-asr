# Veri pipeline'ı — HParl + CSS10 Greek (Samed)

> **Final selected-cohort reporting:** `python scripts/report_benchmark_data.py` generates `docs/DATA_REPORT_TR.md` and `results/data_integrity.json`. The older `python -m greek_asr.data report` command writes a full-source/inventory report to the same Markdown path; after a historical reproduction, run the selected-cohort generator last. Full-source counts are not delivered training hours.

Bu belge veri hazırlama kodunun ne yaptığını, kuralların gerekçesini ve gerçek
verinin nasıl çalıştırılacağını anlatır. Kod `greek_asr/data/` altında,
testler `tests/test_data_pipeline.py` içindedir. Gerçek veri sayıları
`docs/DATA_REPORT_TR.md` dosyasına koddan yazılır. Bu belgede sonuç rakamı
yoktur.

Durum (20 Kasım 2026):

- **Kullanılan veri:** sahibin kilitli seçimi `configs/training_data_selection.json`
  (18 tam resmî train oturumu, seed 42) + değişmeyen dev + dondurulmuş test +
  yalnız train'de CSS10. Tam ~99 saatlik train manifesti yalnız kaynak aday
  envanteridir; sesi teslim edilmez.
- **CSS10 float32 düzeltmesi:** 1.844 CSS10 WAV'ı IEEE float32'dir (RIFF format
  tag 3). Eski denetim bunları integer PCM32 gibi okuyordu ve CSS10 sinyal
  istatistikleri yanlıştı. Python 3.12'deki `wave` modülü ise bu dosyaları hiç
  açmıyordu. Artık ses `greek_asr/audio.py` RIFF okuyucusuyla okunur ve format
  tag'e göre çözülür (bölüm 1b). Yeniden denetim gerçek seslerle 20 Kasım'da
  çalıştı (Python 3.12.10, NumPy 1.26.4).
- **Sonuç:** CSS10 1.844/1.844 korundu. Temiz CSS10 ve 25.446 kayıt /
  35,020678 saatlik karışım önceki dosyalarla bayt-özdeş çıktı. Kilitli HParl
  seçimi ve dev bayt-özdeş yeniden üretildi, test freeze sesle `ok` verdi.
  Kayıt: `configs/css10_mixture_lineage.json`, kanıt:
  `docs/reviews/css10_reaudit_2026-11-20/`, sayılar: `docs/DATA_REPORT_TR.md`.
- **Ses:** test, dev ve CSS10 Git LFS ile repoda. Seçili 23.615 ham train
  WAV'ının yüklemesini repo sahibi yürütüyor.

## 1. Samed'in bilgisayarında çalıştırma

Gerekenler: Python 3.12, Git, Git LFS ve en az ~50 GB boş disk. Bu alan
11,9 GB ZIP, ~14 GB açılmış ses, CSS10 ve Git/LFS kopyası içindir.

```bash
git clone https://github.com/kaanksn1/greek-parliamentary-asr.git
cd greek-parliamentary-asr
git switch feature/p3-data            # bu branch'i push ettikten sonra
python -m venv .venv-data && source .venv-data/bin/activate   # Windows: .venv-data\Scripts\activate
pip install -r requirements-data.txt
python -m unittest discover -s tests -v
```

### Aşama 1 — İndirme ve test dondurma (temizlikten önce)

```bash
python -m greek_asr.data hparl-download     # ~11,9 GB; kesilirse aynı komut kaldığı yerden sürer
python -m greek_asr.data prepare-freeze     # extract + ham manifestler + test freeze + doğrulama
```

`prepare-freeze` HParl ZIP'ini `data/raw/hparl/release/` içine CRC kontrolüyle
açar ve üç resmî bölümün ham manifestlerini yazar. Ardından
`data/manifests/test.jsonl`, `references/test.jsonl` ve `test.freeze.json`
dosyalarını üretir. Bölüm klasörleri otomatik bulunamazsa komut durur. O
durumda `python -m greek_asr.data hparl-inspect` çıktısına bakılır ve
`--layout layout.json` verilir.

Testte kaynak sorunu varsa, örneğin sesi olmayan bir test satırı, dondurma
yapılmaz ve hata mesajı basılır. Bu durumda **testi düzeltme**, sorunu
Serdar/Ozan'a ilet.

Ayrı commit, yalnız dondurma:

```bash
git add data/manifests/test.jsonl data/manifests/references/test.jsonl data/manifests/test.freeze.json \
        data/manifests/raw data/raw/hparl/*.json data/raw/hparl/*.tsv
# Test sesi ve Kaldi test dosyaları: Serdar'la anlaşılan yönteme göre (bkz. bölüm 7)
git commit -m "Freeze original HParl test before any train/dev cleaning"
git push -u origin feature/p3-data
```

Commit kimliğini Serdar'a ilet ve taslak PR aç. Bundan sonraki hiçbir komut bu
dosyaları değiştirmez. `clean` komutu dondurmayı başta ve sonda yeniden
doğrular.

### Aşama 2 — CSS10, denetim, temizlik, rapor

CSS10 Greek ZIP'i Kaggle'dan oturum açılarak indirilir. Web sayfasındaki
Download düğmesi veya `kaggle datasets download -d bryanpark/greek-single-speaker-speech-dataset`
kullanılabilir. Token veya kullanıcı adı repoya girmez.

```bash
python -m greek_asr.data prepare-clean --css10-archive ~/Downloads/greek-single-speaker-speech-dataset.zip
```

Bu komut sırasıyla şunları yapar: dondurma kontrolü, CSS10 hash+extract+ham
manifest, `audit`, `clean`, `report` ve `verify`. Arşivde birden çok
`transcript.txt` varsa `--transcript <göreli yol>` istenir.

Sonra:

1. `docs/DATA_REPORT_TR.md` oku. Ölçülen süreleri üretici bildirimiyle karşılaştır.
2. `data/clean/review_queue.jsonl` içinden birkaç kaydı **dinle**. Kararını
   `configs/review_decisions.jsonl` dosyasına yaz (biçim bölüm 5'te).
   `python -m greek_asr.data clean && python -m greek_asr.data report && python -m greek_asr.data verify`
   ile yeniden üret. Dinlemediğin kaydı karara bağlama.
3. Serdar'ın donanım bütçesi tam train'i kaldırmıyorsa oturum bazlı alt küme seç:
   `python -m greek_asr.data subset --hours 20 --seed 42`. Seed ve ID listesi
   `data/manifests/subsets/*.selection.json` içine yazılır. Saat değerini
   Serdar'ın ölçümüne göre seç.
4. `python -m greek_asr.data delivery-plan` ile teslim boyutunu ölç (bölüm 7).
5. Temizlik commit'i, kendi katkı dosyan ve PR açıklaması.

## 1b. Güncel zincir: kilitli seçimle (`--selection`)

Repoda teslim edilen ses yalnız seçili ham train, dev, test ve CSS10'dur.
`audit`, `clean` ve `verify` komutları `--selection <plan>` ile bu kapsamda
çalışır. Kullanılmayan train sesini istemez.

```bash
python -m greek_asr.data verify-source-metadata
python -m greek_asr.data verify-freeze                      # test sesleri dahil
python -m greek_asr.data audit  --selection configs/training_data_selection.json
python -m greek_asr.data clean  --selection configs/training_data_selection.json
python -m greek_asr.data verify --selection configs/training_data_selection.json
python -m greek_asr.data report
# hepsi + unittest, her komutun çıktısı docs/reviews/<klasör>/ altına:
python scripts/run_selected_data_checks.py
```

- **Plan kontrolü:** komutlar önce plandaki altı kilitli dosyanın (ham/temiz
  seçili manifest + referans, karışım manifest + referans) SHA-256'sını
  doğrular. Seçili ham satırların tam envanterin satırlarıyla birebir aynı
  olduğunu ve seçili oturumların bütün kayıtlarını içerdiğini de kontrol eder.
  Hash tutmazsa durur.
- **`audit --selection`:** train tarafı yalnız seçili ham oturumlardır. Dev,
  CSS10 ve test hash'leri tamdır. Çıktı:
  `data/audit/selected_<etiket>/` (şema `p3-data-audit-v2`; decoder, kodlama
  sayıları ve Python/NumPy ortamı yazılır).
- **`clean --selection`:**
  - Aynı kurallarla yeniden hesaplar. Kilitli temiz HParl seçimi ve dev ile
    **yalnız bayt karşılaştırması** yapar; onları, planı ve testi asla yazmaz.
  - Fark çıkarsa farklı ID'leri ve bayraklarını lineage dosyasına yazar ve
    hata verir. Otomatik güncelleme yok.
  - CSS10 ve karışım (kilitli temiz HParl + güncel CSS10, ID sıralı) önceki
    dosyayla bayt-özdeşse aynı yolda kalır. Değiştiyse yeni
    `*.css10audit-v2.jsonl` dosyasına yazılır.
  - Günlük/kuyruk/özet `data/clean/selected_<etiket>/` altına gider. Kabul
    kaydı `configs/css10_mixture_lineage.json`: eski → yeni hash'ler, eski ve
    yeni CSS10 bayrakları, freeze öncesi/sonrası.
- **`verify --selection`:**
  - Ham/temiz seçili HParl, güncel CSS10, güncel karışım ve dev'in hash'lerini
    plan/lineage ile karşılaştırır. Referans hizasını, test ID ayrılığını ve
    train'de dev/test oturumu olmadığını kontrol eder.
  - Temizin ham satırları yeniden kullandığını ve karışımın tam olarak
    HParl + CSS10 olduğunu kontrol eder.
  - Her WAV'ı bir kez açar; hash, başlık, format ve NaN/Inf kontrolü yapar.
  - Freeze'i sesle doğrular.
- **Ses okuma:** `greek_asr/audio.py` RIFF okuyucusu kullanılır, `wave`
  kullanılmaz. Kodlama format tag'den gelir:
  - PCM16 → ÷2^15, PCM32 → ÷2^31.
  - IEEE float32 değerleri olduğu gibi kullanılır. Kazanç veya tepe
    normalizasyonu yok.
  - NaN/Inf, kesik/tutarsız RIFF, desteklenmeyen kodlama ve LFS pointer
    görünür `DataError` verir.
  - `pcm_sha256` data chunk'ının aynı baytlarıdır, yani eski tekrar analiziyle
    uyumludur. Özgün WAV'lar dönüştürülmez.

Eski tam-envanter denetimi (`data/audit/*.json[.gz]`, şema v1) tarihsel kayıttır;
CSS10 sinyal değerleri geçersizdir (`data/audit/README.md`).

## 2. Aşamalar ve çıktılar

| Komut | Okur | Yazar |
|---|---|---|
| `hparl-download` | resmî CLARIN:EL arşivi | `data/downloads/hparl.zip` (gitignore), `data/raw/hparl/ACQUISITION.json`, katalog kaydı |
| `hparl-extract` | ZIP | `data/raw/hparl/release/**`, `archive_members.tsv`, `EXTRACTION.json` |
| `hparl-manifests` | Kaldi `text`, `utt2spk` (+ varsa `segments`), WAV'lar | `data/manifests/raw/hparl_{train,dev,test}.jsonl`, `raw/references/*`, `raw/hparl_inventory.json`, `data/raw/hparl/LAYOUT.json` |
| `freeze-test` | ham test manifesti/referansı, Kaldi test metni | `data/manifests/test.jsonl`, `references/test.jsonl`, `test.freeze.json` |
| `css10-acquire` | Kaggle ZIP (+ halka açık metadata) | `data/raw/css10_el/release/**`, `ACQUISITION.json`, `kaggle_metadata.json` |
| `css10-manifest` | `transcript.txt` | `data/manifests/raw/css10_el.jsonl`, `css10_el_texts.jsonl`, `references/css10_el.jsonl`, envanter |
| `audit` | ham manifestler | `data/audit/items_*.jsonl.gz`, `data/audit/audit_summary.json` |
| `clean` | audit + config + inceleme kararları | `data/manifests/{train,dev,css10_el_train,train_hparl_css10}.jsonl` + referanslar, `conditions.json`, `data/clean/cleaning_log.jsonl`, `review_queue.jsonl`, `clean_summary.json` |
| `report` | yukarıdaki kayıtlar | `docs/DATA_REPORT_TR.md` |
| `verify` | bütün manifestler | yalnız kontrol: sözleşme, ses hash'i, test ayrımı |
| `audit --selection P` | plandaki seçili ham train, dev, CSS10, test hash'leri | `data/audit/selected_<etiket>/` |
| `clean --selection P` | seçili audit + config + kararlar + kilitli dosyalar | `data/clean/selected_<etiket>/`, `configs/css10_mixture_lineage.json`, değişirse `*.css10audit-v2.jsonl` |
| `verify --selection P` | plan + lineage'daki bütün kullanılan manifestler | yalnız kontrol (bütün kullanılan WAV'lar + freeze) |

Clean manifestler ham sesin **kendisini** gösterir. Train/dev için ses
türetilmedi: resampling, kırpma veya gürültü azaltma yok. Bu yüzden "temiz
veri" iki parçadır: korunan kayıtların listesi ve düzeltilmiş eğitim
metinleri. Ham kayıtla ilişki `cleaning_log.jsonl` içinde durur. Ham kaynak
dosyalarının üzerine hiçbir zaman yazılmaz. Var olan farklı baytla karşılaşan
her yazma işlemi hata verir.

Kimlikler `hparl:<özgün_id>` ve `css10_el:<dosya_adı>` biçimindedir. Kaldi'deki
`validation` veya `dev` adı ortak sözleşmedeki `dev` adına eşlenir. Özgün
klasör adı `LAYOUT.json` ve dondurma kaydında korunur.

Oturum kimliği: kaynakta Kaldi `segments` varsa oradan alınır. Yoksa utterance
ID'si `<konuşmacı>-<oturum>_<başlangıç>_<bitiş>_<n>_<m>` desenine uyduğunda ve
konuşmacı öneki `utt2spk` ile aynı olduğunda oturum ID'den türetilir (örn.
`olomeleia_37`). Uymayan kayıt `null` kalır. Yöntem ve başarısız sayısı
envanterdedir.

## 3. Kurallar ve gerekçeleri

Config: `configs/data_cleaning.json` (`policy_version` her logda yazılır).
Bir kayıt birden çok çıkarma kuralına uyarsa, config sırasındaki ilk kural
kaydedilir.

| Kural | Nerede | İşlem | Gerekçe |
|---|---|---|---|
| `source:missing_audio`, `source:audio_unreadable`, `zero_frames`, `duplicate_id` | train, dev, CSS10 | çıkarma (manifeste hiç girmez) | Çözülemeyen veya boş ses sözleşmeye giremez. Tahminle onarılmaz. |
| `source:audio_without_transcript` | train, dev, CSS10 | dahil edilmez | Etiketsiz ses eğitim örneği değildir. |
| `overlap_test_audio` | train, dev, CSS10 | çıkarma | Test sesinin bayt veya PCM kopyası sızıntıdır. Korunan taraf testtir. |
| `overlap_dev_audio` | train | çıkarma | Dev ile seçim yapılacağı için dev kopyası train'den çıkar. |
| `overlap_hparl_audio` | CSS10 | çıkarma | İki kaynak arasında aynı ses. |
| `duplicate_audio` / `duplicate_pcm` | bölüm içinde | ID sırasında ilki kalır, diğerleri çıkar | Aynı ses iki kez öğrenilmesin. `duplicate_pcm` yalnız WAV başlığı farklı kopyadır. |
| `digital_silence` | train, CSS10 | çıkarma | Bütün örnekler sıfır; metni içeremez. |
| `cps_extreme` (>60 karakter/sn) | train, CSS10 | çıkarma (dev'de yalnız işaret) | **Otomatik kalite sezgiseli**, kanıt değil. Saniye başına boşluk dışı karakter train p99'unun (≈20) yaklaşık üç katını aşıyor; bu kayıtlarda birkaç saniyelik sese yüzlerce–binlerce karakter düşüyor, yani hizalama sorunu adayı. Kayıtlar dinlenmedi. |
| `cps_boundary` (40–60 karakter/sn) | train, CSS10, dev | **korunur**, inceleme kuyruğuna girer | Belirsiz bölge: hızlı konuşma ile hizalama hatası ayırt edilemiyor. v2'de bu kayıtlar çıkarılmıştı; v3 onları geri aldı. |
| `duration_long` (>30 sn) | train, CSS10 | çıkarma (dev'de yalnız işaret) | Yaygın fine-tuning pencereleri (ör. Whisper 30 sn) uzun parçayı kırpar ve hedef metinle uyumu bozar. |
| `empty_reference`, `annotation_only` | train, CSS10 | çıkarma (dev'de korunur) | Konuşulmuş kelime içermeyen hedef. |
| `strip_annotation_tags` | yalnız train hedef metni | metin değişikliği | `<spoken_noise>` bir kelime değildir; modelin bunu yazı olarak üretmesi istenmez. |
| CSS10 uyumlaması | CSS10 | metin değişikliği | Bölüm 4. |

Yalnız işaretlenip korunan ve inceleme kuyruğuna giren bayraklar:
`duration_short` (<0,5 sn), `low_level` (RMS < −50 dBFS), `clipping`
(örneklerin >%0,1'i tam ölçekte), `silence_heavy` (konuşma çerçevesi <%20),
`long_edge_silence` (baş/son sessizlik >2 sn), `cps_high` ve `cps_low`
(saniye başına karakterde, train medyan/MAD'ına göre |z| > 5),
`duplicate_suspect` (aynı konuşmacı + aynı metin + ±0,05 sn), `non_greek_characters`,
`format_nonstandard` (16 kHz/mono/16 bit dışı) ve `reported_duration_mismatch`
(CSS10 bildirilen/ölçülen süre).

Konuşma hızı kuralı için not: bir modelin uzun hedefle eğitilip eğitilemeyeceği
(ör. CTC'de hedef uzunluğu ile çerçeve sayısı ilişkisi) model ve tokenizer'a
bağlıdır. Bu kural o konuda genel bir ispat olarak kullanılmaz.

Bu eşikler kaba aday seçicidir. Gerçek veri raporunda dağılım yüzdelikleri
görülür. Bir eşik değiştirilecekse yalnız train/dev dağılımına bakılarak,
gerekçesiyle ve yeni `policy_version` ile değiştirilir. **Test istatistiğine
bakılmaz.**

Dev'de çıkarma yalnız yapısal sorunlar, test kopyaları ve bölüm içi birebir
tekrarlar için yapılır. Dev referansları değiştirilmez, çünkü dev testle aynı
koşulda puanlanmalıdır. `<spoken_noise>` gibi işaretlerin puanlamada nasıl ele
alınacağına Ozan'ın puanlama politikası karar verir.

### Bölüm, konuşmacı ve test cümlesi örtüşmesi

- **Kullanılan bölüm:** HParl'in resmî oturum bazlı bölümü. Konuşmacıdan
  bağımsız değildir. Hocanın mailindeki How 1 seçeneği belgelenmiş resmî
  bölümü kullanmaya izin veriyor. Dondurulmuş test yeniden bölünmez. Sonuçlar
  yeni konuşmacılara genelleme ölçümü olarak sunulmaz.
- **Sayılar:** `python -m greek_asr.data split-summary` (ses gerekmez)
  konuşmacı, örnek, saat ve oturum sayılarını; ortak konuşmacı/oturumları;
  konuşmacısı train'de de olan test örneklerini her bölüm ve eğitim koşulu için
  verir. `clean` aynı içeriği `data/clean/cohort_summary.json` dosyasına yazar.
- **Bilinmeyen kimlikler:** CSS10'da örnek bazında konuşmacı/kayıt kimliği yok.
  Bunlar uydurulmaz, ayrı sütunda bilinmeyen olarak sayılır. Üretici tek okuyucu
  ve tek kitap bildiriyor, bu bilgi raporda kaynağıyla yazılır.
- **Literal örtüşme:** test referans string'i, hiçbir dönüşüm olmadan eğitim
  koşulunun bir hedef string'ine eşitse eşleşmedir. Eşleşen test örneği sayısı
  ve benzersiz metin sayısı ayrı verilir.
- **Normalize örtüşme:** `<...>` silme + NFC + casefold + boşluk birleştirme.
  Ayrı bir ölçüdür, literal yerine sunulmaz.

## 4. CSS10 metin uyumlaması

1. Üreticinin normalize metin alanı alınır (sayılar yazıyla açılmış hali).
   Özgün metin `css10_el_texts.jsonl` içinde değişmeden durur.
2. Unicode NFC uygulanır.
3. Küçük harfe çevirme `auto` ayarlıdır. HParl train referanslarında büyük
   harf oranı ≤ %0,1 ise uygulanır, oran raporda yazar. Python `str.lower`
   Yunanca sonda `ς` kuralını uygular.
4. Harf, işaret (aksan) ve rakam korunur. HParl train temiz referanslarında
   hiç geçmeyen noktalama/simge karakterleri boşluğa çevrilir. Boşluklar
   tekleştirilir.

Her değişen kayıt `before_text` ve `after_text` alanlarıyla günlüğe yazılır.
Uyumlama metni tahminle düzeltmez; yalnız biçimi HParl yazım düzenine çeker.
`--selection` kapsamında "HParl train referansları" seçili temiz HParl
hedefleridir. 20 Kasım koşusunda büyük harf oranı 0, korunan harf dışı karakter
yoktu. Sonuç tam envanter koşusundaki CSS10 metinleriyle bayt-özdeş çıktı.

## 5. İnsan inceleme kararları

`configs/review_decisions.jsonl`, satır başına bir karar:

```json
{"id":"hparl:...","decision":"drop","reviewer":"SmtOrtakaya","date":"2026-11-20","method":"kulaklıkla dinlendi","note":"ses ile metin farklı cümle"}
{"id":"hparl:...","decision":"correct_text","corrected_text":"...","reviewer":"SmtOrtakaya","date":"2026-11-20","method":"kulaklıkla iki kez dinlendi","note":"eksik kelime"}
{"id":"hparl:...","decision":"keep","reviewer":"SmtOrtakaya","date":"2026-11-20","method":"kulaklıkla dinlendi","note":"sessizlik normal"}
```

`reviewer`, `date` ve `method` zorunludur. Test ID'sine karar yazılırsa
`clean` hata verir. Duyarak güvenilir biçimde doğrulanamayan Yunanca metin
`correct_text` ile düzeltilmez. Model çıktısı karar kanıtı sayılmaz.

## 6. Testin dondurulması

`test.freeze.json` şunları kaydeder:

- Kaynak sürüm ve arşiv SHA-256'sı
- Kaldi test klasörü ve metadata dosyalarının hash'leri
- Kayıt sayısı ve toplam süre
- Sıralı ID listesinin hash'i
- `test.jsonl` ve `references/test.jsonl` hash'leri
- Kayıt bazında ses yolu, bayt sayısı ve SHA-256
- Referans metni ve özgün Kaldi satırının hash'i
- Konuşmacı ve oturum
- Dondurma zamanı

`verify-freeze` bunların hepsini yeniden hesaplar.

Güvenceler:

- Dondurma dosyası bir kez yazılır. Farklı içerikle yeniden üretilmeye
  çalışılırsa hata verir.
- Test referansı Kaldi satırından aynen alınır, normalize edilmez.
- CSS10 ve train/dev işlemleri test dosyalarını yazmaz.
- `clean` önce ve sonra dondurmayı doğrular; fark varsa durur.
- Testlerde ses baytı veya referans değiştirildiğinde doğrulamanın başarısız
  olduğu sınanır.

Bu erken **veri dondurması**, model sahiplerinin test çıkarımından önce
yazacağı `docs/FINAL_PROTOCOL.json` ile aynı şey değildir. O protokolde
gereken `test_manifest_sha256` değeri dondurma kaydında hazırdır.

## 7. Ham/temiz verinin repoda teslimi

Hoca ham ve temiz verinin public repoda olmasını istiyor. Bağlantı, Release
dosyası veya indirme script'i tek başına teslim sayılmaz.

**Gerçek boyutlar:** `python -m greek_asr.data delivery-plan`. Bu komut
kaydedilmiş arşiv listelerinden hesaplar, ses gerekmez. Rapordaki tablo da
aynı fonksiyondan üretilir.

**Karar kaydı:** `configs/data_delivery.json`. Her grup için yöntem ve durum
(`in_repo`, `pending`, `blocked`) yazar. Bir durum yalnız push'lanmış repoda
gerçekten doğruysa değiştirilir.

**Teslim yolları:**

- **Metin kaynakları (Kaldi `text`, `utt2spk`, `spk2utt`, `wav.scp`,
  `spk2gender` ve CSS10 `transcript.txt`):** düz Git ile teslim edilir.
  `.gitattributes` içindeki `-text` kuralı satır sonu dönüşümünü kapatır.
  Temiz clone'da `python -m greek_asr.data verify-source-metadata` her dosyanın
  bayt hash'ini kayıtlarla karşılaştırır.
- **Ses:** `.gitattributes` kaynak WAV'larını Git LFS'e yönlendirir (yalnız
  eklenen dosyalara uygulanır). GitHub Free'de LFS için 10 GiB depolama ve
  ayda 10 GiB indirme var. İkisi de repo sahibinin hesabından düşer.
  **Ücretli paket alınmaz.**
  - Test + dev + kullanılan CSS10 kopyası kotaya sığar.
  - Tam train eklenince kota aşılır.
  - Bu yüzden train sesinden yalnız eğitimde gerçekten kullanılacak alt küme
    teslim edilecek. Alt küme Serdar'ın ölçtüğü GPU bütçesine göre
    `python -m greek_asr.data subset --hours <saat> --seed <seed>` ile, bütün
    oturumlar halinde ve kayıtlı seed ile seçilir. O zamana kadar durum
    `pending`.
- **Ses değişmediği için** ham ve temiz manifestler aynı baytları paylaşır;
  ikinci bir ses kopyası üretilmez. CSS10 arşivindeki bayt bayt aynı ikinci
  kopya teslim edilmez, `archive_members.tsv` içinde listelidir.

LFS sesi eklendikten sonra temiz bir clone'da gerçek dosyalarla doğrulama:

```bash
git lfs pull
python -m greek_asr.data verify-source-metadata
python -m greek_asr.data verify-freeze
python -m greek_asr.data verify --only dev.jsonl test.jsonl css10_el_train.jsonl
python -m greek_asr.data verify --selection configs/training_data_selection.json   # seçili train sesi de gelince
```

LFS pointer dosyası ses sanılmaz, pipeline pointer görünce "Git LFS pointer"
hatası verir. `pack`/`unpack` komutları hash kontrollü parça üretir. Bunlar
yalnız taşıma aracıdır, tek başına "veri repoda" şartını karşılamaz.

## 8. Testler

`python -m unittest discover -s tests -v`. Sentetik fixture
(`tests/data_fixtures.py`) gerçek arşivin klasör düzenini taklit eder ve
bilinen sorunlar içerir: eksik/bozuk ses, etiketsiz ses, bayt ve PCM tekrarı,
test ve dev kopyası, 31 sn parça, sıfır ses, kırpılmış ses, `<spoken_noise>`,
CSS10 tekrarı ve noktalama.

Sınananlar:

- Ham kaydın ID, transkript, hash, süre ve Yunanca metnini koruması
- Kaynak sorunlarının görünür kalması
- Her sorunun beklenen kurala gitmesi ve sağlam kaydın değişmemesi
- Sayı ve süre toplamlarının günlükle tutması
- CSS10'un yalnız train'e girmesi
- Test ID kümesinin resmî kümeyle tam eşitliği
- Test sesi veya referansı değişince doğrulamanın başarısız olması
- Dondurmanın yeniden yazılamaması
- Testte eksik ses varsa dondurmanın durması
- Aynı girdiyle yeniden çalıştırmanın bayt bayt aynı çıktı vermesi
- HParl-only ve karışık koşulun yalnız CSS10 kadar farklı olması
- Paylaşılan `greek_asr.infer` runner'ının bütün manifestleri kabul etmesi
- Zip-slip ve link reddi
- İndirmenin kaldığı yerden devam etmesi ve Range'e uymayan sunucunun reddi
- Release parçalarının gidiş-dönüşü ve bozulma tespiti
- Literal örtüşmede aynı cümleyi taşıyan iki test örneğinin tek benzersiz metin
  sayılması; etiket ve büyük/küçük harf farkının literal eşleşme sayılmaması
- Konuşmacı/oturum özetinde bilinmeyen CSS10 kimliklerinin ayrıca sayılması
- 40–60 karakter/sn sınır kaydının korunup inceleme kuyruğuna girmesi
- Kaynak metin dosyasında satır sonu değişince `verify-source-metadata`
  kontrolünün başarısız olması

Fixture'daki CSS10 klipleri gerçek veri gibi IEEE float32'dir. 20 Kasım'da
eklenen testler:

- `tests/test_data_audio.py`:
  - aynı 4 bayt genişlikte float32 ile integer PCM32'nin format tag'le ayrılması
  - bilinen genlikte RMS/dBFS/tepe/kırpılma/sessizlik değerleri; kazanç yok
  - PCM16/PCM32 desteği
  - Python 3.12'de `wave` olmadan CSS10 okuma
  - NaN/Inf, kesik WAV, LFS pointer ve desteklenmeyen kodlama hataları
- `tests/test_data_selection.py`:
  - seçili zincirin seçilmeyen train sesi olmadan çalışması
  - kilitli HParl/dev'in bayt-özdeş üretilmesi
  - aynı girdiyle bayt-özdeş yeniden çalışma
  - CSS10 değişince sürümlü dosya ve eski → yeni hash
  - oturumlar arası tekrar yüzünden HParl farkının yazılmadan raporlanması
  - bozuk ses/pointer'da `verify` hatası
  - kurcalanmış kilitli dosyanın reddi
- `tests/test_data_locked_repo.py`:
  - gerçek repodaki kilitli seçim, dev, test ve freeze hash'lerinin değişmemesi
  - lineage tutarlılığı
  - gerçek CSS10 sesi varsa 1.844 dosyanın float32 başlık ve hash kontrolü

## 9. Açık konular

- **Seçili train sesi:** 23.615 ham WAV'ın LFS yüklemesini repo sahibi
  yürütüyor. Yüklendikten sonra temiz clone'da
  `verify --selection configs/training_data_selection.json` çalıştırılmalı.
- **İnceleme kuyruğu:** hiçbir kayıt dinlenmedi, insan inceleme kararı sıfır.
  Güncel kuyruk: `data/clean/selected_train_hparl_30.5h_seed42/review_queue.jsonl`.
  Dinlenen her karar `configs/review_decisions.jsonl` dosyasına yazılır.
- **Karışım politikası:** oran ve adım politikası model sahiplerinin kararıdır.
- **Puanlama:** `<spoken_noise>` ve diğer normalizasyon kararları ortak
  puanlama politikasındadır. Dondurulmuş dev/test referansları değiştirilmez.
