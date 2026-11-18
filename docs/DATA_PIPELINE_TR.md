# Veri pipeline'ı — HParl + CSS10 Greek (Samed)

Bu belge veri hazırlama kodunun ne yaptığını, kuralların gerekçesini ve gerçek
verinin nasıl çalıştırılacağını anlatır. Kod `greek_asr/data/` altında,
testler `tests/test_data_pipeline.py` içindedir. Gerçek veri sayıları
`docs/DATA_REPORT_TR.md` dosyasına koddan yazılır. Bu belgede sonuç rakamı
yoktur.

Durum: kod ve sentetik fixture üzerindeki uçtan uca testler hazır. Tam HParl
indirme, test dondurma ve temizlik **henüz gerçek veride çalıştırılmadı**.
Aşağıdaki adımlar Samed'in bilgisayarında yapılacak.

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
4. `python -m greek_asr.data publish-plan` ile teslim boyutunu ölç (bölüm 7).
5. Temizlik commit'i, kendi katkı dosyan ve PR açıklaması.

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
| `cps_implausible` (>40 karakter/sn) | train, CSS10 | çıkarma (dev'de yalnız işaret) | Fiziksel olarak imkânsız konuşma hızı: transkript sesin çok ötesinde (birkaç saniyelik sese yüzlerce–binlerce karakter). Eşik train p99'unun (20 kar/sn) iki katı. CTC'de etiket çerçeveden uzun olunca kayıp tanımsızdır. |
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

Bu eşikler kaba aday seçicidir. Gerçek veri raporunda dağılım yüzdelikleri
görülür. Bir eşik değiştirilecekse yalnız train/dev dağılımına bakılarak,
gerekçesiyle ve yeni `policy_version` ile değiştirilir. **Test istatistiğine
bakılmaz.**

Dev'de çıkarma yalnız yapısal sorunlar, test kopyaları ve bölüm içi birebir
tekrarlar için yapılır. Dev referansları değiştirilmez, çünkü dev testle aynı
koşulda puanlanmalıdır. `<spoken_noise>` gibi işaretlerin puanlamada nasıl ele
alınacağına Ozan'ın puanlama politikası karar verir.

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

Hoca ham ve temiz verinin public repoda olmasını istiyor. Bağlantı veya
script tek başına yeterli değil. `python -m greek_asr.data publish-plan --quota-gib 10`
gerçek boyutları bölüm bölüm verir.

Kaynak bildirimine göre HParl'in 16 kHz/16 bit sesi yaklaşık 13,8 GB tutar.
Train ~11,5 GB, dev ~1,0 GB, test ~1,3 GB. Bu rakamlar ölçüm değildir;
`publish-plan` gerçek değeri verir. GitHub Free hesaplarda Git LFS için
10 GiB depolama ve ayda 10 GiB indirme dahildir. İndirme kotası her clone ve
pull ile repo sahibinin hesabından düşer. Güncel kota Serdar'ın hesabında
Settings → Billing altında görülür. **Ücretli paket alınmaz.**

Serdar'la seçilecek iki uygulanabilir yol:

1. **Test, dev, CSS10 ve metin dosyaları Git LFS'te; HParl train sesi GitHub
   Release dosyalarında.**

   ```bash
   git lfs install
   git lfs track "data/raw/**/*.wav"            # .gitattributes'a yazar; commit'e ekle
   echo "data/raw/hparl/release/train_wavs/" >> .git/info/exclude   # yerel; train sesi yanlışlıkla eklenmesin
   python -m greek_asr.data pack --source data/raw/hparl/release/train_wavs --output ../p3-release-shards
   git add .gitattributes data/raw/hparl/release data/raw/css10_el data/release_index
   ```

   `pack` her parçayı 1,9 GB altında tutar ve `data/release_index/*.json`
   indeksini repoya yazar. Parçalar Serdar tarafından repo Release'ine
   yüklenir. İndiren kişi `python -m greek_asr.data unpack --index ... --shards ...`
   ile hash kontrollü açar. Gerçek klasör adları `LAYOUT.json` içindedir;
   `train_wavs` pilot kaydında görülen addır.
2. **Kota içinde kalan alt küme**: oturum bazlı `subset` ile seçilmiş train
   alt kümesi ve dev/test LFS'te. Bu yol eğitim verisini küçültür. Gerekçesi
   donanım bütçesiyle yazılmalıdır.

Her iki yolda da LFS indirme kotasını korumak için ekip üyeleri sesi
gerekmedikçe `GIT_LFS_SKIP_SMUDGE=1 git clone ...` ile klonlayabilir. Kendi
makinesinde sesi aynı arşivden `prepare-freeze` ile yeniden açan herkes aynı
hash'leri elde eder. `verify-freeze` bunu kanıtlar.

Temiz bir clone'da gerçek ses baytlarını doğrulama:

```bash
git lfs pull && python -m greek_asr.data verify
```

LFS pointer dosyası ses sanılmaz. Pipeline pointer'ı görünce
"Git LFS pointer" hatası verir.

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

## 9. Açık konular

- Gerçek veri henüz çalıştırılmadı; bütün sayılar `DATA_REPORT_TR.md`
  üretildiğinde ortaya çıkacak.
- Teslim yöntemi (bölüm 7) Serdar'la kararlaştırılacak.
- İnceleme kuyruğunun ne kadarının dinleneceği, gerçek kuyruğun büyüklüğüne
  göre belirlenecek. Dinlenen her karar kaydedilecek.
- Eğitim bütçesi ve alt küme ihtiyacı Serdar'ın ölçümüne bağlı.
- Karışım oranı ve adım politikası Adnan/Serdar'ın kararıdır.
