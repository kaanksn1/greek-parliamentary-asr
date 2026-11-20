"""Generate docs/DATA_REPORT_TR.md from recorded pipeline outputs only.

Every number in the report is read from acquisition, inventory, freeze,
audit, cleaning and conditions files produced by the pipeline. Nothing is
typed by hand, so the report can be regenerated and checked.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .cohort import cohort_summary
from .common import ROOT, TAG_PATTERN, DataError, hms, read_json, read_jsonl, sha256_file, write_output
from .publish import delivery_plan
from .selection import LINEAGE

REPORTED = {"train": "99:31:41", "dev": "09:03:33", "test": "11:12:28"}
CSS10_REPORTED = "04:08:14"
SUBSET_LABEL = {"hparl_train": "HParl train", "hparl_dev": "HParl dev", "css10_el": "CSS10 Greek (yalnız train)"}


def _seconds(text: str) -> int:
    h, m, s = (int(x) for x in text.split(":"))
    return h * 3600 + m * 60 + s


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} sa ({hms(seconds)})"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines + [""]


def _short(text: str | None, limit: int = 90) -> str:
    if text is None:
        return "—"
    text = text.replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _source_sections(root: Path, inventory: dict, freeze: dict, freeze_check: tuple[str, str]) -> list[str]:
    acquisition = root / "data/raw/hparl/ACQUISITION.json"
    hparl_acq = read_json(acquisition) if acquisition.exists() else {}
    css_acq_path, css_inv_path = root / "data/raw/css10_el/ACQUISITION.json", root / "data/manifests/raw/css10_el_inventory.json"
    css_acq = read_json(css_acq_path) if css_acq_path.exists() else None
    css_inv = read_json(css_inv_path) if css_inv_path.exists() else None
    out = []
    out += ["## 1. Kaynaklar, sürümler ve lisanslar", ""]
    rows = [["HParl", inventory["dataset"]["version"], inventory["dataset"]["license"],
             f"`{hparl_acq.get('archive_sha256', '—')}`", hparl_acq.get("archive_bytes", "—"),
             hparl_acq.get("downloaded_utc", "—")]]
    if css_acq:
        meta = css_acq.get("kaggle_metadata_fields") or {}
        rows.append(["CSS10 Greek (Kaggle)", meta.get("currentVersionNumber", "kayıt yok"),
                     meta.get("licenseName", css_acq["dataset"]["license_declared_by_distributor"]),
                     f"`{css_acq['archive_sha256']}`", css_acq["archive_bytes"], css_acq["acquired_utc"]])
    out += _table(["Kaynak", "Sürüm", "Lisans", "Arşiv SHA-256", "Bayt", "Edinme (UTC)"], rows)
    out += ["HParl transkriptleri üreticinin insan transkriptleridir; CSS10 metinleri üreticinin kitap metniyle "
            "insan eliyle eşleştirmesidir. Bu projede yeni insan anotasyonu yapılmadı (aşağıdaki insan inceleme "
            "kararları hariç). Lisans ayrıntısı: `DATA_LICENSE.md`.", ""]

    out += ["## 2. Testin erken dondurulması", ""]
    out += _table(["Alan", "Değer"], [
        ["Dondurma zamanı (UTC)", freeze["created_utc"]], ["Kayıt sayısı", freeze["record_count"]],
        ["Toplam süre", _hours(freeze["total_duration_seconds"])],
        ["Konuşmacı / oturum", f"{freeze['speakers']} / {freeze['recordings_or_sessions']}"],
        ["ID listesi SHA-256", f"`{freeze['ids_sha256']}`"],
        ["`test.jsonl` SHA-256", f"`{freeze['test_manifest_sha256']}`"],
        ["`references/test.jsonl` SHA-256", f"`{freeze['references_sha256']}`"],
        ["`test.freeze.json` SHA-256", f"`{sha256_file(root / "data/manifests/test.freeze.json")}`"],
        ["Temizlik öncesi/sonrası doğrulama", f"{freeze_check[0]} / {freeze_check[1]}"],
        ["Kaynakta not edilen (değiştirilmeyen) durumlar", len(freeze["source_notes"])]])
    out += ["Test sesleri ve özgün transkript satırları değiştirilmedi; kayıt eklenmedi/çıkarılmadı, yeniden "
            "bölünmedi, normalize edilmedi. CSS10 teste girmedi. `python -m greek_asr.data verify-freeze` her "
            "hash'i yeniden hesaplar; temizlik komutu bu kontrolü başta ve sonda çalıştırır.", ""]

    out += ["## 3. Ham veri envanteri (yerel ölçüm)", ""]
    rows = []
    for split in ("train", "dev", "test"):
        data = inventory["splits"][split]
        diff = data["audio_seconds"] - _seconds(REPORTED[split])
        rows.append([split, data["transcript_lines"], data["manifest_records"], _hours(data["audio_seconds"]),
                     REPORTED[split], f"{diff:+.0f} sn", data["speakers"], data["recordings_or_sessions"],
                     ", ".join(f"{k}: {v}" for k, v in data["issue_counts"].items()) or "yok"])
    if css_inv:
        diff = css_inv["audio_seconds"] - _seconds(CSS10_REPORTED)
        rows.append(["CSS10", css_inv["transcript_lines"], css_inv["manifest_records"], _hours(css_inv["audio_seconds"]),
                     CSS10_REPORTED, f"{diff:+.0f} sn", "1 (üretici bildirimi)", "—",
                     ", ".join(f"{k}: {v}" for k, v in css_inv["issue_counts"].items()) or "yok"])
    out += _table(["Bölüm", "Transkript satırı", "Sesli kayıt", "Ölçülen süre", "Üretici bildirimi", "Fark",
                   "Konuşmacı", "Oturum", "Kaynak sorunları"], rows)
    method = inventory["splits"]["train"]["recording_id_method"]
    out += [f"Süre yöntemi: her WAV başlığındaki `num_frames / sampling_rate`. Oturum kimliği yöntemi: `{method}`. "
            "Fark sütunu yerel ölçüm ile üretici bildirimi arasındaki farktır; açıklaması aşağıdaki sorunlar ve "
            "yuvarlama ile birlikte okunmalıdır.", ""]
    if css_inv:
        out += [f"CSS10 örnekleme hızları: {css_inv['sampling_rates']}; arşivde transkriptte geçmeyen "
                f"{css_inv['unreferenced_wav_files']} WAV dosyası var (kullanılmadı, silinmedi).", ""]

    return out


def build_report(root: Path = ROOT) -> str:
    root = Path(root)
    if (root / LINEAGE).exists():
        from .report_selected import build_selected_report

        return build_selected_report(root, _source_sections)
    need = {"inventory": root / "data/manifests/raw/hparl_inventory.json",
            "freeze": root / "data/manifests/test.freeze.json", "audit": root / "data/audit/audit_summary.json",
            "clean": root / "data/clean/clean_summary.json", "conditions": root / "data/manifests/conditions.json"}
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        raise DataError(f"report inputs missing: {missing}; run the pipeline first")
    inventory, freeze, audit_summary, clean_summary, conditions = (read_json(need[k]) for k in
                                                                  ("inventory", "freeze", "audit", "clean", "conditions"))
    log = read_jsonl(root / "data/clean/cleaning_log.jsonl")
    stats = clean_summary["subsets"]
    out = ["# Veri raporu — HParl + CSS10 Greek (Samed, veri sorumlusu)", "",
           "Bu dosya `python -m greek_asr.data report` komutuyla pipeline çıktılarından üretilir; içindeki "
           "sayılar elle yazılmamıştır. Model başarımı veya bonus iddiası içermez.", ""]

    out += _source_sections(root, inventory, freeze, (clean_summary["test_freeze_before"]["status"],
                                                      clean_summary["test_freeze_after"]["status"]))

    out += ["## 4. Denetim bulguları (train/dev/CSS10)", "",
            "Bayraklar adaydır; tek başına hatalı veri kanıtı değildir. Bir kayıt birden çok bayrak taşıyabilir.", ""]
    flag_names = sorted({f for s in audit_summary["subsets"].values() for f in s["flag_counts"]})
    rows = []
    for name in flag_names:
        row = [f"`{name}`"]
        for subset in audit_summary["subsets"]:
            data = audit_summary["subsets"][subset]
            count = data["flag_counts"].get(name, 0)
            row.append(f"{count} ({data['flag_seconds'].get(name, 0) / 60:.1f} dk)" if count else "0")
        rows.append(row)
    out += _table(["Bayrak"] + [SUBSET_LABEL[s] for s in audit_summary["subsets"]], rows)
    dup = audit_summary["duplicates"]
    out += [f"Aynı baytlı ses grupları: {dup['bytes']['groups']} grup / {dup['bytes']['records_in_groups']} kayıt "
            f"(test dahil karşılaştırma). Aynı PCM içerikli gruplar: {dup['pcm']['groups']}. Yöntem SHA-256 ile birebir "
            "tekrarları ve yalnız WAV başlığı farklı kopyaları yakalar; yeniden kodlanmış, kesilmiş veya ses düzeyi "
            "değişmiş tekrarları yakalamaz. Bunlar için aynı konuşmacı + aynı metin + ±tolerans süre eşleşmesi "
            "`duplicate_suspect` olarak işaretlenir ve dinlemeye bırakılır.", ""]
    tag_rows = []
    for split in ("dev", "test"):
        refs = [r["reference"] for r in read_jsonl(root / f"data/manifests/references/{split}.jsonl")]
        tagged = sum(1 for r in refs if TAG_PATTERN.search(r))
        tags = sum(len(TAG_PATTERN.findall(r)) for r in refs)
        tag_rows.append(f"{split}: {tagged}/{len(refs)} referans (%{100 * tagged / max(1, len(refs)):.1f}), {tags} işaret")
    out += ["**Puanlama için önemli:** kaynak referanslarında `<spoken_noise>` gibi işaretler sık geçiyor (yalnız okundu, "
            "değiştirilmedi): " + "; ".join(tag_rows) + ". Train hedeflerinden çıkarıldıkları için modeller bu işaretleri "
            "üretmeyecek. İşaretler puanlamada referansta bırakılırsa her biri silme hatası sayılır ve WER yapay olarak "
            "yükselir. Bu işaretlerin referans ve hipotezde aynı kuralla ele alınması Ozan'ın puanlama politikasında, "
            "dev üzerinde ve test dosyalarına yazılmadan kararlaştırılmalıdır.", ""]
    cohort = cohort_summary(root)
    out += ["## 4b. Bölüm, konuşmacı ve test cümlesi örtüşmesi", "",
            "Kullanılan bölüm: **HParl 1.0.0'ın resmî oturum bazlı train/dev/test bölümü**. Bu bölüm konuşmacıdan "
            "bağımsız (speaker-disjoint) **değildir**: aynı milletvekilleri farklı oturumlarda birden çok bölümde "
            "konuşuyor. Hocanın mailindeki How 1 seçeneği belgelenmiş resmî bölümün kullanılmasına izin veriyor; "
            "dondurulmuş test bu nedenle yeniden bölünmedi. Sonuçlar yeni konuşmacılara genelleme ölçümü olarak "
            "sunulmamalıdır; ölçülen şey, test konuşmacılarının train'de de bulunabildiği yeni oturumlardaki başarımdır "
            "(ortak konuşmacı sayıları aşağıda). Sayılar manifestlerden koddan hesaplanır (ses okunmaz).", ""]
    rows = []
    labels = {"train": "ham HParl train", "dev": "HParl dev", "test": "HParl test (dondurulmuş)",
              "clean_dev": "temiz dev (kullanılan)"}
    for name, data in cohort["splits"].items():
        rows.append([labels.get(name, name), data["records"], f"{data['hours']:.3f}", data["known_speakers"],
                     data["records_without_speaker_id"], data["known_sessions_or_recordings"]])
    for name, data in cohort["training_conditions"].items():
        if name == "raw_hparl_train":
            continue
        rows.append([f"eğitim koşulu `{name}`", data["records"], f"{data['hours']:.3f}", data["known_speakers"],
                     data["records_without_speaker_id"], data["known_sessions_or_recordings"]])
    out += _table(["Küme", "Örnek", "Saat", "Bilinen konuşmacı", "Konuşmacı ID'si boş örnek", "Oturum"], rows)
    out += ["Konuşmacı ID'si boş örnekler bilinen konuşmacı sayısına katılmaz; sayıları ayrı sütunda yazılır. "
            + cohort["css10_identity_note"], ""]
    pair_rows = [[pair.replace("&", " ∩ "), data["speakers"], data["sessions_or_recordings"]]
                 for pair, data in cohort["pairwise_overlaps"].items()]
    out += _table(["Bölüm çifti", "Ortak konuşmacı", "Ortak oturum"], pair_rows)
    raw_train = cohort["training_conditions"].get("raw_hparl_train", {})
    if raw_train:
        out += [f"Testteki {cohort['test']['records']} örneğin "
                f"{raw_train['test_examples_whose_speaker_is_in_this_training']} tanesinin konuşmacısı train'de de var "
                f"({raw_train['test_speakers_also_in_this_training']} ortak konuşmacı).", ""]
    out += ["**Test cümlelerinin eğitimde birebir bulunması.** Literal eşitlik: dondurulmuş test referans "
            "string'i hiçbir dönüşüm uygulanmadan (etiket silme, büyük/küçük harf, Unicode veya boşluk düzeltmesi "
            "yok) eğitim koşulunun en az bir hedef string'ine eşitse eşleşme sayılır. Aynı cümle birden çok test "
            "örneğinde geçebildiği için örnek sayısı ve benzersiz metin sayısı ayrı verilir.", ""]
    rows = []
    for name, data in cohort["training_conditions"].items():
        literal, normalized = data["literal_test_overlap"], data["normalized_test_overlap"]
        rows.append([f"`{name}`", literal["matched_test_examples"], literal["matched_distinct_test_texts"],
                     normalized["matched_test_examples"], normalized["matched_distinct_normalized_texts"],
                     f"`{data['references']['path']}` `{data['references']['sha256'][:16]}…`"])
    out += _table(["Eğitim hedefi", "Literal eşleşen test örneği", "Literal benzersiz metin",
                   "Normalize eşleşen test örneği", "Normalize benzersiz metin", "Karşılaştırılan referans dosyası"],
                  rows)
    out += [f"Test referansları: `{cohort['test']['references']['path']}` "
            f"`{cohort['test']['references']['sha256']}` ({cohort['test']['records']} örnek, "
            f"{cohort['test']['distinct_reference_texts']} benzersiz metin). Normalize sütunlar ayrı bir ölçüdür: iki "
            "tarafta da `<...>` işaretleri silinir, Unicode NFC, casefold ve boşluk birleştirme uygulanır; literal "
            "eşitlik yerine sunulmaz. Hiçbir eşleşen cümle testten silinmedi; dondurulmuş test değişmedi. "
            "Makine okunur kayıt: `data/clean/cohort_summary.json`.", ""]
    rows = [[name, *(audit_summary["overlaps"][name].get(pair, "—") for pair in
                     ("hparl_train&hparl_dev", "hparl_train&hparl_test", "hparl_dev&hparl_test"))]
            for name in ("original_ids", "distinct_texts_after_tag_strip_nfc_casefold_space")
            if name in audit_summary["overlaps"]]
    out += ["Denetim adımındaki ek kontroller (ham bölümler arası; `distinct_texts_after_tag_strip_nfc_casefold_space` "
            "normalize benzersiz metin kesişimidir, literal değildir):", ""]
    out += _table(["Tür", "train∩dev", "train∩test", "dev∩test"], rows)

    out += ["## 5. Temizlik sonucu", "",
            "Bir kayıt birden çok çıkarma kuralına uyuyorsa yalnız config'teki ilk kurala yazılır (çifte sayım yok).", ""]
    rows = []
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        if subset not in stats:
            continue
        s = stats[subset]
        rows.append([SUBSET_LABEL[subset], s["raw_records"], _hours(s["raw_seconds"]), s["kept_records"],
                     _hours(s["kept_seconds"]), s["raw_records"] - s["kept_records"],
                     f"{(s['raw_seconds'] - s['kept_seconds']) / 60:.1f} dk", s["modified_records"]])
    out += _table(["Bölüm", "Ham kayıt", "Ham süre", "Temiz kayıt", "Temiz süre", "Çıkarılan", "Çıkarılan süre",
                   "Metni değişen"], rows)
    rows = []
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        if subset not in stats:
            continue
        for rule, count in stats[subset]["dropped_by_rule"].items():
            rows.append([SUBSET_LABEL[subset], "çıkarma", f"`{rule}`", count,
                         f"{stats[subset]['dropped_seconds_by_rule'][rule] / 60:.1f} dk"])
        for step, count in stats[subset]["modification_steps"].items():
            rows.append([SUBSET_LABEL[subset], "metin", f"`{step}`", count, "—"])
        for flag, count in stats[subset]["review_queue_by_flag"].items():
            rows.append([SUBSET_LABEL[subset], "inceleme (korundu)", f"`{flag}`", count, "—"])
    source_drops = defaultdict(int)
    for record in log:
        if record["rule"].startswith("source:"):
            source_drops[(record["subset"], record["action"], record["rule"])] += 1
    for (subset, action, rule), count in sorted(source_drops.items()):
        rows.append([SUBSET_LABEL.get(subset, subset), "kaynakta kullanılamaz" if action == "drop" else "etiketsiz ses",
                     f"`{rule}`", count, "—"])
    out += _table(["Bölüm", "İşlem", "Kural", "Kayıt", "Süre"], rows or [["—", "—", "—", 0, "—"]])
    out += [f"Günlük: `data/clean/cleaning_log.jsonl` ({sum(clean_summary['log_entries'].values())} satır, "
            f"SHA-256 `{clean_summary['log_sha256']}`). İnceleme kuyruğu: `data/clean/review_queue.jsonl` "
            f"({clean_summary['review_queue_records']} kayıt). Uygulanan insan inceleme kararı: "
            f"{clean_summary['review_decisions_applied']}.", ""]
    harmonization = stats.get("css10_harmonization")
    if harmonization:
        out += [f"CSS10 metin uyumlaması: üreticinin normalize metni (sayılar yazıyla) alındı, NFC uygulandı, "
                f"küçük harfe çevirme = {harmonization['lowercase']} (HParl train büyük harf oranı "
                f"{harmonization['hparl_train_uppercase_ratio']}), HParl train referanslarında hiç geçmeyen "
                f"harf dışı karakterler boşluğa çevrildi. HParl train'de korunan harf dışı karakterler: "
                f"`{harmonization['allowed_non_letter_characters'] or 'yok'}`. Özgün ve normalize metin "
                "`data/manifests/raw/css10_el_texts.jsonl` içinde değişmeden durur.", ""]

    config = read_json(root / "configs/data_cleaning.json")
    flags_cfg = config["flags"]
    rate_rows = []
    for record in read_jsonl(root / "data/clean/review_queue.jsonl"):
        if "cps_boundary" in record["flags"]:
            evidence = record["flags"]["cps_boundary"]
            rate_rows.append([SUBSET_LABEL.get(record["subset"], record["subset"]), f"`{record['id']}`",
                              f"{evidence['chars_per_second']:.2f}", f"{evidence['duration_seconds']:.2f}",
                              evidence["chars"], "korundu, inceleme bekliyor"])
    extreme = {subset: data["dropped_by_rule"].get("cps_extreme", 0) for subset, data in stats.items()
               if isinstance(data, dict) and "dropped_by_rule" in data}
    out += ["### Konuşma hızı kuralı (otomatik kalite sezgiseli)", "",
            f"Saniye başına boşluk dışı karakter > {flags_cfg['cps_extreme_min']:g} olan train/CSS10 kayıtları "
            "hizalama sorunu **adayı** olarak eğitimden çıkarılır (train p99 değerinin yaklaşık üç katı; bu "
            "kayıtlarda birkaç saniyelik sese yüzlerce–binlerce karakter düşüyor). Bu bir sezgiseldir: kayıtlar "
            "dinlenmedi, etiketlerin yanlış olduğu kanıtlanmadı. "
            f"{flags_cfg['cps_boundary_min']:g}–{flags_cfg['cps_extreme_min']:g} aralığı belirsiz kabul edilir; "
            "bu kayıtlar eğitimde **korunur** ve inceleme kuyruğuna yazılır. Dev'de hiçbir kayıt bu kuralla "
            "çıkarılmaz. Bir modelin bu hedeflerle eğitilip eğitilemeyeceği model/tokenizer'a bağlıdır; kural bunun "
            "genel ispatı olarak sunulmaz. Çıkarılan: "
            + ", ".join(f"{SUBSET_LABEL.get(k, k)} {v}" for k, v in extreme.items()) + ".", ""]
    if rate_rows:
        out += _table(["Bölüm", "Sınır kaydı (korundu)", "kar/sn", "Süre (sn)", "Karakter", "Durum"], rate_rows)

    out += ["## 6. Gerçek örnekler (günlükten)", ""]
    seen = defaultdict(int)
    rows = []
    for record in log:
        key = (record["subset"], record["action"], record["rule"])
        if record["action"] not in ("drop", "modify_text") or seen[key] >= 2:
            continue
        seen[key] += 1
        evidence = record["evidence"]
        rows.append([SUBSET_LABEL.get(record["subset"], record["subset"]), record["action"], f"`{record['rule']}`",
                     f"`{record['id']}`", _short(record.get("before_text")), _short(record.get("after_text")),
                     _short(str(evidence) if evidence is not None else None, 60)])
    out += _table(["Bölüm", "İşlem", "Kural", "ID", "Önce", "Sonra", "Kanıt"], rows or [["—"] * 7])
    out += ["Önce/sonra metin yalnız metni değişen kayıtlarda günlükte tutulur; çıkarılan kayıtlarda metin hash'i "
            "ve ses hash'i vardır. Dinleyerek doğrulanmayan Yunanca metin tahminle düzeltilmedi.", ""]

    out += ["## 7. Model sahiplerine teslim edilen koşullar", ""]
    def fmt(mapping):
        return ", ".join(f"{k}: {v}" for k, v in mapping.items())

    rows = []
    for name in ("hparl_only", "hparl_css10"):
        data = conditions["conditions"][name]
        rows.append([name, f"`{data['train_manifest']}`", fmt(data["records"]), _hours(data["total_seconds"]),
                     fmt(data["sampling_rates"]), f"`{data['manifest_sha256'][:16]}…`"])
    dev = conditions["shared_dev"]
    rows.append(["ortak dev", f"`{dev['manifest']}`", fmt(dev["records"]), _hours(dev["total_seconds"]),
                 fmt(dev["sampling_rates"]), f"`{dev['manifest_sha256'][:16]}…`"])
    test = conditions["frozen_test"]
    rows.append(["dondurulmuş test", f"`{test['manifest']}`", test["records"], _hours(freeze["total_duration_seconds"]),
                 "—", f"`{test['manifest_sha256'][:16]}…`"])
    out += _table(["Koşul", "Manifest", "Kayıt", "Süre", "Örnekleme hızı", "SHA-256"], rows)
    out += ["Karışım oranı, örnekleme, adım/epoch sayısı ve iki koşulun eşlenmiş eğitim bütçesi veri sorumlusu "
            "tarafından belirlenmez; Adnan/Serdar HParl dev üzerinde seçer ve kendi koşu kayıtlarına yazar. "
            "CSS10 sesi özgün örnekleme hızında kalır; model girdi hattı yükleme sırasında yeniden örnekler.", ""]

    out += ["## 8. Sınırlamalar ve bilinen eksikler", "",
            "- Otomatik bayraklar (sessizlik, kırpılma, konuşma hızı aykırılığı, şüpheli tekrar) dinlenmeden hatalı "
            "sayılmadı; kayıtlar korunup inceleme kuyruğuna yazıldı.",
            "- Ses–metin hizası ASR modeliyle kontrol edilmedi; model çıktısı doğrulanmış gold sayılmaz. Konuşma hızı "
            "kuralı otomatik bir sezgiseldir; çıkarılan kayıtların yanlış etiketli olduğu dinlenerek doğrulanmadı.",
            "- Resmî bölüm konuşmacıdan bağımsız değildir (bölüm 4b); sonuçlar yeni konuşmacılara genelleme "
            "ölçümü değildir.",
            "- İnsan inceleme kararı uygulanmadıysa (bölüm 5) hiçbir kayıt dinlenmiş sayılmaz.",
            "- Yeniden kodlanmış/kesilmiş tekrarlar SHA-256 ile yakalanamaz; yalnız şüphe sezgiseli vardır.",
            "- CSS10 tek konuşmacılı sesli kitap okumasıdır; parlamento konuşmasını temsil etmez ve yararı henüz "
            "gösterilmedi. Sadece train'e girdi.",
            "- HParl'de transkript–ses farkları kaynağın kendi segmentasyonundan gelebilir; düzeltme yapılmadı.",
            "- `<spoken_noise>` gibi işaretler yalnız train hedef metninden çıkarıldı; dev ve test referansları kaynakla "
            "aynıdır. Nihai puanlamada bu işaretlerin ele alınışı Ozan'ın puanlama politikasındadır.",
            "- Oturum kimliği kaynakta ayrı alan değilse utterance ID deseninden türetilir; yöntem ve başarısız "
            "türetme sayısı envanterde yazılıdır.", ""]

    plan = delivery_plan(root)
    out += ["## 9. Verinin repoda teslimi", "",
            "Boyutlar kaydedilmiş arşiv listelerinden (`archive_members.tsv`) hesaplanır; durum "
            "`configs/data_delivery.json` karar kaydından okunur. Ham ve temiz manifestler aynı ses baytlarını "
            "paylaşır (train/dev için ses türetilmedi), bu yüzden ikinci bir ses kopyası yoktur. Yalnız indirme "
            "script'i veya Release bağlantısı teslim sayılmaz.", ""]
    rows = []
    for name, group in plan["groups"].items():
        decision = plan["decisions"].get(name, {})
        rows.append([f"`{name}`", group["files"], f"{group['gib']:.3f}", decision.get("method", "—"),
                     decision.get("status", "teslim edilmez (manifestte yok)" if "unreferenced" in name else "—")])
    out += _table(["Grup", "Dosya", "GiB", "Yöntem", "Durum"], rows)
    cumulative = plan["cumulative_audio_gib_in_priority_order"]
    out += [f"Ücretsiz Git LFS depolama: {plan['free_lfs_storage_gib']:g} GiB. Test + dev + CSS10 sesi kümülatif "
            f"{cumulative.get('css10_used_wav', 0):.3f} GiB "
            f"({'sığıyor' if plan['fits_free_lfs_without_train'] else 'sığmıyor'}); tam train eklenince "
            f"{cumulative.get('hparl_train_wav', 0):.3f} GiB "
            f"({'sığıyor' if plan['fits_free_lfs_with_full_train'] else 'sığmıyor'}). `pending`/`blocked` durumundaki gruplar teslim edilmiş "
            "sayılmaz. Kaynak metin dosyalarının bayt eşitliği temiz clone'da "
            "`python -m greek_asr.data verify-source-metadata` ile kontrol edilir.", ""]
    environment = clean_summary.get("environment")
    if environment:
        out += [f"Bu çıktıları üreten ortam: Python {environment['python']}, NumPy {environment['numpy']}, "
                f"{environment['system']}.", ""]

    out += ["## 10. Yeniden üretme komutları", "", "```bash",
            "python -m greek_asr.data hparl-download            # ~11.9 GB, kaldığı yerden devam eder",
            "python -m greek_asr.data prepare-freeze            # extract + ham manifest + test freeze + doğrulama",
            "python -m greek_asr.data prepare-clean --css10-archive <kaggle.zip> --transcript el/transcript.txt",
            "python -m greek_asr.data split-summary             # konuşmacı/oturum ve literal örtüşme (ses gerekmez)",
            "python -m greek_asr.data verify-source-metadata    # Kaldi/CSS10 kaynak metin dosyalarının bayt kontrolü",
            "python -m greek_asr.data delivery-plan             # gerçek teslim boyutları",
            "python -m greek_asr.data verify                    # tüm manifestleri sözleşme ve sesle doğrular",
            "python -m unittest discover -s tests -v", "```", "",
            "Ayrıntılı açıklama: `docs/DATA_PIPELINE_TR.md`.", ""]

    train = stats["hparl_train"]
    mixed = conditions["conditions"]["hparl_css10"]
    out += ["## 11. Sunum için veri slaytı taslağı ve konuşmacı notu", "",
            "**Slayt 1 — Veri ve test dondurma**", "",
            "- HParl 1.0.0 (CC BY-NC 4.0): resmî oturum bazlı train/dev/test korundu; bölüm konuşmacıdan bağımsız "
            f"değil (testteki {cohort['splits'].get('test', {}).get('known_speakers', '?')} konuşmacının "
            f"{cohort['pairwise_overlaps'].get('train&test', {}).get('speakers', '?')} tanesi train'de de var).",
            f"- Test erkenden donduruldu: {freeze['record_count']} kayıt, {_hours(freeze['total_duration_seconds'])}; "
            "her ses ve referans SHA-256 ile kayıtlı, hiçbir işlem yok.",
            "- Ek kaynak CSS10 Greek yalnız eğitimde.", "",
            "**Slayt 2 — Temizlik**", "",
            f"- HParl train: {train['raw_records']} → {train['kept_records']} kayıt "
            f"({_hours(train['raw_seconds'])} → {_hours(train['kept_seconds'])}); "
            f"{train['modified_records']} transkriptte işaret temizliği.",
            "- Çıkarma kuralları: " + (", ".join(f"{k}: {v}" for k, v in train["dropped_by_rule"].items()) or "yok") + ".",
            f"- Karışık eğitim koşulu: {_hours(mixed['total_seconds'])}; dev ve test iki koşulda da aynı.",
            "- Testte literal olarak eğitimde de bulunan örnek: "
            + ", ".join(f"{name} {data['literal_test_overlap']['matched_test_examples']}"
                        for name, data in cohort["training_conditions"].items()) + ".", "",
            "**Konuşmacı notu (Ozan için):** Testi her şeyden önce dondurduk ve sonra hiç dokunmadık; bunu "
            "hash'lerle kanıtlıyoruz. Temizliği yalnız train/dev'de, kural ve gerekçesiyle yaptık; her değişiklik "
            "günlükte. Otomatik bayrakları hata saymadık, dinlemeye bıraktık. Resmî bölüm konuşmacıdan bağımsız "
            "olmadığı için sonuçları yeni konuşmacıya genelleme diye sunmuyoruz. CSS10'un katkısı ayrı deneyle ölçülecek.", ""]
    return "\n".join(out)


def write_report(root: Path = ROOT) -> Path:
    path = Path(root) / "docs/DATA_REPORT_TR.md"
    write_output(path, build_report(root).encode("utf-8"))
    return path
