"""Generate docs/DATA_REPORT_TR.md from recorded pipeline outputs only.

Every number in the report is read from acquisition, inventory, freeze,
audit, cleaning and conditions files produced by the pipeline. Nothing is
typed by hand, so the report can be regenerated and checked.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .common import ROOT, DataError, hms, read_json, read_jsonl, sha256_file, write_output

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


def build_report(root: Path = ROOT) -> str:
    root = Path(root)
    need = {"inventory": root / "data/manifests/raw/hparl_inventory.json",
            "freeze": root / "data/manifests/test.freeze.json", "audit": root / "data/audit/audit_summary.json",
            "clean": root / "data/clean/clean_summary.json", "conditions": root / "data/manifests/conditions.json"}
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        raise DataError(f"report inputs missing: {missing}; run the pipeline first")
    inventory, freeze, audit_summary, clean_summary, conditions = (read_json(need[k]) for k in
                                                                  ("inventory", "freeze", "audit", "clean", "conditions"))
    acquisition = root / "data/raw/hparl/ACQUISITION.json"
    hparl_acq = read_json(acquisition) if acquisition.exists() else {}
    css_acq_path, css_inv_path = root / "data/raw/css10_el/ACQUISITION.json", root / "data/manifests/raw/css10_el_inventory.json"
    css_acq = read_json(css_acq_path) if css_acq_path.exists() else None
    css_inv = read_json(css_inv_path) if css_inv_path.exists() else None
    log = read_jsonl(root / "data/clean/cleaning_log.jsonl")
    stats = clean_summary["subsets"]
    out = ["# Veri raporu — HParl + CSS10 Greek (Samed, veri sorumlusu)", "",
           "Bu dosya `python -m greek_asr.data report` komutuyla pipeline çıktılarından üretilir; içindeki "
           "sayılar elle yazılmamıştır. Model başarımı veya bonus iddiası içermez.", ""]

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
        ["`test.freeze.json` SHA-256", f"`{sha256_file(need['freeze'])}`"],
        ["Temizlik öncesi/sonrası doğrulama", f"{clean_summary['test_freeze_before']['status']} / "
                                              f"{clean_summary['test_freeze_after']['status']}"],
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
    rows = [[name, *(audit_summary["overlaps"][name].get(pair, "—") for pair in
                     ("hparl_train&hparl_dev", "hparl_train&hparl_test", "hparl_dev&hparl_test"))]
            for name in ("original_ids", "speakers", "recordings_or_sessions", "distinct_texts_exact_after_casefold")]
    out += ["Bölümler arası örtüşme (yalnız raporlanır; test tarafı hiçbir zaman değiştirilmez):", ""]
    out += _table(["Tür", "train∩dev", "train∩test", "dev∩test"], rows)
    out += ["Konuşmacı örtüşmesi resmî bölümün konuşmacıdan bağımsız olmadığını gösterir; bu bir sızıntı düzeltmesi "
            "gerektirmez ama sonuç yorumunda belirtilmelidir. Aynı metin örtüşmesi kısa, sık kalıplardan doğabilir.", ""]

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
            "- Ses–metin hizası ASR modeliyle kontrol edilmedi; model çıktısı doğrulanmış gold sayılmaz.",
            "- Yeniden kodlanmış/kesilmiş tekrarlar SHA-256 ile yakalanamaz; yalnız şüphe sezgiseli vardır.",
            "- CSS10 tek konuşmacılı sesli kitap okumasıdır; parlamento konuşmasını temsil etmez ve yararı henüz "
            "gösterilmedi. Sadece train'e girdi.",
            "- HParl'de transkript–ses farkları kaynağın kendi segmentasyonundan gelebilir; düzeltme yapılmadı.",
            "- `<spoken_noise>` gibi işaretler yalnız train hedef metninden çıkarıldı; dev ve test referansları kaynakla "
            "aynıdır. Nihai puanlamada bu işaretlerin ele alınışı Ozan'ın puanlama politikasındadır.",
            "- Oturum kimliği kaynakta ayrı alan değilse utterance ID deseninden türetilir; yöntem ve başarısız "
            "türetme sayısı envanterde yazılıdır.", ""]

    out += ["## 9. Yeniden üretme komutları", "", "```bash",
            "python -m greek_asr.data hparl-download            # ~11.9 GB, kaldığı yerden devam eder",
            "python -m greek_asr.data prepare-freeze            # extract + ham manifest + test freeze + doğrulama",
            "python -m greek_asr.data prepare-clean --css10-archive <kaggle.zip>   # CSS10 + audit + clean + rapor + verify",
            "python -m greek_asr.data verify                    # tüm manifestleri sözleşme ve sesle doğrular",
            "python -m unittest discover -s tests -v", "```", "",
            "Ayrıntılı açıklama: `docs/DATA_PIPELINE_TR.md`.", ""]

    train = stats["hparl_train"]
    mixed = conditions["conditions"]["hparl_css10"]
    out += ["## 10. Sunum için veri slaytı taslağı ve konuşmacı notu", "",
            "**Slayt 1 — Veri ve test dondurma**", "",
            "- HParl 1.0.0 (CC BY-NC 4.0): resmî oturum bazlı train/dev/test korundu.",
            f"- Test erkenden donduruldu: {freeze['record_count']} kayıt, {_hours(freeze['total_duration_seconds'])}; "
            "her ses ve referans SHA-256 ile kayıtlı, hiçbir işlem yok.",
            "- Ek kaynak CSS10 Greek yalnız eğitimde.", "",
            "**Slayt 2 — Temizlik**", "",
            f"- HParl train: {train['raw_records']} → {train['kept_records']} kayıt "
            f"({_hours(train['raw_seconds'])} → {_hours(train['kept_seconds'])}); "
            f"{train['modified_records']} transkriptte işaret temizliği.",
            "- Çıkarma kuralları: " + (", ".join(f"{k}: {v}" for k, v in train["dropped_by_rule"].items()) or "yok") + ".",
            f"- Karışık eğitim koşulu: {_hours(mixed['total_seconds'])}; dev ve test iki koşulda da aynı.", "",
            "**Konuşmacı notu (Ozan için):** Testi her şeyden önce dondurduk ve sonra hiç dokunmadık; bunu "
            "hash'lerle kanıtlıyoruz. Temizliği yalnız train/dev'de, kural ve gerekçesiyle yaptık; her değişiklik "
            "günlükte. Otomatik bayrakları hata saymadık, dinlemeye bıraktık. CSS10'un katkısı ayrı deneyle ölçülecek.", ""]
    return "\n".join(out)


def write_report(root: Path = ROOT) -> Path:
    path = Path(root) / "docs/DATA_REPORT_TR.md"
    write_output(path, build_report(root).encode("utf-8"))
    return path
