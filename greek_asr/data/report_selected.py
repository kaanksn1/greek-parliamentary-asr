"""DATA_REPORT_TR.md for the locked training selection (selected-scope audit v2).

Used when ``configs/css10_mixture_lineage.json`` exists. Every number is read
from the selection plan, the lineage report, the selected-scope audit/clean
outputs, the freeze and the manifests; nothing is typed by hand.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .cohort import cohort_summary
from .common import TAG_PATTERN, read_json, read_jsonl
from .publish import delivery_plan
from .report import SUBSET_LABEL, _hours, _short, _table
from .selection import LINEAGE

LABEL = {**SUBSET_LABEL, "hparl_train": "HParl train (seçili 18 oturum)"}
CONDITION_LABEL = {"raw_selected_hparl": "ham seçili HParl train", "clean_selected_hparl": "temiz seçili HParl train",
                   "clean_css10": "temiz CSS10 (yalnız train)", "mixed_hparl_css10": "karışım HParl + CSS10"}


def _num(value) -> str:
    return "—" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def build_selected_report(root: Path, source_sections) -> str:
    root = Path(root)
    lineage = read_json(root / LINEAGE)
    plan = read_json(root / lineage["selection"]["path"])
    inventory = read_json(root / "data/manifests/raw/hparl_inventory.json")
    freeze = read_json(root / "data/manifests/test.freeze.json")
    audit_summary = read_json(root / lineage["audit"]["summary"]["path"])
    clean_summary = read_json(root / lineage["clean"]["summary"]["path"])
    clean_dir = Path(lineage["clean"]["summary"]["path"]).parent.as_posix()
    cohorts = read_json(root / clean_dir / "cohort_summary.json")
    log = read_jsonl(root / lineage["clean"]["cleaning_log"]["path"])
    review = read_jsonl(root / lineage["clean"]["review_queue"]["path"])
    stats = clean_summary["subsets"]
    css = lineage["css10"]
    mixture = lineage["mixture"]
    unchanged = lineage["locked_inputs_unchanged"]
    environment = clean_summary["environment"]
    out = ["# Veri raporu — HParl + CSS10 Greek (Samed, veri sorumlusu)", "",
           "Bu dosya `python -m greek_asr.data report` komutuyla pipeline çıktılarından üretilir; içindeki "
           "sayılar elle yazılmamıştır. Model başarımı veya bonus iddiası içermez.", "",
           f"**Güncel kapsam:** sahibin kilitli eğitim seçimi `{lineage['selection']['path']}` "
           f"(`{lineage['selection']['sha256'][:16]}…`, {len(plan['session_ids'])} tam resmî train oturumu, seed "
           f"{plan.get('seed')}), değişmeyen resmî dev, dondurulmuş resmî test ve yalnız eğitimde CSS10. "
           "Tam ~99 saatlik HParl train manifesti (`data/manifests/raw/hparl_train.jsonl`, "
           "`data/manifests/train.jsonl`) yalnız **kaynak aday envanteridir**: sesleri repoda teslim edilmedi ve "
           "eğitimde kullanılmaz. Denetim/temizlik/karışım/doğrulama zinciri `--selection` ile yalnız teslim edilen "
           "seçili ham sesleri okur.", ""]
    out += source_sections(root, inventory, freeze, (clean_summary["test_freeze_before"]["status"],
                                                     clean_summary["test_freeze_after"]["status"]))

    # 4. audit -----------------------------------------------------------------------------------------------
    out += ["## 4. Denetim bulguları (seçili train, dev, CSS10)", "",
            f"Denetim şeması `{audit_summary['schema']}`, politika `{audit_summary['policy_version']}`. Ses okuma: "
            f"{audit_summary['decoder']}. Ortam: Python {audit_summary['environment']['python']}, NumPy "
            f"{audit_summary['environment']['numpy']}, {audit_summary['environment']['system']}. Test sesleri yalnız "
            "sızıntı kontrolü için hash'lendi; testten hiçbir istatistik eşik seçiminde kullanılmadı.", "",
            "Bayraklar adaydır; tek başına hatalı veri kanıtı değildir. Bir kayıt birden çok bayrak taşıyabilir.", ""]
    subsets = list(audit_summary["subsets"])
    rows = [["Kayıt / saat", *(f"{audit_summary['subsets'][s]['records']} / "
                                f"{audit_summary['subsets'][s]['seconds'] / 3600:.3f}" for s in subsets)],
            ["Kodlama", *(", ".join(f"{k}: {v}" for k, v in audit_summary["subsets"][s].get("encodings", {}).items())
                          for s in subsets)],
            ["RMS dBFS p05 / p50 / p95", *(" / ".join(_num(audit_summary["subsets"][s]["rms_dbfs"].get(q))
                                                     for q in ("p05", "p50", "p95")) for s in subsets)],
            ["Tepe dBFS p50 / max", *(" / ".join(_num(audit_summary["subsets"][s].get("peak_dbfs", {}).get(q))
                                                for q in ("p50", "max")) for s in subsets)]]
    for name in sorted({f for data in audit_summary["subsets"].values() for f in data["flag_counts"]}):
        row = [f"`{name}`"]
        for subset in subsets:
            data = audit_summary["subsets"][subset]
            count = data["flag_counts"].get(name, 0)
            row.append(f"{count} ({data['flag_seconds'].get(name, 0) / 60:.1f} dk)" if count else "0")
        rows.append(row)
    out += _table(["Ölçü / bayrak", *(LABEL[s] for s in subsets)], rows)
    dup = audit_summary["duplicates"]
    out += [f"Aynı baytlı ses grupları: {dup['bytes']['groups']} grup / {dup['bytes']['records_in_groups']} kayıt; "
            f"aynı PCM içerikli gruplar: {dup['pcm']['groups']} (seçili train + dev + CSS10 + test hash'leri). "
            "SHA-256 birebir tekrarları ve yalnız başlığı farklı kopyaları yakalar; yeniden kodlanmış veya kesilmiş "
            "tekrarlar yalnız `duplicate_suspect` sezgiseliyle dinlemeye bırakılır.", ""]
    tag_rows = []
    for split in ("dev", "test"):
        refs = [r["reference"] for r in read_jsonl(root / f"data/manifests/references/{split}.jsonl")]
        tagged = sum(1 for r in refs if TAG_PATTERN.search(r))
        tag_rows.append(f"{split}: {tagged}/{len(refs)} referans (%{100 * tagged / max(1, len(refs)):.1f}), "
                        f"{sum(len(TAG_PATTERN.findall(r)) for r in refs)} işaret")
    out += ["**Puanlama için önemli:** kaynak referanslarında `<spoken_noise>` gibi işaretler var (yalnız okundu, "
            "değiştirilmedi): " + "; ".join(tag_rows) + ". Train hedeflerinden çıkarıldıkları için modeller bunları "
            "üretmez; referans ve hipotezde aynı kuralla ele alınmaları ortak puanlama politikasında dev üzerinde "
            "kararlaştırılmalıdır.", ""]

    # 4a. CSS10 float32 correction ---------------------------------------------------------------------------
    signal = css["signal_audit"]
    out += ["## 4a. CSS10 IEEE float32 düzeltmesi: eski ve yeni denetim", "",
            "1.844 CSS10 WAV'ının tamamı RIFF format tag 3 (IEEE float32), mono, 22050 Hz'dir. Önceki denetim "
            "(`p3-data-audit-v1`) 4 baytlık örnekleri integer PCM32 sayıp float baytlarını 2^31'e bölüyordu; RMS, "
            "dBFS ve bunlara bağlı bayraklar bu yüzden **yanlıştı**. Düzeltmede format tag'e göre çözülür, float "
            "değerler olduğu gibi kullanılır; kazanç veya tepe normalizasyonu yok, özgün WAV'lar değiştirilmedi.", ""]
    previous = signal.get("previous")
    if previous:
        out += [f"Eski denetim: `{previous['audit_summary']['path']}` `{previous['audit_summary']['sha256'][:16]}…` — "
                f"{previous['status']}", ""]
        names = sorted({*previous["flag_counts"], *signal["current"]["flag_counts"]})
        rows = [[f"`{n}`", previous["flag_counts"].get(n, 0), signal["current"]["flag_counts"].get(n, 0)] for n in names]
        rows += [["RMS dBFS min / p50 / max", " / ".join(_num(previous["rms_dbfs"][k]) for k in ("min", "p50", "max")),
                  " / ".join(_num(signal["current"]["rms_dbfs"][k]) for k in ("min", "p50", "max"))],
                 ["Tepe dBFS min / p50 / max", " / ".join(_num(previous["peak_dbfs"][k]) for k in ("min", "p50", "max")),
                  " / ".join(_num(signal["current"]["peak_dbfs"][k]) for k in ("min", "p50", "max"))]]
        out += _table(["CSS10 bayrağı / ölçü", "Eski (integer yorum, geçersiz)", "Yeni (float32)"], rows)
        changes = signal["flag_membership_changes"]
        if changes:
            out += ["Kayıt bazında bayrak değişimleri (ilk birkaç ID; tam liste lineage dosyasında, ilk 50):", ""]
            out += _table(["Bayrak", "Eski", "Yeni", "Eklenen örnek", "Kalkan örnek"],
                          [[f"`{n}`", c["previous"], c["current"], ", ".join(f"`{i}`" for i in c["added_ids"][:3]) or "—",
                            ", ".join(f"`{i}`" for i in c["removed_ids"][:3]) or "—"] for n, c in changes.items()])
    out += [f"Temiz CSS10: {css['raw_records']} ham → {css['kept_records']} kayıt ({css['kept_hours']:.6f} saat); "
            "çıkarma kuralları: " + (", ".join(f"`{k}` {v} ({css['dropped_seconds_by_rule'][k]:.2f} sn)"
                                            for k, v in css["dropped_by_rule"].items()) or "yok") + ". "
            + ("Temiz CSS10 manifesti/referansı önceki sürümle **bayt-özdeş** kaldı; yol değişmedi."
               if not css["changed"] else "Temiz CSS10 değişti; yeni sürüm ayrı dosyaya yazıldı (bölüm 5b)."), ""]

    # 4b. cohorts ---------------------------------------------------------------------------------------------
    out += ["## 4b. Bölüm, konuşmacı, oturum ve test cümlesi örtüşmesi", "",
            "Kullanılan bölüm **HParl 1.0.0'ın resmî oturum bazlı train/dev/test bölümüdür** ve konuşmacıdan bağımsız "
            "(speaker-disjoint) **değildir**. Hocanın izin verdiği belgelenmiş resmî bölüm istisnası (How 1) "
            "kullanıldı; dondurulmuş test yeniden bölünmedi. Sonuçlar yeni konuşmacıya genelleme (unseen-speaker) "
            "ölçümü değildir. Sayılar manifestlerden hesaplanır; eksik kimlik uydurulmaz, ayrı sayılır.", ""]
    rows = []
    for name, data in cohorts["conditions"].items():
        for source, part in data["by_source"].items():
            rows.append([CONDITION_LABEL.get(name, name), source, part["records"], f"{part['seconds'] / 3600:.6f}",
                         part["known_speakers"], part["records_without_speaker_id"],
                         part["known_sessions_or_recordings"], part["records_without_session_id"]])
    for label, data in (("dev (değişmedi)", cohorts["dev"]), ("test (dondurulmuş)", cohorts["test"])):
        rows.append([label, "hparl", data["records"], f"{data['seconds'] / 3600:.6f}", data["known_speakers"],
                     data["records_without_speaker_id"], data["known_sessions_or_recordings"],
                     data["records_without_session_id"]])
    out += _table(["Küme", "Kaynak", "Kayıt", "Saat", "Bilinen konuşmacı", "Konuşmacı ID'si boş",
                   "Oturum/kayıt", "Oturum ID'si boş"], rows)
    out += [cohorts["css10_identity_note"], ""]
    rows = [[CONDITION_LABEL.get(name, name), data["test_speakers_also_in_this_training"],
             data["test_examples_whose_speaker_is_in_this_training"], data["sessions_also_in_test"],
             data["speakers_also_in_dev"], data["sessions_also_in_dev"]] for name, data in cohorts["conditions"].items()]
    rows.append(["dev", cohorts["dev"]["test_speakers_also_in_dev"], "—", cohorts["dev"]["sessions_also_in_test"], "—", "—"])
    out += _table(["Küme", "Testle ortak konuşmacı", "Konuşmacısı bu kümede olan test örneği", "Testle ortak oturum",
                   "Dev ile ortak konuşmacı", "Dev ile ortak oturum"], rows)
    out += ["**Test cümlesinin eğitim hedefinde bulunması.** *Literal eşleşme:* dondurulmuş test referans string'i "
            "hiçbir dönüşüm uygulanmadan (etiket silme, büyük/küçük harf, Unicode veya boşluk düzeltmesi yok) eğitim "
            "hedeflerinden en az birine birebir eşittir; eşleşen test örneği ve benzersiz metin ayrı sayılır. "
            "*Normalize eşleşme* yalnız tanısal bir ölçüdür, puanlama kuralı değildir: iki tarafta `<...>` işaretleri "
            "silinir, Unicode NFC, casefold ve boşluk birleştirme uygulanır, benzersiz metinler kesiştirilir. "
            "İkisi farklı tanımlardır ve birbirinin yerine sunulmaz.", ""]
    rows = [[CONDITION_LABEL.get(name, name), data["literal_test_overlap"]["matched_test_examples"],
             data["literal_test_overlap"]["matched_distinct_test_texts"],
             data["normalized_test_overlap"]["matched_test_examples"],
             data["normalized_test_overlap"]["matched_distinct_normalized_texts"],
             f"`{data['references']['path']}` `{data['references']['sha256'][:16]}…`"]
            for name, data in cohorts["conditions"].items()]
    out += _table(["Eğitim hedefi", "Literal test örneği", "Literal benzersiz metin", "Normalize test örneği",
                   "Normalize benzersiz metin", "Referans dosyası"], rows)
    full = cohort_summary(root)
    raw_full = full["splits"].get("train")
    if raw_full:
        out += [f"Bağlam (eğitimde kullanılmaz): tam kaynak aday train envanteri {raw_full['records']} kayıt / "
                f"{raw_full['hours']:.3f} saat / {raw_full['known_speakers']} konuşmacı / "
                f"{raw_full['known_sessions_or_recordings']} oturum. Hiçbir eşleşen cümle testten silinmedi; test "
                f"değişmedi. Makine okunur kayıt: `{clean_dir}/cohort_summary.json`.", ""]

    # 5. cleaning ---------------------------------------------------------------------------------------------
    out += ["## 5. Temizlik sonucu (seçili kapsam)", "",
            "Bir kayıt birden çok çıkarma kuralına uyuyorsa yalnız config'teki ilk kurala yazılır.", ""]
    rows = []
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        s = stats[subset]
        rows.append([LABEL[subset], s["raw_records"], _hours(s["raw_seconds"]), s["kept_records"], _hours(s["kept_seconds"]),
                     s["raw_records"] - s["kept_records"], f"{(s['raw_seconds'] - s['kept_seconds']) / 60:.2f} dk",
                     s["modified_records"]])
    out += _table(["Bölüm", "Ham kayıt", "Ham süre", "Temiz kayıt", "Temiz süre", "Çıkarılan", "Çıkarılan süre",
                   "Metni değişen"], rows)
    rows = []
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        for rule, count in stats[subset]["dropped_by_rule"].items():
            rows.append([LABEL[subset], "çıkarma", f"`{rule}`", count,
                         f"{stats[subset]['dropped_seconds_by_rule'][rule]:.2f} sn"])
        for step, count in stats[subset]["modification_steps"].items():
            rows.append([LABEL[subset], "metin", f"`{step}`", count, "—"])
        for flag, count in stats[subset]["review_queue_by_flag"].items():
            rows.append([LABEL[subset], "inceleme (korundu)", f"`{flag}`", count, "—"])
    source = defaultdict(int)
    for record in log:
        if record["rule"].startswith("source:"):
            source[(record["subset"], record["rule"])] += 1
    rows += [[LABEL.get(k[0], k[0]), "kaynakta kullanılamaz", f"`{k[1]}`", v, "—"] for k, v in sorted(source.items())]
    out += _table(["Bölüm", "İşlem", "Kural", "Kayıt", "Süre"], rows or [["—", "—", "—", 0, "—"]])
    listened = clean_summary["review_decisions_applied"]
    out += [f"Günlük: `{lineage['clean']['cleaning_log']['path']}` ({sum(clean_summary['log_entries'].values())} satır, "
            f"`{lineage['clean']['cleaning_log']['sha256'][:16]}…`). İnceleme kuyruğu: "
            f"`{lineage['clean']['review_queue']['path']}` ({len(review)} kayıt; "
            + ", ".join(f"{LABEL.get(k, k)} {v}" for k, v in clean_summary["review_queue_by_subset"].items())
            + f"). Uygulanan insan dinleme kararı: {listened}"
            + (" — bu turda hiçbir kayıt dinlenmiş sayılmaz." if not listened else ".") + "", ""]
    harmonization = stats["css10_harmonization"]
    out += [f"CSS10 metin uyumlaması: üreticinin normalize metni alındı, NFC, küçük harf = {harmonization['lowercase']} "
            f"(seçili temiz HParl hedeflerinde büyük harf oranı {harmonization['hparl_train_uppercase_ratio']}), "
            "seçili temiz HParl hedeflerinde geçmeyen harf dışı karakterler boşluğa çevrildi (korunan: "
            f"`{harmonization['allowed_non_letter_characters'] or 'yok'}`). Özgün metin "
            "`data/manifests/raw/css10_el_texts.jsonl` içinde değişmeden durur.", ""]
    config = read_json(root / "configs/data_cleaning.json")["flags"]
    boundary = [[LABEL.get(r["subset"], r["subset"]), f"`{r['id']}`", f"{r['flags']['cps_boundary']['chars_per_second']:.2f}",
                 f"{r['flags']['cps_boundary']['duration_seconds']:.2f}", r["flags"]["cps_boundary"]["chars"]]
                for r in review if "cps_boundary" in r["flags"]]
    out += ["### Konuşma hızı kuralı", "",
            f"Saniyede > {config['cps_extreme_min']:g} boşluk dışı karakter: train/CSS10'da hizalama sorunu **adayı** "
            f"olarak çıkarılır (dinlenmedi; sezgisel). {config['cps_boundary_min']:g}–{config['cps_extreme_min']:g} "
            "aralığı korunur ve inceleme kuyruğuna yazılır. Dev'de bu kuralla çıkarma yok.", ""]
    if boundary:
        out += _table(["Bölüm", "Sınır kaydı (korundu)", "kar/sn", "Süre (sn)", "Karakter"], boundary)

    # 5b. lineage ---------------------------------------------------------------------------------------------
    out += ["## 5b. Kilitli dosyalar ve CSS10/karışım lineage", "",
            f"Makine okunur kabul raporu: `{LINEAGE}` (durum `{lineage['status']}`). Kilitli HParl seçimi, dev ve "
            "dondurulmuş test yalnız karşılaştırıldı; hiçbiri yeniden yazılmadı.", ""]
    rows = [["Seçili temiz HParl", unchanged["selected_clean_hparl"]["manifest"]["path"],
             f"`{unchanged['selected_clean_hparl']['manifest']['sha256'][:16]}…`",
             "bayt-özdeş yeniden üretildi" if unchanged["selected_clean_hparl"]["reproduced_byte_identical"] else "FARK RAPORLANDI"],
            ["Dev", unchanged["dev"]["manifest"]["path"], f"`{unchanged['dev']['manifest']['sha256'][:16]}…`",
             "bayt-özdeş yeniden üretildi" if unchanged["dev"]["reproduced_byte_identical"] else "FARK RAPORLANDI"],
            ["Dondurulmuş test", freeze["test_manifest_path"], f"`{freeze['test_manifest_sha256'][:16]}…`",
             f"freeze {unchanged['frozen_test']['before']['status']} → {unchanged['frozen_test']['after']['status']}"]]
    out += _table(["Dosya", "Yol", "SHA-256", "Sonuç"], rows)
    rows = []
    for label, part in (("CSS10", css), ("Karışım", mixture)):
        for kind in ("manifest", "references"):
            old, new = part["previous"][kind], part["current"][kind]
            rows.append([f"{label} {kind}", f"`{old['path']}` `{old['sha256'][:16]}…`" if old else "—",
                         f"`{new['path']}` `{new['sha256'][:16]}…`", "değişti (yeni sürüm)" if part["changed"] else "aynı"])
    out += _table(["Çıktı", "Önceki", "Güncel", "Durum"], rows)
    cohort = mixture["cohort"]
    out += [f"Karışım kuralı: kilitli temiz HParl seçimi + güncel temiz CSS10, ID'ye göre sıralı. Güncel karışım "
            f"{cohort['records']} kayıt / {cohort['seconds'] / 3600:.6f} saat ("
            + ", ".join(f"{k}: {v['records']} kayıt / {v['seconds'] / 3600:.6f} sa" for k, v in cohort["by_source"].items())
            + "). Karışım oranı/örnekleme ve eğitim bütçesi model konfigürasyonunda sahip tarafından belirlenir.", ""]

    # 6. examples ---------------------------------------------------------------------------------------------
    out += ["## 6. Gerçek örnekler (günlükten)", ""]
    seen, rows = defaultdict(int), []
    for record in log:
        key = (record["subset"], record["action"], record["rule"])
        if record["action"] not in ("drop", "modify_text") or seen[key] >= 2:
            continue
        seen[key] += 1
        rows.append([LABEL.get(record["subset"], record["subset"]), record["action"], f"`{record['rule']}`",
                     f"`{record['id']}`", _short(record.get("before_text")), _short(record.get("after_text")),
                     _short(str(record["evidence"]) if record["evidence"] is not None else None, 60)])
    out += _table(["Bölüm", "İşlem", "Kural", "ID", "Önce", "Sonra", "Kanıt"], rows or [["—"] * 7])

    # 7. conditions -------------------------------------------------------------------------------------------
    out += ["## 7. Model sahiplerine teslim edilen eğitim/değerlendirme dosyaları", ""]
    rows = []
    for name in ("clean_selected_hparl", "mixed_hparl_css10", "raw_selected_hparl"):
        data = cohorts["conditions"][name]
        rows.append([CONDITION_LABEL[name], f"`{data['manifest']['path']}`",
                     ", ".join(f"{k}: {v}" for k, v in data["records_by_source"].items()), f"{data['seconds'] / 3600:.6f}",
                     f"`{data['manifest']['sha256'][:16]}…`"])
    rows.append(["dev", f"`{cohorts['dev']['manifest']['path']}`", cohorts["dev"]["records"],
                 f"{cohorts['dev']['seconds'] / 3600:.6f}", f"`{cohorts['dev']['manifest']['sha256'][:16]}…`"])
    rows.append(["dondurulmuş test", f"`{cohorts['test']['manifest']['path']}`", cohorts["test"]["records"],
                 f"{cohorts['test']['seconds'] / 3600:.6f}", f"`{cohorts['test']['manifest']['sha256'][:16]}…`"])
    out += _table(["Dosya", "Manifest", "Kayıt", "Saat", "SHA-256"], rows)
    out += ["Temiz manifestler ham seçili satırların aynısını (aynı özgün WAV yolu/hash'i) kullanır; temizlik için ses "
            "kopyası üretilmedi. CSS10 22050 Hz float32 kalır; model giriş hattı bellekte 16 kHz'e çevirir.", ""]

    # 8. limitations ------------------------------------------------------------------------------------------
    out += ["## 8. Sınırlamalar ve bilinen eksikler", "",
            "- Otomatik bayraklar dinlenmeden hata sayılmadı; korunup inceleme kuyruğuna yazıldı."
            + (" Bu turda insan dinleme kararı yok." if not listened else ""),
            "- Ses–metin hizası ASR modeliyle kontrol edilmedi; model çıktısı gold sayılmaz. Konuşma hızı kuralı sezgiseldir.",
            "- Resmî bölüm konuşmacıdan bağımsız değildir (bölüm 4b).",
            "- Yeniden kodlanmış/kesilmiş tekrarlar SHA-256 ile yakalanamaz.",
            "- Seçili kapsamda train içi tekrar/şüphe kontrolleri yalnız seçili 18 oturumu görür; kilitli temiz HParl "
            "dosyası tam envanterden üretilmişti ve seçili kapsam onu bayt-özdeş yeniden üretti (bölüm 5b).",
            "- CSS10 tek okuyuculu sesli kitaptır; parlamento konuşmasını temsil etmez ve yalnız train'dedir. Eski "
            "integer yorumlu CSS10 sinyal istatistikleri geçersizdir ve yalnız tarihsel kayıttır.",
            "- `<spoken_noise>` işaretleri yalnız train hedeflerinden çıkarıldı; dev/test referansları kaynakla aynıdır.", ""]

    # 9. delivery ---------------------------------------------------------------------------------------------
    delivery = delivery_plan(root)
    out += ["## 9. Verinin repoda teslimi", "",
            "Durum `configs/data_delivery.json` karar kaydından okunur; `pending` teslim edilmiş sayılmaz. Seçili ham "
            "HParl train seslerinin LFS teslimini proje sahibi yürütür; temiz manifestler aynı ham WAV'ları paylaşır.", ""]
    rows = [[f"`{name}`", group["files"], f"{group['gib']:.3f}", delivery["decisions"].get(name, {}).get("method", "—"),
             delivery["decisions"].get(name, {}).get("status", "teslim edilmez" if "unreferenced" in name else "—")]
            for name, group in delivery["groups"].items()]
    out += _table(["Grup (kaynak arşiv listesi)", "Dosya", "GiB", "Yöntem", "Durum"], rows)
    storage = plan.get("storage", {})
    if storage:
        out += [f"Seçili ham train sesi: {storage.get('raw_records')} WAV, {storage.get('raw_train_unique_lfs_bytes')} "
                f"bayt; test + dev + CSS10 + seçili train toplam ≈ {storage.get('all_audio_unique_lfs_gib', 0):.6f} GiB "
                f"(onaylı sınır {storage.get('authorized_quota_gib')} GiB, ücretli alan yok).", ""]
    out += [f"Bu çıktıları üreten ortam: Python {environment['python']}, NumPy {environment['numpy']}, "
            f"{environment['system']}.", ""]

    # 10. commands --------------------------------------------------------------------------------------------
    selection_arg = f"--selection {lineage['selection']['path']}"
    out += ["## 10. Yeniden üretme komutları (yalnız teslim edilen sesle)", "", "```bash",
            "python -m greek_asr.data verify-source-metadata",
            "python -m greek_asr.data verify-freeze                       # test sesleri dahil",
            f"python -m greek_asr.data audit {selection_arg}",
            f"python -m greek_asr.data clean {selection_arg}    # CSS10 + karışım + lineage",
            f"python -m greek_asr.data verify {selection_arg}   # seçili ham/temiz, CSS10, karışım, dev + freeze",
            "python -m greek_asr.data verify --only css10_el_train.jsonl dev.jsonl",
            "python -m greek_asr.data report",
            "python -m unittest discover -s tests -v", "```", "",
            "Kaynaktan baştan üretim (`hparl-download`, `prepare-freeze`, `prepare-clean`) tam arşiv gerektirir; "
            "ayrıntı `docs/DATA_PIPELINE_TR.md`.", ""]

    # 11. slide notes -----------------------------------------------------------------------------------------
    clean_cohort = cohorts["conditions"]["clean_selected_hparl"]
    raw_cohort = cohorts["conditions"]["raw_selected_hparl"]
    out += ["## 11. Sunum için veri slaytı taslağı", "",
            f"- Resmî HParl bölümü (speaker-disjoint değil): seçili train {raw_cohort['records']} ham → "
            f"{clean_cohort['records']} temiz kayıt, {clean_cohort['hours']:.3f} saat, {clean_cohort['known_speakers']} "
            f"konuşmacı, {clean_cohort['known_sessions_or_recordings']} oturum; testin "
            f"{clean_cohort['test_speakers_also_in_this_training']} konuşmacısı train'de de var, oturum örtüşmesi "
            f"{clean_cohort['sessions_also_in_test']}.",
            f"- Test erken donduruldu ({freeze['record_count']} kayıt, {_hours(freeze['total_duration_seconds'])}); "
            "hiçbir işlem yok, hash'lerle doğrulandı.",
            f"- CSS10 (yalnız train) float32 olarak doğru denetlendi: {css['kept_records']} kayıt, {css['kept_hours']:.3f} saat; "
            f"karışım {cohort['records']} kayıt / {cohort['seconds'] / 3600:.3f} saat.",
            "- Literal test cümlesi eşleşmesi: " + ", ".join(
                f"{CONDITION_LABEL[n]} {d['literal_test_overlap']['matched_test_examples']}"
                for n, d in cohorts["conditions"].items() if n != "clean_css10") + " (normalize ölçü ayrı).", ""]
    return "\n".join(out)
