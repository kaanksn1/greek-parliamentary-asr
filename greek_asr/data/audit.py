"""Measure train/dev/CSS10 quality and leakage without changing any data.

Signal statistics, text statistics, duplicate groups and cross-split overlaps
are written as flags. A flag is a candidate for a rule or for listening; it is
not proof that a record is wrong. Thresholds come only from the config and
train/dev/CSS10 distributions. Test audio is read solely to hash it so that
train/dev copies of test recordings can be detected; no test statistic is
used to choose a threshold.
"""
from __future__ import annotations

import gzip
import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .common import (DataError, ROOT, annotation_tags, character_inventory, is_greek_letter, iter_chunks,
                     json_bytes, jsonl_bytes, probe_wav, read_json, read_jsonl, resolve_inside, sha256_bytes,
                     sha256_file, strip_annotation_tags, write_output)

SUBSETS = {"hparl_train": ("hparl_train.jsonl", "references/hparl_train.jsonl"),
           "hparl_dev": ("hparl_dev.jsonl", "references/hparl_dev.jsonl"),
           "css10_el": ("css10_el.jsonl", "references/css10_el.jsonl")}
OUT = ROOT / "data/audit"


def overlap_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", strip_annotation_tags(text)).casefold().split())


def signal_stats(payload: bytes, width: int, channels: int, rate: int, signal: dict) -> dict:
    import numpy as np  # data-pipeline dependency, see requirements-data.txt

    if width == 2:
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 1:
        samples = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 4:
        samples = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 3:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        value = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        samples = np.where(value & 0x800000, value - (1 << 24), value).astype(np.float64) / 8388608.0
    else:
        raise DataError(f"unsupported sample width {width}")
    clip_fraction = float(np.mean(np.abs(samples) >= signal["clip_threshold"])) if samples.size else 0.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        return {"rms_dbfs": None, "peak_dbfs": None, "clip_fraction": 0.0, "speech_fraction": 0.0,
                "leading_silence_seconds": 0.0, "trailing_silence_seconds": 0.0}
    floor = 1e-10
    rms = math.sqrt(float(np.mean(samples * samples)))
    peak = float(np.max(np.abs(samples)))
    frame = max(1, int(round(rate * signal["frame_seconds"])))
    count = max(1, samples.size // frame)
    frames = samples[:count * frame].reshape(count, frame) if samples.size >= frame else samples.reshape(1, -1)
    frame_db = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames * frames, axis=1)), floor))
    voiced = frame_db > signal["silence_frame_dbfs"]
    hop = frames.shape[1] / rate
    if voiced.any():
        leading = int(np.argmax(voiced)) * hop
        trailing = int(np.argmax(voiced[::-1])) * hop
    else:
        leading = trailing = samples.size / rate
    return {"rms_dbfs": round(20 * math.log10(max(rms, floor)), 3),
            "peak_dbfs": round(20 * math.log10(max(peak, floor)), 3),
            "clip_fraction": round(clip_fraction, 6), "speech_fraction": round(float(np.mean(voiced)), 4),
            "leading_silence_seconds": round(leading, 4), "trailing_silence_seconds": round(trailing, 4)}


def _measure(job: tuple) -> list[dict]:
    root, rows, signal, with_stats = job
    results = []
    for row in rows:
        path = resolve_inside(Path(root), row["audio_path"])
        audio = probe_wav(path)
        payload = audio.pop("_payload")
        if audio["audio_sha256"] != row["audio_sha256"]:
            raise DataError(f"{row['id']}: audio changed since the raw manifest was built")
        item = {"id": row["id"], "pcm_sha256": audio["pcm_sha256"],
                "sample_width_bytes": audio["sample_width_bytes"]}
        if with_stats:
            item.update(signal_stats(payload, audio["sample_width_bytes"], audio["channels"],
                                     audio["sampling_rate"], signal))
        results.append(item)
    return results


def measure(root: Path, rows: list[dict], signal: dict, with_stats: bool, workers: int) -> dict[str, dict]:
    jobs = [(str(root), chunk, signal, with_stats) for chunk in iter_chunks(rows, 256)]
    if workers <= 1 or len(jobs) <= 1:
        results = [item for job in jobs for item in _measure(job)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = [item for batch in pool.map(_measure, jobs) for item in batch]
    return {item["id"]: item for item in results}


def percentiles(values: list[float]) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {}
    def pick(q):
        return round(values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))], 4)
    return {"min": round(values[0], 4), "p01": pick(0.01), "p05": pick(0.05), "p50": pick(0.5),
            "p95": pick(0.95), "p99": pick(0.99), "max": round(values[-1], 4)}


def robust(values: list[float]) -> tuple[float, float]:
    values = [v for v in values if v is not None]
    if not values:
        return 0.0, 0.0
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return median, mad


def load_config(root: Path, path: Path | None = None) -> tuple[dict, str]:
    path = path or root / "configs/data_cleaning.json"
    return read_json(path), sha256_file(path)


def run_audit(root: Path = ROOT, config_path: Path | None = None, workers: int = 1, out: Path | None = None) -> dict:
    root = Path(root)
    out = out or root / "data/audit"
    config, config_sha = load_config(root, config_path)
    flags_cfg, signal = config["flags"], config["signal"]
    raw = root / "data/manifests/raw"
    subsets, inputs = {}, {}
    for subset, (manifest, references) in SUBSETS.items():
        if not (raw / manifest).exists():
            if subset == "css10_el":
                continue
            raise DataError(f"missing raw manifest {manifest}; build raw manifests first")
        rows = read_jsonl(raw / manifest)
        refs = {r["id"]: r["reference"] for r in read_jsonl(raw / references)}
        if set(refs) != {r["id"] for r in rows}:
            raise DataError(f"{subset}: references and manifest IDs differ")
        subsets[subset] = (rows, refs)
        inputs[manifest] = sha256_file(raw / manifest)
        inputs[references] = sha256_file(raw / references)
    test_rows = read_jsonl(raw / "hparl_test.jsonl")
    inputs["hparl_test.jsonl"] = sha256_file(raw / "hparl_test.jsonl")
    css10_texts = {}
    if (raw / "css10_el_texts.jsonl").exists():
        css10_texts = {r["id"]: r for r in read_jsonl(raw / "css10_el_texts.jsonl")}

    items: dict[str, dict] = {}
    for subset, (rows, refs) in subsets.items():
        measured = measure(root, rows, signal, True, workers)
        for row in rows:
            text = refs[row["id"]]
            spoken = strip_annotation_tags(text)
            letters = sum(1 for c in spoken if not c.isspace())
            item = {"id": row["id"], "subset": subset, "source": row["source"], "split": row["split"],
                    "audio_sha256": row["audio_sha256"], "audio_path": row["audio_path"],
                    "duration_seconds": row["duration_seconds"], "sampling_rate": row["sampling_rate"],
                    "channels": row["channels"], "speaker_id": row["speaker_id"],
                    "source_recording_id": row["source_recording_id"],
                    "n_words": len(spoken.split()), "n_chars": letters,
                    "chars_per_second": round(letters / row["duration_seconds"], 4),
                    "annotation_tags": annotation_tags(text),
                    "non_greek_characters": "".join(sorted({c for c in spoken if not c.isspace() and not is_greek_letter(c)})),
                    "is_nfc": unicodedata.is_normalized("NFC", text), "flags": {}}
            item.update(measured[row["id"]])
            items[row["id"]] = item
    test_measured = measure(root, test_rows, signal, False, workers)

    def flag(identity, name, evidence=None):
        items[identity]["flags"].setdefault(name, evidence if evidence is not None else True)

    # per-item flags
    cps_reference = {"hparl": robust([i["chars_per_second"] for i in items.values() if i["subset"] == "hparl_train"]),
                     "css10_el": robust([i["chars_per_second"] for i in items.values() if i["subset"] == "css10_el"])}
    expected = flags_cfg["expected_hparl_format"]
    for item in items.values():
        identity, text = item["id"], subsets[item["subset"]][1][item["id"]]
        if not text.strip():
            flag(identity, "empty_reference")
        elif not strip_annotation_tags(text):
            flag(identity, "annotation_only", item["annotation_tags"])
        if item["annotation_tags"]:
            flag(identity, "annotation_tag", item["annotation_tags"])
        if not item["is_nfc"]:
            flag(identity, "not_nfc")
        if item["non_greek_characters"]:
            flag(identity, "non_greek_characters", item["non_greek_characters"])
        if item["source"] == "hparl" and any(item[k] != v for k, v in expected.items()):
            flag(identity, "format_nonstandard", {k: item[k] for k in expected})
        if item["duration_seconds"] < flags_cfg["duration_short_seconds"]:
            flag(identity, "duration_short", item["duration_seconds"])
        if item["duration_seconds"] > flags_cfg["duration_long_seconds"]:
            flag(identity, "duration_long", item["duration_seconds"])
        if item["peak_dbfs"] is not None and item["peak_dbfs"] <= -199.0:
            flag(identity, "digital_silence", "every sample is zero")
        if item["rms_dbfs"] is not None and item["rms_dbfs"] < flags_cfg["low_level_rms_dbfs"]:
            flag(identity, "low_level", item["rms_dbfs"])
        if item["clip_fraction"] > flags_cfg["clipping_fraction"]:
            flag(identity, "clipping", item["clip_fraction"])
        if item["speech_fraction"] < flags_cfg["silence_heavy_speech_fraction"]:
            flag(identity, "silence_heavy", item["speech_fraction"])
        if max(item["leading_silence_seconds"], item["trailing_silence_seconds"]) > flags_cfg["long_edge_silence_seconds"]:
            flag(identity, "long_edge_silence", [item["leading_silence_seconds"], item["trailing_silence_seconds"]])
        median, mad = cps_reference[item["source"]]
        if mad > 0 and text.strip():
            z = (item["chars_per_second"] - median) / (1.4826 * mad)
            if z > flags_cfg["cps_robust_z"]:
                flag(identity, "cps_high", round(z, 2))
            elif z < -flags_cfg["cps_robust_z"]:
                flag(identity, "cps_low", round(z, 2))
        if item["source"] == "css10_el":
            reported = css10_texts.get(identity, {}).get("reported_duration_seconds")
            if reported is not None and abs(reported - item["duration_seconds"]) > flags_cfg["reported_duration_tolerance_seconds"]:
                flag(identity, "reported_duration_mismatch", {"reported": reported, "measured": item["duration_seconds"]})

    # duplicate and leakage groups (bytes first, then PCM payload)
    members = [(i["subset"], i["id"], i["audio_sha256"], i["pcm_sha256"]) for i in items.values()]
    members += [("hparl_test", r["id"], r["audio_sha256"], test_measured[r["id"]]["pcm_sha256"]) for r in test_rows]
    duplicate_stats = {}
    for kind, index in (("bytes", 2), ("pcm", 3)):
        groups = defaultdict(list)
        for member in members:
            groups[member[index]].append(member)
        multi = [group for group in groups.values() if len(group) > 1]
        duplicate_stats[kind] = {"groups": len(multi), "records_in_groups": sum(len(g) for g in multi)}
        for group in multi:
            if kind == "pcm" and len({m[2] for m in group}) == 1:
                continue  # already handled as identical bytes
            test_ids = sorted(m[1] for m in group if m[0] == "hparl_test")
            dev_ids = sorted(m[1] for m in group if m[0] == "hparl_dev")
            hparl_ids = sorted(m[1] for m in group if m[0] != "css10_el")
            for subset, identity, *_ in group:
                if subset == "hparl_test":
                    continue
                if test_ids:
                    flag(identity, "overlap_test_audio", {"kind": kind, "test_ids": test_ids[:5]})
                if subset == "hparl_train" and dev_ids:
                    flag(identity, "overlap_dev_audio", {"kind": kind, "dev_ids": dev_ids[:5]})
                if subset == "css10_el" and hparl_ids:
                    flag(identity, "overlap_hparl_audio", {"kind": kind, "hparl_ids": hparl_ids[:5]})
            for subset in {m[0] for m in group} - {"hparl_test"}:
                ids = sorted(m[1] for m in group if m[0] == subset)
                for identity in ids[1:]:
                    if "duplicate_audio" not in items[identity]["flags"]:
                        flag(identity, "duplicate_audio" if kind == "bytes" else "duplicate_pcm",
                             {"duplicate_of": ids[0]})

    # same speaker + same words + near-identical duration: suspected re-encoded duplicate
    tolerance = flags_cfg["duplicate_suspect_duration_tolerance_seconds"]
    by_text = defaultdict(list)
    for item in items.values():
        key = overlap_key(subsets[item["subset"]][1][item["id"]])
        if key:
            by_text[(item["subset"], key)].append(item)
    text_duplicates = Counter()
    for (subset, _key), group in by_text.items():
        if len(group) < 2:
            continue
        text_duplicates[subset] += len(group)
        by_speaker = defaultdict(list)
        for item in group:
            if item["speaker_id"]:
                by_speaker[item["speaker_id"]].append(item)
        for same in by_speaker.values():
            same.sort(key=lambda i: (i["duration_seconds"], i["id"]))
            for left, right in zip(same, same[1:]):
                if (right["duration_seconds"] - left["duration_seconds"] <= tolerance
                        and left["audio_sha256"] != right["audio_sha256"]):
                    keeper, other = sorted((left, right), key=lambda i: i["id"])
                    flag(other["id"], "duplicate_suspect", {"similar_to": keeper["id"]})

    # cross-split identity overlaps (reported; test is never altered)
    def ids_of(subset, field):
        if subset == "hparl_test":
            return {r[field] for r in test_rows if r.get(field)}
        return {i[field] for i in items.values() if i["subset"] == subset and i.get(field)}

    test_texts = {overlap_key(r["reference"]) for r in read_jsonl(raw / "references/hparl_test.jsonl")} - {""}
    text_sets = {s: {overlap_key(refs[i]) for i in refs} - {""} for s, (_rows, refs) in subsets.items()}
    text_sets["hparl_test"] = test_texts
    pairs = (("hparl_train", "hparl_dev"), ("hparl_train", "hparl_test"), ("hparl_dev", "hparl_test"))
    overlaps = {}
    for field, name in (("speaker_id", "speakers"), ("source_recording_id", "recordings_or_sessions")):
        overlaps[name] = {f"{a}&{b}": len(ids_of(a, field) & ids_of(b, field)) for a, b in pairs}
    overlaps["original_ids"] = {f"{a}&{b}": len({x.split(":", 1)[1] for x in ids_of(a, "id")} &
                                               {x.split(":", 1)[1] for x in ids_of(b, "id")}) for a, b in pairs}
    overlaps["distinct_texts_exact_after_casefold"] = {f"{a}&{b}": len(text_sets.get(a, set()) & text_sets[b])
                                                       for a, b in pairs}

    summary_subsets = {}
    for subset, (rows, refs) in subsets.items():
        chosen = [items[r["id"]] for r in rows]
        flag_counts, flag_seconds = Counter(), Counter()
        for item in chosen:
            for name in item["flags"]:
                flag_counts[name] += 1
                flag_seconds[name] += item["duration_seconds"]
        texts = [strip_annotation_tags(refs[r["id"]]) for r in rows]
        summary_subsets[subset] = {
            "records": len(rows), "seconds": sum(i["duration_seconds"] for i in chosen),
            "flag_counts": dict(sorted(flag_counts.items())),
            "flag_seconds": {k: round(v, 3) for k, v in sorted(flag_seconds.items())},
            "records_with_any_flag": sum(1 for i in chosen if i["flags"]),
            "duration_seconds": percentiles([i["duration_seconds"] for i in chosen]),
            "rms_dbfs": percentiles([i["rms_dbfs"] for i in chosen]),
            "speech_fraction": percentiles([i["speech_fraction"] for i in chosen]),
            "chars_per_second": percentiles([i["chars_per_second"] for i in chosen]),
            "sampling_rates": dict(sorted(Counter(str(i["sampling_rate"]) for i in chosen).items())),
            "annotation_tags": dict(sorted(Counter(t for r in rows for t in annotation_tags(refs[r["id"]])).items())),
            "character_inventory": character_inventory(texts),
            "uppercase_letters": sum(1 for t in texts for c in t if c.isupper()),
            "letters": sum(1 for t in texts for c in t if c.isalpha()),
            "text_duplicate_records": text_duplicates.get(subset, 0),
            "speakers": len({i["speaker_id"] for i in chosen if i["speaker_id"]}),
        }
    summary = {"schema": "p3-data-audit-v1", "policy_version": config["policy_version"],
               "config_sha256": config_sha, "inputs_sha256": dict(sorted(inputs.items())),
               "cps_reference": {k: {"median": round(m, 4), "mad": round(d, 4)} for k, (m, d) in cps_reference.items()},
               "subsets": summary_subsets, "duplicates": duplicate_stats, "overlaps": overlaps,
               "test_audio_hashed_for_leakage_only": len(test_rows),
               "note": "Flags are candidates, not proven errors. Test audio was only hashed."}
    out.mkdir(parents=True, exist_ok=True)
    for subset in subsets:
        rows = [items[r["id"]] for r in subsets[subset][0]]
        write_output(out / f"items_{subset}.jsonl.gz", gzip.compress(jsonl_bytes(rows), mtime=0))
    write_output(out / "audit_summary.json", json_bytes(summary))
    return summary


def load_items(root: Path = ROOT) -> dict[str, dict]:
    items = {}
    for subset in SUBSETS:
        path = Path(root) / "data/audit" / f"items_{subset}.jsonl.gz"
        if path.exists():
            for line in gzip.decompress(path.read_bytes()).decode("utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    items[item["id"]] = item
    return items


def audit_inputs_current(root: Path, summary: dict) -> bool:
    raw = Path(root) / "data/manifests/raw"
    return all((raw / name).exists() and sha256_file(raw / name) == digest
               for name, digest in summary["inputs_sha256"].items())


__all__ = ["run_audit", "load_items", "audit_inputs_current", "overlap_key", "signal_stats", "sha256_bytes"]
