"""Owner's session-cohort/storage decision; reuses Samed's subset algorithm.

Select without consulting test performance. Deliver every RAW training WAV in
the chosen sessions, including examples removed by the existing cleaning rule.
Clean manifests reuse the same original WAV bytes; no duplicate cleaned audio.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from greek_asr.data import clean, cohort, freeze
from greek_asr.data.common import json_bytes, jsonl_bytes, read_jsonl, sha256_bytes, sha256_file


def sizes():
    result = {}
    for source in ("hparl", "css10_el"):
        with (ROOT / f"data/raw/{source}/archive_members.tsv").open(encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle, delimiter="\t"):
                result[f"data/raw/{source}/release/" + item["path"]] = int(item["bytes"])
    return result


def objects(rows, byte_sizes):
    return {row["audio_sha256"]: byte_sizes[row["audio_path"]] for row in rows}


def candidates(seed, reserve_gib, quota_gib):
    rows = read_jsonl(ROOT / "data/manifests/train.jsonl")
    raw = read_jsonl(ROOT / "data/manifests/raw/hparl_train.jsonl")
    byte_sizes = sizes()
    existing = [*read_jsonl(ROOT / "data/manifests/dev.jsonl"), *read_jsonl(ROOT / "data/manifests/test.jsonl"),
                *read_jsonl(ROOT / "data/manifests/css10_el_train.jsonl")]
    existing_objects = objects(existing, byte_sizes)
    by_session = defaultdict(list)
    raw_by_session = defaultdict(list)
    for row in rows:
        by_session[row["source_recording_id"]].append(row)
    for row in raw:
        raw_by_session[row["source_recording_id"]].append(row)
    keys = sorted(by_session)
    random.Random(seed).shuffle(keys)
    selected, selected_raw, result = [], [], []
    for session in keys:
        selected.extend(by_session[session])
        selected_raw.extend(raw_by_session[session])
        train_objects = objects(selected_raw, byte_sizes)
        all_objects = {**existing_objects, **train_objects}
        lfs_gib = sum(all_objects.values()) / 1024**3
        result.append({"sessions": len({row["source_recording_id"] for row in selected}),
                       "hours": sum(row["duration_seconds"] for row in selected) / 3600,
                       "clean_records": len(selected), "raw_records": len(selected_raw),
                       "raw_train_unique_lfs_bytes": sum(train_objects.values()),
                       "all_audio_unique_lfs_gib": lfs_gib,
                       "checkpoint_and_headroom_reserve_gib": reserve_gib,
                       "within_storage_budget": lfs_gib + reserve_gib <= quota_gib})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota-gib", type=float, default=10.)
    parser.add_argument("--reserve-gib", type=float, default=2.6)
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    table = candidates(args.seed, args.reserve_gib, args.quota_gib)
    if args.list_candidates:
        print(json.dumps(table, ensure_ascii=False, indent=2))
        return
    if args.hours is None or args.hours <= 0:
        parser.error("--hours must be positive (or use --list-candidates)")
    frozen_before = freeze.verify_freeze(ROOT, check_audio=False)
    selection = clean.select_subset(ROOT, args.hours, args.seed, True)
    tag = f"train_hparl_{args.hours:g}h_seed{args.seed}"
    base = ROOT / "data/manifests/subsets"
    sessions = set(selection["group_keys"])
    raw = [row for row in read_jsonl(ROOT / "data/manifests/raw/hparl_train.jsonl")
           if row["source_recording_id"] in sessions]
    raw_refs = {row["id"]: row for row in read_jsonl(ROOT / "data/manifests/raw/references/hparl_train.jsonl")}
    selected = read_jsonl(base / f"{tag}.jsonl")
    refs = read_jsonl(base / f"references/{tag}.jsonl")
    mixed = read_jsonl(base / f"{tag}_css10.jsonl")
    mixed_refs = read_jsonl(base / f"references/{tag}_css10.jsonl")
    raw = sorted(raw, key=lambda row: row["id"])
    (base / "raw/references").mkdir(parents=True, exist_ok=True)
    raw_manifest = base / f"raw/{tag}.jsonl"
    raw_references = base / f"raw/references/{tag}.jsonl"
    raw_manifest.write_bytes(jsonl_bytes(raw))
    raw_references.write_bytes(jsonl_bytes(raw_refs[row["id"]] for row in raw))
    byte_sizes = sizes()
    kept_ids = {item["id"] for item in selected}
    delivery = [{"id": row["id"], "audio_path": row["audio_path"], "audio_sha256": row["audio_sha256"],
                 "bytes": byte_sizes[row["audio_path"]], "source_recording_id": row["source_recording_id"],
                 "kept_after_cleaning": row["id"] in kept_ids}
                for row in raw]
    delivery_path = base / f"{tag}.delivery.jsonl"
    delivery_path.write_bytes(jsonl_bytes(delivery))
    chosen_budget = next(row for row in table if row["sessions"] == selection["groups_selected"])
    if not chosen_budget["within_storage_budget"]:
        raise ValueError("selected source audio plus checkpoint reserve exceeds the authorized storage budget")
    test_rows = read_jsonl(ROOT / "data/manifests/test.jsonl")
    test_refs = [(row["id"], row["reference"]) for row in read_jsonl(ROOT / "data/manifests/references/test.jsonl")]
    dev_rows = read_jsonl(ROOT / "data/manifests/dev.jsonl")
    test_speakers = {row["speaker_id"] for row in test_rows if row["speaker_id"]}
    selected_speakers = {row["speaker_id"] for row in selected if row["speaker_id"]}
    train_sessions = {row["source_recording_id"] for row in selected}
    if train_sessions & {row["source_recording_id"] for row in test_rows + dev_rows}:
        raise ValueError("a training session overlaps official dev/test")
    for label, rows in (("raw_hparl", raw), ("clean_hparl", selected), ("mixed_hparl_css10", mixed)):
        if any(row["split"] != "train" for row in rows):
            raise ValueError(f"non-training example in selected {label}")
    def identity(path):
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
    plan = {"schema": "p3-selected-training-delivery-v1", "status": "selected_audio_upload_pending",
            "selection_rule": "Samed's seeded whole-session prefix; no test performance used",
            "source_data_submission": "4db1a4aa61eb78acbb531b705c45e3bbd090854e",
            "seed": args.seed, "target_hours": args.hours, "session_ids": sorted(sessions),
            "subset_selection": selection, "storage": {**chosen_budget, "authorized_quota_gib": args.quota_gib,
                "remaining_account_quota_verified": False, "paid_storage_authorized": False,
                "note": "10 GiB conservative account storage cap; bandwidth is separate and must be checked before further LFS downloads"},
            "artifacts": {"raw_manifest": identity(raw_manifest), "raw_references": identity(raw_references),
                          "clean_manifest": identity(base / f"{tag}.jsonl"),
                          "clean_references": identity(base / f"references/{tag}.jsonl"),
                          "mixed_manifest": identity(base / f"{tag}_css10.jsonl"),
                          "mixed_references": identity(base / f"references/{tag}_css10.jsonl"),
                          "raw_audio_delivery_list": identity(delivery_path)},
            "cohorts": {"raw_selected_train": cohort.cohort_stats(raw), "clean_selected_train": cohort.cohort_stats(selected),
                        "mixed_selected_train": cohort.cohort_stats(mixed), "unchanged_clean_dev": cohort.cohort_stats(dev_rows),
                        "unchanged_frozen_test": cohort.cohort_stats(test_rows)},
            "literal_test_overlap_in_actual_selected_targets": {
                "hparl_only": cohort.literal_overlap([row["reference"] for row in refs], test_refs),
                "hparl_css10": cohort.literal_overlap([row["reference"] for row in mixed_refs], test_refs)},
            "normalized_overlap_diagnostic_not_final_scoring": cohort.normalized_overlap([row["reference"] for row in mixed_refs], test_refs),
            "official_split_speaker_overlap": {"test_speakers_also_in_selected_train": len(test_speakers & selected_speakers),
                "test_utterances_with_selected_train_speaker": sum(row["speaker_id"] in selected_speakers for row in test_rows),
                "session_overlap": 0, "speaker_disjoint": False},
            "frozen_test": frozen_before,
            "raw_audio_policy": "Deliver ALL raw WAVs in selected training sessions, including removed examples. Clean targets reuse original WAVs."}
    if freeze.verify_freeze(ROOT, check_audio=False) != frozen_before:
        raise ValueError("test freeze changed during training selection")
    path = ROOT / "configs/training_data_selection.json"
    path.write_bytes(json_bytes(plan))
    print(json.dumps({"plan": path.relative_to(ROOT).as_posix(), "storage": plan["storage"],
                      "cohorts": plan["cohorts"], "session_ids": plan["session_ids"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
