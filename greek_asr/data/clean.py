"""Rule-based train/dev/CSS10 cleaning with a complete machine-readable log.

The test freeze is verified before and after cleaning; the test files are
never opened for writing here. Raw source bytes are never modified: a clean
manifest refers to the original audio of each retained record. Every removal
or transcript change is one log line with rule, reason, evidence and the
before/after hashes. Automatic flags that are not removal rules go to a
review queue for listening; a human decision is applied only when it is
recorded in ``configs/review_decisions.jsonl`` with reviewer and method.
"""
from __future__ import annotations

import platform
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .audit import audit_inputs_current, load_config, load_items
from .cohort import cohort_summary
from .common import (TAG_PATTERN, DataError, ROOT, is_greek_letter, json_bytes, jsonl_bytes, read_json,
                     read_jsonl, sha256_bytes, sha256_file, strip_annotation_tags, write_output)
from .freeze import verify_freeze

STRUCTURAL = {"missing_audio", "audio_unreadable", "zero_frames", "ambiguous_audio", "duplicate_id",
              "malformed_transcript_line", "audio_path_escapes_release"}
NOT_LABELLED = {"audio_without_transcript"}
REVIEW_FLAGS = {"cps_boundary", "cps_extreme", "digital_silence", "duration_short", "duration_long", "low_level",
                "clipping", "silence_heavy", "long_edge_silence", "cps_high", "cps_low", "duplicate_suspect", "non_greek_characters", "format_nonstandard",
                "reported_duration_mismatch", "annotation_only", "empty_reference"}
OUTPUTS = {"hparl_train": "train", "hparl_dev": "dev", "css10_el": "css10_el_train"}
RAW_FILES = {"hparl_train": "hparl_train", "hparl_dev": "hparl_dev", "css10_el": "css10_el"}


def environment() -> dict:
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:  # recorded, the audit step itself needs numpy
        numpy_version = None
    return {"python": platform.python_version(), "numpy": numpy_version, "system": platform.system()}


def text_hash(text: str | None) -> str | None:
    return None if text is None else sha256_bytes(text.encode("utf-8"))


def load_review_decisions(root: Path, config: dict) -> dict[str, dict]:
    path = Path(root) / config.get("review_decisions", "configs/review_decisions.jsonl")
    if not path.exists():
        return {}
    decisions = {}
    for row in read_jsonl(path):
        missing = [k for k in ("id", "decision", "reviewer", "date", "method") if not row.get(k)]
        if missing:
            raise DataError(f"review decision {row.get('id')} lacks {missing}; human review must be attributable")
        if row["decision"] not in ("keep", "drop", "correct_text"):
            raise DataError(f"review decision {row['id']}: unknown decision {row['decision']!r}")
        if row["decision"] == "correct_text" and not str(row.get("corrected_text", "")).strip():
            raise DataError(f"review decision {row['id']}: corrected_text is required")
        if row["id"] in decisions:
            raise DataError(f"duplicate review decision for {row['id']}")
        decisions[row["id"]] = row
    return decisions


def harmonize(text: str, allowed: set[str], lowercase: bool) -> tuple[str, list[str]]:
    """Map a CSS10 producer-normalized line onto the HParl transcription style."""
    steps, result = [], unicodedata.normalize("NFC", text)
    if result != text:
        steps.append("unicode_nfc")
    if lowercase:
        lowered = unicodedata.normalize("NFC", result.lower())  # str.lower handles Greek final sigma
        if lowered != result:
            steps.append("lowercase")
        result = lowered
    kept = []
    for character in result:
        category = unicodedata.category(character)
        if character.isspace() or category[0] in "LMN" or character in allowed:
            kept.append(character)
        else:
            kept.append(" ")
    stripped = "".join(kept)
    if stripped != result:
        steps.append("remove_characters_absent_from_hparl_train")
    collapsed = " ".join(stripped.split())
    if collapsed != stripped:
        steps.append("collapse_whitespace")
    return collapsed, steps


def hparl_text_policy(text: str, policy: str) -> tuple[str, list[str]]:
    if policy == "unchanged":
        return text, []
    if policy != "strip_annotation_tags":
        raise DataError(f"unknown HParl text policy {policy!r}")
    steps, result = [], unicodedata.normalize("NFC", text)
    if result != text:
        steps.append("unicode_nfc")
    stripped = strip_annotation_tags(result)  # also collapses whitespace
    if TAG_PATTERN.search(result):
        steps.append("strip_annotation_tags")
    elif stripped != result:
        steps.append("collapse_whitespace")
    return stripped, steps


def run_clean(root: Path = ROOT, config_path: Path | None = None) -> dict:
    root = Path(root)
    before = verify_freeze(root)
    config, config_sha = load_config(root, config_path)
    audit_summary = read_json(root / "data/audit/audit_summary.json")
    if audit_summary["config_sha256"] != config_sha or not audit_inputs_current(root, audit_summary):
        raise DataError("audit output is stale for the current raw manifests/config; rerun `audit`")
    items = load_items(root)
    decisions = load_review_decisions(root, config)
    raw = root / "data/manifests/raw"
    inventories = {"hparl": read_json(raw / "hparl_inventory.json")}
    if (raw / "css10_el_inventory.json").exists():
        inventories["css10_el"] = read_json(raw / "css10_el_inventory.json")
    css10_texts = ({r["id"]: r for r in read_jsonl(raw / "css10_el_texts.jsonl")}
                   if (raw / "css10_el_texts.jsonl").exists() else {})
    test_ids = {r["id"] for r in read_jsonl(root / "data/manifests/test.jsonl")}
    bad = sorted(set(decisions) & test_ids)
    if bad:
        raise DataError(f"review decisions must never touch the frozen test: {bad[:5]}")
    unknown = sorted(set(decisions) - set(items))
    if unknown:
        raise DataError(f"review decisions reference unknown train/dev/CSS10 IDs: {unknown[:5]}")
    version = config["policy_version"]
    log, review, outputs, stats = [], [], {}, {}

    def entry(subset, row_or_id, action, rule, reason, evidence=None, before_text=None, after_text=None,
              audio=None, duration=None, kept_audio=True):
        identity = row_or_id if isinstance(row_or_id, str) else row_or_id["id"]
        record = {"id": identity, "subset": subset, "action": action, "rule": rule, "rule_version": version,
                  "reason": reason, "evidence": evidence,
                  "before": {"audio_sha256": audio, "text_sha256": text_hash(before_text), "duration_seconds": duration},
                  "after": None if action in ("drop", "not_included") else
                  {"audio_sha256": audio if kept_audio else None, "text_sha256": text_hash(after_text),
                   "duration_seconds": duration}}
        if action == "modify_text":
            record["before_text"], record["after_text"] = before_text, after_text
        log.append(record)

    # Source items that never entered a raw manifest (no usable audio/label).
    for source, subset_by_split in (("hparl", {"train": "hparl_train", "dev": "hparl_dev"}), ("css10_el", None)):
        inventory = inventories.get(source)
        if not inventory:
            continue
        groups = ([(subset_by_split[s], inventory["splits"][s]["issues"]) for s in ("train", "dev")]
                  if subset_by_split else [("css10_el", inventory["issues"])])
        for subset, issues in groups:
            for issue in issues:
                kind = issue["issue"]
                if kind in STRUCTURAL or kind in NOT_LABELLED:
                    original = issue.get("original_id", f"line-{issue.get('line')}")
                    entry(subset, f"{source}:{original}", "drop" if kind in STRUCTURAL else "not_included",
                          f"source:{kind}", "unusable or unlabelled source item; recorded, never repaired by guesswork",
                          evidence={k: v for k, v in issue.items() if k != "issue"})

    allowed_hparl: set[str] = set()
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        manifest = raw / f"{RAW_FILES[subset]}.jsonl"
        if not manifest.exists():
            continue
        rules = config["subsets"][subset]
        rows = read_jsonl(manifest)
        refs = {r["id"]: r["reference"] for r in read_jsonl(raw / "references" / f"{RAW_FILES[subset]}.jsonl")}
        lowercase = False
        if subset == "css10_el":
            harmonization = config["css10_harmonization"]
            hparl_summary = audit_summary["subsets"]["hparl_train"]
            ratio = hparl_summary["uppercase_letters"] / max(1, hparl_summary["letters"])
            lowercase = (harmonization["lowercase"] is True or
                         (harmonization["lowercase"] == "auto" and ratio <= harmonization["auto_lowercase_max_uppercase_ratio"]))
            stats["css10_harmonization"] = {"hparl_train_uppercase_ratio": round(ratio, 6), "lowercase": lowercase,
                                            "allowed_non_letter_characters": "".join(sorted(
                                                c for c in allowed_hparl if unicodedata.category(c)[0] not in "LMN"
                                                and not c.isspace()))}
        kept_rows, kept_refs, counters = [], [], defaultdict(Counter)
        for row in rows:
            identity, item, text = row["id"], items[row["id"]], refs[row["id"]]
            counters["raw"]["records"] += 1
            counters["raw"]["seconds"] += row["duration_seconds"]
            decision = decisions.get(identity)
            drop_rule = next((rule for rule in rules["drop"] if rule in item["flags"]), None)
            if drop_rule:
                entry(subset, row, "drop", drop_rule, f"automatic rule `{drop_rule}`",
                      evidence=item["flags"][drop_rule], before_text=text, audio=row["audio_sha256"],
                      duration=row["duration_seconds"])
                counters["drop"][drop_rule] += 1
                counters["drop_seconds"][drop_rule] += row["duration_seconds"]
                continue
            if decision and decision["decision"] == "drop":
                entry(subset, row, "drop", "human_review", decision.get("note", "listened and rejected"),
                      evidence={k: decision[k] for k in ("reviewer", "date", "method")}, before_text=text,
                      audio=row["audio_sha256"], duration=row["duration_seconds"])
                counters["drop"]["human_review"] += 1
                counters["drop_seconds"]["human_review"] += row["duration_seconds"]
                continue
            if subset == "css10_el":
                source_text = css10_texts.get(identity, {}).get(config["css10_harmonization"]["source_field"], text)
                new_text, steps = harmonize(source_text, allowed_hparl, lowercase)
            else:
                new_text, steps = hparl_text_policy(text, rules["text"])
            if decision and decision["decision"] == "correct_text":
                new_text, steps = decision["corrected_text"], steps + ["human_review_correction"]
            if not new_text.strip() and rules["drop_if_empty_after_text_policy"]:
                rule = "annotation_only" if item["flags"].get("annotation_only") else "empty_after_text_policy"
                entry(subset, row, "drop", rule, "no spoken words remain to train on",
                      evidence=item["flags"].get("annotation_only"), before_text=text, audio=row["audio_sha256"],
                      duration=row["duration_seconds"])
                counters["drop"][rule] += 1
                counters["drop_seconds"][rule] += row["duration_seconds"]
                continue
            if new_text != text:
                rule = "+".join(steps) or "text_policy"
                evidence = ({k: decision[k] for k in ("reviewer", "date", "method")}
                            if decision and decision["decision"] == "correct_text" else None)
                entry(subset, row, "modify_text", rule, f"text policy `{rules['text']}`", evidence=evidence,
                      before_text=text, after_text=new_text, audio=row["audio_sha256"],
                      duration=row["duration_seconds"])
                for step in steps:
                    counters["modify"][step] += 1
                counters["modified"]["records"] += 1
            open_flags = {name: value for name, value in item["flags"].items() if name in REVIEW_FLAGS}
            if subset == "css10_el" and "non_greek_characters" in open_flags:
                remaining = "".join(sorted({c for c in new_text if not c.isspace() and not is_greek_letter(c)}))
                if remaining:
                    open_flags["non_greek_characters"] = remaining
                else:
                    open_flags.pop("non_greek_characters")
            if decision and decision["decision"] == "keep":
                counters["review"]["human_keep"] += 1
            elif open_flags:
                review.append({"id": identity, "subset": subset, "flags": open_flags, "audio_path": row["audio_path"],
                               "duration_seconds": row["duration_seconds"], "reference": new_text})
                for name in open_flags:
                    counters["review"][name] += 1
            kept_rows.append(row)
            kept_refs.append({"id": identity, "reference": new_text})
            counters["kept"]["records"] += 1
            counters["kept"]["seconds"] += row["duration_seconds"]
        if subset == "hparl_train":
            allowed_hparl = {c for r in kept_refs for c in r["reference"]}
        name = OUTPUTS[subset]
        outputs[subset] = (kept_rows, kept_refs)
        hashes = {"manifest_sha256": write_output(root / f"data/manifests/{name}.jsonl", jsonl_bytes(kept_rows)),
                  "references_sha256": write_output(root / f"data/manifests/references/{name}.jsonl",
                                                    jsonl_bytes(kept_refs))}
        stats[subset] = {"raw_records": counters["raw"]["records"], "raw_seconds": round(counters["raw"]["seconds"], 3),
                         "kept_records": counters["kept"]["records"], "kept_seconds": round(counters["kept"]["seconds"], 3),
                         "dropped_by_rule": dict(sorted(counters["drop"].items())),
                         "dropped_seconds_by_rule": {k: round(v, 3) for k, v in sorted(counters["drop_seconds"].items())},
                         "modified_records": counters["modified"]["records"],
                         "modification_steps": dict(sorted(counters["modify"].items())),
                         "review_queue_by_flag": dict(sorted(counters["review"].items())),
                         "manifest": f"data/manifests/{name}.jsonl", **hashes}
        if stats[subset]["raw_records"] != stats[subset]["kept_records"] + sum(counters["drop"].values()):
            raise DataError(f"{subset}: lineage does not reconcile (raw != kept + dropped)")

    # The two training conditions; dev and frozen test are shared and unchanged here.
    hparl_rows, hparl_refs = outputs["hparl_train"]
    css_rows, css_refs = outputs.get("css10_el", ([], []))
    order = sorted(range(len(hparl_rows) + len(css_rows)), key=lambda i: (hparl_rows + css_rows)[i]["id"])
    mixed_rows = [(hparl_rows + css_rows)[i] for i in order]
    mixed_refs = [(hparl_refs + css_refs)[i] for i in order]
    mixed_hashes = {"manifest_sha256": write_output(root / "data/manifests/train_hparl_css10.jsonl", jsonl_bytes(mixed_rows)),
                    "references_sha256": write_output(root / "data/manifests/references/train_hparl_css10.jsonl",
                                                      jsonl_bytes(mixed_refs))}

    def describe(rows):
        by_source = defaultdict(lambda: [0, 0.0])
        for row in rows:
            by_source[row["source"]][0] += 1
            by_source[row["source"]][1] += row["duration_seconds"]
        return {"records": {k: v[0] for k, v in sorted(by_source.items())},
                "seconds": {k: round(v[1], 3) for k, v in sorted(by_source.items())},
                "total_seconds": round(sum(v[1] for v in by_source.values()), 3),
                "sampling_rates": dict(sorted(Counter(str(r["sampling_rate"]) for r in rows).items()))}

    freeze = read_json(root / "data/manifests/test.freeze.json")
    conditions = {
        "schema": "p3-training-conditions-v1", "policy_version": version,
        "conditions": {
            "hparl_only": {"train_manifest": "data/manifests/train.jsonl",
                           "train_references": "data/manifests/references/train.jsonl",
                           "manifest_sha256": stats["hparl_train"]["manifest_sha256"],
                           "references_sha256": stats["hparl_train"]["references_sha256"], **describe(hparl_rows)},
            "hparl_css10": {"train_manifest": "data/manifests/train_hparl_css10.jsonl",
                            "train_references": "data/manifests/references/train_hparl_css10.jsonl",
                            **mixed_hashes, **describe(mixed_rows)}},
        "shared_dev": {"manifest": "data/manifests/dev.jsonl", "references": "data/manifests/references/dev.jsonl",
                       "manifest_sha256": stats["hparl_dev"]["manifest_sha256"],
                       "references_sha256": stats["hparl_dev"]["references_sha256"],
                       **describe(outputs["hparl_dev"][0])},
        "frozen_test": {"manifest": freeze["test_manifest_path"], "manifest_sha256": freeze["test_manifest_sha256"],
                        "references_sha256": freeze["references_sha256"], "records": freeze["record_count"],
                        "freeze_sha256": sha256_file(root / "data/manifests/test.freeze.json")},
        "mixing_policy": ("Not set by the data owner. Mixing ratio, sampling, steps/epochs and the matched "
                          "optimisation budget are chosen by Adnan/Serdar on HParl dev and recorded in their runs."),
        "note": "CSS10 audio keeps its original sampling rate; model input pipelines resample at load time.",
    }
    write_output(root / "data/manifests/conditions.json", json_bytes(conditions))
    cohort_sha = write_output(root / "data/clean/cohort_summary.json", json_bytes(cohort_summary(root)))
    order_key = {"hparl_train": 0, "hparl_dev": 1, "css10_el": 2}
    log.sort(key=lambda r: (order_key.get(r["subset"], 9), r["id"], r["action"]))
    review.sort(key=lambda r: (order_key.get(r["subset"], 9), r["id"]))
    summary = {"schema": "p3-data-clean-v1", "policy_version": version, "config_sha256": config_sha,
               "audit_summary_sha256": sha256_file(root / "data/audit/audit_summary.json"),
               "review_decisions_applied": len(decisions), "subsets": stats, "cohort_summary_sha256": cohort_sha,
               "environment": environment(),
               "log_sha256": write_output(root / "data/clean/cleaning_log.jsonl", jsonl_bytes(log)),
               "review_queue_sha256": write_output(root / "data/clean/review_queue.jsonl", jsonl_bytes(review)),
               "log_entries": dict(sorted(Counter(r["action"] for r in log).items())),
               "review_queue_records": len(review),
               "test_freeze_before": before}
    after = verify_freeze(root)
    if after != before:
        raise DataError("test freeze verification changed during cleaning")
    summary["test_freeze_after"] = after
    write_output(root / "data/clean/clean_summary.json", json_bytes(summary))
    return summary


def select_subset(root: Path = ROOT, hours: float = 10.0, seed: int = 42, with_css10: bool = True) -> dict:
    """Seeded selection of whole HParl sessions from the clean train (budget experiments)."""
    root = Path(root)
    rows = read_jsonl(root / "data/manifests/train.jsonl")
    refs = {r["id"]: r for r in read_jsonl(root / "data/manifests/references/train.jsonl")}
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_recording_id"] or row["speaker_id"] or row["id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    chosen, seconds, target = [], 0.0, hours * 3600
    for key in keys:
        if seconds >= target:
            break
        chosen.append(key)
        seconds += sum(r["duration_seconds"] for r in groups[key])
    selected = sorted((r for key in chosen for r in groups[key]), key=lambda r: r["id"])
    tag = f"train_hparl_{hours:g}h_seed{seed}"
    base = root / "data/manifests/subsets"
    result = {"seed": seed, "target_hours": hours, "selected_seconds": round(seconds, 3),
              "group_field": "source_recording_id (fallback speaker_id, then id)", "groups_selected": len(chosen),
              "group_keys": sorted(chosen), "records": len(selected),
              "ids_sha256": sha256_bytes("\n".join(r["id"] for r in selected).encode("utf-8")),
              "source_manifest_sha256": sha256_file(root / "data/manifests/train.jsonl"),
              "manifest_sha256": write_output(base / f"{tag}.jsonl", jsonl_bytes(selected)),
              "references_sha256": write_output(base / f"references/{tag}.jsonl",
                                                jsonl_bytes(refs[r["id"]] for r in selected))}
    if with_css10 and (root / "data/manifests/css10_el_train.jsonl").exists():
        css = read_jsonl(root / "data/manifests/css10_el_train.jsonl")
        css_refs = read_jsonl(root / "data/manifests/references/css10_el_train.jsonl")
        mixed = sorted(selected + css, key=lambda r: r["id"])
        mixed_refs = {r["id"]: r for r in [*(refs[r["id"]] for r in selected), *css_refs]}
        result["mixed_manifest_sha256"] = write_output(base / f"{tag}_css10.jsonl", jsonl_bytes(mixed))
        result["mixed_references_sha256"] = write_output(base / f"references/{tag}_css10.jsonl",
                                                         jsonl_bytes(mixed_refs[r["id"]] for r in mixed))
    write_output(base / f"{tag}.selection.json", json_bytes(result))
    return result
