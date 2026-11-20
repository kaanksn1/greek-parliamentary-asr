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
from .common import (TAG_PATTERN, DataError, ROOT, is_greek_letter, json_bytes, jsonl_bytes, keep_bytes, read_json,
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


class _Ledger:
    """Cleaning log + review queue shared by the full and the selected-scope runs."""

    def __init__(self, version: str):
        self.version, self.log, self.review = version, [], []

    def entry(self, subset, row_or_id, action, rule, reason, evidence=None, before_text=None, after_text=None,
              audio=None, duration=None, kept_audio=True):
        identity = row_or_id if isinstance(row_or_id, str) else row_or_id["id"]
        record = {"id": identity, "subset": subset, "action": action, "rule": rule, "rule_version": self.version,
                  "reason": reason, "evidence": evidence,
                  "before": {"audio_sha256": audio, "text_sha256": text_hash(before_text), "duration_seconds": duration},
                  "after": None if action in ("drop", "not_included") else
                  {"audio_sha256": audio if kept_audio else None, "text_sha256": text_hash(after_text),
                   "duration_seconds": duration}}
        if action == "modify_text":
            record["before_text"], record["after_text"] = before_text, after_text
        self.log.append(record)

    def source_issues(self, subset: str, source: str, issues: list[dict]) -> None:
        for issue in issues:
            kind = issue["issue"]
            if kind in STRUCTURAL or kind in NOT_LABELLED:
                original = issue.get("original_id", f"line-{issue.get('line')}")
                self.entry(subset, f"{source}:{original}", "drop" if kind in STRUCTURAL else "not_included",
                           f"source:{kind}", "unusable or unlabelled source item; recorded, never repaired by guesswork",
                           evidence={k: v for k, v in issue.items() if k != "issue"})

    def sorted_outputs(self) -> tuple[list[dict], list[dict]]:
        order_key = {"hparl_train": 0, "hparl_dev": 1, "css10_el": 2}
        log = sorted(self.log, key=lambda r: (order_key.get(r["subset"], 9), r["id"], r["action"]))
        review = sorted(self.review, key=lambda r: (order_key.get(r["subset"], 9), r["id"]))
        return log, review


def css10_text_settings(config: dict, audit_summary: dict, allowed_hparl: set[str]) -> tuple[bool, dict]:
    harmonization = config["css10_harmonization"]
    hparl_summary = audit_summary["subsets"]["hparl_train"]
    ratio = hparl_summary["uppercase_letters"] / max(1, hparl_summary["letters"])
    lowercase = (harmonization["lowercase"] is True or
                 (harmonization["lowercase"] == "auto" and ratio <= harmonization["auto_lowercase_max_uppercase_ratio"]))
    return lowercase, {"hparl_train_uppercase_ratio": round(ratio, 6), "lowercase": lowercase,
                       "allowed_non_letter_characters": "".join(sorted(
                           c for c in allowed_hparl if unicodedata.category(c)[0] not in "LMN" and not c.isspace()))}


def clean_subset(ledger: _Ledger, subset: str, rows: list[dict], refs: dict[str, str], items: dict, config: dict,
                 decisions: dict, css10_texts: dict, allowed_hparl: set[str], lowercase: bool):
    """Apply the drop rules, text policy and human decisions to one raw subset."""
    rules = config["subsets"][subset]
    kept_rows, kept_refs, counters = [], [], defaultdict(Counter)
    for row in rows:
        identity, item, text = row["id"], items[row["id"]], refs[row["id"]]
        counters["raw"]["records"] += 1
        counters["raw"]["seconds"] += row["duration_seconds"]
        decision = decisions.get(identity)
        drop_rule = next((rule for rule in rules["drop"] if rule in item["flags"]), None)
        if drop_rule:
            ledger.entry(subset, row, "drop", drop_rule, f"automatic rule `{drop_rule}`",
                         evidence=item["flags"][drop_rule], before_text=text, audio=row["audio_sha256"],
                         duration=row["duration_seconds"])
            counters["drop"][drop_rule] += 1
            counters["drop_seconds"][drop_rule] += row["duration_seconds"]
            continue
        if decision and decision["decision"] == "drop":
            ledger.entry(subset, row, "drop", "human_review", decision.get("note", "listened and rejected"),
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
            ledger.entry(subset, row, "drop", rule, "no spoken words remain to train on",
                         evidence=item["flags"].get("annotation_only"), before_text=text, audio=row["audio_sha256"],
                         duration=row["duration_seconds"])
            counters["drop"][rule] += 1
            counters["drop_seconds"][rule] += row["duration_seconds"]
            continue
        if new_text != text:
            rule = "+".join(steps) or "text_policy"
            evidence = ({k: decision[k] for k in ("reviewer", "date", "method")}
                        if decision and decision["decision"] == "correct_text" else None)
            ledger.entry(subset, row, "modify_text", rule, f"text policy `{rules['text']}`", evidence=evidence,
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
            ledger.review.append({"id": identity, "subset": subset, "flags": open_flags, "audio_path": row["audio_path"],
                                  "duration_seconds": row["duration_seconds"], "reference": new_text})
            for name in open_flags:
                counters["review"][name] += 1
        kept_rows.append(row)
        kept_refs.append({"id": identity, "reference": new_text})
        counters["kept"]["records"] += 1
        counters["kept"]["seconds"] += row["duration_seconds"]
    stats = {"raw_records": counters["raw"]["records"], "raw_seconds": round(counters["raw"]["seconds"], 3),
             "kept_records": counters["kept"]["records"], "kept_seconds": round(counters["kept"]["seconds"], 3),
             "dropped_by_rule": dict(sorted(counters["drop"].items())),
             "dropped_seconds_by_rule": {k: round(v, 3) for k, v in sorted(counters["drop_seconds"].items())},
             "modified_records": counters["modified"]["records"],
             "modification_steps": dict(sorted(counters["modify"].items())),
             "review_queue_by_flag": dict(sorted(counters["review"].items()))}
    if stats["raw_records"] != stats["kept_records"] + sum(counters["drop"].values()):
        raise DataError(f"{subset}: lineage does not reconcile (raw != kept + dropped)")
    return kept_rows, kept_refs, stats


def check_decisions(root: Path, decisions: dict, known_ids: set[str]) -> None:
    test_ids = {r["id"] for r in read_jsonl(Path(root) / "data/manifests/test.jsonl")}
    bad = sorted(set(decisions) & test_ids)
    if bad:
        raise DataError(f"review decisions must never touch the frozen test: {bad[:5]}")
    unknown = sorted(set(decisions) - known_ids)
    if unknown:
        raise DataError(f"review decisions reference unknown train/dev/CSS10 IDs: {unknown[:5]}")


def run_clean(root: Path = ROOT, config_path: Path | None = None) -> dict:
    root = Path(root)
    before = verify_freeze(root)
    config, config_sha = load_config(root, config_path)
    audit_summary = read_json(root / "data/audit/audit_summary.json")
    if audit_summary["config_sha256"] != config_sha or not audit_inputs_current(root, audit_summary):
        raise DataError("audit output is stale for the current raw manifests/config; rerun `audit`")
    if audit_summary.get("scope", {}).get("kind", "full_source_inventory") != "full_source_inventory":
        raise DataError("data/audit/audit_summary.json is not a full-inventory audit")
    items = load_items(root)
    decisions = load_review_decisions(root, config)
    raw = root / "data/manifests/raw"
    inventories = {"hparl": read_json(raw / "hparl_inventory.json")}
    if (raw / "css10_el_inventory.json").exists():
        inventories["css10_el"] = read_json(raw / "css10_el_inventory.json")
    css10_texts = ({r["id"]: r for r in read_jsonl(raw / "css10_el_texts.jsonl")}
                   if (raw / "css10_el_texts.jsonl").exists() else {})
    check_decisions(root, decisions, set(items))
    version = config["policy_version"]
    ledger, outputs, stats = _Ledger(version), {}, {}

    # Source items that never entered a raw manifest (no usable audio/label).
    for split in ("train", "dev"):
        ledger.source_issues(f"hparl_{split}", "hparl", inventories["hparl"]["splits"][split]["issues"])
    if "css10_el" in inventories:
        ledger.source_issues("css10_el", "css10_el", inventories["css10_el"]["issues"])

    allowed_hparl: set[str] = set()
    for subset in ("hparl_train", "hparl_dev", "css10_el"):
        manifest = raw / f"{RAW_FILES[subset]}.jsonl"
        if not manifest.exists():
            continue
        rows = read_jsonl(manifest)
        refs = {r["id"]: r["reference"] for r in read_jsonl(raw / "references" / f"{RAW_FILES[subset]}.jsonl")}
        lowercase = False
        if subset == "css10_el":
            lowercase, stats["css10_harmonization"] = css10_text_settings(config, audit_summary, allowed_hparl)
        kept_rows, kept_refs, subset_stats = clean_subset(ledger, subset, rows, refs, items, config, decisions,
                                                          css10_texts, allowed_hparl, lowercase)
        if subset == "hparl_train":
            allowed_hparl = {c for r in kept_refs for c in r["reference"]}
        name = OUTPUTS[subset]
        outputs[subset] = (kept_rows, kept_refs)
        hashes = {"manifest_sha256": write_output(root / f"data/manifests/{name}.jsonl", jsonl_bytes(kept_rows)),
                  "references_sha256": write_output(root / f"data/manifests/references/{name}.jsonl",
                                                    jsonl_bytes(kept_refs))}
        stats[subset] = {**subset_stats, "manifest": f"data/manifests/{name}.jsonl", **hashes}

    # The two training conditions; dev and frozen test are shared and unchanged here.
    hparl_rows, hparl_refs = outputs["hparl_train"]
    css_rows, css_refs = outputs.get("css10_el", ([], []))
    mixed_rows, mixed_refs = merge_by_id(hparl_rows, hparl_refs, css_rows, css_refs)
    mixed_hashes = {"manifest_sha256": write_output(root / "data/manifests/train_hparl_css10.jsonl", jsonl_bytes(mixed_rows)),
                    "references_sha256": write_output(root / "data/manifests/references/train_hparl_css10.jsonl",
                                                      jsonl_bytes(mixed_refs))}
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
    log, review = ledger.sorted_outputs()
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


VERSION_TAG = "css10audit-v2"
PREVIOUS_AUDIT = "data/audit/audit_summary.json"


def _identity(root: Path, relative: str) -> dict:
    return {"path": relative, "sha256": sha256_file(root / relative)}


def _versioned(relative: str) -> str:
    return relative.removesuffix(".jsonl") + f".{VERSION_TAG}.jsonl"


def _place(root: Path, previous: tuple[str, str], data: tuple[bytes, bytes]) -> dict:
    """Keep the previous path when bytes are identical; otherwise write a new versioned pair."""
    old = [root / previous[0], root / previous[1]]
    same = all(path.is_file() and path.read_bytes() == content for path, content in zip(old, data))
    current = previous if same else (_versioned(previous[0]), _versioned(previous[1]))
    if not same:
        for relative, content in zip(current, data):
            keep_bytes(root / relative, content)
    return {"changed": not same,
            "previous": {"manifest": _identity(root, previous[0]) if old[0].is_file() else None,
                         "references": _identity(root, previous[1]) if old[1].is_file() else None},
            "current": {"manifest": _identity(root, current[0]), "references": _identity(root, current[1])}}


def _compare_locked(root: Path, label: str, locked: tuple[str, str], rows: list[dict], refs: list[dict],
                    items: dict) -> dict:
    """Byte comparison with a locked file pair; differences are reported, never written over it."""
    manifest, references = (root / locked[0]).read_bytes(), (root / locked[1]).read_bytes()
    same = manifest == jsonl_bytes(rows) and references == jsonl_bytes(refs)
    result = {"manifest": _identity(root, locked[0]), "references": _identity(root, locked[1]),
              "reproduced_byte_identical": same}
    if not same:
        old_rows = {r["id"]: r for r in read_jsonl(root / locked[0])}
        old_refs = {r["id"]: r["reference"] for r in read_jsonl(root / locked[1])}
        new_rows, new_refs = {r["id"]: r for r in rows}, {r["id"]: r["reference"] for r in refs}
        def why(identity):
            return {"flags": sorted(items.get(identity, {}).get("flags", {}))}
        result["differences"] = {
            "only_in_locked": [{"id": i, **why(i)} for i in sorted(set(old_rows) - set(new_rows))],
            "only_in_recomputed": [{"id": i, **why(i)} for i in sorted(set(new_rows) - set(old_rows))],
            "row_changed": sorted(i for i in set(old_rows) & set(new_rows) if old_rows[i] != new_rows[i]),
            "reference_changed": sorted(i for i in set(old_refs) & set(new_refs) if old_refs[i] != new_refs[i])}
        result["action"] = f"{label}: locked file kept unchanged; difference reported for the owner"
    return result


def _flag_counts(items: list[dict]) -> dict:
    return dict(sorted(Counter(name for item in items for name in item["flags"]).items()))


def css10_flag_comparison(root: Path, current_items: dict) -> dict:
    """Old integer-misread CSS10 flags vs the format-aware audit, item by item."""
    previous_summary = root / PREVIOUS_AUDIT
    if not previous_summary.is_file() or read_json(previous_summary).get("schema") != "p3-data-audit-v1":
        return {"previous": None}
    old_items = root / "data/audit/items_css10_el.jsonl.gz"
    if not old_items.is_file():
        return {"previous": None}
    old = {item["id"]: item for item in load_items(root, old_items.parent, ("css10_el",)).values() if item["subset"] == "css10_el"}
    new = {i: item for i, item in current_items.items() if item["subset"] == "css10_el"}
    names = sorted({n for item in [*old.values(), *new.values()] for n in item["flags"]})
    changes = {}
    for name in names:
        before = {i for i, item in old.items() if name in item["flags"]}
        after = {i for i, item in new.items() if name in item["flags"]}
        if before != after:
            changes[name] = {"previous": len(before), "current": len(after),
                             "added_ids": sorted(after - before)[:50], "removed_ids": sorted(before - after)[:50]}
    def pct(items, key):
        values = sorted(item[key] for item in items if item.get(key) is not None)
        return None if not values else {"min": values[0], "p50": values[len(values) // 2], "max": values[-1]}
    return {"previous": {"audit_summary": _identity(root, PREVIOUS_AUDIT),
                         "status": ("INVALID signal statistics: IEEE float32 sample bytes were read as integer "
                                    "PCM32 (audit schema v1). Kept only as history."),
                         "flag_counts": _flag_counts(list(old.values())),
                         "rms_dbfs": pct(old.values(), "rms_dbfs"), "peak_dbfs": pct(old.values(), "peak_dbfs")},
            "current": {"flag_counts": _flag_counts(list(new.values())),
                        "rms_dbfs": pct(new.values(), "rms_dbfs"), "peak_dbfs": pct(new.values(), "peak_dbfs")},
            "flag_membership_changes": changes,
            "identical_records": len(set(old) & set(new)), "records_previous": len(old), "records_current": len(new)}


def run_clean_selected(root: Path, selection: dict, config_path: Path | None = None) -> dict:
    """Selected-scope clean + CSS10 mixture; locked HParl/dev/test files are only compared, never written."""
    from .audit import audit_dir
    from .cohort import selected_cohort_summary
    from .selection import LINEAGE, check_raw_subset

    root = Path(root)
    before = verify_freeze(root)
    check_raw_subset(root, selection)
    config, config_sha = load_config(root, config_path)
    directory = audit_dir(root, selection)
    audit_summary_path = directory / "audit_summary.json"
    if not audit_summary_path.is_file():
        raise DataError(f"run `audit --selection {selection['path']}` first")
    audit_summary = read_json(audit_summary_path)
    if (audit_summary.get("scope", {}).get("selection_sha256") != selection["sha256"]
            or audit_summary["config_sha256"] != config_sha or not audit_inputs_current(root, audit_summary)):
        raise DataError("selected-scope audit is stale for this selection/config/raw manifests; rerun `audit`")
    items = load_items(root, directory)
    decisions = load_review_decisions(root, config)
    full_train_ids = {r["id"] for r in read_jsonl(root / "data/manifests/raw/hparl_train.jsonl")}
    check_decisions(root, decisions, set(items) | full_train_ids)
    out_of_scope = sorted(set(decisions) - set(items))
    decisions = {k: v for k, v in decisions.items() if k in items}
    raw = root / "data/manifests/raw"
    css10_texts = ({r["id"]: r for r in read_jsonl(raw / "css10_el_texts.jsonl")}
                   if (raw / "css10_el_texts.jsonl").exists() else {})
    version = config["policy_version"]
    ledger, stats = _Ledger(version), {}
    ledger.source_issues("hparl_dev", "hparl", read_json(raw / "hparl_inventory.json")["splits"]["dev"]["issues"])
    if (raw / "css10_el_inventory.json").exists():
        ledger.source_issues("css10_el", "css10_el", read_json(raw / "css10_el_inventory.json")["issues"])
    sources = {"hparl_train": (root / selection["artifacts"]["raw_manifest"]["path"],
                               root / selection["artifacts"]["raw_references"]["path"]),
               "hparl_dev": (raw / "hparl_dev.jsonl", raw / "references/hparl_dev.jsonl"),
               "css10_el": (raw / "css10_el.jsonl", raw / "references/css10_el.jsonl")}
    outputs, allowed_hparl = {}, set()
    for subset, (manifest, references) in sources.items():
        rows = read_jsonl(manifest)
        refs = {r["id"]: r["reference"] for r in read_jsonl(references)}
        lowercase = False
        if subset == "css10_el":
            lowercase, stats["css10_harmonization"] = css10_text_settings(config, audit_summary, allowed_hparl)
        kept_rows, kept_refs, stats[subset] = clean_subset(ledger, subset, rows, refs, items, config, decisions,
                                                           css10_texts, allowed_hparl, lowercase)
        if subset == "hparl_train":
            allowed_hparl = {c for r in kept_refs for c in r["reference"]}
        outputs[subset] = (kept_rows, kept_refs)

    locked = selection["artifacts"]
    hparl_check = _compare_locked(root, "selected clean HParl",
                                  (locked["clean_manifest"]["path"], locked["clean_references"]["path"]),
                                  *outputs["hparl_train"], items)
    dev_check = _compare_locked(root, "dev", ("data/manifests/dev.jsonl", "data/manifests/references/dev.jsonl"),
                                *outputs["hparl_dev"], items)
    css_rows, css_refs = outputs["css10_el"]
    css10 = _place(root, ("data/manifests/css10_el_train.jsonl", "data/manifests/references/css10_el_train.jsonl"),
                   (jsonl_bytes(css_rows), jsonl_bytes(css_refs)))
    # The mixture always uses the LOCKED clean HParl selection, never a recomputed one.
    hparl_rows = read_jsonl(root / locked["clean_manifest"]["path"])
    hparl_refs = read_jsonl(root / locked["clean_references"]["path"])
    mixed_rows, mixed_refs = merge_by_id(hparl_rows, hparl_refs, css_rows, css_refs)
    if len({r["id"] for r in mixed_rows}) != len(mixed_rows) or [r["id"] for r in mixed_rows] != [r["id"] for r in mixed_refs]:
        raise DataError("mixture IDs are duplicated or references are misaligned")
    mixture = _place(root, (locked["mixed_manifest"]["path"], locked["mixed_references"]["path"]),
                     (jsonl_bytes(mixed_rows), jsonl_bytes(mixed_refs)))

    clean_dir = root / "data/clean" / selection["scope_dir"]
    log, review = ledger.sorted_outputs()
    conditions = {"raw_selected_hparl": (locked["raw_manifest"]["path"], locked["raw_references"]["path"]),
                  "clean_selected_hparl": (locked["clean_manifest"]["path"], locked["clean_references"]["path"]),
                  "clean_css10": (css10["current"]["manifest"]["path"], css10["current"]["references"]["path"]),
                  "mixed_hparl_css10": (mixture["current"]["manifest"]["path"],
                                        mixture["current"]["references"]["path"])}
    cohorts = selected_cohort_summary(root, conditions)
    cohort_sha = write_output(clean_dir / "cohort_summary.json", json_bytes(cohorts))
    summary = {"schema": "p3-data-clean-selected-v1", "policy_version": version, "config_sha256": config_sha,
               "selection": {"path": selection["path"], "sha256": selection["sha256"]},
               "audit_summary": _identity(root, audit_summary_path.relative_to(root).as_posix()),
               "audit_schema": audit_summary["schema"], "decoder": audit_summary.get("decoder"),
               "review_decisions_applied": len(decisions), "review_decisions_outside_scope": len(out_of_scope),
               "source_inventory_train_issues_outside_selected_raw_manifest": len(
                   read_json(raw / "hparl_inventory.json")["splits"]["train"]["issues"]),
               "subsets": stats, "cohort_summary_sha256": cohort_sha, "environment": environment(),
               "log_sha256": write_output(clean_dir / "cleaning_log.jsonl", jsonl_bytes(log)),
               "review_queue_sha256": write_output(clean_dir / "review_queue.jsonl", jsonl_bytes(review)),
               "log_entries": dict(sorted(Counter(r["action"] for r in log).items())),
               "review_queue_records": len(review),
               "review_queue_by_subset": dict(sorted(Counter(r["subset"] for r in review).items())),
               "test_freeze_before": before}
    after = verify_freeze(root)
    if after != before:
        raise DataError("test freeze verification changed during cleaning")
    summary["test_freeze_after"] = after
    summary_sha = write_output(clean_dir / "clean_summary.json", json_bytes(summary))
    lineage = {
        "schema": "p3-css10-mixture-lineage-v1",
        "status": ("css10_reaudited_mixture_ready" if hparl_check["reproduced_byte_identical"]
                   and dev_check["reproduced_byte_identical"] else "locked_hparl_difference_reported"),
        "selection": summary["selection"], "policy_version": version, "config_sha256": config_sha,
        "audit": {"summary": summary["audit_summary"], "schema": audit_summary["schema"],
                  "decoder": audit_summary.get("decoder"), "environment": audit_summary.get("environment")},
        "clean": {"summary": {"path": (clean_dir / "clean_summary.json").relative_to(root).as_posix(),
                              "sha256": summary_sha},
                  "cleaning_log": {"path": (clean_dir / "cleaning_log.jsonl").relative_to(root).as_posix(),
                                   "sha256": summary["log_sha256"]},
                  "review_queue": {"path": (clean_dir / "review_queue.jsonl").relative_to(root).as_posix(),
                                   "sha256": summary["review_queue_sha256"], "records": len(review)},
                  "environment": summary["environment"]},
        "locked_inputs_unchanged": {"selected_clean_hparl": hparl_check, "dev": dev_check,
                                    "frozen_test": {"before": before, "after": after}},
        "css10": {**css10, "raw_records": stats["css10_el"]["raw_records"],
                  "raw_hours": round(stats["css10_el"]["raw_seconds"] / 3600, 6),
                  "kept_records": stats["css10_el"]["kept_records"],
                  "kept_hours": round(stats["css10_el"]["kept_seconds"] / 3600, 6),
                  "dropped_by_rule": stats["css10_el"]["dropped_by_rule"],
                  "dropped_seconds_by_rule": stats["css10_el"]["dropped_seconds_by_rule"],
                  "harmonization": stats["css10_harmonization"],
                  "signal_audit": css10_flag_comparison(root, items)},
        "mixture": {**mixture, "rule": "locked clean HParl selection + current clean CSS10, sorted by id",
                    "cohort": cohorts["conditions"]["mixed_hparl_css10"]},
        "note": ("CSS10 is training-only. The locked HParl selection, dev and the frozen test are compared "
                 "byte-for-byte and never rewritten. Model configs and budgets are updated by the owner."),
    }
    write_output(root / LINEAGE, json_bytes(lineage))
    if lineage["status"] != "css10_reaudited_mixture_ready":
        raise DataError(f"recomputed locked HParl/dev output differs; see {LINEAGE} (locked files were not changed)")
    return summary


def merge_by_id(rows_a, refs_a, rows_b, refs_b) -> tuple[list[dict], list[dict]]:
    order = sorted(range(len(rows_a) + len(rows_b)), key=lambda i: (rows_a + rows_b)[i]["id"])
    return [(rows_a + rows_b)[i] for i in order], [(refs_a + refs_b)[i] for i in order]


def describe(rows):
    by_source = defaultdict(lambda: [0, 0.0])
    for row in rows:
        by_source[row["source"]][0] += 1
        by_source[row["source"]][1] += row["duration_seconds"]
    return {"records": {k: v[0] for k, v in sorted(by_source.items())},
            "seconds": {k: round(v[1], 3) for k, v in sorted(by_source.items())},
            "total_seconds": round(sum(v[1] for v in by_source.values()), 3),
            "sampling_rates": dict(sorted(Counter(str(r["sampling_rate"]) for r in rows).items()))}


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
