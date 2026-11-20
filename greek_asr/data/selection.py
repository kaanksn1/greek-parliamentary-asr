"""Locked training-selection scope for audit -> clean -> mixture -> verify.

``configs/training_data_selection.json`` (owner's session/storage decision,
built with ``subset --hours 30.5 --seed 42``) names the raw and clean HParl
manifests that are actually delivered. With ``--selection`` the data commands
read only those raw train records plus the unchanged dev, the frozen test and
CSS10, so the unused source-inventory train audio is never required.

Every locked artifact is hash-checked before use; this module never writes
or regenerates the selection, its HParl manifests, dev or the frozen test.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import (DataError, ROOT, read_json, read_jsonl, relative_posix, resolve_inside, sha256_file,
                     validate_manifest_row)

ARTIFACTS = ("raw_manifest", "raw_references", "clean_manifest", "clean_references", "mixed_manifest",
             "mixed_references")
LINEAGE = "configs/css10_mixture_lineage.json"


def load_selection(root: Path = ROOT, path: Path | str = "configs/training_data_selection.json") -> dict:
    root = Path(root)
    path = Path(path)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise DataError(f"selection plan not found: {path}")
    plan = read_json(path)
    artifacts = plan.get("artifacts", {})
    missing = [name for name in ARTIFACTS if name not in artifacts]
    if missing or not plan.get("session_ids"):
        raise DataError(f"{relative_posix(path, root)}: selection plan lacks {missing or ['session_ids']}")
    locked = {}
    for name in ARTIFACTS:
        relative, expected = artifacts[name]["path"], artifacts[name]["sha256"]
        file = resolve_inside(root, relative)
        if not file.is_file():
            raise DataError(f"locked selection artifact missing: {relative}")
        if sha256_file(file) != expected:
            raise DataError(f"locked selection artifact changed (hash differs from the plan): {relative}")
        locked[name] = {"path": relative, "sha256": expected}
    sessions = set(plan["session_ids"])
    raw_rows = read_jsonl(resolve_inside(root, locked["raw_manifest"]["path"]))
    if {row["source_recording_id"] for row in raw_rows} != sessions:
        raise DataError("selected raw manifest sessions differ from the plan's session_ids")
    if any(row["source"] != "hparl" or row["split"] != "train" for row in raw_rows):
        raise DataError("selected raw manifest must contain only official HParl train records")
    tag = Path(locked["clean_manifest"]["path"]).name.removesuffix(".jsonl")
    return {"path": relative_posix(path, root), "sha256": sha256_file(path), "tag": tag,
            "scope_dir": f"selected_{tag}", "session_ids": sorted(sessions), "artifacts": locked,
            "seed": plan.get("seed"), "raw_rows": raw_rows}


def check_raw_subset(root: Path, selection: dict) -> None:
    """The selected raw rows must be byte-for-byte rows of the full raw train manifest."""
    full = {row["id"]: row for row in read_jsonl(Path(root) / "data/manifests/raw/hparl_train.jsonl")}
    for row in selection["raw_rows"]:
        if full.get(row["id"]) != row:
            raise DataError(f"{row['id']}: selected raw row differs from data/manifests/raw/hparl_train.jsonl")
    sessions = set(selection["session_ids"])
    expected = sorted(identity for identity, row in full.items() if row["source_recording_id"] in sessions)
    if expected != sorted(row["id"] for row in selection["raw_rows"]):
        raise DataError("selected raw manifest is not every raw record of the selected whole sessions")


def current_training_manifests(root: Path, selection: dict) -> dict:
    """Paths of the CSS10 and mixture manifests that are current for this selection."""
    root = Path(root)
    lineage_path = root / LINEAGE
    if lineage_path.is_file():
        lineage = read_json(lineage_path)
        if lineage.get("selection", {}).get("sha256") != selection["sha256"]:
            raise DataError(f"{LINEAGE} was produced for a different selection plan")
        current = {"css10_manifest": lineage["css10"]["current"]["manifest"],
                   "css10_references": lineage["css10"]["current"]["references"],
                   "mixed_manifest": lineage["mixture"]["current"]["manifest"],
                   "mixed_references": lineage["mixture"]["current"]["references"]}
        for name, identity in current.items():
            if sha256_file(resolve_inside(root, identity["path"])) != identity["sha256"]:
                raise DataError(f"{identity['path']} differs from the hash in {LINEAGE}")
        return {"source": LINEAGE, **current}
    def identity(relative):
        return {"path": relative, "sha256": sha256_file(resolve_inside(root, relative))}
    return {"source": "selection plan (no CSS10 re-audit lineage yet)",
            "css10_manifest": identity("data/manifests/css10_el_train.jsonl"),
            "css10_references": identity("data/manifests/references/css10_el_train.jsonl"),
            "mixed_manifest": selection["artifacts"]["mixed_manifest"],
            "mixed_references": selection["artifacts"]["mixed_references"]}


def verify_selection(root: Path, selection: dict, *, check_audio: bool = True) -> dict:
    """Contract + real-audio check of every manifest the locked selection delivers.

    Raw selected HParl, clean selected HParl, current CSS10, current mixture and
    dev are hash-checked against the plan/lineage, ID-aligned with their
    references and opened as real WAVs (each file once). The frozen test is
    checked separately by ``verify-freeze`` (also called here).
    """
    from .freeze import verify_freeze

    root = Path(root)
    result = {"selection": {"path": selection["path"], "sha256": selection["sha256"]},
              "test_freeze": verify_freeze(root, check_audio=check_audio), "audio_checked": check_audio}
    current = current_training_manifests(root, selection)
    locked = selection["artifacts"]
    checks = {"raw_selected_hparl": (locked["raw_manifest"], locked["raw_references"]),
              "clean_selected_hparl": (locked["clean_manifest"], locked["clean_references"]),
              "clean_css10": (current["css10_manifest"], current["css10_references"]),
              "mixed_hparl_css10": (current["mixed_manifest"], current["mixed_references"]),
              "dev": ({"path": "data/manifests/dev.jsonl", "sha256": None},
                      {"path": "data/manifests/references/dev.jsonl", "sha256": None})}
    result["current_manifest_source"] = current["source"]
    test_rows = read_jsonl(root / "data/manifests/test.jsonl")
    test_ids = {r["id"] for r in test_rows}
    held_out_sessions = {r["source_recording_id"] for r in [*test_rows, *read_jsonl(root / "data/manifests/dev.jsonl")]
                         if r.get("source_recording_id")}
    cache, loaded = {}, {}
    for name, (manifest, references) in checks.items():
        for identity in (manifest, references):
            digest = sha256_file(resolve_inside(root, identity["path"]))
            if identity["sha256"] is not None and digest != identity["sha256"]:
                raise DataError(f"{identity['path']}: hash differs from the selection plan/lineage")
        rows = read_jsonl(resolve_inside(root, manifest["path"]))
        refs = read_jsonl(resolve_inside(root, references["path"]))
        if [r["id"] for r in rows] != [r["id"] for r in refs]:
            raise DataError(f"{manifest['path']}: references do not match manifest IDs/order")
        if len({r["id"] for r in rows}) != len(rows):
            raise DataError(f"{manifest['path']}: duplicate IDs")
        if test_ids & {r["id"] for r in rows}:
            raise DataError(f"{manifest['path']} shares IDs with the frozen test")
        for row in rows:
            if "reference" in row or "text" in row:
                raise DataError(f"{manifest['path']}: gold text must not appear in an audio manifest")
            validate_manifest_row(row, root, check_audio=check_audio, cache=cache)
            if name != "dev" and (row["split"] != "train" or row.get("source_recording_id") in held_out_sessions):
                raise DataError(f"{row['id']}: training manifest holds a dev/test split record or session")
        loaded[name] = (rows, refs)
        result[name] = {"manifest": manifest["path"], "sha256": sha256_file(resolve_inside(root, manifest["path"])),
                        "references": references["path"],
                        "references_sha256": sha256_file(resolve_inside(root, references["path"])),
                        "records": len(rows), "hours": round(sum(r["duration_seconds"] for r in rows) / 3600, 6),
                        "records_by_source": dict(sorted(Counter(r["source"] for r in rows).items()))}
    raw = {r["id"]: r for r in loaded["raw_selected_hparl"][0]}
    clean = loaded["clean_selected_hparl"][0]
    if any(raw.get(r["id"]) != r for r in clean):
        raise DataError("clean selected HParl must reuse the raw selected rows (same original WAV path/hash)")
    css10 = loaded["clean_css10"][0]
    if any(r["source"] != "css10_el" for r in css10):
        raise DataError("CSS10 manifest contains a non-CSS10 record")
    mixed_ids = [r["id"] for r in loaded["mixed_hparl_css10"][0]]
    if mixed_ids != sorted(r["id"] for r in [*clean, *css10]):
        raise DataError("mixture is not exactly the clean selected HParl plus the current CSS10 (sorted by id)")
    refs = {r["id"]: r["reference"] for name in ("clean_selected_hparl", "clean_css10") for r in loaded[name][1]}
    if any(refs[r["id"]] != r["reference"] for r in loaded["mixed_hparl_css10"][1]):
        raise DataError("mixture references differ from the clean HParl/CSS10 references")
    result["unique_audio_files_checked"] = len(cache) if check_audio else 0
    result["unique_audio_bytes_checked"] = sum(m["audio_bytes"] for m in cache.values()) if check_audio else 0
    result["encodings"] = dict(sorted(Counter(m["encoding"] for m in cache.values()).items()))
    return result
