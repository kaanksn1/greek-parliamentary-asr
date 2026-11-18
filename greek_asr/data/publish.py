"""Size the raw/clean data deliverable against a Git LFS quota.

The delivery method (Git LFS, chunked release assets, or a justified subset)
is agreed with Serdar; this only measures the files so that the decision is
based on real numbers rather than source-reported hours. No paid storage is
assumed.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

from .common import (ROOT, DataError, json_bytes, keep_bytes, read_json, read_jsonl, relative_posix,
                     safe_member_target, sha256_file, write_output)

GIB = 1024 ** 3
FILE_LIMIT = 100 * 1024 ** 2  # plain Git hard limit per file on GitHub


def _size(paths) -> tuple[int, int, int]:
    total = count = largest = 0
    for path in paths:
        size = path.stat().st_size
        total, count, largest = total + size, count + 1, max(largest, size)
    return total, count, largest


def plan(root: Path = ROOT, quota_gib: float = 10.0) -> dict:
    root = Path(root)
    groups = {}
    layout_path = root / "data/raw/hparl/LAYOUT.json"
    release = root / "data/raw/hparl/release"
    if layout_path.exists():
        layout = read_json(layout_path)
        splits = {k: v for k, v in layout.items() if not k.startswith("_")}
        for split, place in splits.items():
            wavs = [p for d in place["wav_dirs"] for p in (release / d).rglob("*.wav")]
            groups[f"hparl_{split}_wav"] = _size(wavs)
        wav_set = {p for place in splits.values() for w in place["wav_dirs"] for p in (release / w).rglob("*.wav")}
        groups["hparl_other_files"] = _size(p for p in release.rglob("*") if p.is_file() and p not in wav_set)
    css = root / "data/raw/css10_el/release"
    if css.is_dir():
        groups["css10_el_files"] = _size(p for p in css.rglob("*") if p.is_file())
    text_dirs = [root / "data/manifests", root / "data/clean", root / "data/audit"]
    groups["manifests_logs_audit"] = _size(p for d in text_dirs if d.is_dir() for p in d.rglob("*") if p.is_file())
    table = {name: {"bytes": b, "gib": round(b / GIB, 3), "files": c, "largest_file_bytes": m}
             for name, (b, c, m) in groups.items()}
    audio_total = sum(v["bytes"] for k, v in table.items() if k != "manifests_logs_audit")
    without_train = audio_total - table.get("hparl_train_wav", {"bytes": 0})["bytes"]
    oversize = [k for k, v in table.items() if v["largest_file_bytes"] > FILE_LIMIT]
    return {"groups": table, "audio_and_source_total_gib": round(audio_total / GIB, 3),
            "quota_gib": quota_gib, "fits_quota_all": audio_total <= quota_gib * GIB,
            "fits_quota_without_hparl_train_audio": without_train <= quota_gib * GIB,
            "groups_with_files_over_100MiB": oversize,
            "note": ("Clean manifests reference the original audio bytes, so clean data adds only text files. "
                     "LFS bandwidth (every clone/pull) also counts against the owner's monthly quota. "
                     "Agree the method with Serdar; do not buy storage.")}


def pack(root: Path, source: str, output: Path, max_bytes: int = 1_900_000_000, name: str | None = None) -> dict:
    """Pack a raw data directory into deterministic, size-limited tar shards.

    Intended for release assets (each must stay below GitHub's per-file limit).
    The index lists every member's SHA-256 and its shard; it is committed to
    the repository so that the published bytes can be verified after download.
    """
    root, output = Path(root), Path(output)
    base = (root / source).resolve()
    if not base.is_dir() or not base.is_relative_to(root.resolve()):
        raise DataError(f"pack source must be a directory inside the project: {source}")
    files = sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: relative_posix(p, root))
    if not files:
        raise DataError(f"nothing to pack under {source}")
    name = name or source.strip("/").replace("/", "_")
    output.mkdir(parents=True, exist_ok=True)
    shards, members, current, current_size = [], [], None, 0

    def close():
        if current is not None:
            current[1].close()
            path = current[0]
            shards.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    largest = max(files, key=lambda p: p.stat().st_size)
    if largest.stat().st_size + 1024 > max_bytes:
        raise DataError(f"single file larger than the shard limit: {largest}")
    for path in files:
        size = path.stat().st_size
        if current is None or current_size + size + 1024 > max_bytes:
            close()
            shard_path = output / f"{name}.part{len(shards) + 1:03d}.tar"
            if shard_path.exists():
                raise DataError(f"refusing to overwrite existing shard {shard_path}")
            current, current_size = (shard_path, tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT)), 0
        info = tarfile.TarInfo(relative_posix(path, root))
        info.size, info.mtime, info.mode, info.uid, info.gid, info.uname, info.gname = size, 0, 0o644, 0, 0, "", ""
        with path.open("rb") as stream:
            current[1].addfile(info, stream)
        current_size += size + 1024 + (-size % 512)
        members.append({"path": info.name, "bytes": size, "sha256": sha256_file(path), "shard": current[0].name})
    close()
    index = {"schema": "p3-data-release-index-v1", "source": source, "shard_limit_bytes": max_bytes,
             "shards": shards, "members": members, "total_bytes": sum(m["bytes"] for m in members)}
    write_output(root / "data/release_index" / f"{name}.json", json_bytes(index))
    return {k: v for k, v in index.items() if k != "members"} | {"members": len(members)}


def unpack(root: Path, index_path: Path, shard_dir: Path) -> dict:
    """Verify downloaded shards and restore their files at the recorded paths."""
    root, index = Path(root), read_json(index_path)
    for shard in index["shards"]:
        path = Path(shard_dir) / shard["file"]
        if not path.is_file() or path.stat().st_size != shard["bytes"] or sha256_file(path) != shard["sha256"]:
            raise DataError(f"shard missing or corrupted: {shard['file']}")
    expected = {m["path"]: m for m in index["members"]}
    restored = 0
    for shard in index["shards"]:
        with tarfile.open(Path(shard_dir) / shard["file"], "r") as archive:
            for info in archive:
                if not info.isfile() or info.name not in expected:
                    raise DataError(f"unexpected tar member {info.name!r} in {shard['file']}")
                target = safe_member_target(info.name, root)
                data = archive.extractfile(info).read()
                keep_bytes(target, data)
                if sha256_file(target) != expected[info.name]["sha256"]:
                    raise DataError(f"restored file hash mismatch: {info.name}")
                restored += 1
    if restored != len(expected):
        raise DataError(f"restored {restored} of {len(expected)} indexed files")
    return {"restored_files": restored, "shards": len(index["shards"])}


def _members(path: Path) -> list[dict]:
    import csv
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def delivery_plan(root: Path = ROOT) -> dict:
    """Real deliverable sizes from the recorded archive listings (no audio needed)."""
    root = Path(root)
    groups: dict[str, dict] = {}

    def add(name: str, size: int) -> None:
        group = groups.setdefault(name, {"files": 0, "bytes": 0})
        group["files"] += 1
        group["bytes"] += size

    hparl_listing = root / "data/raw/hparl/archive_members.tsv"
    if hparl_listing.exists():
        layout = read_json(root / "data/raw/hparl/LAYOUT.json")
        wav_dirs = {wav_dir: split for split, place in layout.items() if not split.startswith("_")
                    for wav_dir in place["wav_dirs"]}
        for member in _members(hparl_listing):
            path, size = member["path"], int(member["bytes"])
            if path.lower().endswith(".wav"):
                add(f"hparl_{wav_dirs.get(path.split('/')[0], 'other')}_wav", size)
            else:
                add("hparl_source_metadata", size)
    css_listing = root / "data/raw/css10_el/archive_members.tsv"
    css_manifest = root / "data/manifests/raw/css10_el.jsonl"
    if css_listing.exists():
        used = set()
        if css_manifest.exists():
            used = {row["audio_path"].removeprefix("data/raw/css10_el/release/") for row in read_jsonl(css_manifest)}
        for member in _members(css_listing):
            path, size = member["path"], int(member["bytes"])
            if path.lower().endswith(".wav"):
                add("css10_used_wav" if path in used else "css10_unreferenced_wav", size)
            else:
                add("css10_source_metadata", size)
    for group in groups.values():
        group["gib"] = round(group["bytes"] / GIB, 3)
    decisions_path = root / "configs/data_delivery.json"
    decisions = read_json(decisions_path) if decisions_path.exists() else {}
    quota = decisions.get("free_lfs_storage_gib", 10.0)
    audio = ["hparl_test_wav", "hparl_dev_wav", "css10_used_wav", "hparl_train_wav"]
    running, cumulative = 0, {}
    for name in audio:
        running += groups.get(name, {"bytes": 0})["bytes"]
        cumulative[name] = round(running / GIB, 3)
    return {"groups": dict(sorted(groups.items())), "cumulative_audio_gib_in_priority_order": cumulative,
            "free_lfs_storage_gib": quota,
            "fits_free_lfs_without_train": cumulative.get("css10_used_wav", 0) <= quota,
            "fits_free_lfs_with_full_train": cumulative.get("hparl_train_wav", 0) <= quota,
            "decisions": decisions.get("groups", {})}


def verify_source_metadata(root: Path = ROOT) -> dict:
    """Check committed source text files byte-for-byte against recorded hashes.

    Runs on a clean clone without audio; detects line-ending or encoding
    conversions by Git as well as missing files.
    """
    root = Path(root)
    problems, checked = [], 0
    inventory = read_json(root / "data/manifests/raw/hparl_inventory.json")
    release = root / "data/raw/hparl/release"
    for split, data in inventory["splits"].items():
        for relative, digest in data["metadata_files_sha256"].items():
            path = release / relative
            checked += 1
            if not path.is_file():
                problems.append(f"missing: {relative}")
            elif sha256_file(path) != digest:
                problems.append(f"hash differs (line endings/encoding?): {relative}")
    css_inventory = root / "data/manifests/raw/css10_el_inventory.json"
    if css_inventory.exists():
        data = read_json(css_inventory)
        path = root / data["transcript"]
        checked += 1
        if not path.is_file():
            problems.append(f"missing: {data['transcript']}")
        elif sha256_file(path) != data["transcript_sha256"]:
            problems.append(f"hash differs (line endings/encoding?): {data['transcript']}")
    if problems:
        raise DataError("SOURCE METADATA CHECK FAILED: " + "; ".join(problems[:20]))
    return {"status": "ok", "files_checked": checked}
