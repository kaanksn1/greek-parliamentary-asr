"""Owner acquisition of the already-locked RAW training cohort, with no cleaning.

Fetch bounded/coalesced public ZIP ranges, preserving every selected original
WAV byte (including records removed by Samed's fixed cleaning). Reuse the data
owner's exact delivery list. No split, reference, test or selection is changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import ssl
import struct
import threading
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import fetch_train_pilot as source

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-transfer-gib", type=float, default=8.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be in 1..12")
    root, output = args.root.resolve(), args.output.resolve()
    output.relative_to(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    selection_path = root / "configs/training_data_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    delivery_path = root / selection["artifacts"]["raw_audio_delivery_list"]["path"]
    if hashlib.sha256(delivery_path.read_bytes()).hexdigest() != selection["artifacts"]["raw_audio_delivery_list"]["sha256"]:
        raise ValueError("delivery list differs from the locked selection")
    rows = [json.loads(line) for line in delivery_path.read_text(encoding="utf-8").splitlines()]
    known, pending = [], []
    for row in rows:
        path = (root / row["audio_path"]).resolve()
        path.relative_to(root / "data/raw/hparl/release/train_wavs")
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["audio_sha256"]:
                raise ValueError(f"refusing different existing source bytes: {row['id']}")
            known.append(row)
        else:
            pending.append(row)
    directory_reader = source.RangeReader()
    entries = []
    with zipfile.ZipFile(directory_reader) as archive:
        for row in pending:
            name = "hparl/train_wavs/" + Path(row["audio_path"]).name
            info = archive.getinfo(name)
            if info.file_size != row["bytes"] or info.flag_bits & 1 or info.compress_type not in (0, 8):
                raise ValueError(f"unexpected source ZIP member properties: {name}")
            # Source local headers have small extras. The additional margin
            # avoids per-file header requests; a checked fallback handles larger extras.
            end = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra) + info.compress_size + 128
            entries.append((info.header_offset, min(end, source.TOTAL), info, row))
    entries.sort(key=lambda item: item[0])
    groups = []
    for entry in entries:
        start, end = entry[:2]
        if groups and start - groups[-1][1] <= 32768 and max(end, groups[-1][1]) - groups[-1][0] <= 16 * 1024**2:
            groups[-1][1] = max(end, groups[-1][1])
            groups[-1][2].append(entry)
        else:
            groups.append([start, end, [entry]])
    planned = sum(end - start for start, end, _ in groups)
    ceiling = int(args.max_transfer_gib * 1024**3)
    if planned + directory_reader.downloaded > ceiling:
        raise ValueError("coalesced source requests exceed the transfer ceiling")
    lock, local = threading.Lock(), threading.local()
    state = {"requested_bytes": directory_reader.downloaded, "downloaded_bytes": directory_reader.downloaded,
             "verified_wavs": len(known), "completed_ranges": 0}
    started = time.perf_counter()
    report = {"schema": "p3-selected-hparl-acquisition-v1", "started_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              "delivery_list_sha256": hashlib.sha256(delivery_path.read_bytes()).hexdigest(),
              "expected_raw_wavs": len(rows), "existing_hash_verified": len(known), "pending_wavs": len(pending),
              "range_groups": len(groups), "planned_transfer_bytes": planned,
              "transfer_ceiling_bytes": ceiling, "source_archive_bytes": source.TOTAL,
              "references_or_splits_modified": False, "test_audio_decoded": False,
              "source_url": source.URL, "workers": args.workers}

    def save():
        value = {**report, **state, "elapsed_seconds": time.perf_counter() - started}
        # OneDrive can transiently lock an existing destination against rename.
        # This is a progress record, not source data; readers may retry JSON.
        data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        for attempt in range(10):
            try:
                output.write_text(data, encoding="utf-8", newline="\n")
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.2)

    def request_range(start, end):
        size = end - start
        for attempt in range(4):
            with lock:
                if state["requested_bytes"] + size > ceiling:
                    raise ValueError("source transfer ceiling reached, including retries")
                state["requested_bytes"] += size
            try:
                connection = getattr(local, "connection", None)
                if connection is None:
                    connection = http.client.HTTPSConnection("inventory.clarin.gr", timeout=45, context=ssl.create_default_context())
                    local.connection = connection
                connection.request("GET", "/storage/", headers={**source.HEADERS, "Range": f"bytes={start}-{end-1}"})
                response = connection.getresponse()
                expected = f"bytes {start}-{end-1}/{source.TOTAL}"
                if response.status != 206 or response.getheader("Content-Range") != expected:
                    raise ValueError("source did not honor exact bounded Range")
                data = response.read(size + 1)
                if len(data) != size:
                    raise OSError("short source range")
                with lock:
                    state["downloaded_bytes"] += len(data)
                return data
            except (OSError, http.client.HTTPException, ValueError):
                if getattr(local, "connection", None) is not None:
                    local.connection.close()
                local.connection = None
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    def fetch(group):
        start, end, members = group
        data = request_range(start, end)
        for offset, _, info, row in members:
            pos = offset - start
            header = struct.unpack_from("<4s5H3I2H", data, pos)
            if header[0] != b"PK\x03\x04" or header[2] & 1 or header[3] != info.compress_type:
                raise ValueError("source ZIP local/central header mismatch")
            name_size, extra_size = header[-2:]
            if data[pos+30:pos+30+name_size] != info.filename.encode("utf-8"):
                raise ValueError("source ZIP member name mismatch")
            payload_start = offset + 30 + name_size + extra_size
            payload_end = payload_start + info.compress_size
            compressed = (data[payload_start-start:payload_end-start] if payload_end <= end
                          else request_range(payload_start, payload_end))
            if info.compress_type == 8:
                decoder = zlib.decompressobj(-15)
                original = decoder.decompress(compressed, info.file_size + 1)
                if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError("invalid source deflate boundary")
            else:
                original = compressed
            if (len(original) != info.file_size or zlib.crc32(original) & 0xFFFFFFFF != info.CRC or
                    hashlib.sha256(original).hexdigest() != row["audio_sha256"]):
                raise ValueError(f"source CRC/size/SHA-256 failed: {row['id']}")
            path = root / row["audio_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".wav.partial")
            temporary.write_bytes(original)
            if path.exists():
                raise FileExistsError("source destination appeared during acquisition")
            os.replace(temporary, path)
        with lock:
            state["verified_wavs"] += len(members)
            state["completed_ranges"] += 1
            if state["completed_ranges"] % 16 == 0 or state["verified_wavs"] == len(rows):
                save()
                print(json.dumps({**state, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)

    save()
    print(json.dumps({"plan": report}), flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for future in executor.map(fetch, groups):
                pass
        if state["verified_wavs"] != len(rows):
            raise ValueError("incomplete selected RAW cohort")
        report["status"] = "complete"
        report["all_selected_original_hashes_verified"] = True
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save()
    print(json.dumps({"status": report["status"], **state}), flush=True)


if __name__ == "__main__":
    main()
