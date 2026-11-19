"""Independent original-byte/hash/encoding diagnosis for the data-owner handoff.

This reports RIFF headers and correct-vs-misread sample statistics. It does not
replace the production decoder, alter source audio or regenerate cleaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inspect(data):
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE recording")
    if struct.unpack_from("<I", data, 4)[0] + 8 != len(data):
        raise ValueError("RIFF byte length mismatch")
    position, properties, payload = 12, None, None
    while position + 8 <= len(data):
        tag, size = struct.unpack_from("<4sI", data, position)
        start, end = position + 8, position + 8 + size
        if end > len(data):
            raise ValueError("truncated RIFF chunk")
        if tag == b"fmt ":
            code, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", data, start)
            properties = {"format_tag": code, "channels": channels, "sampling_rate": rate,
                          "bits_per_sample": bits, "block_align": block_align}
        elif tag == b"data":
            payload = data[start:end]
        position = end + size % 2
    if properties is None or payload is None or len(payload) % properties["block_align"]:
        raise ValueError("invalid or missing audio chunks")
    return {**properties, "num_frames": len(payload) // properties["block_align"]}, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts, failures, examples = Counter(), [], []
    rows = [json.loads(line) for line in (ROOT / "data/manifests/css10_el_train.jsonl").read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        data = (ROOT / row["audio_path"]).read_bytes()
        header, payload = inspect(data)
        if hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
            failures.append({"id": row["id"], "issue": "audio_sha256 mismatch"})
        for field in ("channels", "sampling_rate", "num_frames"):
            if header[field] != row[field]:
                failures.append({"id": row["id"], "issue": f"{field} mismatch"})
        counts[(header["format_tag"], header["bits_per_sample"])] += 1
        if index < 3 and header["format_tag"] == 3 and header["bits_per_sample"] == 32:
            correct = [sample[0] for sample in struct.iter_unpack("<f", payload)]
            wrong = [sample[0] / 2147483648. for sample in struct.iter_unpack("<i", payload)]
            def stats(samples):
                return {"rms_dbfs": 10 * math.log10(max(sum(x*x for x in samples) / len(samples), 1e-20)),
                        "peak_dbfs": 20 * math.log10(max(max(abs(x) for x in samples), 1e-10)),
                        "nonfinite_values": sum(not math.isfinite(x) for x in samples)}
            examples.append({"id": row["id"], "audio_sha256": row["audio_sha256"], "header": header,
                             "correct_ieee_float32": stats(correct), "incorrect_int32_reinterpretation": stats(wrong)})
    report = {"schema": "p3-css10-original-audio-diagnosis-v1", "record_count": len(rows),
              "format_counts": [{"format_tag": code, "bits_per_sample": bits, "records": count}
                                for (code, bits), count in sorted(counts.items())],
              "hash_or_header_failures": failures, "all_original_audio_hashes_verified": not failures,
              "examples": examples, "source_audio_changed": False,
              "data_submission": "4db1a4aa61eb78acbb531b705c45e3bbd090854e",
              "impact": "Python 3.12 wave rejects IEEE float; Python 3.15 wave reads it, but width=4 is not proof of integer PCM. Existing signal_stats treats all width=4 bytes as int32, invalidating CSS10 signal statistics.",
              "production_data_verify_passed": False, "production_model_float_decode_supported": False,
              "official_test_audio_or_text_read": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
