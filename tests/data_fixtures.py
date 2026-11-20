"""Small synthetic HParl/CSS10 archives with deliberate, known problems.

The fixture mimics the official layout seen in the real archive index
(`hparl/hparl_segmented_<split>/text|utt2spk`, `hparl/<split>_wavs/*.wav`)
and uses `validation` for the development split to exercise the mapping.
All audio is generated here; no real corpus data is used.
"""
from __future__ import annotations

import io
import math
import random
import shutil
import struct
import wave
import zipfile
from array import array
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def pcm(seconds: float, rate: int = 16000, *, kind: str = "speech", seed: int = 0) -> bytes:
    rng = random.Random(seed)
    count = int(round(seconds * rate))
    values = array("h")
    for n in range(count):
        t = n / rate
        if kind == "silence":
            sample = 0.0
        elif kind == "clipped":
            sample = 1.0 if math.sin(2 * math.pi * 200 * t) >= 0 else -1.0
        else:
            envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t + seed)
            sample = 0.3 * envelope * math.sin(2 * math.pi * (180 + 20 * seed) * t) + 0.01 * rng.uniform(-1, 1)
        values.append(max(-32768, min(32767, int(sample * 32767))))
    return values.tobytes()


def wav_bytes(payload: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(payload)
    return buffer.getvalue()


def float_samples(seconds: float, rate: int = 22050, *, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    values = []
    for n in range(int(round(seconds * rate))):
        t = n / rate
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t + seed)
        values.append(0.3 * envelope * math.sin(2 * math.pi * (180 + 20 * seed) * t) + 0.01 * rng.uniform(-1, 1))
    return values


def float32_wav_bytes(samples, rate: int = 22050, channels: int = 1) -> bytes:
    """IEEE float32 RIFF/WAVE (format tag 3) like the real CSS10 clips: fmt(18) + fact + data."""
    payload = struct.pack(f"<{len(samples)}f", *samples)
    fmt = struct.pack("<4sIHHIIHHH", b"fmt ", 18, 3, channels, rate, rate * 4 * channels, 4 * channels, 32, 0)
    fact = struct.pack("<4sII", b"fact", 4, len(samples) // channels)
    data = struct.pack("<4sI", b"data", len(payload)) + payload
    body = b"WAVE" + fmt + fact + data
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def pcm32_wav_bytes(values, rate: int = 16000) -> bytes:
    """Integer PCM32 (format tag 1): the same 4-byte width as float32, different encoding."""
    payload = struct.pack(f"<{len(values)}i", *values)
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 4, 4, 32)
    data = struct.pack("<4sI", b"data", len(payload)) + payload
    body = b"WAVE" + fmt + data
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def wav_with_list_chunk(payload: bytes, rate: int = 16000) -> bytes:
    """Same PCM as wav_bytes but a different header (extra LIST chunk)."""
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16)
    info = b"INFOISFT\x06\x00\x00\x00fixtr\x00"
    chunk = struct.pack("<4sI", b"LIST", len(info)) + info
    data = struct.pack("<4sI", b"data", len(payload)) + payload
    body = b"WAVE" + fmt + chunk + data
    return struct.pack("<4sI", b"RIFF", len(body)) + body


def uid(speaker: str, session: int, start: int, part: int = 1) -> str:
    return f"{speaker}-olomeleia_{session}_{start:08d}_{start + 3000:08d}_{part}_1"


TRAIN = {
    "good1": (uid("spka", 1, 10000), "εμείς μπορούμε σήμερα", "speech", 2.0, 1),
    "good2": (uid("spkb", 1, 20000), "σε αυτήν την οικονομική δράση", "speech", 2.4, 2),
    "noise": (uid("spkc", 2, 30000), "τους ξεφεύγει η αντικατάσταση της <spoken_noise>", "speech", 2.2, 3),
    "annot": (uid("spkc", 2, 40000), "<spoken_noise>", "speech", 1.0, 4),
    "dup": (uid("spkd", 3, 50000), "ευχαριστώ πολύ κύριε πρόεδρε", "copy:good1", 0, 0),
    "pcmdup": (uid("spke", 3, 60000), "ναι", "pcmcopy:good2", 0, 0),
    "testcopy": (uid("spkf", 4, 70000), "θα ήθελα να πω", "copy:test1", 0, 0),
    "devcopy": (uid("spkg", 4, 80000), "το λόγο έχει ο κύριος υπουργός", "copy:dev1", 0, 0),
    "long": (uid("spkh", 5, 90000), "μια πολύ μεγάλη ομιλία", "speech", 31.0, 5),
    "silent": (uid("spki", 5, 95000), "καλημέρα σας", "silence", 1.5, 0),
    "clip": (uid("spkj", 6, 99000), "ακούστε με", "clipped", 1.2, 0),
    "corrupt": (uid("spkk", 6, 99500), "κάτι χάλασε", "corrupt", 0, 0),
    "missing": (uid("spkl", 7, 99700), "δεν υπάρχει ήχος", "missing", 0, 0),
    "misaligned": (uid("spkr", 8, 99800), " ".join(["η κυβέρνηση φέρνει σήμερα στη βουλή το νομοσχέδιο"] * 6),
                   "speech", 1.0, 13),
    "boundary": (uid("spks", 8, 99900), "η κυβέρνηση φέρνει σήμερα στη βουλή το νομοσχέδιο", "speech", 1.0, 14),
}
UNLABELLED_TRAIN = uid("spkz", 9, 12345)
DEV = {
    "dev1": (uid("spkm", 11, 10000), "η συνεδρίαση αρχίζει", "speech", 2.0, 6),
    "dev2": (uid("spkn", 11, 20000), "<spoken_noise> ο κύριος βουλευτής", "speech", 2.5, 7),
}
TEST = {
    "test1": (uid("spko", 21, 10000), "κυρίες και κύριοι συνάδελφοι", "speech", 2.6, 8),
    "test2": (uid("spkp", 21, 20000), "ψηφίζουμε <spoken_noise> το νομοσχέδιο", "speech", 2.1, 9),
    "test3": (uid("spkq", 22, 30000), "ναι", "speech", 0.8, 10),
    "test4": (uid("spkb", 22, 40000), "ναι", "speech", 0.7, 15),
}
CSS10 = [
    ("Paramythi_horis_onoma/Paramythi_horis_onoma_0000.wav", "Ήταν μια φορά ένας βασιλιάς.",
     "Ήταν μια φορά ένας βασιλιάς.", 2.0, "speech", 11),
    ("Paramythi_horis_onoma/Paramythi_horis_onoma_0001.wav", "Είχε 3 παιδιά!", "Είχε τρία παιδιά!", 1.6, "speech", 12),
    ("Paramythi_horis_onoma/Paramythi_horis_onoma_0002.wav", "«Πού πας;» ρώτησε.", "«Πού πας;» ρώτησε.", 1.8, "copy:0000", 0),
    ("Paramythi_horis_onoma/Paramythi_horis_onoma_0003.wav", "Χάθηκε.", "Χάθηκε.", 1.0, "missing", 0),
]


def _split_files(spec: dict, generated: dict) -> tuple[dict, dict, dict]:
    text, spk, audio = {}, {}, {}
    for key, (utt, words, kind, seconds, seed) in spec.items():
        text[utt] = words
        spk[utt] = utt.split("-", 1)[0]
        if kind == "speech" or kind in ("silence", "clipped"):
            generated[key] = pcm(seconds, kind=kind, seed=seed)
            audio[utt] = wav_bytes(generated[key])
        elif kind.startswith("copy:"):
            audio[utt] = ("copy", kind[5:])
        elif kind.startswith("pcmcopy:"):
            audio[utt] = ("pcmcopy", kind[8:])
        elif kind == "corrupt":
            audio[utt] = wav_bytes(pcm(1.0, seed=99))[:200]
    return text, spk, audio


def build_hparl_zip(path: Path, *, drop_test_audio: bool = False) -> None:
    generated, layout = {}, {"train": ("train", TRAIN), "dev": ("validation", DEV), "test": ("test", TEST)}
    files = {}
    pending = []
    for _split, (name, spec) in layout.items():
        text, spk, audio = _split_files(spec, generated)
        files[f"hparl/hparl_segmented_{name}/text"] = "".join(f"{u} {t}\n" for u, t in sorted(text.items())).encode()
        files[f"hparl/hparl_segmented_{name}/utt2spk"] = "".join(f"{u} {s}\n" for u, s in sorted(spk.items())).encode()
        for utt, data in audio.items():
            member = f"hparl/{name}_wavs/{utt}.wav"
            if isinstance(data, tuple):
                pending.append((member, data))
            else:
                files[member] = data
    keys = {**{k: v[0] for k, v in TRAIN.items()}, **{k: v[0] for k, v in DEV.items()}, **{k: v[0] for k, v in TEST.items()}}
    for member, (mode, key) in pending:
        if mode == "copy":
            source = next(m for m in files if m.endswith(f"/{keys[key]}.wav"))
            files[member] = files[source]
        else:
            files[member] = wav_with_list_chunk(generated[key])
    files[f"hparl/train_wavs/{UNLABELLED_TRAIN}.wav"] = wav_bytes(pcm(1.0, seed=42))
    if drop_test_audio:
        del files[f"hparl/test_wavs/{TEST['test3'][0]}.wav"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("hparl/", b"")
        for name in sorted(files):
            archive.writestr(name, files[name])


def build_css10_zip(path: Path) -> None:
    files, generated = {}, {}
    lines = []
    for relative, original, normalized, seconds, kind, seed in CSS10:
        lines.append(f"{relative}|{original}|{normalized}|{seconds}\n")
        if kind == "speech":  # the real CSS10 clips are IEEE float32, 22050 Hz mono
            generated[relative] = float32_wav_bytes(float_samples(seconds, 22050, seed=seed), 22050)
            files[f"greek/{relative}"] = generated[relative]
        elif kind.startswith("copy:"):
            files[f"greek/{relative}"] = files[f"greek/Paramythi_horis_onoma/Paramythi_horis_onoma_{kind[5:]}.wav"]
    files["greek/transcript.txt"] = "".join(lines).encode("utf-8")
    files["greek/Paramythi_horis_onoma/unused_9999.wav"] = float32_wav_bytes(float_samples(0.5, 22050, seed=77), 22050)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            archive.writestr(name, files[name])


def make_project(root: Path, **options) -> Path:
    """Create a temporary project root with configs and both source archives."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "configs/data_cleaning.json", root / "configs/data_cleaning.json")
    build_hparl_zip(root / "data/downloads/hparl.zip", **options)
    build_css10_zip(root / "downloads/css10_greek.zip")
    return root


def run_stage_one(root: Path) -> None:
    from greek_asr.data import freeze, hparl
    from greek_asr.data.common import sha256_file

    archive = root / "data/downloads/hparl.zip"
    record = {"dataset": hparl.DATASET, "archive_url": "fixture://hparl.zip", "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha256_file(archive), "archive_file": archive.name, "downloaded_utc": "2026-11-18T00:00:00+00:00"}
    hparl.record_acquisition(record, root / "data/raw/hparl", None, "fixture: no catalog")
    hparl.extract(archive, root / "data/raw/hparl")
    hparl.build_raw_manifests(root / "data/raw/hparl", root / "data/manifests/raw", root)
    freeze.freeze_test(root, now="2026-11-18T00:00:00+00:00")


def run_stage_two(root: Path) -> None:
    from greek_asr.data import audit, clean, css10, report

    css10.acquire(root / "downloads/css10_greek.zip", root / "data/raw/css10_el", None, "fixture: offline")
    css10.build_raw_manifest(root / "data/raw/css10_el", root / "data/manifests/raw", root)
    audit.run_audit(root, workers=1)
    clean.run_clean(root)
    report.write_report(root)


def write_selection_plan(root: Path, sessions: list[str], seed: int = 42) -> Path:
    """A locked selection plan shaped like configs/training_data_selection.json for chosen whole sessions.

    The clean subset is the full clean train restricted to the sessions (what ``subset`` writes for
    those sessions); the raw subset holds every raw record of the same sessions, including removed ones.
    """
    from greek_asr.data.common import json_bytes, jsonl_bytes, read_jsonl, sha256_file

    chosen = set(sessions)
    tag = "train_hparl_fixture_seed%d" % seed
    base = root / "data/manifests/subsets"
    def subset(manifest, references):
        rows = sorted((r for r in read_jsonl(root / manifest) if r["source_recording_id"] in chosen),
                      key=lambda r: r["id"])
        refs = {r["id"]: r for r in read_jsonl(root / references)}
        return rows, [refs[r["id"]] for r in rows]
    raw_rows, raw_refs = subset("data/manifests/raw/hparl_train.jsonl", "data/manifests/raw/references/hparl_train.jsonl")
    rows, refs = subset("data/manifests/train.jsonl", "data/manifests/references/train.jsonl")
    css = read_jsonl(root / "data/manifests/css10_el_train.jsonl")
    css_refs = {r["id"]: r for r in read_jsonl(root / "data/manifests/references/css10_el_train.jsonl")}
    mixed = sorted(rows + css, key=lambda r: r["id"])
    mixed_refs = [{**{r["id"]: r for r in refs}, **css_refs}[r["id"]] for r in mixed]
    files = {"raw_manifest": (f"raw/{tag}.jsonl", raw_rows), "raw_references": (f"raw/references/{tag}.jsonl", raw_refs),
             "clean_manifest": (f"{tag}.jsonl", rows), "clean_references": (f"references/{tag}.jsonl", refs),
             "mixed_manifest": (f"{tag}_css10.jsonl", mixed), "mixed_references": (f"references/{tag}_css10.jsonl", mixed_refs)}
    artifacts = {}
    for name, (relative, content) in files.items():
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jsonl_bytes(content))
        artifacts[name] = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
    plan = root / "configs/training_data_selection.json"
    plan.write_bytes(json_bytes({"schema": "p3-selected-training-delivery-v1", "seed": seed,
                                 "session_ids": sorted(chosen), "artifacts": artifacts}))
    return plan
