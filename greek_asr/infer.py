"""Run one pinned standard ASR baseline on a complete, explicit audio manifest."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {
    "whisper": ("whisper", "openai/whisper-small"),
    "qwen": ("qwen", "Qwen/Qwen3-ASR-0.6B"),
    "canary": ("nemo", "nvidia/canary-1b-v2"),
    "parakeet": ("nemo", "nvidia/parakeet-tdt-0.6b-v3"),
    # Kemal's published CTC checkpoints; a dotted name is an absolute module path.
    "wav2vec": ("greek_asr.models.wav2vec", "lighteternal/wav2vec2-large-xlsr-53-greek"),
    "hubert": ("greek_asr.models.hubert", "ntinouldinho/hubert_greek_6gram"),
}
# The historical snapshot configs/model_sources.lock.json is never rewritten;
# checkpoints found later are pinned in their own documented source record.
SOURCE_LOCKS = {"hubert": "configs/kemal_model_sources.json"}


def _backend_module(module_name: str) -> str:
    return module_name if "." in module_name else "greek_asr.pretrained." + module_name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, project_root: Path, split: str) -> list[dict[str, Any]]:
    """Validate all audio inputs before loading weights; references never leave here."""
    root = project_root.resolve()
    records, seen = [], set()
    required = {"id", "source", "original_id", "source_split", "split", "audio_path",
                "audio_sha256", "sampling_rate", "channels", "num_frames", "duration_seconds"}
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"manifest line {line_no}: required fields missing")
        identifier, source = row["id"], row["source"]
        if source not in ("hparl", "css10_el"):
            raise ValueError(f"manifest line {line_no}: unsupported source")
        if (not isinstance(row["original_id"], str) or not row["original_id"] or
                identifier != source + ":" + row["original_id"]):
            raise ValueError(f"manifest line {line_no}: ID must be source:original_id")
        if identifier in seen:
            raise ValueError(f"duplicate ID: {identifier}")
        seen.add(identifier)
        if row["split"] != split or row["source_split"] not in ("train", "dev", "test", "unsplit"):
            raise ValueError(f"split mismatch for {identifier}")
        if source == "css10_el" and split != "train":
            raise ValueError("CSS10 is training-only")
        if source == "hparl" and row["source_split"] != split:
            raise ValueError("HParl official split must be preserved")
        if not isinstance(row["audio_path"], str):
            raise ValueError("audio_path must be a project-relative string")
        relative = Path(row["audio_path"])
        if relative.is_absolute() or re.match(r"^[A-Za-z]:", row["audio_path"]):
            raise ValueError("audio_path must be project-relative")
        audio = (root / relative).resolve()
        if not audio.is_relative_to(root) or not audio.is_file():
            raise ValueError(f"audio path escapes project or is missing: {identifier}")
        expected_hash = row["audio_sha256"]
        if (not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                or sha256(audio) != expected_hash):
            raise ValueError(f"audio hash mismatch: {identifier}")
        for field in ("sampling_rate", "channels", "num_frames"):
            if type(row[field]) is not int or row[field] <= 0:
                raise ValueError(f"invalid {field}: {identifier}")
        duration = row["duration_seconds"]
        if (type(duration) not in (float, int) or not math.isfinite(duration) or duration <= 0
                or abs(duration - row["num_frames"] / row["sampling_rate"]) > 0.001):
            raise ValueError(f"invalid duration: {identifier}")
        with wave.open(str(audio), "rb") as wav:
            if ((wav.getframerate(), wav.getnchannels(), wav.getnframes()) !=
                    (row["sampling_rate"], row["channels"], row["num_frames"])):
                raise ValueError(f"WAV header disagrees with manifest: {identifier}")
            if wav.getcomptype() != "NONE":
                raise ValueError(f"manifest expects uncompressed WAV: {identifier}")
            payload = wav.readframes(wav.getnframes())
            if len(payload) != wav.getnframes() * wav.getnchannels() * wav.getsampwidth():
                raise ValueError(f"truncated WAV payload: {identifier}")
        # Whitelist inputs; even an accidental reference field is never passed to a backend.
        records.append({"id": identifier, "audio": audio, "audio_sha256": expected_hash,
                        "duration_seconds": float(duration)})
    if not records:
        raise ValueError("empty manifest")
    return records


def _test_protocol(path: Path | None, manifest: Path, project_root: Path) -> str:
    if path is None:
        raise ValueError("test inference requires --test-protocol committed before the run")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (protocol.get("selection_fixed") is not True or not protocol.get("scoring_policy")
            or protocol.get("test_manifest_sha256") != sha256(manifest)):
        raise ValueError("test protocol must fix selection, scoring and the exact test manifest")
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    committed = subprocess.run(["git", "show", "HEAD:" + relative], cwd=project_root,
                               capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("test protocol differs from its committed version")
    return sha256(path)


def _environment() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "numpy", "nemo-toolkit", "qwen-asr", "huggingface-hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result: dict[str, Any] = {"python": platform.python_version(), "packages": versions}
    try:
        import torch
        result.update(hip=torch.version.hip, gpu_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
            result["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return result


def run(*, manifest: Path, backend_name: str, split: str, output: Path,
        project_root: Path = ROOT, device: str = "cuda", seed: int = 42,
        purpose: str = "baseline", allow_test: bool = False,
        test_protocol: Path | None = None, loader=None) -> dict[str, Any]:
    if backend_name not in BACKENDS:
        raise ValueError("unknown backend")
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    if purpose == "runtime-smoke" and split != "train":
        raise ValueError("runtime smoke accepts training audio only")
    if split == "test" and not allow_test:
        raise ValueError("test inference requires --allow-test after final protocol commit")
    protocol_hash = _test_protocol(test_protocol, manifest, project_root) if split == "test" else None
    records = load_manifest(manifest, project_root, split)
    if purpose == "runtime-smoke" and len(records) > 100:
        raise ValueError("runtime smoke accepts at most 100 training records")
    lock_path = project_root / SOURCE_LOCKS.get(backend_name, "configs/model_sources.lock.json")
    module_name, model_id = BACKENDS[backend_name]
    module_path = _backend_module(module_name)
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    models = [row for row in lock["models"] if row["model_id"] == model_id]
    if len(models) != 1 or not re.fullmatch(r"[0-9a-f]{40}", models[0]["revision"]):
        raise ValueError("checkpoint lock must contain one immutable revision for this model")
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "predictions.partial.jsonl"
    report: dict[str, Any] = {
        "schema": "greek-asr-inference-v1", "status": "running", "purpose": purpose,
        "created_utc": datetime.now(timezone.utc).isoformat(), "backend": backend_name,
        "model": models[0], "seed": seed, "device": device, "split": split,
        "manifest_sha256": sha256(manifest), "source_lock_sha256": sha256(lock_path),
        "test_protocol_sha256": protocol_hash, "expected_records": len(records),
        "total_audio_seconds": sum(row["duration_seconds"] for row in records),
        "training_performed": False, "references_used_as_model_input": False,
        "wer_cer": None, "completed_records": 0,
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                        for p in (Path(__file__), *sorted((ROOT / "greek_asr/pretrained").glob("*.py")))},
    }
    if not module_path.startswith("greek_asr.pretrained."):
        backend_files = [ROOT / (module_path.replace(".", "/") + ".py"),
                         ROOT / "greek_asr/models/common.py", ROOT / "greek_asr/models/ctc_utils.py"]
        if module_path == "greek_asr.models.hubert":
            backend_files.append(ROOT / "greek_asr/models/wav2vec.py")
        report["code_sha256"].update({str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                                      for p in backend_files if p.is_file()})
    try:
        # Importing this runner and validating manifests requires only the stdlib.
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("requested GPU is unavailable; no CPU fallback")
            torch.cuda.reset_peak_memory_stats()
        if loader is None:
            loader = importlib.import_module(module_path).load_backend
        started = time.perf_counter()
        backend = loader(model_id=model_id, revision=models[0]["revision"], device=device,
                         model_dir=project_root / "models" / (backend_name + "-" + models[0]["revision"][:12]))
        report["download_and_load_seconds"] = time.perf_counter() - started
        report["settings"] = getattr(backend, "settings", {})
        started = time.perf_counter()
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            for row in records:
                text = backend.transcribe(row["audio"])
                if not isinstance(text, str):
                    raise ValueError("backend prediction must be a string, including genuinely empty output")
                stream.write(json.dumps({"id": row["id"], "text": text}, ensure_ascii=False) + "\n")
                stream.flush()
                report["completed_records"] += 1
                print(f"{backend_name}: {report['completed_records']}/{len(records)} {row['id']}", flush=True)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        report["inference_seconds_including_output_io"] = time.perf_counter() - started
        if sha256(manifest) != report["manifest_sha256"]:
            raise ValueError("input manifest changed during inference")
        for row in records:
            if sha256(row["audio"]) != row["audio_sha256"]:
                raise ValueError("source audio changed during inference")
        predictions = output / "predictions.jsonl"
        partial.rename(predictions)
        report.update(status="complete", predictions_sha256=sha256(predictions))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["environment"] = _environment()
        (output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purpose", choices=("baseline", "runtime-smoke"), default="baseline")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--test-protocol", type=Path)
    args = parser.parse_args(argv)
    try:
        run(manifest=args.manifest, backend_name=args.backend, split=args.split,
            output=args.output, device=args.device, seed=args.seed, purpose=args.purpose,
            allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"inference failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
