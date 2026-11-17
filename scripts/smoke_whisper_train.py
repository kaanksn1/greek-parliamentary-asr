"""Run pinned multilingual Whisper on the three training-pilot WAVs, not a benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "data/raw/pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "results/runtime_smoke/whisper_small_train")
    args = parser.parse_args()
    if (args.output / "run.json").exists():
        parser.error("a run already exists; choose a new --output directory")

    from greek_asr.pilot_data import audit_pilot
    records, audit = audit_pilot(args.pilot)
    if len(records) > 100 or any(row["source_split"] != "train" for row in records):
        parser.error("this runtime check accepts only a small official-training pilot")
    lock_path = ROOT / "configs/model_sources.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    source = next(row for row in lock["models"] if row["model_id"] == "openai/whisper-small")

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    if not torch.cuda.is_available():
        parser.error("the GPU preflight is required; no silent CPU fallback")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    disable_progress_bars()
    model_dir = ROOT / "models" / ("whisper-small-" + source["revision"][:12])
    print("Downloading/reusing the pinned Whisper files; no training or test input.", flush=True)
    started = time.perf_counter()
    local = snapshot_download(
        repo_id=source["model_id"], revision=source["revision"], local_dir=model_dir,
        allow_patterns=["*.json", "*.txt", "*.safetensors"], token=False,
    )
    torch.cuda.reset_peak_memory_stats()
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        local, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager", use_safetensors=True,
    ).to("cuda").eval()
    load_seconds = time.perf_counter() - started
    predictions, timings = [], []
    print("Model loaded on " + torch.cuda.get_device_name(0), flush=True)
    for row in records:
        if (row["sampling_rate"], row["channels"], row["sample_width_bytes"]) != (16000, 1, 2):
            raise ValueError("runtime pilot expects the verified mono 16-bit 16 kHz source WAV")
        with wave.open(str(args.pilot / row["audio_path"]), "rb") as handle:
            pcm = handle.readframes(handle.getnframes())
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, language="el", task="transcribe",
                                       max_new_tokens=96, num_beams=1, do_sample=False)
        torch.cuda.synchronize()
        timings.append({"id": row["id"], "cold_inference_seconds": time.perf_counter() - started})
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]
        predictions.append({"id": row["id"], "text": text})
        print(row["id"] + ": " + text, flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output / "predictions.jsonl"
    prediction_bytes = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8") for row in predictions)
    prediction_path.write_bytes(prediction_bytes)
    report = {
        "purpose": "training-pilot runtime smoke only; not a benchmark or model-selection result",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": source, "language": "el", "task": "transcribe", "seed": 42,
        "generation": {"max_new_tokens": 96, "num_beams": 1, "do_sample": False},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": importlib.metadata.version("transformers"),
                        "numpy": np.__version__, "hip": torch.version.hip,
                        "gpu": torch.cuda.get_device_name(0), "dtype": "float32", "attention": "eager"},
        "pilot_audit": audit, "download_and_load_seconds": load_seconds,
        "timings_not_a_warm_benchmark": timings,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "references_used_as_model_input": False, "training_performed": False,
        "official_development_data_read": False, "official_test_data_read": False,
        "wer_cer": None,
        "scoring_note": "no score claimed: tiny train pilot includes an unresolved source annotation marker",
    }
    (args.output / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Runtime check saved to " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
