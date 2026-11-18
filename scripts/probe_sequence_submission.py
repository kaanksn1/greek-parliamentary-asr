"""Owner's independent, train-pilot-only review of submitted sequence code.

Calls the submitted pilot functions unchanged. Reports failures per model,
checks PCM32 decoding and Whisper label prefixes on synthetic inputs, and
never loads official dev/test data or persists a trained checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.submission_root.resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    import torchaudio
    from greek_asr.models import attention, common, transducer, whisper_finetune
    from huggingface_hub import snapshot_download
    from transformers import WhisperConfig, WhisperProcessor
    from transformers.models.whisper.modeling_whisper import shift_tokens_right

    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable; no CPU fallback")
    if args.output.exists():
        raise FileExistsError("choose a new output path; an earlier review must not be overwritten")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "p3-sequence-submission-review-v1",
        "submission_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "device": args.device, "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "transformers": importlib.metadata.version("transformers"), "numpy": np.__version__,
        "hip": torch.version.hip,
        "scope": "Synthetic contract probes and unchanged one-step functions on three official train-pilot clips",
        "official_dev_or_test_used": False, "full_training_performed": False,
        "trained_checkpoints_persisted": False,
        "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root / "scripts/pilot_sequence_models.py", *sorted((root / "greek_asr/models").glob("*.py"))]},
        "train_signatures": {"attention": str(inspect.signature(attention.train)),
                             "transducer": str(inspect.signature(transducer.train))},
        "probes": {},
    }

    # Synthetic 32-bit PCM values: same format width as CSS10's committed audit.
    # No corpus recording or reference is manufactured or changed.
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "pcm32.wav"
        samples = [0, 2 ** 30, -(2 ** 30), 2 ** 31 - 1]
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(4)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<4i", *samples))
        before = path.read_bytes()
        decoded = common.load_waveform(path, target_sample_rate=16000)
        report["probes"]["pcm32"] = {
            "expected_frames": len(samples), "actual_frames": decoded.numel(),
            "expected_values": [x / 2 ** 31 for x in samples], "actual_values": decoded.tolist(),
            "source_bytes_unchanged": path.read_bytes() == before,
        }

    revision = "973afd24965f72e36ca33b3055d56a652f456b4d"
    local = snapshot_download("openai/whisper-small", revision=revision, token=False,
                              allow_patterns=["*.json", "*.txt"])
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    config = WhisperConfig.from_pretrained(local, local_files_only=True)
    labels = whisper_finetune.build_labels(processor, ["καλημέρα"], language="el", task="transcribe")
    decoder_inputs = shift_tokens_right(labels, config.pad_token_id, config.decoder_start_token_id)
    report["probes"]["whisper_prefix"] = {
        "decoder_start_token_id": config.decoder_start_token_id,
        "label_prefix": labels[0, :6].tolist(), "decoder_input_prefix": decoder_inputs[0, :6].tolist(),
        "duplicated_decoder_start": bool((decoder_inputs[0, :2] == config.decoder_start_token_id).all()),
    }
    first = np.zeros(31 * 16000, dtype=np.float32)
    second = first.copy()
    second[30 * 16000:] = 0.5
    extracted = processor.feature_extractor([first, second], sampling_rate=16000, return_tensors="pt")
    report["probes"]["whisper_31_second_inputs"] = {
        "different_last_second": True,
        "feature_shape": list(extracted["input_features"].shape),
        "features_identical": torch.equal(extracted["input_features"][0], extracted["input_features"][1]),
        "input_kind": "two synthetic signals only; not corpus audio",
    }

    spec = importlib.util.spec_from_file_location("submitted_pilot", root / "scripts/pilot_sequence_models.py")
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    utterances = pilot._pilot_utterances()
    report["pilot_ids"] = [u.id for u in utterances]
    report["pilot_audio_seconds"] = sum(u.duration_seconds for u in utterances)
    for name, function in (("attention", pilot.pilot_attention), ("transducer", pilot.pilot_transducer),
                           ("whisper_finetune", pilot.pilot_whisper)):
        gc.collect()
        torch.manual_seed(42)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            result = function(utterances, args.device)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            report["probes"][name] = {"status": "passed", "submitted_measurement": result,
                                      "optimizer_step_completed": True}
        except Exception as error:
            report["probes"][name] = {"status": "failed", "error_type": type(error).__name__,
                                      "error": str(error), "traceback": traceback.format_exc()}
        report["probes"][name]["outer_wall_seconds"] = time.perf_counter() - started
        if args.device.startswith("cuda"):
            report["probes"][name]["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(name + ": " + report["probes"][name]["status"], flush=True)


if __name__ == "__main__":
    main()
