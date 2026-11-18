"""GPU pilot for Adnan's three sequence models: real forward/backward/optimizer
step on Serdar's machine, with finite-loss/gradient, VRAM and step-time
measurement. This is NOT a training-length decision and NOT a benchmark: it
only measures what section 7 of docs/HANDOFF_ADNAN_TR.md asks for, using the
three already-committed train-pilot clips (6.61 seconds total) as the only
real-audio input available before Samed's full data delivery is accepted.

Usage (on Serdar's ROCm machine, inside the isolated sequence environment):

    HSA_ENABLE_DXG_DETECTION=1 python scripts/pilot_sequence_models.py \
        --device cuda --output outputs/sequence_gpu_pilot

Combine the reported ``step_seconds`` and ``peak_gpu_memory_allocated_bytes``
with the real train manifest's total hours (once delivered) to estimate a
non-arbitrary ``max_steps``/``batch_size`` for configs/sequence/*.json; this
script does not choose that budget itself.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from greek_asr.models import attention, common, transducer, whisper_finetune  # noqa: E402
from greek_asr.pilot_data import audit_pilot  # noqa: E402


def _pilot_utterances() -> list[common.Utterance]:
    records, _ = audit_pilot(ROOT / "data/raw/pilot")
    return [
        common.Utterance(id="hparl:" + row["source_utterance_id"],
                         audio_path=(ROOT / "data/raw/pilot" / row["audio_path"]).resolve(),
                         duration_seconds=row["duration_seconds"], reference=row["reference"])
        for row in records
    ]


def _environment(device: str) -> dict:
    import torch

    info = {"python": platform.python_version(), "torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
        info["hip_version"] = torch.version.hip
    return info


def pilot_attention(utterances: list[common.Utterance], device: str) -> dict:
    import torch

    torch_module, nn, functional, AttentionASRModel = attention._build_torch()
    vocab = common.build_char_vocabulary(
        [u.reference for u in utterances], [common.PAD, common.SOS, common.EOS, common.UNK, common.BLANK])
    config = attention.AttentionConfig(vocab_size=vocab.size)
    model = AttentionASRModel(config, vocab.stoi[common.PAD]).to(device)
    frontend = common.LogMelFrontend()
    dataset, collate = attention._dataset_and_collate(utterances, vocab, frontend, frontend.sample_rate)
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss, components = attention.compute_loss(
        torch_module, functional, model, batch, vocab.stoi[common.PAD], config.ctc_weight, vocab.stoi[common.BLANK])
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started
    return {"loss_finite": bool(torch.isfinite(loss)), "step_seconds": elapsed, **components}


def pilot_transducer(utterances: list[common.Utterance], device: str) -> dict:
    import torch

    torch_module, nn, rnnt_loss, TransducerASRModel = transducer._build_torch()
    vocab = common.build_char_vocabulary([u.reference for u in utterances], [common.BLANK, common.SOS, common.UNK])
    config = transducer.TransducerConfig(vocab_size=vocab.size)
    model = TransducerASRModel(config).to(device)
    frontend = common.LogMelFrontend()
    dataset, collate = transducer._dataset_and_collate(
        utterances, vocab, frontend, frontend.sample_rate, vocab.stoi[common.SOS])
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in batch.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    loss = transducer.compute_loss(torch_module, rnnt_loss, model, batch, vocab.stoi[common.BLANK])
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started
    return {"loss_finite": bool(torch.isfinite(loss)), "step_seconds": elapsed, "loss": loss.item()}


def pilot_whisper(utterances: list[common.Utterance], device: str) -> dict:
    import torch

    processor, model = whisper_finetune._default_model_loader(
        "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", device=device)
    model.generation_config.language, model.generation_config.task = "el", "transcribe"
    dataset, collate = whisper_finetune._dataset_and_collate(
        utterances, processor, sample_rate=16000, language="el", task="transcribe")
    batch = collate([dataset[i] for i in range(len(dataset))])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outputs = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device))
    outputs.loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started
    return {"loss_finite": bool(torch.isfinite(outputs.loss)), "step_seconds": elapsed, "loss": outputs.loss.item()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sequence_gpu_pilot")
    args = parser.parse_args()

    import torch

    torch.manual_seed(42)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested GPU is unavailable; no CPU fallback for this pilot")
    utterances = _pilot_utterances()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, pilot_fn in (("attention", pilot_attention), ("transducer", pilot_transducer),
                          ("whisper_finetune", pilot_whisper)):
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        measurement = pilot_fn(utterances, args.device)
        if args.device.startswith("cuda"):
            measurement["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report = {
            "schema": "greek-asr-sequence-gpu-pilot-v1", "model": name,
            "pilot_utterances": len(utterances), "pilot_audio_seconds": sum(u.duration_seconds for u in utterances),
            "measurement": measurement, "environment": _environment(args.device),
            "note": "Forward/backward/optimizer-step smoke on 3 pilot clips only; not a training-length decision.",
        }
        (args.output / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
        print(f"{name}: step={measurement['step_seconds']:.3f}s loss_finite={measurement['loss_finite']}")


if __name__ == "__main__":
    main()
