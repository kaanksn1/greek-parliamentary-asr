"""Our fine-tuning contribution on the pinned openai/whisper-small checkpoint:
two controlled training conditions sharing one training code path.

- Condition 1 (``configs/sequence/whisper_hparl_only.json``): HParl-only train.
- Condition 2 (``configs/sequence/whisper_hparl_css10.json``): the same HParl
  train plus CSS10 Greek, training only.

Both conditions must share the starting checkpoint/tokenizer, seed, HParl
dev/test and scoring, and an explicitly matched training budget (same
``micro_batch_size`` x ``gradient_accumulation_steps`` effective batch and the
same ``max_steps``, i.e. the same number of optimizer updates) so that any
difference is attributable to data composition, not extra optimization. See
docs/SEQUENCE_MODELS.md "Whisper fine-tuning" for the full protocol and
docs/BASELINES.md / AGENTS.md for why the unmodified checkpoint remains a
separate baseline row (owned by Serdar) from this fine-tuning contribution.

Labels: the processor's tokenizer prefix is pinned to (language, task) before
encoding, so BOS/language/task tokens are real target content, not padding.
Only genuine padding positions (via the tokenizer's attention mask, not a
token-id guess) are set to -100; the real EOS token is therefore never masked.
References are tokenized with ``truncation=False``: a reference is never
silently cut to fit a length limit (see ``build_labels``).

Long audio: WhisperFeatureExtractor pads/truncates every input to the
model's fixed 30-second window, matching Serdar's unmodified Whisper baseline
(greek_asr/pretrained/whisper.py); this is Whisper's own architectural input
contract, not a strategy chosen here. No dev/train example is dropped for
being "hard" or long.

Precision: training here runs in float32 only; no AMP/mixed-precision path is
implemented in this code yet. If Serdar enables AMP on the ROCm pilot, the
overflow/NaN check already present in the training loop (``nonfinite_losses_skipped``)
must be watched, and the actual precision used must be recorded in the run
report instead of this module's current fixed ``"precision": "float32"``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from greek_asr.models import common

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WhisperFinetuneConfig:
    model_id: str
    revision: str
    train_manifest: str
    train_references: str
    dev_manifest: str
    dev_references: str
    seed: int
    learning_rate: float
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    warmup_steps: int
    sample_rate: int = 16000
    language: str = "el"
    task: str = "transcribe"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


def _validate_revision(project_root: Path, model_id: str, revision: str) -> None:
    lock = json.loads((project_root / "configs/model_sources.lock.json").read_text(encoding="utf-8-sig"))
    matches = [row for row in lock["models"] if row["model_id"] == model_id]
    if len(matches) != 1 or matches[0]["revision"] != revision:
        raise ValueError(f"{model_id}@{revision} does not match the pinned configs/model_sources.lock.json entry")


def build_labels(processor: Any, texts: list[str], *, language: str, task: str) -> "torch.Tensor":  # noqa: F821
    """Tokenize with the pinned (language, task) prefix; mask only real padding with -100.

    Masking is driven by the tokenizer's attention mask, not by comparing token
    IDs to the pad/eos ID: Whisper's pad and eos tokens are frequently the same
    ID, and a token-ID-based mask would silently delete the real end-of-sequence
    token together with padding.
    """
    processor.tokenizer.set_prefix_tokens(language=language, task=task)
    # truncation=False is explicit: a reference must never be silently cut to fit a
    # model/tokenizer length limit. If a real utterance ever needs more tokens than
    # the model supports, this must surface as a loud error, not a dropped suffix.
    encoded = processor.tokenizer(list(texts), padding=True, truncation=False, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return labels


def _default_model_loader(model_id: str, revision: str, *, device: str):
    """Load the pinned checkpoint exactly like greek_asr.pretrained.whisper, but trainable."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local = snapshot_download(repo_id=model_id, revision=revision, token=False,
                              allow_patterns=["*.json", "*.txt", "*.safetensors"])
    processor = WhisperProcessor.from_pretrained(local, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        local, local_files_only=True, torch_dtype=torch.float32, attn_implementation="eager", use_safetensors=True,
    ).to(device)
    return processor, model


def _save_checkpoint(path: Path, *, model, optimizer, scheduler, step: int, config: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "greek-asr-whisper-finetune-checkpoint-v1",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "step": step, "config": dict(config), "rng_state": common.capture_rng_state(),
    }, path)


def _load_checkpoint(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "greek-asr-whisper-finetune-checkpoint-v1":
        raise ValueError(f"{path}: unrecognized checkpoint schema")
    return checkpoint


def _dataset_and_collate(utterances: list[common.Utterance], processor: Any, *, sample_rate: int,
                         language: str, task: str):
    import numpy as np
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(utterances)

        def __getitem__(self, index: int):
            utterance = utterances[index]
            waveform = common.load_waveform(utterance.audio_path, target_sample_rate=sample_rate).numpy()
            return utterance.id, waveform, utterance.reference

    def collate(batch):
        ids, waveforms, references = zip(*batch)
        features = processor.feature_extractor(list(waveforms), sampling_rate=sample_rate, return_tensors="pt")
        labels = build_labels(processor, list(references), language=language, task=task)
        return {"ids": list(ids), "input_features": features["input_features"], "labels": labels}

    return SequenceDataset(), collate


def _warmup_then_constant(warmup_steps: int) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return min(1.0, (step + 1) / max(warmup_steps, 1))
    return schedule


# --------------------------------------------------------------------------- #
# Train / predict.
# --------------------------------------------------------------------------- #


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path, device: str = "cpu",
          model_loader: Callable[..., tuple[Any, Any]] | None = None) -> dict[str, Any]:
    import torch

    if config.get("max_steps") is None:
        raise ValueError("config.max_steps (optimizer updates) must be set equal for both Whisper conditions "
                         "from a measured pilot run before training (see docs/SEQUENCE_MODELS.md)")
    _validate_revision(project_root, config["model_id"], config["revision"])
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    train_utterances = common.load_utterances(
        project_root / config["train_manifest"], project_root / config["train_references"],
        project_root=project_root, split="train")
    dev_utterances = common.load_utterances(
        project_root / config["dev_manifest"], project_root / config["dev_references"],
        project_root=project_root, split="dev")

    loader = model_loader or _default_model_loader
    processor, model = loader(config["model_id"], config["revision"], device=device)
    language, task = config.get("language", "el"), config.get("task", "transcribe")
    model.generation_config.language, model.generation_config.task = language, task
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_then_constant(int(config["warmup_steps"])))

    sample_rate = int(config.get("sample_rate", 16000))
    train_dataset, collate = _dataset_and_collate(train_utterances, processor, sample_rate=sample_rate,
                                                   language=language, task=task)
    dev_dataset, _ = _dataset_and_collate(dev_utterances, processor, sample_rate=sample_rate,
                                          language=language, task=task)
    micro_batch = int(config["micro_batch_size"])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=micro_batch, shuffle=True, collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=micro_batch, shuffle=False, collate_fn=collate)

    output_dir.mkdir(parents=True, exist_ok=False)
    max_steps = int(config["max_steps"])
    grad_accum = int(config["gradient_accumulation_steps"])
    report: dict[str, Any] = {
        "schema": "greek-asr-whisper-finetune-train-v1", "config": config,
        "model_id": config["model_id"], "revision": config["revision"], "seed": seed,
        "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/common.py"),
        "train_manifest_sha256": common.hash_file(project_root / config["train_manifest"]),
        "train_references_sha256": common.hash_file(project_root / config["train_references"]),
        "dev_manifest_sha256": common.hash_file(project_root / config["dev_manifest"]),
        "dev_references_sha256": common.hash_file(project_root / config["dev_references"]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "effective_batch_size": micro_batch * grad_accum, "max_steps": max_steps, "precision": "float32",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_utterances": len(train_utterances), "dev_utterances": len(dev_utterances),
        "train_audio_seconds": sum(u.duration_seconds for u in train_utterances),
        # Actual source mix (e.g. hparl vs css10_el) in this training condition's manifest,
        # so the CSS10 sampling ratio is a recorded number, not a derived assumption.
        "train_source_breakdown": common.manifest_source_breakdown(project_root / config["train_manifest"]),
        "optimizer_updates_completed": 0, "micro_steps_completed": 0,
        "best_dev_loss": None, "best_optimizer_update": None, "nonfinite_losses_skipped": 0,
        "training_seconds": None,
    }
    micro_step, optimizer_updates, best_dev_loss = 0, 0, None
    started = time.perf_counter()
    while optimizer_updates < max_steps:
        for batch in train_loader:
            outputs = model(input_features=batch["input_features"].to(device), labels=batch["labels"].to(device))
            loss = outputs.loss
            if not torch.isfinite(loss):
                report["nonfinite_losses_skipped"] += 1
                optimizer.zero_grad()
                continue
            (loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_updates += 1
                if optimizer_updates % max(int(config.get("dev_eval_every", 1)), 1) == 0 or optimizer_updates >= max_steps:
                    model.eval()
                    with torch.no_grad():
                        dev_losses = [model(input_features=dev_batch["input_features"].to(device),
                                            labels=dev_batch["labels"].to(device)).loss.item()
                                     for dev_batch in dev_loader]
                    dev_loss = sum(dev_losses) / len(dev_losses)
                    if best_dev_loss is None or dev_loss < best_dev_loss:
                        best_dev_loss, report["best_optimizer_update"] = dev_loss, optimizer_updates
                        _save_checkpoint(output_dir / "checkpoint.best.pt", model=model, optimizer=optimizer,
                                         scheduler=scheduler, step=optimizer_updates, config=config)
                    model.train()
            if optimizer_updates >= max_steps:
                break
    report["training_seconds"] = time.perf_counter() - started
    report["micro_steps_completed"], report["optimizer_updates_completed"] = micro_step, optimizer_updates
    report["best_dev_loss"] = best_dev_loss
    _save_checkpoint(output_dir / "checkpoint.last.pt", model=model, optimizer=optimizer,
                     scheduler=scheduler, step=optimizer_updates, config=config)
    (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _require_test_protocol(path: Path | None, manifest: Path, project_root: Path) -> None:
    if path is None:
        raise ValueError("test prediction requires --test-protocol committed before the run")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if (protocol.get("selection_fixed") is not True or not protocol.get("scoring_policy")
            or protocol.get("test_manifest_sha256") != manifest_hash):
        raise ValueError("test protocol must fix selection, scoring policy and the exact test manifest")
    relative = path.resolve().relative_to(project_root.resolve()).as_posix()
    committed = subprocess.run(["git", "show", "HEAD:" + relative], cwd=project_root,
                               capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("test protocol differs from its committed version")


def predict(checkpoint_path: Path, manifest_path: Path, *, project_root: Path, split: str, output_dir: Path,
           device: str = "cpu", num_beams: int = 1, allow_test: bool = False, test_protocol: Path | None = None,
           model_loader: Callable[..., tuple[Any, Any]] | None = None) -> dict[str, Any]:
    import torch

    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    if split == "test":
        _require_test_protocol(test_protocol, manifest_path, project_root)
    from greek_asr.infer import load_manifest
    records = load_manifest(manifest_path, project_root, split)

    checkpoint = _load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    loader = model_loader or _default_model_loader
    processor, model = loader(config["model_id"], config["revision"], device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    language, task = config.get("language", "el"), config.get("task", "transcribe")
    sample_rate = int(config.get("sample_rate", 16000))

    output_dir.mkdir(parents=True, exist_ok=False)
    ids = [row["id"] for row in records]
    report = {
        "schema": "greek-asr-whisper-finetune-predict-v1", "status": "running", "checkpoint": str(checkpoint_path),
        "split": split, "model_id": config["model_id"], "revision": config["revision"], "num_beams": num_beams,
        "records": len(records), "completed_records": 0, "inference_seconds": None,
        "references_used_as_model_input": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version()},
    }
    writer = common.PartialPredictionWriter(output_dir, ids)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for row in records:
                waveform = common.load_waveform(row["audio"], target_sample_rate=sample_rate).numpy()
                features = processor.feature_extractor([waveform], sampling_rate=sample_rate, return_tensors="pt")
                generated = model.generate(input_features=features["input_features"].to(device), language=language,
                                           task=task, num_beams=num_beams, do_sample=False)
                decoded = processor.batch_decode(generated, skip_special_tokens=True)
                if len(decoded) != 1 or not isinstance(decoded[0], str):
                    raise ValueError(f"{row['id']}: expected a single-record transcription")
                writer.write(row["id"], decoded[0])
                report["completed_records"] += 1
        writer.finish()
        report["status"] = "complete"
    except Exception as error:
        writer.abandon()
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["inference_seconds"] = time.perf_counter() - started
        (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--project-root", type=Path, default=ROOT)
    train_parser.add_argument("--device", default="cpu")

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--num-beams", type=int, default=1)
    predict_parser.add_argument("--allow-test", action="store_true")
    predict_parser.add_argument("--test-protocol", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            train(config, project_root=args.project_root, output_dir=args.output, device=args.device)
        else:
            predict(args.checkpoint, args.manifest, project_root=args.project_root, split=args.split,
                    output_dir=args.output, device=args.device, num_beams=args.num_beams,
                    allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"whisper fine-tuning {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
