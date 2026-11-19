"""HuBERT + CTC head for Greek ASR: published-checkpoint inference and our own CTC fine-tuning.

1. Published candidate (standard inference, no training by us):
   ``ntinouldinho/hubert_greek_6gram`` is the only Hugging Face HuBERT checkpoint
   found with a trained Greek character CTC head (HubertForCTC, 24 layers).
   It has no model card, no stated license and no stated training data; its
   pinned revision and these gaps are recorded in ``configs/kemal_model_sources.json``.
   Running it is the integration owner's licensing decision. Inference uses the
   shared runner: ``python -m greek_asr.infer --backend hubert`` (greedy CTC; the
   repository's bundled 6-gram KenLM is not used).

2. Own CTC fine-tuning recipe (``train``/``predict`` below): a multilingual HuBERT
   encoder whose pretraining covers Greek, ``utter-project/mHuBERT-147``
   (HuBERT base, CC-BY-NC-SA-4.0), plus a new linear CTC head over the train
   character vocabulary, trained on our labelled train split. This is our group's
   training of a standard baseline family and is reported separately from
   off-the-shelf results. The CNN feature encoder stays frozen; the transformer
   layers and the CTC head are fine-tuned (the transformer can additionally be
   frozen for the first ``freeze_transformer_updates`` updates). Encoder masking
   (``mask_time_prob``) follows the standard wav2vec 2.0/HuBERT fine-tuning recipe.
   The CTC loss is computed here with each item's true encoder output length;
   infeasible items follow the configured policy and nonfinite losses stop the run.

A bare HuBERT encoder or an emotion classifier is never used to transcribe.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from greek_asr.models import common, ctc_utils, wav2vec

ROOT = Path(__file__).resolve().parents[2]
SPECIALS = (common.BLANK, common.UNK)
ENCODER_FILES = ["config.json", "preprocessor_config.json", "model.safetensors"]


def load_backend(*, model_id: str, revision: str, device: str, model_dir: Path | None = None):
    """Loader used by ``python -m greek_asr.infer --backend hubert`` (published checkpoint)."""
    return wav2vec.load_ctc_backend(model_id=model_id, revision=revision, device=device, model_dir=model_dir,
                                    model_class="HubertForCTC")


def encoder_output_lengths(config, lengths):
    """Frames produced by the HuBERT/wav2vec 2.0 CNN feature encoder (no padding, floor division)."""
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        lengths = (lengths - kernel) // stride + 1
    return lengths


def _encoder_dir(config: dict, project_root: Path) -> Path:
    if config.get("encoder_local_dir"):
        return project_root / config["encoder_local_dir"]
    wav2vec.validate_load_options(config["encoder_revision"], "cpu")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=config["encoder_model_id"], revision=config["encoder_revision"],
                                  token=False, allow_patterns=ENCODER_FILES,
                                  local_dir=str(project_root / "models" / ("hubert-encoder-" +
                                                                           config["encoder_revision"][:12]))))


def _feature_extractor(directory: Path):
    from transformers import Wav2Vec2FeatureExtractor

    if (directory / "preprocessor_config.json").is_file():
        return Wav2Vec2FeatureExtractor.from_pretrained(directory, local_files_only=True)
    return Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                    return_attention_mask=False)


def _set_transformer_trainable(model, trainable: bool) -> None:
    for name, parameter in model.hubert.named_parameters():
        if not name.startswith("feature_extractor"):
            parameter.requires_grad = trainable


def _batch_inputs(feature_extractor, waveforms, device):
    """Pad raw audio; an attention mask is passed only when the checkpoint was trained with one."""
    import torch

    use_mask = bool(feature_extractor.return_attention_mask)
    inputs = feature_extractor([w.numpy() for w in waveforms], sampling_rate=feature_extractor.sampling_rate,
                               padding=True, return_tensors="pt", return_attention_mask=use_mask)
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    return {key: value.to(device) for key, value in inputs.items()}, lengths


def _ctc_loss(model, feature_extractor, waveforms, labels, blank_id: int, device, ids: Sequence[str]):
    import torch
    import torch.nn.functional as F

    inputs, lengths = _batch_inputs(feature_extractor, waveforms, device)
    logits = model(**inputs).logits
    out_lengths = encoder_output_lengths(model.config, lengths).to(device)
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    targets = torch.cat([torch.tensor(label, dtype=torch.long) for label in labels]).to(device)
    target_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long, device=device)
    loss = F.ctc_loss(log_probs, targets, out_lengths, target_lengths, blank=blank_id, reduction="mean",
                      zero_infinity=False)
    if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite CTC loss for batch {list(ids)[:5]!r}; feasibility check missed an item")
    return loss


def _items(utterances, rows, encoder_config, vocab, sample_rate: int, *, strip_tags: bool):
    import torch

    from greek_asr.models.ctc import resampled_samples

    items, checks = [], []
    for utterance in utterances:
        text = ctc_utils.strip_annotation_tags(utterance.reference) if strip_tags else utterance.reference
        labels = vocab.encode(" ".join(text.split()))
        row = rows[utterance.id]
        samples = resampled_samples(row["num_frames"], row["sampling_rate"], sample_rate)
        frames = int(encoder_output_lengths(encoder_config, torch.tensor(samples)))
        items.append((utterance.id, utterance.audio_path, labels, utterance.duration_seconds))
        checks.append((utterance.id, frames, labels))
    return items, checks


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path, device: str = "cpu",
          max_steps_override: int | None = None) -> dict[str, Any]:
    import torch
    from transformers import HubertForCTC

    from greek_asr.models.ctc import _lr_lambda, _manifest_rows

    max_steps = max_steps_override if max_steps_override is not None else config.get("max_steps")
    if max_steps is None:
        raise ValueError("max_steps (optimizer updates) must be set from a measured pilot (config or --max-steps)")
    max_steps = int(max_steps)
    resolved = common.resolve_device(device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    train_manifest, dev_manifest = project_root / config["train_manifest"], project_root / config["dev_manifest"]
    train_utterances = common.load_utterances(train_manifest, project_root / config["train_references"],
                                              project_root=project_root, split="train")
    dev_utterances = common.load_utterances(dev_manifest, project_root / config["dev_references"],
                                            project_root=project_root, split="dev")
    vocab = common.build_char_vocabulary([" ".join(u.reference.split()) for u in train_utterances], SPECIALS)
    blank_id = vocab.stoi[common.BLANK]

    encoder_dir = _encoder_dir(config, project_root)
    feature_extractor = _feature_extractor(encoder_dir)
    model = HubertForCTC.from_pretrained(
        encoder_dir, local_files_only=True, vocab_size=vocab.size, pad_token_id=blank_id,
        ctc_loss_reduction="mean", ctc_zero_infinity=False,
        mask_time_prob=float(config.get("mask_time_prob", 0.05)),
        layerdrop=float(config.get("layerdrop", 0.0)), final_dropout=float(config.get("final_dropout", 0.0)))
    model.freeze_feature_encoder()
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    model.to(resolved)
    sample_rate = int(feature_extractor.sampling_rate)

    policy = config.get("infeasible_policy", "exclude_and_report")
    train_items, train_checks = _items(train_utterances, _manifest_rows(train_manifest), model.config, vocab,
                                       sample_rate, strip_tags=False)
    excluded_train, train_problems = ctc_utils.check_ctc_feasibility(
        train_checks, policy=policy, max_excluded_fraction=float(config.get("max_excluded_fraction", 0.01)))
    dev_items, dev_checks = _items(dev_utterances, _manifest_rows(dev_manifest), model.config, vocab, sample_rate,
                                   strip_tags=True)
    excluded_dev, dev_problems = ctc_utils.check_ctc_feasibility(dev_checks, policy="exclude_and_report",
                                                                 max_excluded_fraction=1.0)
    train_items = [item for item in train_items if item[0] not in excluded_train]
    dev_items = [item for item in dev_items if item[0] not in excluded_dev]
    if not train_items or not dev_items:
        raise ValueError("no usable train or dev items after the feasibility check")

    freeze_updates = int(config.get("freeze_transformer_updates", 0))
    _set_transformer_trainable(model, freeze_updates == 0)
    optimizer = torch.optim.AdamW([p for name, p in model.named_parameters()
                                   if not name.startswith("hubert.feature_extractor")],
                                  lr=float(config["learning_rate"]), weight_decay=float(config.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(int(config.get("warmup_steps", 0)), max_steps))
    batch_size, grad_accum = int(config["batch_size"]), int(config.get("gradient_accumulation_steps", 1))

    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema": "greek-asr-hubert-ctc-train-v1", "status": "running", "config": config,
        "provenance": "own CTC fine-tuning of a pretrained HuBERT encoder (not an off-the-shelf result)",
        "encoder": {"model_id": config.get("encoder_model_id"), "revision": config.get("encoder_revision"),
                    "local_dir": config.get("encoder_local_dir")},
        "max_steps": max_steps, "max_steps_source": "cli" if max_steps_override is not None else "config",
        "vocab_size": vocab.size, "seed": seed, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/common.py",
                                               ROOT / "greek_asr/models/ctc_utils.py"),
        "train_manifest_sha256": common.hash_file(train_manifest),
        "train_references_sha256": common.hash_file(project_root / config["train_references"]),
        "dev_manifest_sha256": common.hash_file(dev_manifest),
        "dev_references_sha256": common.hash_file(project_root / config["dev_references"]),
        "train_source_breakdown": common.manifest_source_breakdown(train_manifest),
        "train_items": len(train_items), "train_excluded_infeasible": train_problems,
        "dev_loss_items": len(dev_items), "dev_loss_excluded": dev_problems,
        "dev_reference_preparation": "annotation tags removed in memory for dev loss only",
        "frozen": ["hubert.feature_extractor (CNN)"] + (
            [f"transformer for the first {freeze_updates} updates"] if freeze_updates else []),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters_after_warmup": sum(p.numel() for n, p in model.named_parameters()
                                                 if not n.startswith("hubert.feature_extractor")),
        "effective_batch_size": batch_size * grad_accum, "precision": "float32",
        "device_requested": device, "device_used": str(resolved), "optimizer_updates_completed": 0,
        "best_dev_loss": None, "best_optimizer_update": None, "dev_history": [],
    }

    def load_audio(items_subset):
        return [common.load_waveform(item[1], target_sample_rate=sample_rate) for item in items_subset]

    def dev_loss() -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(dev_items), batch_size):
                chunk = dev_items[start:start + batch_size]
                loss = _ctc_loss(model, feature_extractor, load_audio(chunk), [c[2] for c in chunk], blank_id,
                                 resolved, [c[0] for c in chunk])
                total += loss.item() * len(chunk)
        model.train()
        return total / len(dev_items)

    started = time.perf_counter()
    micro, updates, epoch, best = 0, 0, 0, None
    model.train()
    try:
        while updates < max_steps:
            for batch in ctc_utils.length_bucketed_batches([item[3] for item in train_items], batch_size,
                                                           seed=seed + epoch):
                chunk = [train_items[i] for i in batch]
                loss = _ctc_loss(model, feature_extractor, load_audio(chunk), [c[2] for c in chunk], blank_id,
                                 resolved, [c[0] for c in chunk])
                (loss / grad_accum).backward()
                micro += 1
                if micro % grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                updates += 1
                if freeze_updates and updates == freeze_updates:
                    _set_transformer_trainable(model, True)
                if updates % max(int(config.get("dev_eval_every", 1)), 1) == 0 or updates >= max_steps:
                    dev_started = time.perf_counter()
                    value = dev_loss()
                    report["dev_history"].append({"optimizer_update": updates, "dev_loss": value,
                                                  "train_loss": loss.item(),
                                                  "dev_seconds": time.perf_counter() - dev_started})
                    if best is None or value < best:
                        best = value
                        report["best_dev_loss"], report["best_optimizer_update"] = value, updates
                        _save(output_dir / "checkpoint-best", model, feature_extractor, vocab, config, updates)
                    print(f"hubert-ctc update {updates}/{max_steps}: train {loss.item():.4f} dev {value:.4f}",
                          flush=True)
                if updates >= max_steps:
                    break
            epoch += 1
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["training_seconds"] = time.perf_counter() - started
        report["dev_eval_seconds"] = sum(row["dev_seconds"] for row in report["dev_history"])
        report["seconds_per_update_excluding_dev"] = (
            (report["training_seconds"] - report["dev_eval_seconds"]) / updates if updates else None)
        report["optimizer_updates_completed"], report["micro_steps_completed"] = updates, micro
        report["environment"] = ctc_utils.environment_record()
        if report["status"] == "complete":
            _save(output_dir / "checkpoint-last", model, feature_extractor, vocab, config, updates)
        (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _save(directory: Path, model, feature_extractor, vocab: common.Vocabulary, config: dict, step: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    feature_extractor.save_pretrained(directory)
    (directory / "greek_asr_ctc.json").write_text(json.dumps({
        "schema": "greek-asr-hubert-ctc-checkpoint-v1", "vocab": vocab.to_json(), "step": step,
        "train_config": config}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_trained(directory: Path, device):
    from transformers import HubertForCTC

    metadata = json.loads((directory / "greek_asr_ctc.json").read_text(encoding="utf-8"))
    if metadata.get("schema") != "greek-asr-hubert-ctc-checkpoint-v1":
        raise ValueError(f"{directory}: not a trained HuBERT-CTC checkpoint from this project")
    model = HubertForCTC.from_pretrained(directory, local_files_only=True, use_safetensors=True).to(device).eval()
    return model, _feature_extractor(directory), common.Vocabulary.from_json(metadata["vocab"]), metadata


def predict(checkpoint_dir: Path, manifest_path: Path, *, project_root: Path, split: str, output_dir: Path,
            device: str = "cpu", batch_size: int = 4, allow_test: bool = False,
            test_protocol: Path | None = None) -> dict[str, Any]:
    import torch

    from greek_asr.infer import _test_protocol, load_manifest
    from greek_asr.models.ctc import decode_ids

    resolved = common.resolve_device(device)
    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    protocol_hash = _test_protocol(test_protocol, manifest_path, project_root) if split == "test" else None
    records = load_manifest(manifest_path, project_root, split)
    model, feature_extractor, vocab, metadata = load_trained(checkpoint_dir, resolved)
    blank_id = vocab.stoi[common.BLANK]
    sample_rate = int(feature_extractor.sampling_rate)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "greek-asr-hubert-ctc-predict-v1", "status": "running", "checkpoint": str(checkpoint_dir),
        "checkpoint_step": metadata["step"], "provenance": "own CTC fine-tuning (not an off-the-shelf result)",
        "split": split, "decode": "greedy CTC best path over valid encoder frames", "batch_size": batch_size,
        "manifest_sha256": common.hash_file(manifest_path), "test_protocol_sha256": protocol_hash,
        "records": len(records), "completed_records": 0, "references_used_as_model_input": False,
        "device_requested": device, "device_used": str(resolved), "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/ctc_utils.py"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    writer = common.PartialPredictionWriter(output_dir, [row["id"] for row in records])
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start:start + batch_size]
                waveforms = [common.load_waveform(row["audio"], target_sample_rate=sample_rate) for row in chunk]
                inputs, lengths = _batch_inputs(feature_extractor, waveforms, resolved)
                logits = model(**inputs).logits
                out_lengths = encoder_output_lengths(model.config, lengths)
                for row, ids in zip(chunk, ctc_utils.ctc_greedy_decode(logits, out_lengths, blank_id)):
                    writer.write(row["id"], decode_ids(vocab, ids))
                    report["completed_records"] += 1
        writer.finish()
        report["status"] = "complete"
    except Exception as error:
        writer.abandon()
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["inference_seconds"] = time.perf_counter() - started
        report["environment"] = ctc_utils.environment_record()
        (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--max-steps", type=int, help="measured budget; recorded as max_steps_source=cli")
    train_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint-best directory")
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--batch-size", type=int, default=4)
    predict_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser.add_argument("--allow-test", action="store_true")
    predict_parser.add_argument("--test-protocol", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train(json.loads(args.config.read_text(encoding="utf-8")), project_root=args.project_root,
                  output_dir=args.output, device=args.device, max_steps_override=args.max_steps)
        else:
            predict(args.checkpoint, args.manifest, project_root=args.project_root, split=args.split,
                    output_dir=args.output, device=args.device, batch_size=args.batch_size,
                    allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"hubert {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
