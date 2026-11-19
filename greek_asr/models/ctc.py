"""Plain supervised CTC ASR baseline: log-mel -> conv subsampling -> BiLSTM -> CTC.

Standard architecture (Graves et al., 2006, "Connectionist Temporal
Classification"; Amodei et al., 2016, "Deep Speech 2" for the conv + bidirectional
recurrent encoder; Park et al., 2019, SpecAugment), trained from scratch on
our labelled train split. It does not use any pretrained checkpoint and is a
separate baseline row from the published wav2vec 2.0/HuBERT CTC models.

- Input: 80-dim log-mel (greek_asr.models.common.LogMelFrontend, 25 ms / 10 ms),
  global mean/variance normalization estimated on train only.
- Encoder: two stride-2 3x3 conv layers (4x time subsampling, 40 ms frames),
  linear projection, stacked bidirectional LSTM; linear CTC head.
- Vocabulary: blank + characters of the train references (+ <unk>, never a target).
- Loss: torch ``ctc_loss`` with true input/target lengths and ``zero_infinity``
  off; infeasible training items are reported and handled by the configured
  policy, nonfinite losses stop the run.
- Decoding: greedy best path over each utterance's valid frames only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from greek_asr.models import common, ctc_utils

ROOT = Path(__file__).resolve().parents[2]
SPECIALS = (common.BLANK, common.UNK)


@dataclass(frozen=True)
class CTCConfig:
    vocab_size: int
    input_dim: int = 80
    conv_channels: int = 32
    model_dim: int = 320
    lstm_hidden: int = 320
    lstm_layers: int = 4
    dropout: float = 0.1


def subsampled_length(lengths):
    """Output length of the two (kernel 3, stride 2, padding 1) conv layers."""
    for _ in range(2):
        lengths = (lengths - 1) // 2 + 1
    return lengths


def _build_torch():
    import torch
    import torch.nn as nn

    class ConvSubsampling(nn.Module):
        """Two stride-2 conv layers; frames past each item's true length are zeroed after every
        layer, so an utterance's output does not depend on how much padding its batch has."""

        def __init__(self, input_dim: int, channels: int, model_dim: int):
            super().__init__()
            self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
            self.projection = nn.Linear(channels * subsampled_length(input_dim), model_dim)

        @staticmethod
        def _mask(x, lengths):
            valid = common.padding_mask(lengths, x.shape[2])  # (B, T)
            return x * valid[:, None, :, None]

        def forward(self, features, lengths):
            x = self._mask(features.unsqueeze(1), lengths)
            lengths = (lengths - 1) // 2 + 1
            x = self._mask(torch.relu(self.conv1(x)), lengths)
            lengths = (lengths - 1) // 2 + 1
            x = self._mask(torch.relu(self.conv2(x)), lengths)  # (B, C, T', F')
            batch, channels, time, freq = x.shape
            x = self.projection(x.permute(0, 2, 1, 3).reshape(batch, time, channels * freq))
            return x, lengths

    class CTCModel(nn.Module):
        def __init__(self, config: CTCConfig):
            super().__init__()
            self.config = config
            self.subsampling = ConvSubsampling(config.input_dim, config.conv_channels, config.model_dim)
            self.lstm = nn.LSTM(config.model_dim, config.lstm_hidden, num_layers=config.lstm_layers,
                                batch_first=True, bidirectional=True,
                                dropout=config.dropout if config.lstm_layers > 1 else 0.0)
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(2 * config.lstm_hidden, config.vocab_size)

        def forward(self, features, lengths):
            x, out_lengths = self.subsampling(features, lengths)
            if int(out_lengths.min()) < 1:
                raise ValueError("an utterance is too short for the conv subsampling")
            packed = nn.utils.rnn.pack_padded_sequence(x, out_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
            return self.head(self.dropout(x)), out_lengths

    return torch, CTCModel


# --------------------------------------------------------------------------- #
# Data.
# --------------------------------------------------------------------------- #


def _manifest_rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                                       if line.strip())}


def resampled_samples(num_frames: int, source_rate: int, target_rate: int) -> int:
    """Sample count after torchaudio's in-memory resampling (ceil rule)."""
    return num_frames if source_rate == target_rate else int(math.ceil(target_rate * num_frames / source_rate))


def output_frames(row: dict, frontend: common.LogMelFrontend) -> int:
    samples = resampled_samples(row["num_frames"], row["sampling_rate"], frontend.sample_rate)
    return int(subsampled_length(frontend.output_length(samples)))


def spec_augment(features, generator: random.Random, *, freq_masks: int, freq_width: int,
                 time_masks: int, time_ratio: float):
    """SpecAugment frequency/time masking on a normalized (time, mel) tensor (mask value 0 = mean)."""
    features = features.clone()
    time, freq = features.shape
    for _ in range(freq_masks):
        width = generator.randint(0, min(freq_width, freq))
        start = generator.randint(0, freq - width)
        features[:, start:start + width] = 0.0
    max_time = int(time * time_ratio)
    for _ in range(time_masks):
        width = generator.randint(0, max_time) if max_time > 0 else 0
        start = generator.randint(0, max(time - width, 0))
        features[start:start + width, :] = 0.0
    return features


class _Featurizer:
    def __init__(self, frontend: common.LogMelFrontend, mean, std):
        self.frontend, self.mean, self.std = frontend, mean, std

    def __call__(self, audio_path: Path):
        waveform = common.load_waveform(audio_path, target_sample_rate=self.frontend.sample_rate)
        return (self.frontend(waveform) - self.mean) / self.std


def estimate_cmvn(utterances: Sequence[common.Utterance], frontend: common.LogMelFrontend, *, seed: int,
                  max_utterances: int):
    import torch

    chosen = list(utterances)
    random.Random(seed).shuffle(chosen)
    chosen = chosen[:max_utterances]
    total, first, second = 0, None, None
    for utterance in chosen:
        x = frontend(common.load_waveform(utterance.audio_path, target_sample_rate=frontend.sample_rate)).double()
        first = x.sum(0) if first is None else first + x.sum(0)
        second = (x * x).sum(0) if second is None else second + (x * x).sum(0)
        total += x.shape[0]
    mean = first / total
    std = torch.sqrt((second / total - mean * mean).clamp_min(1e-8))
    return mean.float(), std.float(), len(chosen)


def _dataset(items: Sequence[tuple[str, Path, list[int]]], featurizer: _Featurizer, augment: dict | None, seed: int):
    import torch
    from torch.utils.data import Dataset

    class CTCDataset(Dataset):
        def __init__(self):
            self.calls = 0  # per-process counter: a new mask on every visit, reproducible for a fixed worker setup

        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int):
            identity, audio_path, labels = items[index]
            features = featurizer(audio_path)
            if augment:
                self.calls += 1
                features = spec_augment(features, random.Random(f"{seed}-{index}-{self.calls}"), **augment)
            return identity, features, torch.tensor(labels, dtype=torch.long)

    return CTCDataset()


def collate(batch):
    import torch

    ids, feature_list, label_list = zip(*batch)
    features, lengths = common.pad_sequences(list(feature_list))
    target_lengths = torch.tensor([len(labels) for labels in label_list], dtype=torch.long)
    return {"ids": list(ids), "features": features, "lengths": lengths,
            "targets": torch.cat(label_list), "target_lengths": target_lengths}


def ctc_loss(model, batch, blank_id: int):
    import torch
    import torch.nn.functional as F

    logits, out_lengths = model(batch["features"], batch["lengths"])
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    loss = F.ctc_loss(log_probs, batch["targets"], out_lengths, batch["target_lengths"], blank=blank_id,
                      reduction="mean", zero_infinity=False)
    if not torch.isfinite(loss):
        raise RuntimeError(f"nonfinite CTC loss for batch {batch['ids'][:5]!r}; feasibility check missed an item")
    return loss


# --------------------------------------------------------------------------- #
# Train / predict.
# --------------------------------------------------------------------------- #


def _labelled_items(utterances, rows, frontend, vocab, *, strip_tags: bool):
    items, checks = [], []
    for utterance in utterances:
        text = ctc_utils.strip_annotation_tags(utterance.reference) if strip_tags else utterance.reference
        labels = vocab.encode(" ".join(text.split()))
        items.append((utterance.id, utterance.audio_path, labels))
        checks.append((utterance.id, output_frames(rows[utterance.id], frontend), labels))
    return items, checks


def _lr_lambda(warmup_steps: int, max_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(max_steps - warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / remaining)
    return schedule


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path, device: str = "cpu",
          max_steps_override: int | None = None) -> dict[str, Any]:
    torch, CTCModel = _build_torch()
    max_steps = max_steps_override if max_steps_override is not None else config.get("max_steps")
    if max_steps is None:
        raise ValueError("max_steps (optimizer updates) must be set from a measured pilot (config or --max-steps); "
                         "no arbitrary training budget is assumed")
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
    frontend = common.LogMelFrontend(sample_rate=int(config.get("sample_rate", 16000)))
    policy = config.get("infeasible_policy", "exclude_and_report")
    max_fraction = float(config.get("max_excluded_fraction", 0.01))
    train_items, train_checks = _labelled_items(train_utterances, _manifest_rows(train_manifest), frontend, vocab,
                                                strip_tags=False)
    excluded_train, train_problems = ctc_utils.check_ctc_feasibility(train_checks, policy=policy,
                                                                     max_excluded_fraction=max_fraction)
    dev_items, dev_checks = _labelled_items(dev_utterances, _manifest_rows(dev_manifest), frontend, vocab,
                                            strip_tags=True)
    excluded_dev, dev_problems = ctc_utils.check_ctc_feasibility(dev_checks, policy="exclude_and_report",
                                                                 max_excluded_fraction=1.0)
    train_items = [item for item in train_items if item[0] not in excluded_train]
    dev_items = [item for item in dev_items if item[0] not in excluded_dev]
    if not train_items or not dev_items:
        raise ValueError("no usable train or dev items after the feasibility check")

    started = time.perf_counter()
    mean, std, cmvn_count = estimate_cmvn(train_utterances, frontend, seed=seed,
                                          max_utterances=int(config.get("cmvn_utterances", 2000)))
    featurizer = _Featurizer(frontend, mean, std)
    model_config = CTCConfig(vocab_size=vocab.size, **config.get("model", {}))
    model = CTCModel(model_config).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config.get("weight_decay", 0.01)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(int(config.get("warmup_steps", 0)), max_steps))
    augment = config.get("spec_augment")
    train_dataset = _dataset(train_items, featurizer, augment, seed)
    dev_dataset = _dataset(dev_items, featurizer, None, seed)
    batch_size, grad_accum = int(config["batch_size"]), int(config.get("gradient_accumulation_steps", 1))
    workers = int(config.get("num_workers", 0))
    durations = {u.id: u.duration_seconds for u in train_utterances}
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=collate, num_workers=workers)

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_config = {**config, "max_steps": max_steps, "model_config": asdict(model_config),
                         "cmvn_mean": mean.tolist(), "cmvn_std": std.tolist(),
                         "sample_rate": frontend.sample_rate}
    report: dict[str, Any] = {
        "schema": "greek-asr-ctc-train-v1", "status": "running", "config": config,
        "max_steps": max_steps, "max_steps_source": "cli" if max_steps_override is not None else "config",
        "model_config": asdict(model_config), "vocab_size": vocab.size, "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(), "code_commit": common.git_commit_hash(project_root),
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
        "cmvn_utterances": cmvn_count, "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "effective_batch_size": batch_size * grad_accum, "precision": "float32",
        "device_requested": device, "device_used": str(resolved), "optimizer_updates_completed": 0,
        "best_dev_loss": None, "best_optimizer_update": None, "dev_history": [],
    }
    micro, updates, epoch = 0, 0, 0
    best = None
    model.train()
    try:
        while updates < max_steps:
            batches = ctc_utils.length_bucketed_batches([durations[item[0]] for item in train_items], batch_size,
                                                        seed=seed + epoch)
            loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batches, collate_fn=collate,
                                                 num_workers=workers)
            for batch in loader:
                batch = common.move_batch_to_device(batch, resolved)
                loss = ctc_loss(model, batch, blank_id)
                (loss / grad_accum).backward()
                micro += 1
                if micro % grad_accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config.get("max_grad_norm", 5.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                updates += 1
                if updates % max(int(config.get("dev_eval_every", 1)), 1) == 0 or updates >= max_steps:
                    dev_loss = _dev_loss(torch, model, dev_loader, resolved, blank_id)
                    report["dev_history"].append({"optimizer_update": updates, "dev_loss": dev_loss,
                                                  "train_loss": loss.item()})
                    if best is None or dev_loss < best:
                        best = dev_loss
                        report["best_dev_loss"], report["best_optimizer_update"] = dev_loss, updates
                        common.save_checkpoint(output_dir / "checkpoint.best.pt", model=model, optimizer=optimizer,
                                               scheduler=scheduler, step=updates, config=checkpoint_config,
                                               vocab=vocab)
                    print(f"ctc update {updates}/{max_steps}: train {loss.item():.4f} dev {dev_loss:.4f}", flush=True)
                    model.train()
                if updates >= max_steps:
                    break
            epoch += 1
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["training_seconds"] = time.perf_counter() - started
        report["optimizer_updates_completed"], report["micro_steps_completed"] = updates, micro
        report["epochs_started"] = epoch + (1 if updates < max_steps else 0)
        report["environment"] = ctc_utils.environment_record()
        if report["status"] == "complete":
            common.save_checkpoint(output_dir / "checkpoint.last.pt", model=model, optimizer=optimizer,
                                   scheduler=scheduler, step=updates, config=checkpoint_config, vocab=vocab)
        (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _dev_loss(torch, model, loader, device, blank_id: int) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = common.move_batch_to_device(batch, device)
            total += ctc_loss(model, batch, blank_id).item() * len(batch["ids"])
            count += len(batch["ids"])
    return total / count


def decode_ids(vocab: common.Vocabulary, ids: Sequence[int]) -> str:
    """Characters of a collapsed CTC path; blank/unknown symbols are not text."""
    skip = {common.BLANK, common.UNK, common.PAD, common.SOS, common.EOS}
    return " ".join("".join(vocab.itos[i] for i in ids if vocab.itos[i] not in skip).split())


def load_model(checkpoint_path: Path, device):
    torch, CTCModel = _build_torch()
    checkpoint = common.load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    vocab = common.Vocabulary.from_json(checkpoint["vocab"])
    model = CTCModel(CTCConfig(**config["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frontend = common.LogMelFrontend(sample_rate=int(config["sample_rate"]))
    featurizer = _Featurizer(frontend, torch.tensor(config["cmvn_mean"]), torch.tensor(config["cmvn_std"]))
    return model, vocab, featurizer, checkpoint


def predict(checkpoint_path: Path, manifest_path: Path, *, project_root: Path, split: str, output_dir: Path,
            device: str = "cpu", batch_size: int = 8, allow_test: bool = False,
            test_protocol: Path | None = None) -> dict[str, Any]:
    import torch

    from greek_asr.infer import _test_protocol, load_manifest

    resolved = common.resolve_device(device)
    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    protocol_hash = _test_protocol(test_protocol, manifest_path, project_root) if split == "test" else None
    records = load_manifest(manifest_path, project_root, split)
    model, vocab, featurizer, checkpoint = load_model(checkpoint_path, resolved)
    blank_id = vocab.stoi[common.BLANK]
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "greek-asr-ctc-predict-v1", "status": "running", "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": common.hash_file(checkpoint_path), "checkpoint_step": checkpoint["step"],
        "split": split, "decode": "greedy CTC best path over valid frames", "batch_size": batch_size,
        "manifest_sha256": common.hash_file(manifest_path), "test_protocol_sha256": protocol_hash,
        "records": len(records), "completed_records": 0, "references_used_as_model_input": False,
        "device_requested": device, "device_used": str(resolved),
        "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/ctc_utils.py"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    writer = common.PartialPredictionWriter(output_dir, [row["id"] for row in records])
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                chunk = records[start:start + batch_size]
                features, lengths = common.pad_sequences([featurizer(row["audio"]) for row in chunk])
                logits, out_lengths = model(features.to(resolved), lengths.to(resolved))
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
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--batch-size", type=int, default=8)
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
        parser.exit(1, f"ctc {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
