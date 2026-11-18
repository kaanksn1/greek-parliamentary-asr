"""Standard supervised RNN-T (Recurrent Neural Network Transducer) ASR.

Architecture (Graves, 2012, "Sequence Transduction with Recurrent Neural
Networks", arXiv:1211.3711; streaming-style encoder per He et al., 2019,
"Streaming End-to-end Speech Recognition for Mobile Devices"):
- Input features: 80-dim log-mel filterbank (greek_asr.models.common.LogMelFrontend).
- Acoustic encoder: unidirectional, pyramidal LSTM (each reduction layer halves
  time resolution by concatenating adjacent frame pairs).
- Prediction network: unidirectional LSTM over the label history, fed a
  dedicated start token before the first real label.
- Joint network: encoder/predictor projections, summed, tanh, linear to the
  blank-inclusive vocabulary.
- Loss: ``torchaudio.functional.rnnt_loss`` (BSD-2-Clause, the official
  torchaudio implementation; CPU forward/backward verified locally and it is
  already part of Serdar's locked ROCm environment, see docs/SEQUENCE_MODELS.md).
- Decoding: greedy transducer decoding (one time step advances only on blank;
  bounded symbols-per-step to avoid an infinite loop), never fed gold text.
- Tokenizer/vocabulary: character-level, learned only from train references.

This is a vanilla baseline-family architecture (see docs/SEQUENCE_MODELS.md and
docs/BASELINES.md); it is explicitly distinct from torchaudio/NeMo's CTC-based
or pretrained Parakeet TDT pipelines. Our own implementation/training labour is
credited in contributions/ADNAN_DEMIR_220911784_adnan-demir.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from greek_asr.models import common

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TransducerConfig:
    vocab_size: int
    input_dim: int = 80
    encoder_hidden: int = 256
    encoder_layers: int = 3
    predictor_embedding: int = 256
    predictor_hidden: int = 256
    predictor_layers: int = 1
    joint_dim: int = 256
    max_symbols_per_step: int = 5
    max_total_symbols: int = 400

    @property
    def encoder_output_dim(self) -> int:
        return self.encoder_hidden


def _build_torch():
    import torch
    import torch.nn as nn
    from torchaudio.functional import rnnt_loss

    class UnidirectionalPyramidalEncoder(nn.Module):
        """Unidirectional LSTM stack; each reduction halves T by pairwise concatenation."""

        def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
            super().__init__()
            layers, in_dim = [], input_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True))
                in_dim = hidden_size * 2  # concatenated pairwise for the next layer's input
            self.layers = nn.ModuleList(layers)
            self.num_layers = num_layers

        @staticmethod
        def _reduce(x, lengths):
            batch, length, dim = x.shape
            if length % 2 == 1:
                x = x[:, : length - 1, :]
                length -= 1
            x = x.reshape(batch, length // 2, dim * 2)
            return x, torch.div(lengths, 2, rounding_mode="floor")

        def forward(self, x, lengths):
            for index, lstm in enumerate(self.layers):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
                packed_out, _ = lstm(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
                if index < self.num_layers - 1:
                    x, lengths = self._reduce(x, lengths)
            return x, lengths

    class PredictionNetwork(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)

        def forward(self, label_inputs, state=None):
            embedded = self.embedding(label_inputs)
            return self.rnn(embedded, state)

    class JointNetwork(nn.Module):
        def __init__(self, encoder_dim: int, predictor_dim: int, joint_dim: int, vocab_size: int):
            super().__init__()
            self.encoder_proj = nn.Linear(encoder_dim, joint_dim)
            self.predictor_proj = nn.Linear(predictor_dim, joint_dim)
            self.output_proj = nn.Linear(joint_dim, vocab_size)

        def forward(self, encoder_out, predictor_out):
            time_steps, label_steps = encoder_out.shape[1], predictor_out.shape[1]
            encoded = self.encoder_proj(encoder_out).unsqueeze(2).expand(-1, -1, label_steps, -1)
            predicted = self.predictor_proj(predictor_out).unsqueeze(1).expand(-1, time_steps, -1, -1)
            return self.output_proj(torch.tanh(encoded + predicted))

    class TransducerASRModel(nn.Module):
        def __init__(self, config: TransducerConfig):
            super().__init__()
            self.config = config
            self.encoder = UnidirectionalPyramidalEncoder(config.input_dim, config.encoder_hidden, config.encoder_layers)
            self.predictor = PredictionNetwork(
                config.vocab_size, config.predictor_embedding, config.predictor_hidden, config.predictor_layers)
            self.joint = JointNetwork(
                config.encoder_output_dim, config.predictor_hidden, config.joint_dim, config.vocab_size)

        def forward(self, features, feature_lengths, predictor_inputs):
            encoded, encoded_lengths = self.encoder(features, feature_lengths)
            predictor_out, _ = self.predictor(predictor_inputs)
            return self.joint(encoded, predictor_out), encoded_lengths

        def greedy_decode(self, encoder_out, start_id: int, blank_id: int) -> list[int]:
            """Standard RNN-T greedy decoding (Graves, 2012); the gold transcript is never an input."""
            if encoder_out.shape[0] != 1:
                raise ValueError("greedy_decode expects one utterance at a time")
            device = encoder_out.device
            state = None
            predictor_out, state = self.predictor(torch.tensor([[start_id]], device=device), state)
            tokens: list[int] = []
            for t in range(encoder_out.shape[1]):
                frame = encoder_out[:, t : t + 1, :]
                for _ in range(self.config.max_symbols_per_step):
                    if len(tokens) >= self.config.max_total_symbols:
                        return tokens
                    logits = self.joint(frame, predictor_out)
                    next_id = int(logits[0, 0, 0].argmax())
                    if next_id == blank_id:
                        break
                    tokens.append(next_id)
                    predictor_out, state = self.predictor(torch.tensor([[next_id]], device=device), state)
            return tokens

    return torch, nn, rnnt_loss, TransducerASRModel


# --------------------------------------------------------------------------- #
# Dataset/collate.
# --------------------------------------------------------------------------- #


def _dataset_and_collate(utterances: list[common.Utterance], vocab: common.Vocabulary,
                         frontend: "common.LogMelFrontend", sample_rate: int, start_id: int):
    import torch
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        def __len__(self) -> int:
            return len(utterances)

        def __getitem__(self, index: int):
            utterance = utterances[index]
            waveform = common.load_waveform(utterance.audio_path, target_sample_rate=sample_rate)
            features = frontend(waveform)
            char_ids = vocab.encode(utterance.reference)
            if not char_ids:
                raise ValueError(f"{utterance.id}: reference must not be empty")
            return utterance.id, features, char_ids

    def collate(batch):
        ids, feature_list, char_id_lists = zip(*batch)
        features, feature_lengths = common.pad_sequences(list(feature_list))
        predictor_inputs, _ = common.pad_sequences(
            [torch.tensor([start_id, *c], dtype=torch.long) for c in char_id_lists])
        targets, target_lengths = common.pad_sequences([torch.tensor(c, dtype=torch.long) for c in char_id_lists])
        return {"ids": list(ids), "features": features, "feature_lengths": feature_lengths,
                "predictor_inputs": predictor_inputs, "targets": targets, "target_lengths": target_lengths}

    return SequenceDataset(), collate


def compute_loss(torch_module, rnnt_loss_fn, model, batch, blank_id: int):
    joint_logits, encoded_lengths = model(batch["features"], batch["feature_lengths"], batch["predictor_inputs"])
    loss = rnnt_loss_fn(
        joint_logits, batch["targets"].to(torch_module.int32), encoded_lengths.to(torch_module.int32),
        batch["target_lengths"].to(torch_module.int32), blank=blank_id, reduction="mean")
    return loss


# --------------------------------------------------------------------------- #
# Train / predict.
# --------------------------------------------------------------------------- #


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path) -> dict[str, Any]:
    if config.get("max_steps") is None:
        raise ValueError("config.max_steps must be set from a measured pilot run before training "
                         "(see docs/SEQUENCE_MODELS.md); no arbitrary step budget is assumed")
    torch, nn, rnnt_loss, TransducerASRModel = _build_torch()
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    train_utterances = common.load_utterances(
        project_root / config["train_manifest"], project_root / config["train_references"],
        project_root=project_root, split="train")
    dev_utterances = common.load_utterances(
        project_root / config["dev_manifest"], project_root / config["dev_references"],
        project_root=project_root, split="dev")

    specials = [common.BLANK, common.SOS, common.UNK]  # blank=0, SOS doubles as the predictor start token
    vocab = common.build_char_vocabulary([u.reference for u in train_utterances], specials)
    blank_id, start_id = vocab.stoi[common.BLANK], vocab.stoi[common.SOS]

    model_config = TransducerConfig(vocab_size=vocab.size)
    model = TransducerASRModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    frontend = common.LogMelFrontend(sample_rate=int(config.get("sample_rate", 16000)))
    train_dataset, collate = _dataset_and_collate(train_utterances, vocab, frontend, frontend.sample_rate, start_id)
    dev_dataset, _ = _dataset_and_collate(dev_utterances, vocab, frontend, frontend.sample_rate, start_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate)

    output_dir.mkdir(parents=True, exist_ok=False)
    max_steps, grad_accum = int(config["max_steps"]), int(config.get("gradient_accumulation_steps", 1))
    report: dict[str, Any] = {
        "schema": "greek-asr-transducer-train-v1", "config": config, "model_config": asdict(model_config),
        "vocab_size": vocab.size, "seed": seed, "created_utc": datetime.now(timezone.utc).isoformat(),
        "train_utterances": len(train_utterances), "dev_utterances": len(dev_utterances),
        "max_steps": max_steps, "steps_completed": 0, "best_dev_loss": None, "best_step": None,
    }
    step, best_dev_loss = 0, None
    model.train()
    while step < max_steps:
        for batch in train_loader:
            loss = compute_loss(torch, rnnt_loss, model, batch, blank_id)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
            step += 1
            if step % max(int(config.get("dev_eval_every", 1)), 1) == 0 or step >= max_steps:
                model.eval()
                with torch.no_grad():
                    dev_losses = [compute_loss(torch, rnnt_loss, model, dev_batch, blank_id).item()
                                 for dev_batch in dev_loader]
                dev_loss = sum(dev_losses) / len(dev_losses)
                if best_dev_loss is None or dev_loss < best_dev_loss:
                    best_dev_loss, report["best_step"] = dev_loss, step
                    common.save_checkpoint(output_dir / "checkpoint.best.pt", model=model, optimizer=optimizer,
                                           scheduler=None, step=step, config=config, vocab=vocab)
                model.train()
            if step >= max_steps:
                break
    report["steps_completed"], report["best_dev_loss"] = step, best_dev_loss
    common.save_checkpoint(output_dir / "checkpoint.last.pt", model=model, optimizer=optimizer,
                           scheduler=None, step=step, config=config, vocab=vocab)
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
           allow_test: bool = False, test_protocol: Path | None = None) -> dict[str, Any]:
    torch, nn, rnnt_loss, TransducerASRModel = _build_torch()
    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    if split == "test":
        _require_test_protocol(test_protocol, manifest_path, project_root)
    from greek_asr.infer import load_manifest
    records = load_manifest(manifest_path, project_root, split)

    checkpoint = common.load_checkpoint(checkpoint_path)
    vocab = common.Vocabulary.from_json(checkpoint["vocab"])
    model_config = TransducerConfig(vocab_size=vocab.size)
    model = TransducerASRModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    train_config = checkpoint["config"]
    frontend = common.LogMelFrontend(sample_rate=int(train_config.get("sample_rate", 16000)))
    blank_id, start_id = vocab.stoi[common.BLANK], vocab.stoi[common.SOS]
    texts: dict[str, str] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for row in records:
            waveform = common.load_waveform(row["audio"], target_sample_rate=frontend.sample_rate)
            features = frontend(waveform).unsqueeze(0)
            lengths = torch.tensor([features.shape[1]], dtype=torch.long)
            encoded, _ = model.encoder(features, lengths)
            tokens = model.greedy_decode(encoded, start_id, blank_id)
            texts[row["id"]] = vocab.decode(tokens)
    elapsed = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = [row["id"] for row in records]
    common.write_predictions(output_dir / "predictions.jsonl", ids, texts)
    report = {
        "schema": "greek-asr-transducer-predict-v1", "checkpoint": str(checkpoint_path), "split": split,
        "records": len(records), "inference_seconds": elapsed, "references_used_as_model_input": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version()},
    }
    (output_dir / "run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--project-root", type=Path, default=ROOT)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--split", choices=("train", "dev", "test"), required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--project-root", type=Path, default=ROOT)
    predict_parser.add_argument("--allow-test", action="store_true")
    predict_parser.add_argument("--test-protocol", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            train(config, project_root=args.project_root, output_dir=args.output)
        else:
            predict(args.checkpoint, args.manifest, project_root=args.project_root, split=args.split,
                    output_dir=args.output, allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"transducer model {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
