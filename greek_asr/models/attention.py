"""Standard supervised attention encoder-decoder ASR, with an optional hybrid
CTC/attention auxiliary loss (Watanabe et al., 2017, "Hybrid CTC/Attention
Architecture for End-to-End Speech Recognition", IEEE JSTSP).

Architecture (Listen, Attend and Spell -- Chan et al., 2016, arXiv:1508.01211):
- Input features: 80-dim log-mel filterbank (greek_asr.models.common.LogMelFrontend).
- Encoder ("Listener"): pyramidal bidirectional LSTM; each reduction layer
  concatenates adjacent time-step pairs, halving the time resolution.
- Attention: single-head dot-product attention over encoder outputs.
- Decoder ("Speller"): multi-layer unidirectional LSTM; teacher forcing during
  training/dev-loss, free-running (no gold input) during prediction.
- Auxiliary CTC head: linear projection of encoder outputs, trained jointly
  with torch.nn.functional.ctc_loss when ``ctc_weight`` > 0.
- Tokenizer/vocabulary: character-level, learned only from train references
  (greek_asr.models.common.build_char_vocabulary).

This is a vanilla baseline-family architecture (see docs/SEQUENCE_MODELS.md and
docs/BASELINES.md); our own implementation/training labour is credited in
contributions/ADNAN_DEMIR_220911784_adnan-demir.md, not as a new architecture.
The auxiliary CTC head above is part of *this* hybrid CTC/attention baseline
row, not a substitute for Kemal's separate plain-CTC-encoder baseline
(docs/BASELINES.md "Plain CTC system"); training this model does not complete
that deliverable.

Long audio: no truncation is applied to either features or references at any
stage here. An utterance's full length is encoded/decoded as-is; only the
padding introduced by batching is masked out (see common.padding_mask and the
CTC/attention loss masking in compute_loss). There is currently no maximum
utterance-length cap.
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
class AttentionConfig:
    vocab_size: int
    input_dim: int = 80
    encoder_hidden: int = 256
    encoder_pyramid_layers: int = 3
    decoder_embedding: int = 256
    decoder_hidden: int = 512
    decoder_layers: int = 2
    attention_dim: int = 256
    ctc_weight: float = 0.3
    max_decode_length: int = 400

    @property
    def encoder_output_dim(self) -> int:
        return self.encoder_hidden * 2


# --------------------------------------------------------------------------- #
# Model.
# --------------------------------------------------------------------------- #


def _build_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class PyramidalBiLSTMEncoder(nn.Module):
        """Each reduction layer halves time length by concatenating adjacent pairs."""

        def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
            super().__init__()
            layers, in_dim = [], input_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(in_dim, hidden_size, batch_first=True, bidirectional=True))
                in_dim = hidden_size * 2 * 2  # bidirectional output, concatenated pairwise for the next layer
            self.layers = nn.ModuleList(layers)
            self.num_layers = num_layers

        @staticmethod
        def _reduce(x: "torch.Tensor", lengths: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            batch, length, dim = x.shape
            if length % 2 == 1:
                x = x[:, : length - 1, :]
                length -= 1
            x = x.reshape(batch, length // 2, dim * 2)
            return x, torch.div(lengths, 2, rounding_mode="floor")

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            for index, lstm in enumerate(self.layers):
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
                packed_out, _ = lstm(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
                if index < self.num_layers - 1:
                    x, lengths = self._reduce(x, lengths)
            return x, lengths

    class DotProductAttention(nn.Module):
        def __init__(self, query_dim: int, key_dim: int, attn_dim: int):
            super().__init__()
            self.query_proj = nn.Linear(query_dim, attn_dim)
            self.key_proj = nn.Linear(key_dim, attn_dim)

        def forward(self, query: "torch.Tensor", keys: "torch.Tensor",
                    mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            scores = torch.bmm(self.query_proj(query).unsqueeze(1), self.key_proj(keys).transpose(1, 2)).squeeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
            return context, weights

    class Speller(nn.Module):
        def __init__(self, vocab_size: int, embedding_dim: int, encoder_dim: int, hidden_size: int,
                    num_layers: int, attention_dim: int, pad_id: int):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
            self.rnn = nn.LSTM(embedding_dim + encoder_dim, hidden_size, num_layers=num_layers, batch_first=True)
            self.attention = DotProductAttention(hidden_size, encoder_dim, attention_dim)
            self.output_proj = nn.Linear(hidden_size + encoder_dim, vocab_size)
            self.hidden_size, self.num_layers, self.encoder_dim = hidden_size, num_layers, encoder_dim

        def _init_state(self, batch: int, device):
            h = torch.zeros(self.num_layers, batch, self.hidden_size, device=device)
            c = torch.zeros_like(h)
            context = torch.zeros(batch, self.encoder_dim, device=device)
            return h, c, context

        def _step(self, token: "torch.Tensor", h, c, context, encoder_out, encoder_mask):
            embedded = self.embedding(token)
            step_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
            output, (h, c) = self.rnn(step_input, (h, c))
            context, weights = self.attention(h[-1], encoder_out, encoder_mask)
            logits = self.output_proj(torch.cat([output.squeeze(1), context], dim=-1))
            return logits, h, c, context, weights

        def forward(self, decoder_inputs: "torch.Tensor", encoder_out: "torch.Tensor",
                    encoder_mask: "torch.Tensor") -> "torch.Tensor":
            """Teacher-forced training/dev-loss pass; never used for prediction."""
            batch, length = decoder_inputs.shape
            h, c, context = self._init_state(batch, decoder_inputs.device)
            logits = []
            for t in range(length):
                step_logits, h, c, context, _ = self._step(
                    decoder_inputs[:, t], h, c, context, encoder_out, encoder_mask)
                logits.append(step_logits)
            return torch.stack(logits, dim=1)

        def greedy_decode(self, encoder_out, encoder_mask, sos_id: int, eos_id: int,
                          max_length: int) -> "torch.Tensor":
            """Free-running decode; the gold transcript is never fed back in."""
            batch, device = encoder_out.shape[0], encoder_out.device
            h, c, context = self._init_state(batch, device)
            token = torch.full((batch,), sos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch, dtype=torch.bool, device=device)
            outputs = []
            for _ in range(max_length):
                logits, h, c, context, _ = self._step(token, h, c, context, encoder_out, encoder_mask)
                token = torch.where(finished, torch.full_like(logits.argmax(-1), eos_id), logits.argmax(-1))
                outputs.append(token)
                finished = finished | (token == eos_id)
                if bool(torch.all(finished)):
                    break
            return torch.stack(outputs, dim=1)

        def beam_search_decode(self, encoder_out, encoder_mask, sos_id: int, eos_id: int,
                               max_length: int, beam_size: int) -> list[int]:
            """Single-utterance length-normalized beam search; gold text is never an input."""
            if encoder_out.shape[0] != 1:
                raise ValueError("beam_search_decode expects one utterance at a time")
            device = encoder_out.device
            h0, c0, context0 = self._init_state(1, device)
            beams = [{"tokens": [sos_id], "score": 0.0, "h": h0, "c": c0, "context": context0, "done": False}]
            completed: list[dict] = []
            for _ in range(max_length):
                candidates = []
                for beam in beams:
                    if beam["done"]:
                        completed.append(beam)
                        continue
                    token = torch.tensor([beam["tokens"][-1]], device=device)
                    logits, h, c, context, _ = self._step(
                        token, beam["h"], beam["c"], beam["context"], encoder_out, encoder_mask)
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    top_log_probs, top_ids = log_probs.topk(min(beam_size, log_probs.shape[-1]))
                    for log_prob, token_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                        candidates.append({"tokens": beam["tokens"] + [token_id], "score": beam["score"] + log_prob,
                                          "h": h, "c": c, "context": context, "done": token_id == eos_id})
                if not candidates:
                    break
                candidates.sort(key=lambda b: b["score"], reverse=True)
                beams = candidates[:beam_size]
                if all(beam["done"] for beam in beams):
                    completed.extend(beams)
                    beams = []
                    break
            completed.extend(beams)
            best = max(completed, key=lambda b: b["score"] / max(len(b["tokens"]), 1))
            return best["tokens"]

    class AttentionASRModel(nn.Module):
        def __init__(self, config: AttentionConfig, pad_id: int):
            super().__init__()
            self.config, self.pad_id = config, pad_id
            self.encoder = PyramidalBiLSTMEncoder(config.input_dim, config.encoder_hidden, config.encoder_pyramid_layers)
            self.speller = Speller(config.vocab_size, config.decoder_embedding, config.encoder_output_dim,
                                   config.decoder_hidden, config.decoder_layers, config.attention_dim, pad_id)
            self.ctc_head = nn.Linear(config.encoder_output_dim, config.vocab_size) if config.ctc_weight > 0 else None

        def encode(self, features, lengths):
            encoded, encoded_lengths = self.encoder(features, lengths)
            mask = common.padding_mask(encoded_lengths, encoded.shape[1])
            return encoded, encoded_lengths, mask

        def forward(self, features, lengths, decoder_inputs):
            encoded, encoded_lengths, mask = self.encode(features, lengths)
            logits = self.speller(decoder_inputs, encoded, mask)
            ctc_logits = self.ctc_head(encoded) if self.ctc_head is not None else None
            return logits, ctc_logits, encoded_lengths

    return torch, nn, F, AttentionASRModel


# --------------------------------------------------------------------------- #
# Dataset/collate.
# --------------------------------------------------------------------------- #


def _dataset_and_collate(utterances: list[common.Utterance], vocab: common.Vocabulary,
                         frontend: "common.LogMelFrontend", sample_rate: int):
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
        sos, eos = vocab.stoi[common.SOS], vocab.stoi[common.EOS]
        decoder_inputs, _ = common.pad_sequences(
            [torch.tensor([sos, *c], dtype=torch.long) for c in char_id_lists])
        decoder_targets, _ = common.pad_sequences(
            [torch.tensor([*c, eos], dtype=torch.long) for c in char_id_lists])
        ctc_targets, ctc_lengths = common.pad_sequences(
            [torch.tensor(c, dtype=torch.long) for c in char_id_lists])
        return {"ids": list(ids), "features": features, "feature_lengths": feature_lengths,
                "decoder_inputs": decoder_inputs, "decoder_targets": decoder_targets,
                "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths}

    return SequenceDataset(), collate


def compute_loss(torch_module, functional, model, batch, pad_id: int, ctc_weight: float, blank_id: int | None):
    logits, ctc_logits, encoded_lengths = model(batch["features"], batch["feature_lengths"], batch["decoder_inputs"])
    attention_loss = functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), batch["decoder_targets"].reshape(-1), ignore_index=pad_id)
    if ctc_logits is None:
        return attention_loss, {"attention_loss": attention_loss.item(), "ctc_loss": None}
    log_probs = functional.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
    ctc_loss = functional.ctc_loss(log_probs, batch["ctc_targets"], encoded_lengths, batch["ctc_lengths"],
                                   blank=blank_id, zero_infinity=True)
    total = (1 - ctc_weight) * attention_loss + ctc_weight * ctc_loss
    return total, {"attention_loss": attention_loss.item(), "ctc_loss": ctc_loss.item()}


# --------------------------------------------------------------------------- #
# Train / predict.
# --------------------------------------------------------------------------- #


def train(config: dict[str, Any], *, project_root: Path, output_dir: Path, device: str = "cpu") -> dict[str, Any]:
    if config.get("max_steps") is None:
        raise ValueError("config.max_steps (optimizer updates) must be set from a measured pilot run "
                         "before training (see docs/SEQUENCE_MODELS.md); no arbitrary step budget is assumed")
    torch, nn, F, AttentionASRModel = _build_torch()
    resolved_device = common.resolve_device(device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    random.seed(seed)

    train_utterances = common.load_utterances(
        project_root / config["train_manifest"], project_root / config["train_references"],
        project_root=project_root, split="train")
    dev_utterances = common.load_utterances(
        project_root / config["dev_manifest"], project_root / config["dev_references"],
        project_root=project_root, split="dev")

    specials = [common.PAD, common.SOS, common.EOS, common.UNK, common.BLANK]
    vocab = common.build_char_vocabulary([u.reference for u in train_utterances], specials)
    pad_id, blank_id = vocab.stoi[common.PAD], vocab.stoi[common.BLANK]

    model_config = AttentionConfig(vocab_size=vocab.size, ctc_weight=float(config.get("ctc_weight", 0.3)),
                                   max_decode_length=int(config.get("max_decode_length", 400)))
    model = AttentionASRModel(model_config, pad_id).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    frontend = common.LogMelFrontend(sample_rate=int(config.get("sample_rate", 16000)))
    train_dataset, collate = _dataset_and_collate(train_utterances, vocab, frontend, frontend.sample_rate)
    dev_dataset, _ = _dataset_and_collate(dev_utterances, vocab, frontend, frontend.sample_rate)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate)

    output_dir.mkdir(parents=True, exist_ok=False)
    # max_steps counts optimizer updates (not microbatches); gradient_accumulation_steps
    # microbatches make up one update. The loop below only ever stops right after a
    # completed accumulation cycle (never mid-cycle), so an accumulation count that
    # doesn't evenly divide one epoch's worth of data carries the partial gradient
    # over into the next epoch instead of dropping it -- see
    # tests/test_models_attention.py::test_trailing_partial_accumulation_is_never_dropped.
    max_steps, grad_accum = int(config["max_steps"]), int(config.get("gradient_accumulation_steps", 1))
    report: dict[str, Any] = {
        "schema": "greek-asr-attention-train-v1", "config": config, "model_config": asdict(model_config),
        "vocab_size": vocab.size, "seed": seed, "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": common.git_commit_hash(project_root),
        "code_sha256": common.code_file_hashes(Path(__file__), ROOT / "greek_asr/models/common.py"),
        "train_manifest_sha256": common.hash_file(project_root / config["train_manifest"]),
        "train_references_sha256": common.hash_file(project_root / config["train_references"]),
        "dev_manifest_sha256": common.hash_file(project_root / config["dev_manifest"]),
        "dev_references_sha256": common.hash_file(project_root / config["dev_references"]),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "effective_batch_size": int(config["batch_size"]) * grad_accum, "precision": "float32",
        "train_utterances": len(train_utterances), "dev_utterances": len(dev_utterances),
        "device_requested": device, "device_used": str(resolved_device),
        "max_steps": max_steps, "optimizer_updates_completed": 0, "micro_steps_completed": 0,
        "best_dev_loss": None, "best_optimizer_update": None, "training_seconds": None,
    }
    micro_step, optimizer_updates, best_dev_loss = 0, 0, None
    started = time.perf_counter()
    model.train()
    while optimizer_updates < max_steps:
        for batch in train_loader:
            batch = common.move_batch_to_device(batch, resolved_device)
            loss, components = compute_loss(torch, F, model, batch, pad_id, model_config.ctc_weight, blank_id)
            (loss / grad_accum).backward()
            micro_step += 1
            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_updates += 1
                if optimizer_updates % max(int(config.get("dev_eval_every", 1)), 1) == 0 or optimizer_updates >= max_steps:
                    model.eval()
                    with torch.no_grad():
                        dev_losses = [compute_loss(torch, F, model, common.move_batch_to_device(dev_batch, resolved_device),
                                                   pad_id, model_config.ctc_weight, blank_id)[0].item()
                                     for dev_batch in dev_loader]
                    dev_loss = sum(dev_losses) / len(dev_losses)
                    if best_dev_loss is None or dev_loss < best_dev_loss:
                        best_dev_loss, report["best_optimizer_update"] = dev_loss, optimizer_updates
                        common.save_checkpoint(output_dir / "checkpoint.best.pt", model=model, optimizer=optimizer,
                                               scheduler=None, step=optimizer_updates, config=config, vocab=vocab)
                    model.train()
            if optimizer_updates >= max_steps:
                break
    report["training_seconds"] = time.perf_counter() - started
    report["micro_steps_completed"], report["optimizer_updates_completed"] = micro_step, optimizer_updates
    report["best_dev_loss"] = best_dev_loss
    common.save_checkpoint(output_dir / "checkpoint.last.pt", model=model, optimizer=optimizer,
                           scheduler=None, step=optimizer_updates, config=config, vocab=vocab)
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
           decode: str = "greedy", beam_size: int = 5, device: str = "cpu", allow_test: bool = False,
           test_protocol: Path | None = None) -> dict[str, Any]:
    torch, nn, F, AttentionASRModel = _build_torch()
    resolved_device = common.resolve_device(device)
    if split == "test" and not allow_test:
        raise ValueError("test prediction requires --allow-test after the final protocol commit")
    if split == "test":
        _require_test_protocol(test_protocol, manifest_path, project_root)
    from greek_asr.infer import load_manifest
    records = load_manifest(manifest_path, project_root, split)

    checkpoint = common.load_checkpoint(checkpoint_path)  # map_location="cpu": loads regardless of training device
    vocab = common.Vocabulary.from_json(checkpoint["vocab"])
    train_config = checkpoint["config"]
    model_config = AttentionConfig(vocab_size=vocab.size, ctc_weight=float(train_config.get("ctc_weight", 0.3)),
                                   max_decode_length=int(train_config.get("max_decode_length", 400)))
    model = AttentionASRModel(model_config, vocab.stoi[common.PAD]).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    frontend = common.LogMelFrontend(sample_rate=int(train_config.get("sample_rate", 16000)))
    sos_id, eos_id = vocab.stoi[common.SOS], vocab.stoi[common.EOS]
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = [row["id"] for row in records]
    report = {
        "schema": "greek-asr-attention-predict-v1", "status": "running", "checkpoint": str(checkpoint_path),
        "split": split, "decode": decode, "beam_size": beam_size if decode == "beam" else None,
        "records": len(records), "completed_records": 0, "inference_seconds": None,
        "references_used_as_model_input": False, "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_requested": device, "device_used": str(resolved_device),
        "environment": {"python": platform.python_version()},
    }
    writer = common.PartialPredictionWriter(output_dir, ids)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for row in records:
                waveform = common.load_waveform(row["audio"], target_sample_rate=frontend.sample_rate)
                features = frontend(waveform).unsqueeze(0).to(resolved_device)
                lengths = torch.tensor([features.shape[1]], dtype=torch.long, device=resolved_device)
                encoded, _, mask = model.encode(features, lengths)
                if decode == "greedy":
                    tokens = model.speller.greedy_decode(
                        encoded, mask, sos_id, eos_id, model_config.max_decode_length)[0].tolist()
                elif decode == "beam":
                    tokens = model.speller.beam_search_decode(
                        encoded, mask, sos_id, eos_id, model_config.max_decode_length, beam_size)
                else:
                    raise ValueError("decode must be 'greedy' or 'beam'")
                writer.write(row["id"], vocab.decode(tokens, stop_tokens=(common.EOS,)))
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
    predict_parser.add_argument("--decode", choices=("greedy", "beam"), default="greedy")
    predict_parser.add_argument("--beam-size", type=int, default=5)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--allow-test", action="store_true")
    predict_parser.add_argument("--test-protocol", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            train(config, project_root=args.project_root, output_dir=args.output, device=args.device)
        else:
            predict(args.checkpoint, args.manifest, project_root=args.project_root, split=args.split,
                    output_dir=args.output, decode=args.decode, beam_size=args.beam_size, device=args.device,
                    allow_test=args.allow_test, test_protocol=args.test_protocol)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"attention model {args.command} failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
