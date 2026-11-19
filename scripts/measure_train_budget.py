"""Measure unchanged submitted training kernels on real TRAIN audio, not test.

Each invocation owns one model/process, fixes its seed, synchronizes the GPU,
separates warmup, records input IDs and measures forward/backward/AdamW and
loss-only evaluation. No trained model or ASR result from this pilot is reported.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["attention", "transducer", "whisper", "ctc", "hubert", "gmm"])
    parser.add_argument("--suite", action="store_true", help="run one isolated model process at a time, with timeouts")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--condition", choices=["random", "long", "css10"], default="random")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--random-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, default=ROOT / "outputs/budget/pilot")
    args = parser.parse_args()
    if args.suite:
        suite(args)
        return
    if args.model is None:
        parser.error("--model or --suite is required")
    import torch
    from greek_asr.models import common, ctc_utils

    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    device = common.resolve_device("cpu" if args.model == "gmm" else args.device)
    gpu = device.type == "cuda"
    if gpu:
        # Keep WDDM/ROCm out of host-memory spill; an oversized trial fails here.
        torch.cuda.set_per_process_memory_fraction(.85)
        torch.cuda.reset_peak_memory_stats()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("choose a new measurement output")
    report = {"schema": "p3-representative-training-budget-v1", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
              "model": args.model, "batch_size": args.batch_size, "condition": args.condition,
              "random_offset": args.random_offset,
              "device": str(device), "precision": "float32", "optimizer": "AdamW",
              "warmup_updates": 1, "requested_measured_updates": args.steps,
              "official_dev_or_test_read": False, "benchmark_result": False,
              "checkpoint_saved": False, "gpu_allocator_fraction_limit": .85 if gpu else None,
              "environment": ctc_utils.environment_record(),
              "runtime_environment": {name: os.environ.get(name) for name in
                                      ("HSA_ENABLE_DXG_DETECTION", "MIOPEN_FIND_MODE", "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_source_sha256": common.code_file_hashes(*sorted((ROOT / "greek_asr/models").glob("*.py"))),
              "updates": [], "evaluation_batches": []}

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def sync():
        if gpu:
            torch.cuda.synchronize()

    save()
    try:
        pilot_rows = read_rows(args.pilot / "train.jsonl")
        refs = {row["id"]: row["reference"] for row in read_rows(args.pilot / "references.jsonl")}
        acquisition = json.loads((args.pilot / "acquisition.json").read_text())
        if args.condition == "css10":
            rows = read_rows(ROOT / "data/manifests/css10_el_train.jsonl")
            refs = {row["id"]: row["reference"] for row in read_rows(ROOT / "data/manifests/references/css10_el_train.jsonl")}
            # Longest genuine secondary-source inputs stress memory and target lengths.
            rows.sort(key=lambda row: (-row["duration_seconds"], row["id"]))
        elif args.condition == "long":
            rows = sorted(pilot_rows, key=lambda row: (-row["duration_seconds"] * len(refs[row["id"]]), row["id"]))
        else:
            by_id = {row["id"]: row for row in pilot_rows}
            rows = [by_id[identifier] for identifier in acquisition["random_ids"]]
            offset = args.random_offset % len(rows)
            rows = rows[offset:] + rows[:offset]
        if any(row["split"] != "train" or
               (row["source"] == "hparl" and row["source_split"] != "train") or
               (row["source"] == "css10_el" and row["source_split"] != "unsplit") or
               row["source"] not in ("hparl", "css10_el") for row in rows):
            raise ValueError("only HParl TRAIN or training-only unsplit CSS10 may enter measurements")
        for row in rows:
            data = (ROOT / row["audio_path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != row["audio_sha256"]:
                raise ValueError(f"missing, LFS pointer or changed pilot audio: {row['id']}")
        utterances = [common.Utterance(row["id"], ROOT / row["audio_path"], row["duration_seconds"], refs[row["id"]]) for row in rows]
        # The exact full candidate TRAIN vocabulary matters for parameter/memory cost.
        all_text = [row["reference"] for row in read_rows(ROOT / "data/manifests/references/train.jsonl")]
        frontend = common.LogMelFrontend()
        init_started = time.perf_counter()
        if args.model in ("attention", "transducer"):
            from greek_asr.models import attention, transducer
            module = attention if args.model == "attention" else transducer
            specials = ([common.PAD, common.SOS, common.EOS, common.UNK, common.BLANK]
                        if args.model == "attention" else [common.BLANK, common.SOS, common.UNK])
            vocab = common.build_char_vocabulary(all_text, specials)
            if args.model == "attention":
                _, _, functional, cls = module._build_torch()
                config = module.AttentionConfig(vocab_size=vocab.size)
                model = cls(config, vocab.stoi[common.PAD]).to(device)
                dataset, collate = module._dataset_and_collate(utterances, vocab, frontend, 16000)
                def loss_fn(batch):
                    return module.compute_loss(torch, functional, model, batch, vocab.stoi[common.PAD],
                                               config.ctc_weight, vocab.stoi[common.BLANK])[0]
            else:
                _, _, rnnt_loss, cls = module._build_torch()
                config = module.TransducerConfig(vocab_size=vocab.size)
                model = cls(config).to(device)
                dataset, collate = module._dataset_and_collate(utterances, vocab, frontend, 16000, vocab.stoi[common.SOS])
                def loss_fn(batch):
                    return module.compute_loss(torch, rnnt_loss, model, batch, vocab.stoi[common.BLANK])
            lr, clip = 3e-4, 5.
        elif args.model == "whisper":
            from greek_asr.models import whisper_finetune
            processor, model = whisper_finetune._default_model_loader(
                "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", device=str(device))
            dataset, collate = whisper_finetune._dataset_and_collate(
                utterances, processor, sample_rate=16000, language="el", task="transcribe",
                decoder_start_token_id=model.config.decoder_start_token_id)
            def loss_fn(batch):
                return model(input_features=batch["input_features"], labels=batch["labels"]).loss
            lr, clip = 1e-5, 1.
        elif args.model == "ctc":
            from greek_asr.models import ctc
            cfg = json.loads((ROOT / "configs/kemal/ctc.json").read_text())
            vocab = common.build_char_vocabulary([" ".join(text.split()) for text in all_text], ctc.SPECIALS)
            items, checks = ctc._labelled_items(utterances, {row["id"]: row for row in rows}, frontend, vocab, strip_tags=False)
            excluded, problems = ctc_utils.check_ctc_feasibility(checks, policy="exclude_and_report", max_excluded_fraction=1.)
            report["timing_cohort_ctc_exclusions"] = problems
            items = [item for item in items if item[0] not in excluded]
            by_id = {u.id: u for u in utterances}
            utterances = [by_id[item[0]] for item in items]
            mean, std, count = ctc.estimate_cmvn(utterances, frontend, seed=42, max_utterances=min(128, len(utterances)))
            _, cls = ctc._build_torch()
            model = cls(ctc.CTCConfig(vocab_size=vocab.size, **cfg["model"])).to(device)
            dataset = ctc._dataset(items, ctc._Featurizer(frontend, mean, std), cfg["spec_augment"], 42)
            collate = ctc.collate
            def loss_fn(batch):
                return ctc.ctc_loss(model, batch, vocab.stoi[common.BLANK])
            lr, clip = .001, 5.
        elif args.model == "hubert":
            from greek_asr.models import hubert
            from transformers import HubertForCTC
            directory = Path("/home/serda/p3-reviews/kemal-680b875d/models/hubert-encoder-7ad3fc0bc510")
            cfg = json.loads((ROOT / "configs/kemal/hubert_ctc.json").read_text())
            if not directory.is_dir():
                directory = hubert._encoder_dir(cfg, ROOT)
            processor = hubert._feature_extractor(directory)
            vocab = common.build_char_vocabulary([" ".join(text.split()) for text in all_text], hubert.SPECIALS)
            model = HubertForCTC.from_pretrained(directory, local_files_only=True, vocab_size=vocab.size,
                                                pad_token_id=vocab.stoi[common.BLANK], ctc_loss_reduction="mean",
                                                ctc_zero_infinity=False, mask_time_prob=.05, layerdrop=0., final_dropout=0.)
            model.freeze_feature_encoder()
            model.to(device)
            items, checks = hubert._items(utterances, {row["id"]: row for row in rows}, model.config, vocab, 16000, strip_tags=False)
            excluded, problems = ctc_utils.check_ctc_feasibility(checks, policy="exclude_and_report", max_excluded_fraction=1.)
            report["timing_cohort_ctc_exclusions"] = problems
            items = [item for item in items if item[0] not in excluded]
            by_id = {u.id: u for u in utterances}
            utterances = [by_id[item[0]] for item in items]
            dataset = items
            def collate(chunk):
                return chunk
            def loss_fn(batch):
                return hubert._ctc_loss(model, processor, [common.load_waveform(item[1], target_sample_rate=16000) for item in batch],
                                        [item[2] for item in batch], vocab.stoi[common.BLANK], device, [item[0] for item in batch])
            lr, clip = 5e-5, 1.
            report["hubert_mode"] = "full transformer fine-tuning, CNN frozen; head-only warmup is cheaper"
        else:
            from greek_asr.models import gmm_hmm
            # Use the entire random cohort, retaining the real length distribution.
            cfg = json.loads((ROOT / "configs/kemal/gmm_hmm.json").read_text())
            cfg.update(train_manifest=str(args.pilot / "train.jsonl"),
                       train_references=str(args.pilot / "references.jsonl"), train_subset_hours=None)
            cfg["training"].update(num_iterations=5, split_iterations=[0, 1, 2, 3], max_gaussians_per_state=16,
                                   max_excluded_fraction=1.)
            run = gmm_hmm.train(cfg, project_root=ROOT, output_dir=args.output.with_suffix(""), device="cpu", progress=True)
            report.update(status="complete", gmm_train=run, checkpoint_saved=True,
                          checkpoint_scope="local diagnostic model for TRAIN decode timing; not a benchmark checkpoint",
                          checkpoint_path=str(args.output.with_suffix("") / "model.pt"))
            save()
            return
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=.01 if args.model == "ctc" else 0. if args.model == "hubert" else .01)
        report["initialization_seconds"] = time.perf_counter() - init_started
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report["pool_utterances"] = len(utterances)
        report["pool_audio_hours"] = sum(u.duration_seconds for u in utterances) / 3600
        report["pool_duration_max_seconds"] = max(u.duration_seconds for u in utterances)
        model.train()
        for index in range(args.steps + 1):
            if args.condition == "random":
                indices = [(index * args.batch_size + j) % len(dataset) for j in range(args.batch_size)]
            else:
                indices = [j % len(dataset) for j in range(args.batch_size)]
            selected = [utterances[i] for i in indices]
            sync()
            started = time.perf_counter()
            batch = collate([dataset[i] for i in indices])
            if isinstance(batch, dict):
                batch = common.move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite measured training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            sync()
            observation = {"warmup": index == 0, "seconds": time.perf_counter() - started,
                           "ids": [u.id for u in selected], "audio_seconds": sum(u.duration_seconds for u in selected),
                           "max_audio_seconds": max(u.duration_seconds for u in selected),
                           "max_target_characters": max(len(u.reference) for u in selected),
                           "loss": loss.item(), "gradient_norm": float(grad_norm)}
            report["updates"].append(observation)
            print(json.dumps({"model": args.model, "condition": args.condition, "update": index, **observation}), flush=True)
            save()
            del batch, loss
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            for index in range(min(args.steps, 3)):
                indices = [(index * args.batch_size + j) % len(dataset) for j in range(args.batch_size)]
                sync()
                started = time.perf_counter()
                batch = collate([dataset[i] for i in indices])
                if isinstance(batch, dict):
                    batch = common.move_batch_to_device(batch, device)
                loss = loss_fn(batch)
                sync()
                report["evaluation_batches"].append({"seconds": time.perf_counter() - started,
                                                       "audio_seconds": sum(utterances[i].duration_seconds for i in indices),
                                                       "loss": float(loss), "loss_finite": bool(torch.isfinite(loss))})
                del batch, loss
        measured = report["updates"][1:]
        seconds = sum(item["seconds"] for item in measured)
        audio = sum(item["audio_seconds"] for item in measured)
        report.update(status="complete", mean_update_seconds=statistics.mean(item["seconds"] for item in measured),
                      max_update_seconds=max(item["seconds"] for item in measured),
                      training_seconds_per_audio_hour=seconds / audio * 3600,
                      evaluation_seconds_per_audio_hour=sum(item["seconds"] for item in report["evaluation_batches"]) /
                      sum(item["audio_seconds"] for item in report["evaluation_batches"]) * 3600)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if gpu:
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            report["gpu_name"] = torch.cuda.get_device_name()
        save()
    if report["status"] != "complete":
        sys.exit(2)


def suite(args):
    """Sequential measurements avoid GPU contention; preserve every failed trial."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("whisper", 2, "random", 32), ("whisper", 2, "long", 3), ("whisper", 2, "css10", 3),
        ("attention", 16, "random", 6), ("attention", 16, "long", 2),
        ("ctc", 32, "random", 3), ("ctc", 32, "long", 2),
        ("hubert", 8, "random", 8), ("hubert", 8, "long", 2),
        ("hubert", 4, "random", 8), ("hubert", 4, "long", 2),
        ("transducer", 8, "random", 3), ("transducer", 4, "long", 1),
        ("gmm", 1, "random", 1),
    ]
    index = []
    for model, batch, condition, steps in cases:
        target = output / f"{model}_{condition}_b{batch}_n{steps}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--model", model,
                   "--batch-size", str(batch), "--condition", condition, "--steps", str(steps),
                   "--device", args.device, "--pilot", str(args.pilot), "--output", str(target)]
        record = {"model": model, "batch_size": batch, "condition": condition, "steps": steps,
                  "output": target.name, "command": command, "timeout_seconds": 600}
        if target.exists():
            record["status"] = "existing_output_preserved"
        else:
            print(json.dumps({"starting": target.name}), flush=True)
            with target.with_suffix(".log").open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                    record.update(status="process_finished", exit_code=completed.returncode)
                except subprocess.TimeoutExpired:
                    record["status"] = "timeout_process_stopped"
        index.append(record)
        (output / "suite.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
