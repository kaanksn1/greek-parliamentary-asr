"""Owner review of Kemal's submitted baselines on three original TRAIN clips.

Temporary train/dev roles are code-path fixtures, not the official splits or
a benchmark. Use the licensed mHuBERT encoder, never the unlicensed published
HuBERT candidate. Tiny trained weights are removed; public weights stay cached.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.submission_root.resolve()
    if args.output.exists():
        raise FileExistsError("choose a new review output")
    sys.path.insert(0, str(root))
    import torch
    from greek_asr import infer
    from greek_asr.models import ctc, ctc_utils, gmm_hmm, hubert
    from greek_asr.pilot_data import audit_pilot

    torch.set_num_threads(2)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable; no CPU fallback")
    rows, _ = audit_pilot(root / "data/raw/pilot")
    report = {
        "schema": "p3-kemal-submission-review-v1",
        "submission_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [root / "greek_asr/infer.py", *sorted((root / "greek_asr/models").glob("*.py"))]},
        "environment": ctc_utils.environment_record(), "device_requested": args.device,
        "scope": "Three original TRAIN pilot clips; temporary train/dev code-path fixtures only",
        "official_dev_or_test_used": False, "full_training_performed": False,
        "representative_budget_measurement": False, "trained_checkpoints_persisted": False,
        "unlicensed_published_hubert_loaded": False,
        "pilot_ids": ["hparl:" + row["source_utterance_id"] for row in rows],
        "pilot_audio_seconds": sum(row["duration_seconds"] for row in rows), "models": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="p3-kemal-review-") as directory:
        temp = Path(directory)

        def fixture(name, role, *, clean_targets):
            manifest, references = temp / (name + ".jsonl"), temp / (name + "_refs.jsonl")
            audio_rows, text_rows = [], []
            for row in rows:
                identifier = "hparl:" + row["source_utterance_id"]
                with wave.open(str(root / "data/raw/pilot" / row["audio_path"]), "rb") as handle:
                    frames = handle.getnframes()
                audio_rows.append({
                    "id": identifier, "source": "hparl", "original_id": row["source_utterance_id"],
                    "source_split": role, "split": role, "audio_path": "data/raw/pilot/" + row["audio_path"],
                    "audio_sha256": row["audio_sha256"], "sampling_rate": row["sampling_rate"],
                    "channels": row["channels"], "num_frames": frames,
                    "duration_seconds": row["duration_seconds"], "speaker_id": row["source_speaker_id"],
                })
                text = ctc_utils.strip_annotation_tags(row["reference"]) if clean_targets else row["reference"]
                text_rows.append({"id": identifier, "reference": text})
            for path, content in ((manifest, audio_rows), (references, text_rows)):
                path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in content), encoding="utf-8")
            return manifest, references

        train, train_refs = fixture("train", "train", clean_targets=True)
        dev, dev_refs = fixture("dev_role_train_fixture", "dev", clean_targets=False)
        prediction_manifest, _ = fixture("prediction_train", "train", clean_targets=False)
        common_config = {"train_manifest": str(train), "train_references": str(train_refs),
                         "dev_manifest": str(dev), "dev_references": str(dev_refs)}
        expected_ids = set(report["pilot_ids"])

        def predictions(path):
            result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(result) != len(expected_ids) or {r["id"] for r in result} != expected_ids:
                raise AssertionError("prediction IDs are incomplete or duplicated")
            return result

        def gmm_check():
            config = json.loads((root / "configs/kemal/gmm_hmm.json").read_text())
            config.update(common_config, train_subset_hours=None)
            config["training"].update(num_iterations=3, split_iterations=[1],
                                      max_gaussians_per_state=2, min_gaussian_occupancy=1.0)
            output = temp / "gmm"
            training = gmm_hmm.train(config, project_root=root, output_dir=output, device="cpu", progress=False)
            prediction = gmm_hmm.predict(output / "model.pt", prediction_manifest, project_root=root, split="train",
                                         output_dir=temp / "gmm_prediction", settings=gmm_hmm.DecoderSettings(), device="cpu")
            return {"train": training, "predict": prediction,
                    "predictions": predictions(temp / "gmm_prediction/predictions.jsonl")}

        def ctc_check():
            config = json.loads((root / "configs/kemal/ctc.json").read_text())
            config.update(common_config, max_steps=2, batch_size=3, gradient_accumulation_steps=1,
                          warmup_steps=0, dev_eval_every=1, cmvn_utterances=3, num_workers=0)
            output = temp / "ctc"
            training = ctc.train(config, project_root=root, output_dir=output, device=args.device)
            prediction = ctc.predict(output / "checkpoint.best.pt", prediction_manifest, project_root=root,
                                     split="train", output_dir=temp / "ctc_prediction", device=args.device, batch_size=3)
            return {"train": training, "predict": prediction,
                    "predictions": predictions(temp / "ctc_prediction/predictions.jsonl")}

        def mhubert_check():
            config = json.loads((root / "configs/kemal/hubert_ctc.json").read_text())
            if config["encoder_model_id"] != "utter-project/mHuBERT-147":
                raise AssertionError("review must use the documented licensed encoder")
            config.update(common_config, max_steps=2, batch_size=2, gradient_accumulation_steps=1,
                          warmup_steps=0, freeze_transformer_updates=0, dev_eval_every=1)
            output = temp / "mhubert"
            training = hubert.train(config, project_root=root, output_dir=output, device=args.device)
            prediction = hubert.predict(output / "checkpoint-best", prediction_manifest, project_root=root,
                                        split="train", output_dir=temp / "mhubert_prediction", device=args.device, batch_size=2)
            return {"train": training, "predict": prediction,
                    "predictions": predictions(temp / "mhubert_prediction/predictions.jsonl")}

        def wav2vec_check():
            output = temp / "wav2vec"
            result = infer.run(manifest=prediction_manifest, backend_name="wav2vec", split="train",
                               output=output, project_root=root, device=args.device, purpose="runtime-smoke")
            return {"infer": result, "predictions": predictions(output / "predictions.jsonl")}

        for name, function in (("gmm_hmm_cpu", gmm_check), ("ctc_gpu", ctc_check),
                               ("mhubert_ctc_gpu", mhubert_check), ("wav2vec_gpu", wav2vec_check)):
            gc.collect()
            torch.manual_seed(42)
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            observations = []
            original_step = torch.optim.AdamW.step

            def observed_step(optimizer, *positional, **keywords):
                parameters = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                finite = bool(parameters) and bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
                before = parameters[0].detach().clone() if parameters else None
                result = original_step(optimizer, *positional, **keywords)
                observations.append({"parameters_with_gradient": len(parameters), "all_gradients_finite": finite,
                                     "first_parameter_changed": bool(parameters) and not torch.equal(before, parameters[0].detach())})
                return result

            torch.optim.AdamW.step = observed_step
            started = time.perf_counter()
            try:
                result = function()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                if name in ("ctc_gpu", "mhubert_ctc_gpu") and (
                    not observations or not all(r["all_gradients_finite"] and r["first_parameter_changed"] for r in observations)
                ):
                    raise AssertionError("finite gradient / parameter update observation failed")
                report["models"][name] = {"status": "passed", "result": result, "optimizer_observations": observations}
            except Exception as error:
                report["models"][name] = {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error), "traceback": traceback.format_exc(),
                                         "optimizer_observations": observations}
            finally:
                torch.optim.AdamW.step = original_step
            report["models"][name]["outer_wall_seconds"] = time.perf_counter() - started
            if args.device.startswith("cuda"):
                report["models"][name]["peak_gpu_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(name + ": " + report["models"][name]["status"], flush=True)


if __name__ == "__main__":
    main()
