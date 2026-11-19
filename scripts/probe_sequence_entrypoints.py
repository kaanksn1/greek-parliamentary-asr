"""Exercise submitted train/predict CLIs on train-pilot code-path fixtures.

The temporary dev-role fixture uses an original TRAIN clip, as the submitted
unit tests do. It is not the official development split or an ASR benchmark.
No source audio/reference is modified and temporary trained weights are removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.submission_root.resolve()
    if args.output.exists():
        raise FileExistsError("choose a new diagnostic output")
    sys.path.insert(0, str(root))
    import torch
    from greek_asr.pilot_data import audit_pilot

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested GPU unavailable; no CPU fallback")
    rows, _ = audit_pilot(root / "data/raw/pilot")
    report = {
        "schema": "p3-sequence-entrypoint-review-v1",
        "submission_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "review_probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "device_requested": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "scope": "Submitted CLI training/checkpoint/prediction on temporary train-pilot fixtures only",
        "official_development_or_test_used": False,
        "dev_role_is_original_train_pilot": True,
        "full_training_performed": False, "trained_checkpoints_persisted": False,
        "models": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run(arguments):
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, *arguments], cwd=root, env=env,
                                   text=True, capture_output=True, timeout=120)
        return {"arguments": arguments, "exit_code": completed.returncode,
                "wall_seconds": time.perf_counter() - started,
                "stdout": completed.stdout, "stderr": completed.stderr}

    with tempfile.TemporaryDirectory(prefix="p3-sequence-cli-review-") as directory:
        temp = Path(directory)

        def fixture(name, selected, role):
            manifest, references = temp / (name + ".jsonl"), temp / (name + "_refs.jsonl")
            audio_rows, text_rows = [], []
            for row in selected:
                identifier = "hparl:" + row["source_utterance_id"]
                with wave.open(str(root / "data/raw/pilot" / row["audio_path"]), "rb") as handle:
                    frames = handle.getnframes()
                audio_rows.append({
                    "id": identifier, "source": "hparl", "original_id": row["source_utterance_id"],
                    "source_split": role, "split": role,
                    "audio_path": "data/raw/pilot/" + row["audio_path"],
                    "audio_sha256": row["audio_sha256"], "sampling_rate": row["sampling_rate"],
                    "channels": row["channels"], "num_frames": frames,
                    "duration_seconds": row["duration_seconds"],
                })
                text_rows.append({"id": identifier, "reference": row["reference"]})
            for path, content in ((manifest, audio_rows), (references, text_rows)):
                path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in content), encoding="utf-8")
            return manifest, references

        train, train_refs = fixture("train", rows[:2], "train")
        dev, dev_refs = fixture("dev_fixture", rows[2:], "dev")
        prediction_manifest, _ = fixture("prediction_train", rows, "train")
        expected_ids = {"hparl:" + row["source_utterance_id"] for row in rows}
        for name in ("attention", "transducer"):
            config = {
                "train_manifest": str(train), "train_references": str(train_refs),
                "dev_manifest": str(dev), "dev_references": str(dev_refs),
                "seed": 42, "learning_rate": 1e-4, "batch_size": 2,
                "gradient_accumulation_steps": 3, "max_steps": 2, "dev_eval_every": 1,
                "sample_rate": 16000, "ctc_weight": 0.3, "max_decode_length": 12,
            }
            config_path = temp / (name + "_config.json")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            training, prediction = temp / (name + "_training"), temp / (name + "_prediction")
            result = {"status": "running", "diagnostic_optimizer_updates": 2,
                      "diagnostic_accumulation": 3}
            try:
                result["train_command"] = run([
                    "-m", "greek_asr.models." + name, "train", "--config", str(config_path),
                    "--project-root", str(root), "--device", args.device, "--output", str(training)])
                if result["train_command"]["exit_code"]:
                    raise RuntimeError("submitted training CLI failed")
                result["train_report"] = json.loads((training / "run.json").read_text())
                result["predict_command"] = run([
                    "-m", "greek_asr.models." + name, "predict", "--checkpoint", str(training / "checkpoint.best.pt"),
                    "--manifest", str(prediction_manifest), "--split", "train", "--project-root", str(root),
                    "--device", args.device, "--output", str(prediction)])
                if result["predict_command"]["exit_code"]:
                    raise RuntimeError("submitted prediction CLI failed")
                result["predict_report"] = json.loads((prediction / "run.json").read_text())
                predictions = [json.loads(line) for line in (prediction / "predictions.jsonl").read_text().splitlines()]
                result["prediction_ids_complete"] = len(predictions) == len(expected_ids) and {r["id"] for r in predictions} == expected_ids
                train_report, predict_report = result["train_report"], result["predict_report"]
                valid = (train_report["optimizer_updates_completed"] == 2 and
                         train_report["micro_steps_completed"] == 6 and
                         train_report["device_used"] == args.device and
                         predict_report["device_used"] == args.device and result["prediction_ids_complete"])
                result["status"] = "passed" if valid else "failed_validation"
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, error=str(error))
            report["models"][name] = result
            args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n")
            print(name + ": " + result["status"], flush=True)


if __name__ == "__main__":
    main()
