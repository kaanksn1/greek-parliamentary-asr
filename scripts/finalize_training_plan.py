"""Generate owner run settings/report from actual measurements and selected data.

These are train/dev budgets, never a final-test authorization. Do not run full
training until Samed's used-data verification and corrected CSS10 audit pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    data = path.read_bytes()
    return json.loads(data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def rounded_updates(examples, effective_batch, passes):
    return int(math.ceil(math.ceil(examples * passes / effective_batch) / 100) * 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, default=ROOT / "outputs/budget")
    parser.add_argument("--evidence", type=Path, default=ROOT / "docs/reviews/budget_2026-11-20")
    args = parser.parse_args()
    inputs, evidence = args.measurements, args.evidence
    selection = read(ROOT / "configs/training_data_selection.json")
    clean_n = selection["cohorts"]["clean_selected_train"]["records"]
    mixed_n = selection["cohorts"]["mixed_selected_train"]["records"]
    dev_hours = selection["cohorts"]["unchanged_clean_dev"]["hours"]
    test_hours = selection["cohorts"]["unchanged_frozen_test"]["hours"]
    artifact = selection["artifacts"]
    cases = {"whisper": inputs / "representative/whisper_random_b2_n32.json",
             "attention": inputs / "representative/attention_random_b16_n6.json",
             "ctc": inputs / "representative/ctc_random_b32_n3.json",
             "hubert": inputs / "hubert_random_b8_fast.json",
             "transducer": inputs / "representative/transducer_random_b8_n3.json"}
    if (inputs / "ctc_random_b32_fast.json").exists():
        fast_ctc = read(inputs / "ctc_random_b32_fast.json")
        if fast_ctc.get("status") == "complete":
            cases["ctc"] = inputs / "ctc_random_b32_fast.json"
    configs = [("attention", "configs/sequence/attention.json", 16, 1, 2, 1000),
               ("transducer", "configs/sequence/transducer.json", 8, 1, 1, 1000),
               ("ctc", "configs/kemal/ctc.json", 32, 1, 3, 1000),
               ("hubert", "configs/kemal/hubert_ctc.json", 8, 4, 2, 500),
               ("whisper", "configs/sequence/whisper_hparl_only.json", 2, 8, 1, 500),
               ("whisper", "configs/sequence/whisper_hparl_css10.json", 2, 8, 1, 500)]
    training = []
    for name, path, batch, accumulation, passes, cadence in configs:
        measurement = read(cases[name])
        if measurement["status"] != "complete":
            raise ValueError(f"incomplete representative measurement for {name}")
        cfg_path = ROOT / path
        cfg = read(cfg_path)
        mixed = path.endswith("whisper_hparl_css10.json")
        cfg["train_manifest"] = artifact["mixed_manifest" if mixed else "clean_manifest"]["path"]
        cfg["train_references"] = artifact["mixed_references" if mixed else "clean_references"]["path"]
        # Both Whisper runs receive the mixed-pool one-pass update count. Their
        # shared checkpoint/optimizer/schedule/seed/batch settings stay matched.
        updates = rounded_updates(mixed_n if name == "whisper" else clean_n, batch * accumulation, passes)
        cfg.update(max_steps=updates, gradient_accumulation_steps=accumulation, dev_eval_every=cadence)
        cfg["micro_batch_size" if name == "whisper" else "batch_size"] = batch
        if name in ("ctc", "hubert"):
            cfg["warmup_steps"] = min(1000, int(updates * .1))
        cfg["_comment"] = ("Owner train/dev budget generated from configs/training_data_selection.json and "
                           "docs/reviews/budget_2026-11-20. Seeded approximate passes are a compute allocation, not "
                           "convergence evidence. Full training waits for corrected CSS10 audit and used-train audio "
                           "delivery/verification. Final test remains locked until its separate committed protocol.")
        write(cfg_path, cfg)
        # Measurement recipes keep the same selected cohort/model, and differ
        # only in their explicitly smaller step/warmup/evaluation budgets.
        if name in ("ctc", "hubert"):
            measure_path = cfg_path.with_name(cfg_path.stem + "_measure.json")
            measure = read(measure_path)
            for key in ("train_manifest", "train_references", "batch_size", "gradient_accumulation_steps"):
                measure[key] = cfg[key]
            write(measure_path, measure)
        evals = int(math.ceil(updates / cadence))
        train_seconds = updates * accumulation * measurement["mean_update_seconds"]
        eval_seconds = evals * dev_hours * measurement["evaluation_seconds_per_audio_hour"]
        training.append({"model": name, "condition": cfg.get("condition", "hparl_only"), "config": path,
                         "micro_batch_size": batch, "gradient_accumulation_steps": accumulation,
                         "effective_batch_size": batch * accumulation, "optimizer_updates": updates,
                         "approximate_pass_target": passes, "dev_evaluations": evals,
                         "measured_micro_update_seconds": measurement["mean_update_seconds"],
                         "estimated_train_hours": train_seconds / 3600,
                         "estimated_dev_loss_hours": eval_seconds / 3600,
                         "measurement": cases[name].name,
                         "limits": "pilot extrapolation; variable-shape overhead, caching and data loading can differ"})
    gmm_path = ROOT / "configs/kemal/gmm_hmm.json"
    gmm_cfg = read(gmm_path)
    gmm_cfg.update(train_manifest=artifact["clean_manifest"]["path"], train_references=artifact["clean_references"]["path"])
    write(gmm_path, gmm_cfg)
    gmm_measure = read(ROOT / "configs/kemal/gmm_hmm_measure.json")
    gmm_measure.update(train_manifest=gmm_cfg["train_manifest"], train_references=gmm_cfg["train_references"])
    write(ROOT / "configs/kemal/gmm_hmm_measure.json", gmm_measure)
    gmm = read(inputs / "representative/gmm_random_b1_n1.json")["gmm_train"]
    pilot_audio_hours = read(inputs / "pilot/acquisition.json")["cohort"]["hours"]
    full_component_iteration = max(row.get("seconds", 0) for row in gmm["iterations"])
    gmm_hours = (full_component_iteration * gmm_cfg["training"]["num_iterations"] + gmm["feature_seconds"]) / pilot_audio_hours * selection["storage"]["hours"] / 3600
    inference = []
    for backend in ("whisper", "qwen", "wav2vec", "parakeet", "canary"):
        path = inputs / f"infer_{backend}_cached/run.json"
        measurement = read(path)
        if measurement["status"] != "complete" or measurement["completed_records"] != 32:
            raise ValueError(f"incomplete inference budget measurement for {backend}")
        ratio = measurement["inference_seconds_including_output_io"] / measurement["total_audio_seconds"]
        inference.append({"backend": backend, "batch_size": measurement.get("batch_size", 1),
                          "train_clips": 32, "audio_seconds": measurement["total_audio_seconds"],
                          "measured_seconds": measurement["inference_seconds_including_output_io"],
                          "estimated_frozen_test_hours": ratio * test_hours,
                          "not_test_inference": True})
    qwen_batch = inputs / "infer_qwen_b4/run.json"
    qwen_consistency = None
    if qwen_batch.exists():
        batch = read(qwen_batch)
        if batch.get("status") == "complete" and batch.get("completed_records") == 32:
            def prediction_rows(path):
                return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            single_rows = prediction_rows(inputs / "infer_qwen_cached/predictions.jsonl")
            batch_rows = prediction_rows(inputs / "infer_qwen_b4/predictions.jsonl")
            if [row["id"] for row in single_rows] != [row["id"] for row in batch_rows]:
                raise ValueError("Qwen batching changed cohort IDs/order")
            qwen_consistency = {"same_ordered_training_ids": True, "records": len(batch_rows),
                                "different_raw_hypotheses": sum(a["text"] != b["text"] for a, b in zip(single_rows, batch_rows)),
                                "empty_batch_hypotheses": sum(not row["text"] for row in batch_rows),
                                "reference_text_used": False, "accuracy_comparison": False}
            inference.append({"backend": "qwen", "batch_size": 4, "train_clips": 32,
                              "audio_seconds": batch["total_audio_seconds"],
                              "measured_seconds": batch["inference_seconds_including_output_io"],
                              "estimated_frozen_test_hours": batch["inference_seconds_including_output_io"] / batch["total_audio_seconds"] * test_hours,
                              "not_test_inference": True})
    estimated_train_dev = sum(row["estimated_train_hours"] + row["estimated_dev_loss_hours"] for row in training) + gmm_hours
    plan = {"schema": "p3-owner-training-budget-v1", "status": "measured_budget_data_corrections_pending",
            "selection_sha256": hashlib.sha256((ROOT / "configs/training_data_selection.json").read_bytes()).hexdigest(),
            "runtime_environment": {"HSA_ENABLE_DXG_DETECTION": "1", "MIOPEN_FIND_MODE": "FAST"},
            "runtime_note": "FAST measured on new random HuBERT input shapes; old default-mode random-shape timings include substantial startup/search cost. No model architecture or loss changed.",
            "seeds": [42], "training_runs": training, "gmm_training_estimated_hours": gmm_hours,
            "all_train_and_dev_loss_estimated_hours": estimated_train_dev,
            "pretrained_inference_measurements": inference,
            "qwen_batch_consistency": qwen_consistency,
            "schedule": {"train_dev_reserve_hours": 22, "final_inference_reserve_hours": 12,
                         "integration_retry_report_reserve_hours": 4,
                         "note": "planning allocations, not professor's exact deadline; exact Saturday evening cutoff remains unknown. Re-estimate after first real run; use dev only for revisions."},
            "start_conditions": ["all selected RAW and cleaned-used audio retrievable from repository", "correct float-aware CSS10 audit/clean reports regenerated by Samed", "used-cohort verify and unchanged frozen-test audio verify pass"],
            "final_test_authorized": False, "final_test_used_for_measurements": False,
            "storage_authorized_gib": 10, "paid_storage_authorized": False}
    write(ROOT / "configs/training_budget.json", plan)
    source = read(ROOT / "configs/data_source.json")
    source.update(status="delivered_test_dev_and_css10_bytes_verified_selected_train_pending_css10_audit_fix",
                  test_data_frozen=True, test_freeze_manifest="data/manifests/test.freeze.json",
                  test_audio_independently_verified=True, full_corpus_locally_verified=False,
                  selected_training_plan="configs/training_data_selection.json",
                  selected_clean_hparl_hours=selection["storage"]["hours"],
                  selected_mixed_training_hours=selection["cohorts"]["mixed_selected_train"]["hours"],
                  selected_train_audio_delivered=False, full_training_completed=False)
    source["secondary_training_corpus"].update(locally_acquired=True, locally_audited=False,
                                               original_audio_hashes_independently_verified=True,
                                               actual_audio_encoding="IEEE float32, RIFF format tag 3; all 1844 clips",
                                               audit_blocker="existing signal_stats reads width=4 as integer PCM; regenerate with format-aware decoding")
    write(ROOT / "configs/data_source.json", source)
    delivery = read(ROOT / "configs/data_delivery.json")
    delivery["groups"]["hparl_train_wav"].update(status="pending", detail=
        "Selected 18 whole official train sessions: 23615 raw WAVs (including 13 removed examples), "
        "23602 clean HParl clips, 30.883408 h; exact list in configs/training_data_selection.json. "
        "All audio is estimated at 6.714255 GiB; 2.6 GiB reserved for checkpoints/headroom inside the authorized 10 GiB cap.")
    write(ROOT / "configs/data_delivery.json", delivery)
    copies = {"acquisition.json": inputs / "pilot/acquisition.json",
              "css10_encoding_diagnosis.json": inputs / "css10_encoding_diagnosis.json",
              "verify_freeze_audio.json": inputs / "verify_freeze_audio.json",
              "verify_hparl_delivered_audio.json": inputs / "verify_hparl_delivered_audio.json",
              "source_metadata_verification.json": inputs / "source_metadata_verification.json",
              "training_candidates.json": inputs / "training_candidates.json",
              "whisper_random_b4_oom.json": inputs / "whisper_random_b4.json",
              "whisper_css10_b2_fixed.json": inputs / "whisper_css10_b2_fixed.json",
              "hubert_random_b8_fast.json": inputs / "hubert_random_b8_fast.json",
              "gmm_decode_train4.json": inputs / "gmm_decode_train4/run.json"}
    copies.update({path.name: path for path in (inputs / "representative").glob("*.json")
                   if path.name not in ("suite.json", "whisper_css10_b2_n3.json")})
    copies.update({f"infer_{name}.json": inputs / f"infer_{name}_cached/run.json"
                   for name in ("whisper", "qwen", "wav2vec", "parakeet", "canary")})
    if qwen_batch.exists() and read(qwen_batch).get("status") == "complete":
        copies["infer_qwen_b4.json"] = qwen_batch
    for name in ("ctc_random_b32_fast.json", "transducer_long_b8_fixed.json"):
        if (inputs / name).exists():
            copies[name] = inputs / name
    for name, path in copies.items():
        write(evidence / name, read(path))
    # Input identities and references for timing are committed. They are TRAIN
    # diagnostic inputs; acquisition code retrieves identical original bytes.
    for name in ("train.jsonl", "references.jsonl"):
        (evidence / ("pilot_" + name)).write_bytes((inputs / "pilot" / name).read_bytes())
    for name in ("inference_train.jsonl", "inference_train4.jsonl"):
        (evidence / name).write_bytes((inputs / name).read_bytes())
    lines = ["# Owner measured training/data budget", "", "Generated by `scripts/finalize_training_plan.py`; timings use TRAIN only.", "",
             "Full training waits for Samed's corrected CSS10 audit and selected RAW audio delivery. Final test is still locked.", "",
             "| Run | Microbatch × accumulation | Updates | Train hours estimate | Dev-loss hours estimate |",
             "|---|---:|---:|---:|---:|"]
    for row in training:
        lines.append(f"| {row['model']} / {row['condition']} | {row['micro_batch_size']} × {row['gradient_accumulation_steps']} | {row['optimizer_updates']} | {row['estimated_train_hours']:.2f} | {row['estimated_dev_loss_hours']:.2f} |")
    lines += ["", f"GMM-HMM train estimate: {gmm_hours:.2f} h; all training + dev-loss estimate: {estimated_train_dev:.2f} h.",
              "These are pilot extrapolations, not measured full runs, convergence claims or benchmark scores.", "",
              "| Standard inference | Batch | Measured TRAIN seconds | Frozen-test hours estimate |", "|---|---:|---:|---:|"]
    for row in inference:
        lines.append(f"| {row['backend']} | {row['batch_size']} | {row['measured_seconds']:.2f} | {row['estimated_frozen_test_hours']:.2f} |")
    lines += ["", "The inference estimates exclude checkpoint loading and do not include our trained-model prediction runs.",
              "Reserve extra time for all trained models, dev selection, failures and reporting. No test output was generated.", "",
              "Whole-session storage decision: see `configs/training_data_selection.json` and `training_candidates.json`.",
              "Approximate-pass targets allocate compute: 1 matched mixed-pool pass for each Whisper condition; 2 attention, 1 RNN-T, 3 plain CTC, 2 HuBERT. From-scratch models can remain undertrained; assess dev curves and report actual completed updates.",
              "", "Raw JSON records are retained here with seeds, batch IDs, exact package/device identities, source hashes and failures (Whisper batch-4 OOM). Historic absolute paths identify the measurement checkout; rerun from the repository root with the documented scripts.",
              "", "The CSS10 diagnosis verifies original audio hashes/headers; its old signal-statistics audit remains invalid until Samed regenerates it with IEEE-float decoding."]
    (evidence / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"training_estimate_hours": estimated_train_dev, "selection_hours": selection["storage"]["hours"],
                      "configs_written": [row["config"] for row in training], "evidence_files": len(copies)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
