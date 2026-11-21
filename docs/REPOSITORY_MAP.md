# Script and document map

Local GitHub/monitor helpers under ignored `outputs/` are not ML methods.

## Current workflow scripts

| Script | Purpose |
|---|---|
| [`acquire_selected_hparl.py`](../scripts/acquire_selected_hparl.py) | Owner acquisition of the already-locked RAW training cohort, with no cleaning. |
| [`benchmark_worker.py`](../scripts/benchmark_worker.py) | Execute exactly one planned ASR system; references never enter test decoders. |
| [`cache_benchmark_models.py`](../scripts/cache_benchmark_models.py) | Fetch the exact upstream checkpoints before the deliberately offline test queue. |
| [`calibrate_batch_resources.py`](../scripts/calibrate_batch_resources.py) | Compare bounded batch 4/8 on DEV, selecting by time/memory, never test WER. |
| [`finalize_training_plan.py`](../scripts/finalize_training_plan.py) | Generate owner run settings/report from actual measurements and selected data. |
| [`freeze_benchmark_protocol.py`](../scripts/freeze_benchmark_protocol.py) | Freeze all systems and scoring only after complete DEV calibration acceptance. |
| [`keep_training_awake.ps1`](../scripts/keep_training_awake.ps1) | Temporary process-scoped sleep prevention; no permanent power-plan change. |
| [`prepare_benchmark.py`](../scripts/prepare_benchmark.py) | Export exact inference weights and select a seeded DEV-only calibration cohort. |
| [`report_benchmark_data.py`](../scripts/report_benchmark_data.py) | Report actual selected cohorts and literal/final-policy overlap; never edit data. |
| [`run_benchmark_queue.py`](../scripts/run_benchmark_queue.py) | Run all planned ASR systems, score complete cohorts, and produce reproducible reports. |
| [`run_primary_training_queue.py`](../scripts/run_primary_training_queue.py) | Run independent CPU GMM and sequential GPU HParl jobs; never run final test. |
| [`run_selected_data_checks.py`](../scripts/run_selected_data_checks.py) | Run the selected-scope data chain on real audio and record every command's output. |
| [`run_training_job.py`](../scripts/run_training_job.py) | Owner's observable full-run wrapper; submitted model algorithms stay unchanged. |
| [`select_training_delivery.py`](../scripts/select_training_delivery.py) | Owner's session-cohort/storage decision; reuses Samed's subset algorithm. |
| [`setup_nemo_rocm.sh`](../scripts/setup_nemo_rocm.sh) | Create the isolated pinned NeMo runtime. |
| [`setup_transformers_rocm.sh`](../scripts/setup_transformers_rocm.sh) | Create the isolated pinned Transformers/ROCm runtime. |
| [`summarize_benchmark.py`](../scripts/summarize_benchmark.py) | Generate tables, figures, paired comparison and the presentation brief from actual outputs. |
| [`summarize_training.py`](../scripts/summarize_training.py) | Generate actual training-budget/selection tables from original run records. |
| [`validate_final_benchmark.py`](../scripts/validate_final_benchmark.py) | Re-score saved predictions and check final cohort/provenance; never run ASR. |
| [`verify_css10_training.py`](../scripts/verify_css10_training.py) | Independently reproduce the submitted CSS10 pipeline in isolated staging. |
| [`verify_hparl_training.py`](../scripts/verify_hparl_training.py) | Reproduce Samed's unchanged cleaning on delivered selected RAW HParl audio. |

## Historical train-pilot, measurement and code-review utilities

These reproduce dated diagnostics or resource measurements using train fixtures, not new test tuning. They are not new model contributions.
`verify_scoring.py` checks the older draft scorer oracle; the final scorer is `greek_asr/evaluation.py`.

| Script | Purpose |
|---|---|
| [`audit_train_pilot.py`](../scripts/audit_train_pilot.py) | Audit the downloaded train-only pilot and write a source-hashed report. |
| [`fetch_train_pilot.py`](../scripts/fetch_train_pilot.py) | Fetch a tiny deterministic HParl training pilot using bounded public ZIP ranges. |
| [`inspect_css10_delivery.py`](../scripts/inspect_css10_delivery.py) | Independent original-byte/hash/encoding diagnosis for the data-owner handoff. |
| [`measure_train_budget.py`](../scripts/measure_train_budget.py) | Measure unchanged submitted training kernels on real TRAIN audio, not test. |
| [`pilot_sequence_models.py`](../scripts/pilot_sequence_models.py) | GPU pilot for Adnan's three sequence models: real forward/backward/optimizer |
| [`prepare_budget_pilot.py`](../scripts/prepare_budget_pilot.py) | Acquire a reproducible TRAIN-only timing cohort; never inspect dev/test. |
| [`prepare_owner_pilot.py`](../scripts/prepare_owner_pilot.py) | Export the already-audited three training clips to the shared inference schema. |
| [`probe_kemal_submission.py`](../scripts/probe_kemal_submission.py) | Owner review of Kemal's submitted baselines on three original TRAIN clips. |
| [`probe_sequence_entrypoints.py`](../scripts/probe_sequence_entrypoints.py) | Exercise submitted train/predict CLIs on train-pilot code-path fixtures. |
| [`probe_sequence_submission.py`](../scripts/probe_sequence_submission.py) | Owner's independent, train-pilot-only review of submitted sequence code. |
| [`review_data_submission.py`](../scripts/review_data_submission.py) | Read-only integration review of a teammate's committed ASR data delivery. |
| [`smoke_whisper_train.py`](../scripts/smoke_whisper_train.py) | Run pinned multilingual Whisper on the three training-pilot WAVs, not a benchmark. |
| [`verify_rnnt_submission.py`](../scripts/verify_rnnt_submission.py) | Owner's independent CPU-only RNN-T value/gradient and padding review. |
| [`verify_scoring.py`](../scripts/verify_scoring.py) | Compare the shared scorer to JiWER on deterministic, non-benchmark fixtures. |

## Current documents

- [Final benchmark](FINAL_BENCHMARK.md): frozen evaluation and reproduction commands.
- [Baseline inventory](BASELINES.md): each required family and its provenance.
- [Selected data report](DATA_REPORT_TR.md): generated actual cohorts, cleaning and overlaps.
- [Data pipeline](DATA_PIPELINE_TR.md): raw-to-clean commands; scope is stated at the top.
- [Protocol erratum](FINAL_PROTOCOL_ERRATA.md): descriptive GMM LM typo and original-byte archive restoration.
- [Teacher audit](TEACHER_FINAL_AUDIT_TR.md): evidence and open items.
- [Serhan handoff](HANDOFF_SERHAN_FINAL_TR.md): presentation/member-owned delivery.

## Historical and ownership documents

`PROTOCOL_DRAFT.md`, `SECONDARY_SOURCE_TR.md`, `INITIAL_WORK_TR.md`, `TOPIC_DECISION_TR.md` and earlier handoffs preserve dated plans/source choices. They do not override the final protocol.
`docs/reviews/` retains dated train-pilot bug reviews, budget measurements, training and final verification evidence. Failed/OOM diagnostics are not final benchmark rows.
`TEAM.md` is the dated ownership plan. Individual contribution reports preserve their authors' dated development work; current central execution is recorded in generated results.
Teacher clarification documents record the supplied rules. Old text-classification-specific novelty examples are not the Project 3 ASR baseline list.

## Reproduction boundaries

Selected weights and frozen inference code reproduce the published benchmark. `validate_final_benchmark.py` accepts the original published execution commit `0ff3196c`; it verifies the archive, not a newly trained or newly executed experiment with another commit identifier.
A fresh ASR replay records its own execution commit. Never overwrite the original archive or tune the published test. New training/export runs use separate output folders; the original pre-export snapshot is described in `FINAL_BENCHMARK.md`.
