# Audit outputs

**Current audit:** `selected_train_hparl_30.5h_seed42/` (schema `p3-data-audit-v2`, 20 November 2026,
Python 3.12.10, NumPy 1.26.4). It was produced by
`python -m greek_asr.data audit --selection configs/training_data_selection.json` on the real audio:
the 18 selected raw train sessions, the unchanged dev, the frozen-test hashes and CSS10.
WAVs are decoded by their RIFF format tag. IEEE float32 CSS10 samples are used as stored.

**Historical, partly invalid:** the files directly in this folder (`audit_summary.json`,
`items_*.jsonl.gz`, schema `p3-data-audit-v1`, 18–19 November) audited the full ~99-hour source train
inventory. They decoded every 4-byte sample as integer PCM32. For CSS10, which is IEEE float32, the
RMS/dBFS/peak values and every signal flag derived from them are therefore **wrong**. HParl (PCM16) values
and the text, duplicate and overlap results are unaffected. These files are kept only because the locked
selection (`data/manifests/train.jsonl` → `configs/training_data_selection.json`) was derived from that run.
Old → new CSS10 comparison: `configs/css10_mixture_lineage.json` and section 4a of `docs/DATA_REPORT_TR.md`.
