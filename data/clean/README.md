# Cleaning outputs

**Current:** `selected_train_hparl_30.5h_seed42/` was produced by
`python -m greek_asr.data clean --selection configs/training_data_selection.json`. It holds the cleaning
log, review queue, summary and cohort summary for the selected train, dev and CSS10. It reproduced the
locked clean HParl selection and dev byte for byte, and wrote CSS10 and the mixture, which are unchanged.
Acceptance record: `configs/css10_mixture_lineage.json`.

**Historical:** the files directly in this folder come from the 18–19 November full-source-inventory
run, which produced `data/manifests/train.jsonl`, the source of the locked selection. Their CSS10 review
flags depend on the invalid integer-PCM32 signal statistics described in `data/audit/README.md`. Of the
CSS10 drop rules, only `digital_silence` reads a signal statistic, and an all-zero float32 clip is also
all-zero when misread as int32. The 20 November float-aware recomputation produced the identical CSS10 manifest.
