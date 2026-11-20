# Data acquisition and delivery

Current 20 November delivery: all 23,615 original RAW train WAVs from the locked
18 HParl sessions are locally acquired, hash-verified and committed through
Git LFS. Remote upload state is in `../configs/hparl_delivery_status.json`.
Original test/dev/CSS10 audio is already in the data branch through Git LFS.
An isolated rerun of Samed's unchanged HParl pipeline reproduced every locked
clean train/dev manifest/reference hash; its committed proof is
`../configs/hparl_training_acceptance.json`. HParl-only training has started.
CSS10's original IEEE float32 audio requires Samed's corrected format-aware
data audit before mixed-corpus training. See `../docs/PRIMARY_TRAINING_TR.md`.

The first three original HParl training WAVs are stored under `raw/pilot/`,
together with their original references and provenance. They total 6.61 seconds
and are only a pipeline/runtime sample, not the final training corpus.
The source is the original release, not a third-party mirror's random split.

`raw/pilot/audio/` retains unmodified WAV bytes. `manifest.jsonl` records source
utterance/speaker IDs, source member, audio hash and header properties.
`selected_source_lines.json` preserves the selected original text/utt2spk lines.
`source_summary.json` records the original training metadata hashes, selection
seed and bounded archive access. No development/test audio or references were read.

Run `python scripts/audit_train_pilot.py --output outputs/train_pilot_audit.json`.
One reference contains `<spoken_noise>`; the audit flags it without rewriting
the source. Its treatment in any final reference/scoring policy is unresolved.

The full-corpus pipeline is `python -m greek_asr.data` (see
`../docs/DATA_PIPELINE_TR.md`). After a real run the layout is:

- `raw/hparl/release/`: CRC-checked extraction of the official ZIP, original
  paths below the archive's top-level folder; `ACQUISITION.json`,
  `EXTRACTION.json`, `LAYOUT.json`, `archive_members.tsv` record provenance.
- `raw/css10_el/release/`: extracted Kaggle CSS10 Greek archive with the same records.
- `manifests/raw/`: uncleaned contract manifests, verbatim references and inventories.
- `manifests/test.jsonl`, `references/test.jsonl`, `test.freeze.json`: frozen test.
- `manifests/{train,dev,css10_el_train,train_hparl_css10}.jsonl` and references:
  clean outputs; `conditions.json` describes the two training conditions.
- `audit/`: per-record measurements and flags; `clean/`: cleaning log, review
  queue and summary.

The original planning layout follows.

Expected layout for the full selected corpus after acquisition:

- `raw/`: acquired source release, original audio/transcripts, license and a
  manifest of source URLs, release identity, bytes and SHA-256 hashes.
- `clean/`: the selected usable audio/transcripts and per-item cleaning log.
- `manifests/`: IDs, original split, source session/recording, speaker where
  available, audio file/hash, duration and reference transcription.

Raw and cleaned data must be included in the eventual public submission as the
professor requires. They are not excluded by this project's `.gitignore`.
A download script or external link by itself does not satisfy that requirement.
The selected RAW/clean cohort and authorized storage calculation are recorded
in `../configs/training_data_selection.json`; full-source candidate inventories
do not imply that all 99 hours of HParl train are delivered or used.

Source-reported official HParl split durations (paper Table II):
train `99:31:41`, development `09:03:33`, test `11:12:28`; total `119:47:42`.
The full-source measurements are in `../docs/DATA_REPORT_TR.md`; active selected
training hours are recorded separately in `../configs/training_data_selection.json`.

The selected second training source is CSS10 Greek v1. Its producer reports
`04:08:14`, from one audiobook reader. All original CSS10 bytes are acquired and
independently hash/header-verified; the old integer-interpreted signal audit
must be corrected. Combined source-reported availability is `123:55:56`, with a
maximum pre-cleaning training candidate pool of `103:39:55` (HParl train plus
CSS10); HParl dev/test are not training hours. These are planning sums, not
our used/cleaned duration. See `../docs/SECONDARY_SOURCE_TR.md`.
CSS10 enters training only; the original HParl test cohort remains unchanged.

Inspect the training side first. Detect actual unreadable/empty/clipped/duplicate
audio and alignment problems; do not invent issues to claim cleaning work.
Preserve the original human transcript and record any justified correction.
Do not promote a model's `rec_text` or a mirror's transcription field to verified
gold without provenance. No official-test model selection or exploratory scoring.
