# Data acquisition and delivery

## Current state (20 November 2026)

Training uses only the owner's locked selection `../configs/training_data_selection.json`
(18 whole official HParl train sessions, seed 42), the unchanged official dev and the frozen
official test. CSS10 Greek is used in training only. The official split is session-based and
**not speaker-disjoint** (49 of 82 test speakers also occur in the selected train).

| Set | Records (raw → clean) | Hours (clean) | Manifest / references |
|---|---|---|---|
| Selected HParl train | 23,615 → 23,602 | 30.883408 | `manifests/subsets/train_hparl_30.5h_seed42.jsonl` (raw: `manifests/subsets/raw/…`) |
| CSS10 Greek (train only) | 1,844 → 1,844 | 4.137269 | `manifests/css10_el_train.jsonl` |
| Mixture HParl + CSS10 | 25,446 | 35.020678 | `manifests/subsets/train_hparl_30.5h_seed42_css10.jsonl` |
| Dev (unchanged) | 7,113 | 9.059189 | `manifests/dev.jsonl` |
| Test (frozen) | 8,679 | 11.207889 | `manifests/test.jsonl`, `manifests/test.freeze.json` |

Each manifest has references under the sibling `references/` folder. Clean manifests point at
the same original raw WAVs, so cleaning made no audio copies. CSS10 stays IEEE float32 / 22050 Hz
as distributed. The model input pipeline decodes it by format tag and resamples in memory.

- Current audit/cleaning: `audit/selected_train_hparl_30.5h_seed42/` and
  `clean/selected_train_hparl_30.5h_seed42/`. They are format-aware (schema v2) and were run on the
  real audio on 20 November.
- The files directly in `audit/` and `clean/` are the historical full-inventory run. Their CSS10
  signal statistics are invalid; see `audit/README.md`.
- Acceptance and old → new hashes: `../configs/css10_mixture_lineage.json`.
- The full ~99-hour train manifests (`manifests/raw/hparl_train.jsonl`, `manifests/train.jsonl`,
  `manifests/train_hparl_css10.jsonl`) are a source candidate inventory. Their audio is not
  delivered and they are not training data.

Audio in the repository (Git LFS): test, dev and the CSS10 copy the manifests use. The 23,615
selected raw train WAVs are uploaded by the repository owner. After `git lfs pull`, everything
used is checked with `python -m greek_asr.data verify --selection configs/training_data_selection.json`
and `verify-freeze`.

## History

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
Publication format/size must be settled before selecting the final subset.

Source-reported official HParl split durations (paper Table II):
train `99:31:41`, development `09:03:33`, test `11:12:28`; total `119:47:42`.
These figures must be independently checked against the acquired release.

The selected second training source is CSS10 Greek v1. Its producer reports
`04:08:14`, from one audiobook reader. (Planning note from before acquisition;
CSS10 was acquired, audited and cleaned on 18–20 November, see above.) Combined source-reported availability is `123:55:56`, with a
maximum pre-cleaning training candidate pool of `103:39:55` (HParl train plus
CSS10); HParl dev/test are not training hours. These are planning sums, not
our used/cleaned duration. See `../docs/SECONDARY_SOURCE_TR.md`.
CSS10 enters training only; the original HParl test cohort remains unchanged.

Inspect the training side first. Detect actual unreadable/empty/clipped/duplicate
audio and alignment problems; do not invent issues to claim cleaning work.
Preserve the original human transcript and record any justified correction.
Do not promote a model's `rec_text` or a mirror's transcription field to verified
gold without provenance. No official-test model selection or exploratory scoring.
