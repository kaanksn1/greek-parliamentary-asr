# Samed Ortakaya — 210901024 — SmtOrtakaya

Role: data owner for Project 3 (HParl + CSS10 Greek acquisition, early test
freeze, train/dev audit and cleaning, training-condition manifests).

## Actual work so far (branch `feature/p3-data`)

- Implemented the data pipeline `greek_asr/data/` with the CLI
  `python -m greek_asr.data`:
  - Resumable HParl download with exact `Content-Range` checks and archive
    SHA-256. CRC-checked extraction that rejects zip-slip paths, links and
    differing existing bytes.
  - Kaldi `text`/`utt2spk` parsing with split-layout detection
    (`validation` → `dev`). Raw contract manifests and verbatim references for
    all official splits. Every source problem (missing/unreadable audio,
    unlabelled audio, duplicate IDs) is listed in an inventory, not dropped
    silently.
  - Early test freeze (`test.freeze.json`) with per-record audio, reference
    and source-line hashes. The freeze cannot be rewritten. `verify-freeze`
    recomputes everything, and cleaning re-verifies it before and after.
  - CSS10 Greek acquisition and a training-only raw manifest. The producer's
    original and normalized text are kept unchanged.
  - Train/dev/CSS10 audit:
    - Signal level, clipping, silence and duration
    - Characters per second
    - Annotation tags and character inventory
    - Byte- and PCM-identical duplicates
    - Train/dev copies of test audio
    - Speaker/session/text overlaps between splits
  - Versioned cleaning policy (`configs/data_cleaning.json`) with a per-record
    machine-readable log, a review queue for flags that need listening and an
    attributable human-review decision file. Outputs are the HParl-only and
    HParl + CSS10 training manifests plus `conditions.json`.
  - Report generator (`docs/DATA_REPORT_TR.md`) that writes all numbers,
    examples, slide draft and speaker note from the recorded outputs.
  - Delivery helpers: `publish-plan` for real sizes against an LFS quota, and
    deterministic release shards (`pack`/`unpack`) with a committed hash index.
- Wrote `docs/DATA_PIPELINE_TR.md`: run order, rule justifications, freeze
  guarantees and delivery options.
- Added 32 data tests (`tests/test_data_pipeline.py`) on a synthetic corpus
  with known problems. They cover the handoff's acceptance list: lineage,
  visible errors, rule outcomes, reconciliation, CSS10 train-only, exact test
  cohort, test-immutability failures, determinism, HParl-only vs mixed,
  zip-slip, portable paths, runner compatibility, resumable download and
  shard integrity, literal overlap, cohort counts and source-metadata bytes. The full suite (76 tests) passes on Python 3.12 with
  numpy 1.26.4 and on Python 3.13.

## Real-data runs (18 November 2026)

- Downloaded the official HParl 1.0.0 archive (11,887,118,613 bytes, SHA-256
  recorded) and CSS10 Greek Kaggle v1 (CC0 metadata recorded). Extracted and
  measured every file: 92,133 HParl WAVs. Local durations match the producer's
  split durations; there were no source problems.
- Froze the original test (8,679 records, 11:12:28) before any cleaning and
  committed it separately (`d28c97f`). The freeze verified `ok` before and
  after cleaning.
- Ran the audit and cleaning (`bdaf2f8`): 17 byte-identical train duplicates
  and annotation-tag removal in train targets. CSS10 was harmonised to the
  HParl transcription style.
- From the real audit, found train segments whose transcripts are far longer
  than their audio (hundreds to thousands of characters in a few seconds).
  Added the `cps_implausible` rule (policy v2) and re-ran the cleaning. The
  current counts are generated in `docs/DATA_REPORT_TR.md`.
- Reported for the scoring owner that `<spoken_noise>` appears in a large
  share of dev/test references, so it must be handled the same way in
  references and hypotheses.

## Review follow-up (PR #2 review at `4e4a80d`)

- Merged the updated `main` into `feature/p3-data` with a merge commit; no
  history rewrite. The frozen test files were not regenerated or changed.
- Added literal (untransformed) test-sentence overlap per training condition
  (matched test examples and distinct texts, with file hashes). The earlier
  normalized distinct-text overlap is kept under its own name. My code
  reproduces the review's independent values on `4e4a80d` (raw 14/14,
  clean 6/6, mixed 6/6).
- Added the speaker/utterance/hour/session summary per split and per training
  condition (`split-summary`, `data/clean/cohort_summary.json`). CSS10's
  missing speaker/recording IDs are counted as unknown with the producer's
  single-reader/single-book note. README and the report now state that the
  official session split is not speaker-disjoint.
- Changed the speech-rate rule to an explicitly heuristic policy v3: only
  records above 60 characters per second are dropped (train/CSS10). The
  40–60 boundary records are kept and queued for review. No record has been
  listened to, and none is described as a verified wrong label.
- Committed the original Kaldi metadata of all splits and the CSS10
  transcript as plain Git with `-text`, so no line-ending conversion happens.
  Added `verify-source-metadata` to byte-check them on a clean clone.
- Added `delivery-plan` (real sizes from the archive listings) and the
  decision record `configs/data_delivery.json`.

## Audio delivery (19 November 2026)

- After the repository owner approved it, added the test, dev and CSS10 audio
  (the copy the manifests reference) through Git LFS, with
  `verify --only` for checking a partial delivery.

## CSS10 IEEE float32 correction and selected-data chain (20 November 2026)

The owner's real-audio check found two problems in my data code:

- All 1,844 CSS10 WAVs are IEEE float32 (RIFF format tag 3), but my audit read
  every 4-byte sample as integer PCM32. That made the CSS10 RMS/dBFS values
  and their signal flags wrong.
- My `wave`-based probe could not open these files on Python 3.12.

Fixes:

- **Reading:** the data probe now uses the shared RIFF reader
  (`greek_asr/audio.py`), and signal statistics decode by format tag.
  - PCM16/PCM32 use integer full-scale scaling.
  - Float32 values are used as stored, with no gain or normalization.
  - NaN/Inf, truncated WAVs and LFS pointers fail visibly.
  - Original WAVs are never converted.
- **Selected scope:** added `--selection configs/training_data_selection.json`
  to `audit`, `clean` and `verify`. The chain now needs only the delivered audio
  (selected raw train, dev, test, CSS10), not the unused 99-hour train.
- **Locked files protected:** `clean --selection` only compares the locked
  clean HParl selection and dev byte for byte; it never rewrites them or the
  frozen test. A changed CSS10 or mixture would go to new versioned files with
  old → new hashes. The acceptance record is `configs/css10_mixture_lineage.json`.
- **Real run** (Python 3.12.10, NumPy 1.26.4, Windows,
  `scripts/run_selected_data_checks.py`, logs in
  `docs/reviews/css10_reaudit_2026-11-20/`):
  - Corrected CSS10 RMS median: −27.67 dBFS (the old misread value was −6.105).
  - One new review flag: `long_edge_silence` on clip 1738.
  - No CSS10 drop; 1,844/1,844 kept (4.137269 h).
  - Clean CSS10 and the 25,446-record / 35.020678 h mixture are byte-identical
    to the previous files.
  - The locked 23,602-record HParl selection and dev were reproduced byte for
    byte. The frozen test verified `ok` with audio before and after cleaning.
  - `verify --selection` opened all 32,572 used WAVs.
  - 229 tests ran with no failures. 99 model tests were skipped because torch
    is not in the data environment; no data test was skipped.
- **Tests added:**
  - float32 vs PCM32 at the same width
  - known-amplitude statistics
  - Python 3.12 CSS10 reading
  - NaN/Inf, truncated and pointer failures
  - the selected chain without unselected audio, determinism, versioning and
    reported HParl differences
  - locked repository hashes

## Not yet done

- **Selected train audio:** the 23,615 selected raw train WAVs are uploaded by
  the repository owner; after that, `verify --selection` should be run on a
  clean clone.
- **Listening review and data slides:** no record has been listened to yet
  (zero human review decisions).

No bonus or model result is claimed.

Coding-agent assistance was used for implementation, tests and checks.
