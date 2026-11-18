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
- Added 29 data tests (`tests/test_data_pipeline.py`) on a synthetic corpus
  with known problems. They cover the handoff's acceptance list: lineage,
  visible errors, rule outcomes, reconciliation, CSS10 train-only, exact test
  cohort, test-immutability failures, determinism, HParl-only vs mixed,
  zip-slip, portable paths, runner compatibility, resumable download and
  shard integrity. The full suite (73 tests) passes on Python 3.12 with
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

## Not yet done

Listening review of the review queue, raw/clean audio delivery (Git LFS /
release shards, agreed with the repository owner) and the data slides are
pending. No used training hours beyond the generated report, bonus or model
result is claimed.

Coding-agent assistance was used for implementation, tests and checks.
