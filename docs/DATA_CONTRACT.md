# Shared data and prediction contract

Version 1, 18 November 2026. This is the minimum shared interface for data,
models and evaluation. Samed implements the full data pipeline; the owner
provides the common inference runner. Nothing here claims completed full-data
acquisition, cleaning or test freezing.

## Audio manifests

UTF-8 JSONL, one complete record per utterance:
`data/manifests/train.jsonl`, `dev.jsonl`, `test.jsonl`.
Paths are relative to the project repository root, never a member's machine.

```json
{"id":"hparl:original-utterance-id","source":"hparl","original_id":"original-utterance-id","source_split":"train","split":"train","audio_path":"data/raw/hparl/train/audio.wav","audio_sha256":"64-lowercase-hex-characters","sampling_rate":16000,"channels":1,"num_frames":40160,"duration_seconds":2.51,"source_recording_id":null,"speaker_id":null}
```

- Required: `id`, `source`, `original_id`, `source_split`, `split`, `audio_path`,
  `audio_sha256`, `sampling_rate`, `channels`, `num_frames`, `duration_seconds`.
  Integers/rate/frames and finite duration must agree with the acquired audio.
- Sources: `hparl` or `css10_el`; ID is exactly `source:original_id`. It must be
  unique. Source namespacing avoids collisions between the corpora.
- Working splits: `train`, `dev`, `test`. Retain original HParl assignment.
  CSS10 has `source_split: "unsplit"` unless its release actually supplies a
  documented split; its working split is **train only**.
- Preserve actual recording/session and speaker IDs when supplied. Missing
  metadata is `null`, not an inferred/fabricated identity. Session derivation
  from an ID must be documented and checked against the producer structure.
- Raw source bytes/transcript lines remain separately available. Clean train/dev
  audio paths refer to the actual deliverable bytes and carry their hashes.
  A cleaning lineage report maps each retained/removed/modified item to raw data.
- Test audio paths refer to unchanged original bytes; preserve every official
  test member/reference. Do not resegment/resample/edit test files. Input feature
  extraction during normal model inference must not overwrite source recordings.

## References and early test freeze

Gold references are separate UTF-8 JSONL at
`data/manifests/references/{train,dev,test}.jsonl`:

```json
{"id":"hparl:original-utterance-id","reference":"original source transcription"}
```

The common runner reads audio inputs only. Reference text, any annotation and
model-generated transcript must never be passed as a prompt or model feature.
Canonical audio manifests should not contain gold text. The runner whitelists
model inputs even if an accidental extra field is present.

Before train/dev cleaning/mixing, Samed produces `data/manifests/test.freeze.json`:
source/release identity, original test membership/IDs, audio and reference hashes,
manifest/reference file hashes, actual recording/session mapping where available,
record count and creation date. The freeze and corresponding raw test data are
committed through Samed's PR and verified by Serdar. Ozan later checks the same
cohort. This early data freeze is still pending.

The late final model/scoring protocol is separate. Before final inference it
must be committed, set `selection_fixed: true`, specify `scoring_policy` and
the exact `test_manifest_sha256`. Model/checkpoint, settings and selection
decisions must also be recorded. The runner requires `--allow-test` and
`--test-protocol docs/FINAL_PROTOCOL.json`; its minimal check is not a substitute
for the scientific protocol or the early data freeze. No test protocol exists yet.

## Model outputs and central runs

Exactly one UTF-8 JSONL prediction per input ID, including actual empty output:

```json
{"id":"hparl:original-utterance-id","text":"model transcription"}
```

No failed/missing record may be dropped. A backend failure leaves a clearly
failed run with `predictions.partial.jsonl`, never a successful benchmark file.
Successful output has `predictions.jsonl` and `run.json` containing source/code/
checkpoint hashes, settings, environment and timing. Output directories must
be new so an earlier experiment is not overwritten.

```bash
python -m greek_asr.infer --manifest data/manifests/dev.jsonl --backend whisper --split dev --output outputs/whisper_dev
```

Owner backends: `whisper`, `canary`, `parakeet`, `qwen`. Other model owners
implement their own training/decoding and produce the same output interface.
Full training and inference are central on Serdar's PC; small local unit tests
and data work remain with their respective owners.

The existing scoring CLI uses `--split validation` for the `dev` manifest;
this is a CLI label only. Ozan owns the final evaluation/scoring policy.
