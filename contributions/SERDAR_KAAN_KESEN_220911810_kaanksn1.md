# Serdar Kaan Kesen — 220911810 — kaanksn1

## Actual owner work

- Set up this independent Greek parliamentary ASR project, source attribution,
  code LICENSE and the announced baseline inventory.
- Verified original HParl archive access and acquired/audited three unchanged
  original training clips (6.61 seconds). Preserved their source transcripts,
  lineage and hashes. This is a startup fixture, not Samed's full data work.
- Prepared the initial shared WER/CER implementation and its verification.
  Ozan owns its later evaluation development, comparisons and presentation.
- Verified the earlier Whisper-small GPU smoke on the training fixture; no
  training, dev/test inference or benchmark score was produced.
- Recorded the professor's reply, assigned the team tasks and prepared the
  shared data/prediction contract and detailed Samed handoff.
- Created the public `kaanksn1/greek-parliamentary-asr` repository and sent write
  collaborator invitations to all four real teammate accounts on 18 November.
- Implemented owner inference integration for standard Whisper, Canary,
  Parakeet and Qwen3-ASR using immutable producer checkpoint revisions, plus
  the shared audio-only CLI and its runtime records/failure handling.
- Created a separate WSL ROCm environment with its actual dependency lock and
  reusable setup script; ran Whisper and Qwen successfully on all three
  original training-pilot clips using the shared runner. Qwen's observed
  composite attention configuration is recorded explicitly.
- Created the separate NeMo ROCm environment and its actual dependency lock;
  ran Canary and Parakeet successfully on all three original training clips.
  Disabled Parakeet's NVIDIA CUDA Graph acceleration before construction on
  ROCm, retaining the standard greedy TDT algorithm, with a regression test.
- Verified the original 43 lightweight tests in a clean public clone without
  ASR packages; the NeMo regression brings the current suite to 44 tests.
  Checked original runtime-evidence hashes and recorded the successful runs.
- Reviewed Samed's PR #2 without implementing or modifying his data pipeline:
  ran its 73 tests in the declared NumPy environment, regenerated its report,
  checked freeze history and independently counted speaker/literal-text
  overlaps. Recorded remaining audio-delivery and documentation issues.
- Updated owner integration requirements to the two new professor emails,
  preserving the official-test freeze and teammates' work ownership.
- Rechecked Samed's corrected `e5aa488` submission (76 tests) and Adnan's
  `0437631` submission (93 tests), and tested their temporary merge (125 tests).
  Verified original metadata/report/freeze consistency without claiming full
  audio delivery. Added independent train-pilot GPU and synthetic PCM32/token
  probes, recorded actual failures and prepared an actionable correction prompt.
  The unchanged submitted Whisper pilot completed one GPU optimizer step;
  attention and RNN-T failed. No full training or official dev/test inference.
- Re-reviewed Adnan's `0c13ce1` fixes on 19 November: 114 tests passed; all
  three unchanged submitted pilots completed real GPU updates with observed
  finite gradients and changed parameters. Exercised attention/RNN-T actual
  train/checkpoint/predict CLIs on temporary train-pilot fixtures and recorded
  an independent exhaustive RNN-T value/gradient oracle. Kept remaining P2
  issues separate from resolved original blockers. No teammate source edits,
  official dev/test inference or full-data training were performed.
  Temporary integration with Samed's `4db1a4a` passed 146 tests; LFS audio
  blobs were not downloaded or independently rehashed in this code check.
- After the user's explicit 19 November assignment, completed the remaining
  sequence review fixes in owner-authored commits: nonfinite Whisper train/dev
  loss abort/reporting, prediction source-lock validation, safe ignored RNN-T
  padding, meaningful regression tests, and matching model/config documentation.
  All 119 local tests passed. Adnan's original model commits and personal
  contribution report remain intact; these limited fixes are attributed here.
  Rechecked the committed fixes on GPU (all three model pilots and the actual
  attention/RNN-T CLI paths), reran the independent RNN-T oracle including
  ignored padding, and passed 151 tests in temporary data-code integration.
  Versioned evidence is in docs/reviews/SEQUENCE_P2_RESOLVED_TR.md.

## Limits and pending dependent work

Samed submitted early test-freeze metadata and his data pipeline in PR #2;
full audio delivery and independent audio-byte verification are pending; the
reporting/source-metadata corrections were verified. His pipeline and cleaning
authorship remain his. Adnan's submitted implementation and the owner's final
review corrections have passed code/pilot checks. Kemal's model work, Ozan's final evaluation and all full
benchmark runs remain pending. No custom-data bonus, trained-model improvement or new
human annotation is claimed. Personal contribution percentages are not results.

Commits made under `kaanksn1` in this repository provide the version history.
Coding-agent assistance was used for implementation, research and checks;
that does not establish teammates' authorship or work they have not performed.
