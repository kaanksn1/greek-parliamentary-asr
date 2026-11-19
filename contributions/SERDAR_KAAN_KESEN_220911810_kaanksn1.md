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
- Reviewed Kemal's PR #9 at `680b875d`: independently passed 164 tests and
  exercised GMM-HMM on CPU, standard-size CTC and real mHuBERT-147 + CTC on
  GPU (finite gradients/parameter updates/checkpoint/predict), and published
  wav2vec inference on GPU using the three original train clips. Verified the
  earlier CPU evidence hashes. Recorded the licensed HuBERT training-source
  decision and added the reused wav2vec helper to shared HuBERT run code hashes.
  Kemal's model implementations, original commits and report remain his.
  Temporary integration with Samed's data code passed 196 tests after resolving
  only the shared contribution-inventory conflict; no personal report or model/
  data implementation changed, and no full LFS audio verification was claimed.

## 20 November: independent audio verification and measured budgets

- Downloaded Samed's actual LFS test/dev/CSS10 audio. Independently verified
  the unchanged 8,679-record frozen test with audio and the 7,113 dev WAVs;
  byte-checked 16 original metadata files. Verified all 1,844 original CSS10
  audio hashes/RIFF headers and detected IEEE float32 rather than integer PCM32.
- Repaired the shared model input contract (`greek_asr/audio.py`, shared runner
  validation and the thin `models/common.py` decoding adapter), preserving raw
  bytes and model architectures. Added semantic float/amplitude/finite/truncation
  tests; all 203 tests passed after that fix. Samed owns the outstanding
  data-probe/audit-statistics correction and regeneration.
- Acquired a deterministic 144-clip official TRAIN timing cohort (128 random
  plus long audio/text tails) with source hashes, and measured actual attention,
  RNN-T, CTC, licensed mHuBERT and Whisper GPU steps/VRAM plus GMM-HMM CPU cost.
  Exercised the longest real CSS10 inputs after the shared decoder correction.
  Measured ROCm MIOpen FAST on previously unmeasured input shapes.
- Reused Samed's whole-session selector to generate the 18-session used-data
  pool, RAW/clean/reference/delivery hashes and conservative 10 GiB storage plan.
  The RAW delivery includes removed examples; cleaning authorship remains Samed's.
  Generated central training allocations and both matched Whisper conditions
  from real measurement JSON, rather than inventing an hour or step limit.
- Measured all five selected published ASR backends on 32 real TRAIN clips.
  Added bounded Qwen batching with ID/order/empty-result/failure checks; batch
  4 produced the same 32 hypotheses as single inference in roughly half the
  measured time. These are runtime observations, not WER/CER benchmark scores.
- Prepared the exact Samed handoff, byte-verification evidence, source-aware
  storage calculations and reproducible budget report. See
  `docs/HANDOFF_SAMED_FINAL_DATA_TR.md` and `docs/reviews/budget_2026-11-20/`.

## Limits and pending dependent work

Samed submitted early test-freeze metadata and his data pipeline in PR #2;
test/dev audio passed independent verification and CSS10 original hashes/headers
matched. Selected RAW train upload, float-aware CSS10 data audit and used-only
pipeline reproduction are pending. His pipeline and cleaning authorship remain
his. Adnan's submitted implementation and the owner's final
review corrections have passed code/pilot checks. Kemal's submitted models passed
code/pilot review; Ozan's final evaluation and all full
benchmark runs remain pending. No custom-data bonus, trained-model improvement or new
human annotation is claimed. Personal contribution percentages are not results.

Commits made under `kaanksn1` in this repository provide the version history.
Coding-agent assistance was used for implementation, research and checks;
that does not establish teammates' authorship or work they have not performed.
