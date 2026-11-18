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

## Limits and pending dependent work

The full original test has not yet been frozen. Samed's full acquisition,
cleaning, logs and data deliverable remain with Samed. Kemal's and Adnan's
model training implementations, Ozan's final evaluation and all full benchmark
runs remain pending. No custom-data bonus, trained-model improvement or new
human annotation is claimed. Personal contribution percentages are not results.

Commits made under `kaanksn1` in this repository provide the version history.
Coding-agent assistance was used for implementation, research and checks;
that does not establish teammates' authorship or work they have not performed.
