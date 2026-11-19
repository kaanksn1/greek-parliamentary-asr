# Adnan Demir — 220911784 — adnan-demir

Model owner for the standard attention encoder-decoder (hybrid CTC/attention),
standard RNN-T transducer, and our own Whisper fine-tuning contribution
(`docs/TEAM.md`, `docs/HANDOFF_ADNAN_TR.md`). This report lists only the work
actually done in this PR; it does not claim Samed's data pipeline, Kemal's
GMM-HMM/CTC/wav2vec/HuBERT models, Ozan's scorer/presentation, Serdar's
integration/central runs, or any teammate's personal report.

## Baseline family (standard architecture, our implementation/training labour)

- Implemented a standard **attention encoder-decoder with an optional hybrid
  CTC head** (`greek_asr/models/attention.py`): Listen-Attend-and-Spell
  encoder-decoder (Chan et al., 2016) with the Watanabe et al. (2017) hybrid
  CTC/attention auxiliary loss. Plain PyTorch/torchaudio, not a third-party
  toolkit; rationale in `docs/SEQUENCE_MODELS.md` "Toolkit choice".
- Implemented a standard **RNN-T transducer** (`greek_asr/models/transducer.py`):
  Graves (2012) RNN-T with a streaming-style unidirectional pyramidal LSTM
  encoder, LSTM prediction network and feed-forward joint network, trained
  with `greek_asr.models.rnnt_loss` -- a pure PyTorch/autograd loss
  (log-space forward algorithm), value/gradient-verified against
  `torchaudio.functional.rnnt_loss`'s own CPU kernel, adopted after that
  kernel's CUDA/ROCm backend turned out to be missing on Serdar's RX 7800 XT
  (see "Serdar's RX 7800 XT review fixes" below); real GPU/ROCm throughput
  is still unmeasured. Explicitly distinct from CTC and from the pretrained
  Parakeet TDT baseline (owned by Serdar).
- Both remain **baseline-family architectures** per `docs/BASELINES.md`; the
  credited work here is the actual implementation, the character-level
  tokenizer/vocabulary construction, the padding/masking contract, the
  greedy/beam decoding paths (neither ever reads the reference transcript),
  the checkpoint format, and the unit tests below — not a new architecture.
- Shared plumbing (`greek_asr/models/common.py`): manifest/reference joining
  that reuses `greek_asr.infer.load_manifest` for the audio-side contract
  (never reimplemented or weakened), in-memory resampling that never touches
  source files, a character-vocabulary builder restricted to train
  references, speaker/recording-grouped seeded subset selection, read-only
  split-statistics/literal-overlap reporting, and the shared prediction
  JSONL writer.

## Contribution (own fine-tuning)

- `greek_asr/models/whisper_finetune.py`: our fine-tuning of the pinned
  `openai/whisper-small` checkpoint (`973afd24965f72e36ca33b3055d56a652f456b4d`,
  verified against `configs/model_sources.lock.json` before any download or
  training) under two controlled conditions sharing one training code path —
  `configs/sequence/whisper_hparl_only.json` (HParl-only) and
  `configs/sequence/whisper_hparl_css10.json` (HParl + CSS10 Greek,
  training-only). The two configs are identical in every field except the
  train manifest/references (enforced by
  `tests/test_whisper_finetune.py::MatchedConditionConfigTests`), with a
  matched effective batch (`micro_batch_size x gradient_accumulation_steps`)
  and the same `max_steps` once that is set from a real pilot measurement.
  Serdar's unmodified Whisper baseline stays untouched and separate.
- Label construction masks only genuine padding (via the tokenizer's
  attention mask, not a pad/eos token-ID guess) with `-100`, keeping the real
  BOS/language/task/EOS tokens as real target content — tested against the
  real pinned tokenizer, not a stand-in.

## Tests (all passing on CPU, `.venv-sequence`)

`tests/test_models_common.py`, `tests/test_models_attention.py`,
`tests/test_models_transducer.py`, `tests/test_whisper_finetune.py`,
`tests/test_rnnt_loss.py` — 70 tests total as of the Serdar-review fixes
(up from 28 at the original submission). They cover: real-pilot-audio
manifest/reference joining and
rejection of the frozen test split; real WAV decoding/resampling without
touching source bytes; vocabulary construction/round-trip and out-of-vocabulary
handling; padding/masking correctness; tiny synthetic forward/backward with
verified parameter updates and finite gradients for both the attention model
(including its CTC head) and the RNN-T loss; an end-to-end train-then-predict
run on the three real train-pilot clips for both models, confirming
checkpoints are written, the vocabulary is learned only from train references,
and prediction never reads the reference text; the `--split test` gate
(refused without `--allow-test`/a committed `--test-protocol`); real-tokenizer
label/padding/prefix-token behaviour and a genuine optimizer step for Whisper
fine-tuning (via a tiny randomly-initialized model sharing the real pinned
tokenizer, the same `model_loader` injection pattern `greek_asr.infer.run`
already uses — not a mocked arithmetic check); the matched-budget/config
parity check between the two Whisper conditions; and the shipped configs'
refusal to train with an unset step budget.

I also ran `python scripts/pilot_sequence_models.py --device cpu` locally as
a code-path smoke (not a GPU pilot, not committed): one real forward/backward/
optimizer step per model, including on the full real `openai/whisper-small`
checkpoint, all with a finite loss. This is not a training-length decision or
a benchmark.

## GPU pilot and real training

`scripts/pilot_sequence_models.py` (one real forward/backward/optimizer step
per model, VRAM/step-time measurement) and the exact `train`/`predict`
commands are packaged for Serdar in `docs/SEQUENCE_MODELS.md` "GPU pilot
package". I did not run this on a GPU myself; Serdar's RX 7800 XT machine is
the only full-training/inference environment for this project
(`docs/TEAM.md`). No real training, dev-based checkpoint/beam selection, or
benchmark score is claimed.

## Fixed, not left as a TODO

- `.gitignore` had a bare `models/` rule that (per gitignore matching rules)
  also silently excluded my assigned source path `greek_asr/models/` (see
  `docs/TEAM.md`'s suggested file ownership), not just the top-level
  model-weight-cache directory `greek_asr/infer.py` actually uses. Anchored
  it to `/models/` so only the top-level cache directory is still ignored.

## Pre-merge audit (self-audit against the original raw prompt)

Before asking Serdar to merge, I re-audited this PR against my original raw
task prompt and the professor's four announcements, verifying every claim by
actually running tests/CLIs rather than trusting the earlier session's
summary. Samed's `feature/p3-data` branch has moved (new Kaldi metadata and
manifests addressing the PR #2 review), but it is still unmerged and still
contains no real audio files (only the same 3 pilot clips) -- data acceptance
status is unchanged for this branch. The audit found and fixed real gaps in
my own code, all in new commits on this branch:

- Training run reports were missing required provenance fields (code commit,
  per-file code hash, train/dev manifest+reference hashes, trainable
  parameter count, effective batch size, precision, training wall-clock
  time) for all three models -- added, with tests.
- `predict()` for all three models had no partial/failed-run marker; a
  mid-run failure left either nothing or an inconsistent state. Added
  `common.PartialPredictionWriter`, mirroring `greek_asr.infer.run`'s
  existing `predictions.partial.jsonl` contract, with tests proving a
  simulated failure leaves the partial file and a `status: "failed"`
  `run.json`, never a complete-looking `predictions.jsonl`.
- The Whisper training report did not record the actual CSS10/HParl mix of
  the training manifest -- added `common.manifest_source_breakdown` and a
  test against a real mixed manifest.
- Whisper label tokenization relied on the tokenizer's default truncation
  behaviour instead of an explicit `truncation=False`; added the explicit
  flag and a test that proves a long reference is not silently cut (the
  tokenizer's own warning, now visible in that test's output, shows this was
  a real risk, not a hypothetical one).
- No test directly proved "checkpoint save/load gives consistent evaluation"
  (listed explicitly as a required check) -- added for all three models.
- No test proved re-running into an existing output directory is refused
  rather than silently overwriting a previous run -- added for all three.
- Documentation gaps: the hybrid CTC head not completing Kemal's separate
  plain-CTC baseline, the long-audio/no-truncation policy for attention and
  RNN-T, the dependency of checkpoint selection on Ozan's not-yet-finalized
  normalization policy, the requirement to coordinate with Serdar/Samed
  before using the subset-selection utility on real data, the lack of an
  AMP/mixed-precision implementation, and the Whisper model card link were
  all missing and have been added to `docs/SEQUENCE_MODELS.md` and/or the
  relevant module docstrings.
- The GPU pilot script reported raw step time but not a usable throughput
  ratio for estimating epoch cost; added
  `estimated_training_seconds_per_audio_hour` to its output.

All 93 tests (up from 72) pass inside `.venv-sequence` as of the pre-merge
audit; 59 of them (the ones that do not need torch/transformers) also pass
with the base interpreter used elsewhere in this repo, with the other 34
skipping cleanly instead of erroring. Everything above is a code/doc fix
within my own scope; nothing in `greek_asr/infer.py`, `greek_asr/pretrained/`,
`greek_asr/score.py`, `greek_asr/scoring.py`, Samed's data files/pipeline, or
any other member's files was touched (verified with `git diff` against
`main`).

After the Serdar review fixes below, the suite grew to **114 tests, all
passing** inside `.venv-sequence` (`python -m unittest discover -s tests -v`);
59 of them also pass with the base (torch-less) interpreter, with the other
55 skipping cleanly. New test modules/cases: `tests/test_rnnt_loss.py` (the
new RNN-T loss backend's value/gradient correctness) plus new cases added to
`tests/test_models_common.py` (PCM16/32 decoding, device contract),
`tests/test_models_attention.py`/`tests/test_models_transducer.py`
(trailing-partial-accumulation regression) and `tests/test_whisper_finetune.py`
(decoder-start-token regression, long-audio guard).

## Serdar's RX 7800 XT review fixes (review dated 18 November 2026)

Serdar reviewed submitted commit `0437631` on his real RX 7800 XT (ROCm)
machine plus two additional synthetic/token-level checks and found eight
concrete defects -- see `docs/reviews/PR2_PR5_FOLLOWUP_TR.md` and the
reproducible evidence in `docs/reviews/ADNAN_0437631_GPU.json`. **I am
reporting these as Serdar's findings, not my own** -- my CPU-only machine
could not have reproduced the GPU-specific ones (2, 3, 4), and the others
(1, 5, 6, 7, 8) were also first identified in his review, not by me before
it. What follows is what I changed in response, on this same branch/PR:

1. **CSS10 PCM32 decode (`common.load_waveform`)**: the function decoded
   every WAV payload as 16-bit regardless of the file's real sample width;
   Serdar's audit of CSS10's 1,844 recordings found all of them 4-byte
   (32-bit) PCM, so they were being misread (doubled frame count, corrupted
   amplitude). Fixed to branch on the WAV header's real `sampwidth` (16-bit
   or 32-bit), reject any other width/format with a clear error, and added
   tests with known sample values plus a source-byte-hash-preserved check
   (`tests/test_models_common.py::WaveformTests`).
2. **Attention GPU device mismatch (`common.padding_mask`)**: built its
   position tensor with a bare `torch.arange` (implicit CPU), compared
   against a lengths tensor that was on GPU in Serdar's real pilot run,
   raising `RuntimeError: Expected all tensors to be on the same device`.
   Fixed to build the position tensor on `lengths.device`; added a
   device-contract test using torch's "meta" device (no real GPU needed to
   exercise the same mismatch code path) since my machine has no GPU.
3. **No device selection in the real train/predict path**: `train()`/
   `predict()` for attention and RNN-T never moved the model or batch
   tensors to a requested device, and the CLIs had no `--device` flag at
   all (a separate one-line `.to(device)` call in Serdar's pilot script
   never touched the real training path). Added `common.resolve_device`
   (explicit error if a requested device is unavailable -- no silent CPU
   fallback -- and ROCm's own `"cuda"` device string is not assumed to mean
   NVIDIA) and `common.move_batch_to_device`; wired both into `train()`/
   `predict()` for all three models, added `--device` to every CLI, and
   record `device_requested`/`device_used` in every run report. I have no
   GPU to verify this end-to-end myself -- see "Known limitations" below for
   the single command Serdar needs to run.
4. **RNN-T loss has no ROCm kernel**: `torchaudio.functional.rnnt_loss`'s
   compiled op raised `NotImplementedError` on Serdar's AMD wheel
   (`'torchaudio::rnnt_loss' ... 'CUDA' backend`); the CPU-only test that
   previously passed was never evidence this ran on GPU. Replaced it with
   `greek_asr/models/rnnt_loss.py`, a pure PyTorch/autograd implementation
   (log-space forward algorithm) built only from ops every backend torch
   ships has a kernel for, so it needs no device-specific compiled op at
   all. Verified its value **and** gradient against `torchaudio`'s own CPU
   kernel on a hand-enumerated two-path lattice and several random
   variable-length batches (`tests/test_rnnt_loss.py`). Real GPU/ROCm
   throughput is still unmeasured -- see "Known limitations".
5. **Whisper decoder start token duplicated**: `build_labels` left the
   tokenizer's own leading `decoder_start_token_id` in the returned labels,
   and the model's internal `shift_tokens_right` call prepends that same
   token again, so the real decoder input started
   `[50258, 50258, 50281, ...]` instead of `[50258, 50281, ...]` (verified
   against the real pinned tokenizer/config in Serdar's review). Fixed by
   stripping that leading token (checked against the real
   `model.config.decoder_start_token_id`, never a guessed `bos_token_id`)
   before the model's own shift re-adds it once; kept the real EOS and
   language/task prefix untouched, masked only true padding. Added a
   regression test that applies tokenization + the model's real shift
   together and checks the start token appears exactly once -- the existing
   "prefix korunuyor" test did not catch this because it never applied the
   model's shift at all.
6. **Whisper silently truncates audio over 30 seconds**: confirmed with two
   synthetic 31-second clips whose different final second produced
   identical extracted features. Added an explicit `<=30s` check
   (`_require_within_whisper_window`) that raises instead of letting that
   truncation happen silently; the reference text is never cut either way.
   The real HParl/HParl+CSS10 manifests are already under 30s (max 26.4s)
   after the PCM32 fix, so this is not expected to fire on current data.
   Also corrected the README/module-docstring claim that this path is "the
   same as" Serdar's unmodified Whisper baseline adapter -- it reuses the
   same feature extractor default, but this module now raises explicitly
   where the baseline adapter does not.
7. **max_steps conflated microbatches with optimizer updates**: attention
   and RNN-T compared a raw microbatch counter directly against `max_steps`
   and broke out of training as soon as that count was hit, even mid
   gradient-accumulation-cycle -- when `max_steps` didn't divide evenly by
   `gradient_accumulation_steps`, the last accumulated-but-unapplied
   gradient was silently dropped. Redefined `max_steps` as optimizer
   updates (matching Whisper's existing `optimizer_updates_completed`/
   `micro_steps_completed` split) and restructured the loop so it only ever
   stops right after a completed accumulation cycle, carrying any partial
   cycle over to the next epoch instead of dropping it. Added a regression
   test with `gradient_accumulation_steps=3` over a 2-example dataset
   (batch size 1) where the completing microbatch only arrives in the
   second epoch, for both attention and RNN-T.
8. **Honest reporting for the controlled Whisper experiment**: this was
   already mostly in place (`train_source_breakdown` records the real
   CSS10/HParl mix per run, `code_commit`/manifest+reference hashes/
   environment/device/duration/checkpoints were already in the report); I
   added the explicit statement that resume-from-checkpoint is **not**
   implemented anywhere in this code (no CLI flag reloads a checkpoint to
   continue training, and the checkpoint's RNG snapshot is torch-CPU +
   Python `random` only, not NumPy/GPU state) so it is not implied by the
   mere presence of a stored RNG field.

## Known limitations / risks for Serdar

- `data/manifests/*.jsonl` do not exist on `main` (Samed's PR #2 is still
  unmerged). Samed's `feature/p3-data` branch now has real train/dev/test
  Kaldi metadata and manifests (76,302-line `train.jsonl`), but as of this
  audit it still contains only the same 3 pilot WAV files -- the manifests
  reference ~76k audio paths that do not exist in the repository, so real
  audio delivery is still the open item, not the metadata/manifest format.
  None of this code has run on real HParl/CSS10 audio.
  `configs/sequence/*.json` point at the documented future paths and set
  `max_steps: null` on purpose — all three `train()` functions refuse to run
  with an unset budget rather than assuming an arbitrary one.
- **I have no GPU and did not verify any of items 2-4 above on real
  hardware.** Serdar needs to re-run the single pilot command in
  `docs/SEQUENCE_MODELS.md` "GPU pilot package" on his RX 7800 XT:
  `HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" scripts/pilot_sequence_models.py --device cuda --output outputs/sequence_gpu_pilot`.
  This exercises the fixed device plumbing and the new pure-PyTorch RNN-T
  loss end-to-end and records real step time/VRAM; none of that is claimed
  here as verified.
- The new RNN-T loss backend (`greek_asr/models/rnnt_loss.py`) is only
  value/gradient-verified against torchaudio's CPU kernel; its real
  throughput on an actual GPU/ROCm device is unmeasured (it is a
  straightforward O(time × label) loop, not the anti-diagonal/compiled
  optimization torchaudio's own kernel uses).
- Official HParl split speaker overlap applies to these models exactly as to
  every other baseline; no speaker-disjoint or unseen-speaker claim is made.

## Slides

Draft 1–2 slides and speaker notes for Ozan: `docs/ADNAN_SLIDES_DRAFT_TR.md`.
I am not taking over the full presentation or the cross-model comparison.
