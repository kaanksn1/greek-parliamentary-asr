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
  with `torchaudio.functional.rnnt_loss` (verified BSD-2-Clause license and
  CPU forward/backward correctness; ROCm correctness still pending Serdar's
  pilot). Explicitly distinct from CTC and from the pretrained Parakeet TDT
  baseline (owned by Serdar).
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
`tests/test_models_transducer.py`, `tests/test_whisper_finetune.py` — 28
tests total. They cover: real-pilot-audio manifest/reference joining and
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

## Known limitations / risks for Serdar

- `data/manifests/*.jsonl` do not exist yet (Samed's PR #2 is unmerged,
  `docs/reviews/SAMED_PR2_REVIEW_TR.md`); none of this code has run on real
  HParl/CSS10 audio. `configs/sequence/*.json` point at the documented future
  paths and set `max_steps: null` on purpose — all three `train()` functions
  refuse to run with an unset budget rather than assuming an arbitrary one.
- The RNN-T loss's ROCm/HIP correctness is unverified; only CPU forward/
  backward is verified so far. See `docs/SEQUENCE_MODELS.md`.
- Official HParl split speaker overlap applies to these models exactly as to
  every other baseline; no speaker-disjoint or unseen-speaker claim is made.

## Slides

Draft 1–2 slides and speaker notes for Ozan: `docs/ADNAN_SLIDES_DRAFT_TR.md`.
I am not taking over the full presentation or the cross-model comparison.
