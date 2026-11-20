# Adnan's sequence models: attention, RNN-T and Whisper fine-tuning

This covers the three model tasks assigned to Adnan Demir in `docs/TEAM.md`
and `docs/HANDOFF_ADNAN_TR.md`: a standard attention encoder-decoder (hybrid
CTC/attention included), a standard RNN-T transducer, and our own fine-tuning
of the pinned `openai/whisper-small` checkpoint under two training conditions.

**Verification at submitted commit `0c13ce1`, 19 November 2026:** 114 tests
passed; attention, RNN-T and Whisper completed real GPU forward/backward and
optimizer checks on Serdar's RX 7800 XT. Attention/RNN-T train, checkpoint
reload and predict CLI paths also passed. Evidence is recorded in
[ADNAN_0c13ce1_REVIEW_TR.md](reviews/ADNAN_0c13ce1_REVIEW_TR.md),
[GPU diagnostics](reviews/ADNAN_0c13ce1_GPU.json) and
[CLI diagnostics](reviews/ADNAN_0c13ce1_ENTRYPOINTS.json).
These checks used only the three original training-pilot clips, not the
official dev/test sets. No full-data training or benchmark result is claimed.

Current owner update, 20 November: representative TRAIN measurements set the
allocations in `configs/training_budget.json`; all configs use the selected
18-session cohort in `configs/training_data_selection.json`. Test/dev audio
verification passed. Selected RAW train upload and a corrected float-aware
CSS10 audit remain with Samed in PR #2. Full runs start after their acceptance.
All full training/inference runs centrally on Serdar's machine (`docs/TEAM.md`).

## Toolkit choice

The implementation uses **PyTorch + torchaudio**, with Transformers for
Whisper, in the existing isolated ROCm runtime. The standard architecture
recipes below are implemented directly and remain inspectable in the repo.
ESPnet, SpeechBrain and WeNet were alternatives listed in the handoff; they
are not dependencies of this implementation. Compatibility was checked on
the actual target machine rather than inferred from the framework choice.

### RNN-T loss backend

Candidates from `docs/HANDOFF_ADNAN_TR.md`: `torchaudio` RNN-T loss, k2/icefall
pruned RNN-T, NeMo, or a referenced custom implementation.

**Originally chosen `torchaudio.functional.rnnt_loss` ran on Serdar's real RX
7800 XT (ROCm) and failed**: `NotImplementedError: Could not run
'torchaudio::rnnt_loss' ... 'CUDA' backend` (see
`docs/reviews/ADNAN_0437631_GPU.json`, finding 3). The op is a custom
compiled kernel, and the project's AMD wheel (`requirements-rocm.lock.txt`:
`torchaudio==2.9.0+rocm7.2.1...`) only has a CPU build of it. The CPU
forward/backward check that previously passed
(`tests/test_models_transducer.py::SyntheticLossBackwardTests`) was never
evidence this ran during real GPU training — it just happened to run on the
one device the kernel supports.

**Current backend: `greek_asr/models/rnnt_loss.py`, recorded as
`pure_pytorch_logspace` in the config and run report.**

- The log-space forward algorithm uses PyTorch/autograd operations rather
  than the unavailable compiled TorchAudio RNN-T op. This avoids that specific
  dependency; it does not remove the need to check numerical behavior,
  device execution, gradients, memory and speed on the target hardware.
- Correctness evidence (`tests/test_rnnt_loss.py`): value **and** gradient
  match `torchaudio.functional.rnnt_loss` on CPU across a hand-enumerated
  two-path lattice (solved independently, no library call) and several
  random variable-length batches (`torch==2.15.0+cpu`,
  `torchaudio==2.11.1+cpu`, this repo's `.venv-sequence`).
- `transducer.py`'s report now records `"rnnt_loss_backend":
  "pure_pytorch_logspace"` so the backend in use is explicit, not implied.
- Independent alignment enumeration also checked loss and gradients for
  variable lengths, empty targets, U>T and a nonzero blank ID;
  [the recorded mathematical checks](reviews/ADNAN_0c13ce1_RNNT_MATH.json)
  passed. Real RX 7800 XT checks at `0c13ce1` then confirmed finite GPU
  loss/gradients, a parameter update and the full train/predict CLI path.
- Full-data throughput and memory remain unmeasured. The dynamic program
  has O(time × label) work, and the joint network materializes a dense
  time/label/vocabulary tensor. Short-pilot success does not establish a
  feasible full training batch or epoch cost. Repeat measurements on
  representative training lengths before choosing the budget:

  ```bash
  HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" scripts/pilot_sequence_models.py \
    --device cuda --output outputs/sequence_gpu_pilot
  ```

  A different backend would need its own license, numerical and ROCm
  verification, plus an explicit config/report change. No alternative
  backend is currently claimed to be verified.

## Architectures

### A) Attention encoder-decoder (hybrid CTC/attention), `greek_asr/models/attention.py`

- Recipe: **Listen, Attend and Spell** (Chan, Jaitly, Le, Vinyals, 2016,
  arXiv:1508.01211) with an auxiliary CTC head (**hybrid CTC/attention**,
  Watanabe et al., 2017, IEEE JSTSP).
- Input: 80-dim log-mel filterbank (`greek_asr.models.common.LogMelFrontend`;
  `n_fft=400`, `hop_length=160`, i.e. 25 ms/10 ms at 16 kHz).
- Encoder ("Listener"): 3-layer pyramidal bidirectional LSTM (hidden 256 per
  direction); two reduction steps concatenate adjacent frame pairs, giving a
  4x time reduction overall.
- Attention: single-head dot-product attention over encoder outputs, masked
  at the padded positions (`common.padding_mask`).
- Decoder ("Speller"): 2-layer unidirectional LSTM (hidden 512); teacher
  forcing for the train/dev loss, free-running (no gold input) for greedy and
  beam-search decoding — `Speller.greedy_decode` / `Speller.beam_search_decode`
  never read the reference text.
- Auxiliary CTC head: linear projection of the encoder output; joint loss is
  `(1 - ctc_weight) * attention_CE + ctc_weight * ctc_loss` via
  `torch.nn.functional.ctc_loss`, `ctc_weight = 0.3` by default
  (`configs/sequence/attention.json`).
- Vocabulary: character-level, learned only from the given train references
  (`common.build_char_vocabulary`); `<pad> <sos> <eos> <unk> <blank>` specials.
- `max_steps` means **optimizer updates**, not microbatches (same convention
  as Whisper's `optimizer_updates_completed`/`micro_steps_completed`); see
  "Training budget" below.
- Device: `--device {cpu,cuda}` on both `train` and `predict` (ROCm PyTorch
  reports its accelerator as `"cuda"` too; this is not an NVIDIA-only flag).
  Requesting an unavailable device raises instead of silently falling back
  to CPU (`common.resolve_device`); checkpoints load via `map_location="cpu"`
  so they can always be loaded regardless of which device trained them, then
  moved to whatever device `predict --device` requests. The run report
  records both `device_requested` and `device_used`.
- CLI: `python -m greek_asr.models.attention train --config configs/sequence/attention.json --output <dir> --device cpu`
  and `... predict --checkpoint <dir>/checkpoint.best.pt --manifest <manifest> --split {train,dev,test} --output <dir> --device cpu`
  (use `--device cuda` on Serdar's ROCm machine).
  `--split test` is refused unless `--allow-test --test-protocol <committed path>` matches the
  exact frozen test manifest hash and is itself the committed version (mirrors `greek_asr/infer.py`'s gate).

### B) RNN-T transducer, `greek_asr/models/transducer.py`

- Recipe: **RNN-T** (Graves, 2012, arXiv:1211.3711), streaming-style encoder
  per He et al. (2019), "Streaming End-to-end Speech Recognition for Mobile
  Devices".
- Acoustic encoder: 3-layer **unidirectional**, pyramidal LSTM (hidden 256),
  same 4x reduction scheme as the attention encoder but single-direction.
- Prediction network: 1-layer unidirectional LSTM over the label history,
  fed a dedicated start token (reusing the `<sos>` id) before the first label.
- Joint network: encoder/predictor linear projections, summed, `tanh`, linear
  to the blank-inclusive vocabulary (`blank` id `0`).
- Loss: `greek_asr.models.rnnt_loss.rnnt_loss`, backend
  `pure_pytorch_logspace` (see above).
- Decoding: standard greedy transducer decoding — one encoder time step only
  advances after a blank is emitted, bounded by `max_symbols_per_step` to
  avoid an infinite loop; `TransducerASRModel.greedy_decode` never reads the
  reference text.
- Vocabulary: character-level, `<blank> <sos> <unk>` specials (`<sos>` doubles
  as the predictor's start token; `<blank> = 0` also serves as the pad-fill value).
- This is explicitly **not** CTC and **not** NeMo/Parakeet TDT inference — see
  the module docstring and `docs/BASELINES.md`.
- `max_steps` means **optimizer updates**, not microbatches; same `--device`
  contract (and `device_requested`/`device_used` report fields) as the
  attention model above.
- CLI: `python -m greek_asr.models.transducer train --config configs/sequence/transducer.json --output <dir> --device cpu`
  and `... predict ...` with the same `--split test` gate and `--device` flag as the attention model.

### C) Whisper fine-tuning contribution, `greek_asr/models/whisper_finetune.py`

- Starting checkpoint: `openai/whisper-small` @ `973afd24965f72e36ca33b3055d56a652f456b4d`
  (matches `configs/model_sources.lock.json`; `train()`/`predict()` call
  `_validate_revision` and refuse to run on any other model/revision before
  any download happens — see `tests/test_whisper_finetune.py::RevisionValidationTests`).
  Model card: https://huggingface.co/openai/whisper-small (multilingual;
  Greek is one of the Whisper paper's supported languages). Serdar's
  unmodified-checkpoint baseline row owns this same verification for the
  baseline inventory (`docs/BASELINES.md`); this is the same checkpoint, not
  a separately-checked one.
  Serdar's unmodified baseline run on this same checkpoint stays a separate,
  untouched baseline row (`docs/BASELINES.md`); this is additional training.
- Two conditions, **one training code path**
  (`configs/sequence/whisper_hparl_only.json`,
  `configs/sequence/whisper_hparl_css10.json`): identical in every field
  except `train_manifest`/`train_references`/`condition`
  (`tests/test_whisper_finetune.py::MatchedConditionConfigTests` enforces
  this at the config level). Both read the same HParl `dev`/`test` manifests.
  CSS10 only ever appears in the `train` split.
- Matched training budget: both configs fix `micro_batch_size=2`,
  `gradient_accumulation_steps=8` (effective batch 16) and the same
  1,600 `max_steps` (optimizer updates). The epoch counts can differ, since
  the two training sets have different sizes.
- Labels: `processor.tokenizer.set_prefix_tokens(language="el", task="transcribe")`
  then tokenize; `-100` is applied from the tokenizer's **attention mask**,
  not from a token-ID comparison, so the real end-of-sequence token is never
  mistaken for padding even when `pad_token_id == eos_token_id` (it is, for
  Whisper). The leading actual `decoder_start_token_id` is removed from
  labels before the model's internal shift adds it once; Greek/task tokens
  and EOS remain. See `build_labels` and `BuildLabelsTests`.
- Fine-tuning train/predict rejects audio over 30 seconds before feature
  extraction, with an explicit error; references are not truncated. The
  standard owner baseline in `greek_asr/pretrained/whisper.py` supports
  longer inputs using `truncation=False`, an explicit attention mask and
  timestamp-based generation. It does not silently cut audio at 30 seconds.
- Fine-tuned prediction currently passes `input_features` without an explicit
  attention mask. Its upstream pad/EOS warning is not evidence that this call
  is identical to the owner baseline, which does pass the mask.
- `model.generate(..., language="el", task="transcribe", do_sample=False)` is
  used for prediction, never fed the reference text.
- A nonfinite Whisper training or dev loss raises an error and records a
  failed `run.json`. It is not skipped while continuing the accumulation
  cycle or waiting indefinitely for the optimizer-step budget to complete.
- Checkpointing: best (by dev loss) and last are separate files; each stores
  model/optimizer/scheduler state, the optimizer-update step and
  `common.capture_rng_state()` (torch CPU + Python `random` state only --
  **not** NumPy's or any GPU/CUDA-generator RNG state).
- Resume (continuing a stopped run from a saved checkpoint): **not
  implemented**. `train()` always starts a fresh model/optimizer/scheduler;
  no CLI flag loads a checkpoint to continue training, and the stored RNG
  state is never re-applied anywhere. This is deliberately reported as
  absent rather than implied: a real resume claim would additionally need
  the NumPy/GPU RNG state captured and a real continuation exercised and
  checked, neither of which this PR does (see
  `docs/reviews/PR2_PR5_FOLLOWUP_TR.md`, "Resume iddiası varsa...").
- CLI: `python -m greek_asr.models.whisper_finetune train --config configs/sequence/whisper_hparl_only.json --output <dir> --device cuda`
  (repeat with `whisper_hparl_css10.json`), and `... predict ...` with the
  same `--allow-test`/`--test-protocol` gate as the other two models.

## Training budget from representative TRAIN measurements

The owner's generated allocations and timing/memory evidence are in
`configs/training_budget.json` and `docs/reviews/budget_2026-11-20/README.md`.
Attention uses batch 16 / 3,000 updates; RNN-T batch 8 / 3,000 updates; both
Whisper conditions microbatch 2 × accumulation 8 / 1,600 updates. Actual full
training/convergence is pending. Any unset `max_steps` still fails immediately;
the regression tests set it to `None` explicitly to exercise that guard.
Source audio, used-cohort hashes and the corrected CSS10 audit must be accepted
before training. These allocations do not authorize final test inference.

## GPU pilot package (for Serdar)

One combined script, using the three already-committed train-pilot clips
(6.61 seconds total; no dev/test audio read) as the only real-audio input
available before data delivery:

```bash
# Inside the isolated ROCm environment used for Whisper/Qwen
# (scripts/setup_transformers_rocm.sh), with the sequence-model code on the path:
HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" scripts/pilot_sequence_models.py \
  --device cuda --output outputs/sequence_gpu_pilot
```

This performs one real forward/backward/optimizer step per model (attention,
transducer, whisper_finetune) and writes
`outputs/sequence_gpu_pilot/{attention,transducer,whisper_finetune}.json`
with: `loss_finite`, `step_seconds`, `peak_gpu_memory_allocated_bytes`, the
pilot audio duration/count, and the environment (Python/Torch/HIP/GPU name).
It performs no training-length decision, no dev/test access and no scoring.

The recorded `0c13ce1` GPU and CLI evidence is linked above. Replays must use
a fresh output path; the three-clip diagnostic does not choose a full-data
training budget or produce WER/CER benchmark scores.

## Environment

`requirements-sequence.txt` lists the CPU dependency pins reported for Adnan's
Windows/Python 3.12.10 development environment (`torch==2.15.0+cpu`,
`torchaudio==2.11.1+cpu`, `numpy==1.26.4`, `transformers==4.57.6`). These are
not the ROCm Torch/TorchAudio versions with the wheel suffix swapped. Build
the CPU development environment with:
Build it with:

```powershell
py -3.12 -m venv .venv-sequence
.venv-sequence\Scripts\python -m pip install -r requirements-sequence.txt
```

Serdar's tests and GPU checks at `0c13ce1` used the existing
`project3-greek-asr-transformers` runtime: PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, TorchAudio
`2.9.0+rocm7.2.1.gite3c6ee2b`, NumPy `1.26.4`, Transformers `4.57.6`.
Its exact installation pins are in `requirements-rocm.lock.txt`; see
`docs/RUNTIME_ENVIRONMENT.md`. Use that lock for ROCm replay, not the CPU
requirements file. No package change was needed for those checks. A new
conflict requires separate investigation rather than replacing that runtime.

## Shared utilities (`greek_asr/models/common.py`)

- `load_utterances`: joins a manifest and its reference file by ID, reusing
  `greek_asr.infer.load_manifest` for the audio-side hash/WAV-header/ID
  validation instead of duplicating it. Raises (never silently drops) on any
  missing/extra ID, and refuses the `test` split outright (`load_utterances`
  is only ever called for `train`/`dev`).
- `load_waveform`: the owner's thin shared-input adapter uses
  `greek_asr.audio` to distinguish PCM16 (HParl), integer PCM32 fixtures and
  the actual IEEE float32 CSS10 files by RIFF format tag, then resamples in memory via
  `torchaudio.functional.resample`. Unsupported formats fail explicitly;
  the source file is never rewritten.
- `select_grouped_subset`: speaker/recording-grouped, seeded subset selection
  for a bounded training budget (section 0.2/7 of the main prompt) — a group
  is never split across the selection boundary; utterances with neither a
  speaker nor a recording ID fall back to utterance-level grouping, and the
  count of that fallback is reported in the returned diagnostics for the
  README. Tested on a synthetic multi-speaker fixture in
  `tests/test_models_common.py`. The owner chose Samed's 18 whole-session
  prefix; its raw/clean IDs and byte/storage bounds are versioned separately.
- `split_statistics` / `literal_overlap`: read-only speaker/utterance/hour
  counts and literal (unnormalized) train/test text-overlap counts, for the
  per-split reporting required in section 0.2 and the PR description. Never
  runs a model on test audio; a pure text/metadata comparison.
- `hash_file` / `git_commit_hash` / `code_file_hashes` / `manifest_source_breakdown`:
  run-provenance helpers wired into all three `train()` reports (code commit,
  per-file code hash, train/dev manifest+reference hashes, trainable parameter
  count, effective batch size, precision, training wall-clock seconds, and --
  for Whisper -- the per-source utterance/hour composition of the train
  manifest). This manifest composition is not a count of which samples have
  been drawn by a bounded training run.
- `PartialPredictionWriter`: all three `predict()` functions now stream to
  `predictions.partial.jsonl` and only rename to `predictions.jsonl` on a full,
  exact-ID-set run, mirroring `greek_asr.infer.run`'s existing contract. A
  failure partway leaves the partial file and a `run.json` with `status:
  "failed"` plus `completed_records`; it is never left looking like a
  complete benchmark file.

## Dev loss selection and common ASR scoring

The training loops save the best checkpoint using the mean of their
per-batch **teacher-forced dev losses**, computed from the supplied reference
targets. They do not generate dev predictions or log dev WER/CER during that
selection. Scoring normalization is not applied to this loss, and changing
the later scorer does not recompute or redefine a recorded dev loss.

Ozan owns the common WER/CER scoring policy for generated predictions,
including annotation tags, punctuation, case and accents. The final policy
must apply equally to every system's references and hypotheses while the
source files remain unchanged. If dev WER/CER or beam search is used for an
additional selection stage, produce real dev predictions and use that common
scorer; record the selection rule before the final test phase. See
`docs/PROTOCOL_DRAFT.md` and `docs/TEACHER_UPDATES_2026-11-18_TR.md`.

## Subset selection requires Serdar/Samed sign-off before use

`common.select_grouped_subset` exists and is tested (synthetic multi-speaker
fixture only), but it must not be invoked against real data on its own
authority: per section 7 of the main prompt, any training-budget subset has
to use "Serdar'ın kabul ettiği seed'li grup/oturum seçimi" coordinated with
the data owner (Samed). If a bounded subset turns out to be necessary once
real data lands, that coordination happens first, in the PR/issue thread, not
silently inside a config change.

## Known limitations / open risks

- No full-data training, real dev-based selection or final benchmark exists.
  GPU optimizer steps and train-pilot CLI checks at `0c13ce1` are recorded
  diagnostics, not a selected production checkpoint or accuracy result.
- The final training budget and batch sizes need measurements at accepted
  training-data lengths. RNN-T GPU execution is verified on the short pilot;
  its dense lattice memory and full-data throughput still need measurement.
- Different hardware, dependency changes or materially different sequence
  lengths require their own checks; the successful RX 7800 XT evidence is
  specific to its recorded code and runtime.
- CSS10 subset selection and the two Whisper conditions' matched-budget
  training have not been exercised on real multi-speaker data; only the
  config-level matching and the grouping logic itself are tested so far.
- Official HParl split speaker overlap (not speaker-disjoint) applies to
  these models exactly as it does to every other baseline; see
  `docs/TEACHER_UPDATES_2026-11-18_TR.md` and the PR description.
- No AMP/mixed-precision path is implemented for any of the three models;
  all training/report code currently assumes/reports `"precision": "float32"`.
  If Serdar enables AMP on the ROCm pilot, that is a code change (not just a
  runtime flag) and the reported precision field must change with it.
- `scripts/pilot_sequence_models.py` reports raw step time/VRAM, not a
  ready-made "epoch cost in hours" number -- see its own docstring/output for
  the throughput ratio Serdar combines with the real train manifest's hours.
