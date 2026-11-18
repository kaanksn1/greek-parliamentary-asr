# Adnan's sequence models: attention, RNN-T and Whisper fine-tuning

This covers the three model tasks assigned to Adnan Demir in `docs/TEAM.md`
and `docs/HANDOFF_ADNAN_TR.md`: a standard attention encoder-decoder (hybrid
CTC/attention included), a standard RNN-T transducer, and our own fine-tuning
of the pinned `openai/whisper-small` checkpoint under two training conditions.

**Status as of 19 November 2026: code, configs and tests are complete and
passing on CPU with synthetic/real-pilot-audio fixtures, including the fixes
from Serdar's real RX 7800 XT review of commit `0437631`
(`docs/reviews/PR2_PR5_FOLLOWUP_TR.md`, `docs/reviews/ADNAN_0437631_GPU.json`)
-- none of those fixes have themselves been re-verified on real GPU/ROCm
hardware yet; see "Known limitations" below. No full training, dev
selection or benchmark run has happened.** Samed's data PR #2 is not
merged (`docs/reviews/SAMED_PR2_REVIEW_TR.md`); `data/manifests/*.jsonl` do
not exist yet. The configs in `configs/sequence/` are prepared for when they
land, with `max_steps` deliberately left `null` (see "Training budget" below).
All full training/inference runs centrally on Serdar's RX 7800 XT/ROCm
machine, same as every other baseline (`docs/TEAM.md`).

## Toolkit choice

`ESPnet`, `SpeechBrain` and `WeNet` were the documented toolkit candidates in
`docs/HANDOFF_ADNAN_TR.md`. This implementation uses **plain PyTorch +
torchaudio** instead, for three concrete reasons:

1. All three heavy toolkits would need their own separate ROCm compatibility
   verification (the project already carries two such separate, hard-won
   environments for Transformers/NeMo; see `docs/RUNTIME_ENVIRONMENT.md` and
   `docs/NEMO_RUNTIME.md`). Adding a fourth large framework before any of
   them is verified on Serdar's AMD GPU is a real risk this project does not
   need, when the architectures themselves are simple enough to implement
   directly.
2. `torch`/`torchaudio` are already locked dependencies of Serdar's working
   ROCm environment (`requirements-rocm.lock.txt`: `torch==2.9.1+rocm7.2.1`,
   `torchaudio==2.9.0+rocm7.2.1`), so no new runtime has to be built for these
   two models.
3. A direct implementation keeps the full forward/backward math auditable by
   teammates (Kemal, Ozan) without needing to learn a third-party framework's
   abstractions, while still being a standard, citable recipe (see below) —
   "standard toolkit kullanılabilir... düz PyTorch/torchaudio ile belgeli
   reçete" is explicitly listed as an acceptable option.

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

**Now chosen: option (a) from the review's three candidates — a pure
PyTorch/autograd RNN-T loss, `greek_asr/models/rnnt_loss.py`.**

- Rationale for (a) over (b) GPU-encoder/CPU-loss or (c) a separately
  documented environment: it is built entirely from ordinary
  `torch.autograd` primitives (`log_softmax`, `gather`, `logaddexp`,
  elementwise add/compare) that every backend torch itself ships has a
  kernel for — CUDA or ROCm — so there is no backend-specific compiled op to
  be missing, by construction, rather than by a verification step that
  could later fail on different hardware.
- Correctness evidence (`tests/test_rnnt_loss.py`): value **and** gradient
  match `torchaudio.functional.rnnt_loss` on CPU across a hand-enumerated
  two-path lattice (solved independently, no library call) and several
  random variable-length batches (`torch==2.15.0+cpu`,
  `torchaudio==2.11.1+cpu`, this repo's `.venv-sequence`).
- `transducer.py`'s report now records `"rnnt_loss_backend":
  "pure_pytorch_logspace"` so the backend in use is explicit, not implied.
- **Not yet measured**: real throughput on an actual GPU/ROCm device. This
  implementation is a straightforward O(time × label) sequential-over-time,
  batch-vectorized forward pass — not the anti-diagonal/compiled-kernel
  optimization torchaudio's own op uses — and has only been timed on CPU
  with the repo's short pilot/test clips. `scripts/pilot_sequence_models.py`
  must be run once on Serdar's RX 7800 XT to record real step time/VRAM
  before any training-budget decision:

  ```bash
  HSA_ENABLE_DXG_DETECTION=1 "$P3_PYTHON" scripts/pilot_sequence_models.py \
    --device cuda --output outputs/sequence_gpu_pilot
  ```

  If this backend turns out to be too slow on real training-length audio,
  the documented fallback is still k2/icefall's pruned RNN-T loss (BSD-
  licensed, CPU+CUDA) — re-verify its license/ROCm support from its own
  repository before switching, and document the switch in the PR rather
  than silently downgrading to CTC.

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
- Loss: `torchaudio.functional.rnnt_loss` (see above).
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
- Matched training budget: both configs fix `micro_batch_size=8`,
  `gradient_accumulation_steps=2` (effective batch 16) and the same
  `max_steps` (optimizer updates) once set — never "same epoch count", since
  the two training sets have different sizes.
- Labels: `processor.tokenizer.set_prefix_tokens(language="el", task="transcribe")`
  then tokenize; `-100` is applied from the tokenizer's **attention mask**,
  not from a token-ID comparison, so the real end-of-sequence token is never
  mistaken for padding even when `pad_token_id == eos_token_id` (it is, for
  Whisper) — see `build_labels` and
  `tests/test_whisper_finetune.py::BuildLabelsTests`.
- A benign upstream warning ("attention mask is not set... pad token is same
  as eos token") appears during `generate()`; it matches the exact call
  pattern already used by `greek_asr/pretrained/whisper.py` (no explicit
  `attention_mask` passed either) and does not affect the single-sequence
  audio-conditioned generation used here.
- `model.generate(..., language="el", task="transcribe", do_sample=False)` is
  used for prediction, never fed the reference text.
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

## Training budget: deliberately left unset

`configs/sequence/*.json` set `max_steps: null`. All three `train()` functions
raise immediately (`"max_steps must be set from a measured pilot run..."`)
rather than silently assuming an arbitrary step count — see
`tests/test_models_attention.py`/`test_models_transducer.py::test_shipped_config_refuses_an_unmeasured_step_budget`
and `test_whisper_finetune.py::MatchedConditionConfigTests::test_shipped_configs_require_a_measured_step_budget_before_training`.
Serdar must run `scripts/pilot_sequence_models.py` (below) and combine the
measured step time/VRAM with the real train manifest's total hours (once
Samed's data is accepted) to choose a real, justified `max_steps`/`batch_size`
— not the three-pilot-clip smoke, and not an arbitrary round number.

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

A CPU smoke of the same script (not a GPU pilot, not committed) already
confirmed the code path end-to-end on this machine, including a real
forward/backward/optimizer step on the full real `openai/whisper-small`
checkpoint (not a stub): finite loss, ~11.4 s/step on CPU. Attention and
transducer CPU steps completed in under a second each. These CPU numbers are
not a substitute for the ROCm VRAM/step-time measurement Serdar must record.

## Environment

`requirements-sequence.txt` pins the exact CPU/Windows/Python 3.12.10
environment used for all synthetic/real-pilot-audio tests in this PR
(`torch==2.15.0+cpu`, `torchaudio==2.11.1+cpu`, `numpy==1.26.4`,
`transformers==4.57.6` — the `numpy`/`transformers` versions were deliberately
pinned down to match Serdar's already-validated Whisper/Qwen ROCm environment
in `requirements-rocm.lock.txt`, not left at the newest available release).
Build it with:

```powershell
py -3.12 -m venv .venv-sequence
.venv-sequence\Scripts\python -m pip install -r requirements-sequence.txt
```

For Serdar: no new environment is required for these three models beyond the
existing `project3-greek-asr-transformers` ROCm environment
(`docs/RUNTIME_ENVIRONMENT.md`) — `torch`, `torchaudio` and `transformers` are
already locked there at compatible versions. Do not install a CUDA wheel over
the ROCm build, and do not downgrade/upgrade that shared environment for this
PR; if a version conflict appears, report it in the PR rather than changing
the environment Whisper/Qwen inference already depends on.

## Shared utilities (`greek_asr/models/common.py`)

- `load_utterances`: joins a manifest and its reference file by ID, reusing
  `greek_asr.infer.load_manifest` for the audio-side hash/WAV-header/ID
  validation instead of duplicating it. Raises (never silently drops) on any
  missing/extra ID, and refuses the `test` split outright (`load_utterances`
  is only ever called for `train`/`dev`).
- `load_waveform`: decodes original PCM16 WAV bytes via the stdlib `wave`
  module (matching `greek_asr/pretrained/whisper.py`'s `_read_pcm16`, since
  this `torchaudio` version's `torchaudio.load` requires the separate
  `torchcodec` package) and resamples only in memory via
  `torchaudio.functional.resample`; the source file is never rewritten.
- `select_grouped_subset`: speaker/recording-grouped, seeded subset selection
  for a bounded training budget (section 0.2/7 of the main prompt) — a group
  is never split across the selection boundary; utterances with neither a
  speaker nor a recording ID fall back to utterance-level grouping, and the
  count of that fallback is reported in the returned diagnostics for the
  README. **Not yet run on real data** (no manifests exist yet); tested on a
  synthetic multi-speaker fixture in `tests/test_models_common.py`.
- `split_statistics` / `literal_overlap`: read-only speaker/utterance/hour
  counts and literal (unnormalized) train/test text-overlap counts, for the
  per-split reporting required in section 0.2 and the PR description. Never
  runs a model on test audio; a pure text/metadata comparison.
- `hash_file` / `git_commit_hash` / `code_file_hashes` / `manifest_source_breakdown`:
  run-provenance helpers wired into all three `train()` reports (code commit,
  per-file code hash, train/dev manifest+reference hashes, trainable parameter
  count, effective batch size, precision, training wall-clock seconds, and --
  for Whisper -- the actual per-source utterance/hour breakdown of the train
  manifest, so the CSS10 mixing ratio is a recorded number, not assumed).
- `PartialPredictionWriter`: all three `predict()` functions now stream to
  `predictions.partial.jsonl` and only rename to `predictions.jsonl` on a full,
  exact-ID-set run, mirroring `greek_asr.infer.run`'s existing contract. A
  failure partway leaves the partial file and a `run.json` with `status:
  "failed"` plus `completed_records`; it is never left looking like a
  complete benchmark file.

## Checkpoint selection depends on an unset normalization policy

Dev-based checkpoint/beam selection for all three models will be run against
whatever text the common scorer (`greek_asr/score.py`, owned by Ozan) and
normalization policy produce. That policy -- handling of `<spoken_noise>`,
punctuation, case, accents -- is explicitly **not yet finalized**
(`docs/PROTOCOL_DRAFT.md` step 5, `docs/TEACHER_UPDATES_2026-11-18_TR.md`
section 3). None of this PR's checkpoint-selection code invents its own
normalization rule or rewrites `greek_asr/scoring.py`; once Ozan's policy is
committed, the *same* dev loss/WER numbers this code already logs must be
recomputed under that policy before a real checkpoint is chosen -- the
current selection-by-dev-loss code path itself does not change.

## Subset selection requires Serdar/Samed sign-off before use

`common.select_grouped_subset` exists and is tested (synthetic multi-speaker
fixture only), but it must not be invoked against real data on its own
authority: per section 7 of the main prompt, any training-budget subset has
to use "Serdar'ın kabul ettiği seed'li grup/oturum seçimi" coordinated with
the data owner (Samed). If a bounded subset turns out to be necessary once
real data lands, that coordination happens first, in the PR/issue thread, not
silently inside a config change.

## Known limitations / open risks

- No real training, dev-based checkpoint/beam selection or benchmark exists.
  Everything above is implementation + synthetic/real-pilot-audio testing.
- `max_steps`/`batch_size` for all three configs need Serdar's GPU pilot
  measurement combined with the real train manifest's hours.
- The RNN-T loss (`greek_asr.models.rnnt_loss`, pure PyTorch/autograd) has
  only been timed on CPU; real GPU/ROCm throughput is unmeasured (see "RNN-T
  loss backend" above).
- The `--device cuda` plumbing added to attention/RNN-T train+predict (and
  the `padding_mask`/PCM32/Whisper-decoder-prefix fixes from Serdar's
  review) has only been exercised on CPU and with torch's "meta" device in
  this PR's own tests -- there is no GPU on this development machine. All of
  it needs the single pilot command in "GPU pilot package" re-run on
  Serdar's RX 7800 XT before it counts as verified.
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
